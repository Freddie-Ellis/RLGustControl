"""On-disk format for training runs.

runs/<YYYYmmdd-HHMMSS>_<name>/   a training run (train.py)
runs/<session>/                  a distributed learner's run: named after its session and carried on when it resumes
    config.json   the exact RunConfig used
    metrics.csv   one row per iteration, appended while training (read with Polars)
    policy.pt     policy weights (state_dict)
    status.json   RunStatus: running / done / stopped / failed
    STOP          created by the UI to ask a running training process to stop
    output.log    everything the process printed, errors included (the distributed learner writes it, so its output
                  can be followed from another machine)
"""

import csv
import io
import math
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TextIO

import polars as pl
from pydantic import BaseModel

from src.config import CONFIG
from src.ml.atomic import atomic_write_text
from src.ml.envs.base import Env
from src.ml.policy import SquashedGaussianPolicy
from src.ml.registry import build_env
from src.ml.run_config import RunConfig

type RunState = Literal["running", "done", "stopped", "failed"]


class RunStatus(BaseModel):
    state: RunState = "running"
    started: datetime
    finished: datetime | None = None
    iterations_done: int = 0
    final_return: float | None = None
    error: str | None = None
    # Written every few seconds by a process that is alive (the distributed learner). A run that says "running"
    # with an old heartbeat was killed without a chance to say so. None for runs that do not write one.
    heartbeat: datetime | None = None
    info: dict[str, float] = {}  # the process's own progress numbers, e.g. the learner's backlog and buffer fill


class RunDir:
    """A handle on one run directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._metric_fields: list[str] | None = None  # metrics.csv's columns, once known

    @classmethod
    def create(cls, cfg: RunConfig, root: Path = CONFIG.runs_dir) -> RunDir:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(root / f"{stamp}_{cfg.name}")._init(cfg)

    @classmethod
    def open_named(cls, cfg: RunConfig, name: str, root: Path = CONFIG.runs_dir) -> RunDir:
        """`root/name`: created if new, otherwise carried on (marked running again; metrics keep appending).
        Raises ValueError if the existing run was made with a different config."""
        run = cls(root / name)
        if not run.status_path.exists():
            return run._init(cfg, exist_ok=True)  # an empty folder is fine to start in
        if run.read_config().model_dump() != cfg.model_dump():
            raise ValueError(f"{run.path} is a run with a different config; use another name or delete it")
        status = run.read_status()
        status.state, status.finished, status.error = "running", None, None
        run.write_status(status)
        return run

    def _init(self, cfg: RunConfig, exist_ok: bool = False) -> RunDir:
        self.path.mkdir(parents=True, exist_ok=exist_ok)
        atomic_write_text(self.config_path, cfg.model_dump_json(indent=2))
        self.write_status(RunStatus(started=datetime.now()))
        return self

    # ----- paths -----
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def config_path(self) -> Path:
        return self.path / "config.json"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.csv"

    @property
    def policy_path(self) -> Path:
        return self.path / "policy.pt"

    @property
    def status_path(self) -> Path:
        return self.path / "status.json"

    @property
    def stop_path(self) -> Path:
        return self.path / "STOP"

    @property
    def log_path(self) -> Path:
        return self.path / "output.log"

    def policy_version_path(self, version: int) -> Path:
        """Where the distributed learner keeps every weight version it published, for good: policies/ in the run."""
        return self.path / "policies" / f"policy_v{version:06d}.pt"

    def tee_output(self) -> None:
        """From now on, also write everything this process prints (stdout and stderr, so an uncaught traceback too)
        to output.log. Appends: a resumed run keeps one log, with a line marking each start."""
        # Line-buffered, so it can be read while the run goes on; open until the process ends, so no `with`.
        log = open(self.log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        log.write(f"--- started {datetime.now():%Y-%m-%d %H:%M:%S} on {socket.gethostname()} ---\n")
        sys.stdout = _Tee(sys.stdout, log)
        sys.stderr = _Tee(sys.stderr, log)

    # ----- read / write -----
    def read_config(self) -> RunConfig:
        return RunConfig.model_validate_json(self.config_path.read_text())

    def read_status(self) -> RunStatus:
        return RunStatus.model_validate_json(self.status_path.read_text())

    def write_status(self, status: RunStatus) -> None:
        text = status.model_dump_json(indent=2)
        try:
            atomic_write_text(self.status_path, text)
        except PermissionError:
            # A reader kept the file open through every try (Windows). Losing the status would be worse than a
            # reader catching it half-written once, which every reader already puts up with.
            self.status_path.write_text(text)

    def append_metrics(self, iteration: int, metrics: dict[str, float]) -> None:
        """Append one row. The first row of a new file sets the columns. Later rows follow the file's header: a
        column the file does not have is left out and one the row does not have is NaN, so a run carried on by
        newer code (which may log more) keeps a valid CSV."""
        row = {"iteration": iteration, **metrics}
        new_file = not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
        if self._metric_fields is None:
            if new_file:
                self._metric_fields = list(row)
            else:
                with open(self.metrics_path, newline="") as f:
                    self._metric_fields = next(csv.reader(f))
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._metric_fields, extrasaction="ignore", restval=math.nan)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def read_metrics(self) -> pl.DataFrame:
        if not self.metrics_path.exists():
            return pl.DataFrame()
        try:
            # Every metric is a number. Read as text then cast, rather than let Polars guess from the first rows: a
            # column that starts with `nan` (losses before the first update) would otherwise come back as strings.
            return pl.read_csv(self.metrics_path, infer_schema=False).with_columns(pl.all().cast(pl.Float64))
        except (pl.exceptions.ComputeError, pl.exceptions.NoDataError, pl.exceptions.InvalidOperationError):
            return pl.DataFrame()  # file caught mid-write by a running training process

    def request_stop(self) -> None:
        self.stop_path.touch()

    def stop_requested(self) -> bool:
        return self.stop_path.exists()


class _Tee(io.TextIOBase):
    """A text stream that writes to several others, e.g. the console and a log file."""

    def __init__(self, *streams: TextIO) -> None:
        super().__init__()
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty()


@dataclass
class RunSummary:
    run: RunDir
    config: RunConfig
    status: RunStatus


def list_runs(root: Path = CONFIG.runs_dir) -> list[RunSummary]:
    """All readable runs, newest first. Folders with missing or invalid files are skipped."""
    if not root.exists():
        return []
    out: list[RunSummary] = []
    for path in root.iterdir():
        run = RunDir(path)
        try:
            out.append(RunSummary(run, run.read_config(), run.read_status()))
        except (OSError, ValueError):
            continue
    # By start time, not name: session-named runs have no timestamp in their name.
    return sorted(out, key=lambda s: s.status.started.replace(tzinfo=None), reverse=True)


def make_env(cfg: RunConfig, n_envs: int, seed: int | None = None) -> Env:
    return build_env(cfg.env, n_envs=n_envs, seed=seed)


@dataclass(frozen=True)
class EnvDims:
    """The sizes of a run's env: all that anything but the process stepping it needs to know about it."""
    obs_dim: int
    act_dim: int
    priv_dim: int
    priv_names: tuple[str, ...]


def env_dims(cfg: RunConfig) -> EnvDims:
    """The sizes of `cfg`'s env, for building networks and reading episodes. The one place an env is built without
    being run: it is never reset, so a wind tunnel env opens no hardware here, and it is thrown away at once. Only
    the process that steps the env (the actor, or train.py) calls `make_env` and keeps what it gets."""
    env = make_env(cfg, n_envs=1)
    return EnvDims(env.obs_dim, env.act_dim, env.priv_dim, env.priv_names)


def make_policy(cfg: RunConfig) -> SquashedGaussianPolicy:
    env = env_dims(cfg)
    return SquashedGaussianPolicy(env.obs_dim, env.act_dim, hidden=cfg.policy.hidden, init_log_std=cfg.policy.init_log_std)


@dataclass
class LoadedRun:
    run: RunDir
    config: RunConfig
    status: RunStatus
    policy: SquashedGaussianPolicy
    metrics: pl.DataFrame


def load_run(path: Path) -> LoadedRun:
    """Rebuild the policy from config.json and load its saved weights."""
    run = RunDir(path)
    cfg = run.read_config()
    policy = make_policy(cfg)
    policy.load(run.policy_path)
    return LoadedRun(run, cfg, run.read_status(), policy, run.read_metrics())
