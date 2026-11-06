"""On-disk format for training runs.

runs/<YYYYmmdd-HHMMSS>_<name>/   a training run (train.py)
runs/<session>/                  a distributed learner's run: named after its session and carried on when it resumes
    config.json   the exact RunConfig used
    metrics.csv   one row per iteration, appended while training (read with Polars)
    policy.pt     policy weights (state_dict)
    status.json   RunStatus: running / done / stopped / failed
    STOP          created by the UI to ask a running training process to stop
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel

from src.config import CONFIG
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


class RunDir:
    """A handle on one run directory."""

    def __init__(self, path: Path) -> None:
        self.path = path

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
        self.config_path.write_text(cfg.model_dump_json(indent=2))
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

    # ----- read / write -----
    def read_config(self) -> RunConfig:
        return RunConfig.model_validate_json(self.config_path.read_text())

    def read_status(self) -> RunStatus:
        return RunStatus.model_validate_json(self.status_path.read_text())

    def write_status(self, status: RunStatus) -> None:
        self.status_path.write_text(status.model_dump_json(indent=2))

    def append_metrics(self, iteration: int, metrics: dict[str, float]) -> None:
        """Append one row. Writes the header on the first call; columns must stay the same afterwards."""
        row = {"iteration": iteration, **metrics}
        new_file = not self.metrics_path.exists()
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
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


def make_policy(cfg: RunConfig) -> SquashedGaussianPolicy:
    env = make_env(cfg, n_envs=1)
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
