"""On-disk format for training runs.

runs/<YYYYmmdd-HHMMSS>_<name>/
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
from src.ml.policies.base import Policy
from src.ml.registry import build_env, build_policy
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
        run = cls(root / f"{stamp}_{cfg.name}")
        run.path.mkdir(parents=True, exist_ok=False)
        run.config_path.write_text(cfg.model_dump_json(indent=2))
        run.write_status(RunStatus(started=datetime.now()))
        return run

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
            return pl.read_csv(self.metrics_path)
        except (pl.exceptions.ComputeError, pl.exceptions.NoDataError):
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
    for path in sorted(root.iterdir(), reverse=True):
        run = RunDir(path)
        try:
            out.append(RunSummary(run, run.read_config(), run.read_status()))
        except (OSError, ValueError):
            continue
    return out


def make_env(cfg: RunConfig, n_envs: int, seed: int | None = None) -> Env:
    return build_env(cfg.env, n_envs=n_envs, seed=seed)


def make_policy(cfg: RunConfig) -> Policy:
    env = make_env(cfg, n_envs=1)
    return build_policy(cfg.policy, env.obs_dim, env.act_dim)


@dataclass
class LoadedRun:
    run: RunDir
    config: RunConfig
    status: RunStatus
    policy: Policy
    metrics: pl.DataFrame


def load_run(path: Path) -> LoadedRun:
    """Rebuild the policy from config.json and load its saved weights."""
    run = RunDir(path)
    cfg = run.read_config()
    policy = make_policy(cfg)
    policy.load(run.policy_path)
    return LoadedRun(run, cfg, run.read_status(), policy, run.read_metrics())
