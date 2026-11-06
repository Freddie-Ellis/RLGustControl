"""The shared folder between the tunnel actor and the laptop learner. The ONLY way the two talk.

    shared/
        config.json                 RunConfig, written by the learner, read by the actor
        outbox/ep_000042.json       EpisodeMetaData        (written first)
        outbox/ep_000042.parquet    the episode's transitions (written last = "episode complete")
        weights/policy_v000007.pt   policy weights, newest version wins
        learner_state.pt            the learner's full SAC state (critics, alpha, optimisers), for resuming
        STOP                        create to stop the actor after its current episode
        learner_run.txt             path of the learner's runs/ directory (for the Session tab)

Every file is written to `<name>.tmp` and then renamed, so a reader never sees a half-written file even over a
network share. Keep the shared folder outside OneDrive: sync can lock or duplicate files mid-write.

With the learner on the HPC, each side has its own copy of this folder and `sync.py` (on the tunnel PC) mirrors them,
plus the learner's run directory. Only learner_state.pt stays on the HPC: only the learner needs it.
"""

import re
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import polars as pl
import torch
from pydantic import BaseModel
from torch import Tensor

from src.config import CONFIG
from src.ml.policy import SquashedGaussianPolicy
from src.ml.run_config import RunConfig

TRANSITION_COLUMNS = ("obs", "priv", "act", "rew", "next_obs", "next_priv", "done")
_WEIGHTS_RE = re.compile(r"policy_v(\d+)\.pt")

# ----- where sessions live: the one place to change it -----
# A session is a shared folder with the same name on both machines: LOCAL_SESSIONS/<name> and HPC_SESSIONS/<name>.
LOCAL_SESSIONS = Path(r"C:\rl_shared")  # tunnel PC (and laptop). Never inside OneDrive.
HPC_SESSIONS = PurePosixPath("/scratch/fe1g22/wt_testing")  # the learner's copies on Iridis
SESSION = "test"  # used when a script is run without a session name
_SESSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # a plain folder name, safe in paths and sftp commands


def _checked(session: str) -> str:
    if not _SESSION_RE.fullmatch(session):
        raise ValueError(f"session name {session!r}: use letters, digits, '_', '.' and '-' only")
    return session


def local_shared(session: str) -> Path:
    return LOCAL_SESSIONS / _checked(session)


def hpc_shared(session: str) -> PurePosixPath:
    return HPC_SESSIONS / _checked(session)


def shared(session: str) -> Path:
    """This machine's copy of the session's folder: Linux = the HPC."""
    return Path(hpc_shared(session)) if sys.platform == "linux" else local_shared(session)


def local_sessions() -> list[str]:
    """Names of the sessions that already have a folder here, newest first."""
    if not LOCAL_SESSIONS.is_dir():
        return []
    folders = [p for p in LOCAL_SESSIONS.iterdir() if p.is_dir() and _SESSION_RE.fullmatch(p.name)]
    return [p.name for p in sorted(folders, key=lambda p: p.stat().st_mtime, reverse=True)]


# ----- layout -----
def outbox_dir(shared: Path) -> Path:
    return shared / "outbox"


def weights_dir(shared: Path) -> Path:
    return shared / "weights"


def config_path(shared: Path) -> Path:
    return shared / "config.json"


def stop_path(shared: Path) -> Path:
    return shared / "STOP"


def learner_run_path(shared: Path) -> Path:
    return shared / "learner_run.txt"


def learner_run_dir(shared: Path, runs_dir: Path = CONFIG.runs_dir) -> Path | None:
    """The learner's run directory, or None before it has started. If the learner ran on another machine (the HPC),
    the path it wrote does not exist here, and the copy `sync.py` keeps is `runs_dir/<same name>` instead."""
    path = learner_run_path(shared)
    if not path.exists():
        return None
    written = path.read_text().strip()
    return Path(written) if Path(written).exists() else runs_dir / PurePosixPath(written.replace("\\", "/")).name


def learner_state_path(shared: Path) -> Path:
    return shared / "learner_state.pt"


def init_shared(shared: Path, cfg: RunConfig) -> None:
    """Create the folder layout and write the run config. Called by the learner when a session starts."""
    outbox_dir(shared).mkdir(parents=True, exist_ok=True)
    weights_dir(shared).mkdir(parents=True, exist_ok=True)
    atomic_write_text(config_path(shared), cfg.model_dump_json(indent=2))


def read_config(shared: Path) -> RunConfig:
    return RunConfig.model_validate_json(config_path(shared).read_text())


# ----- atomic writes -----
def tmp_path(path: Path) -> Path:
    """`ep_1.json` -> `ep_1.json.tmp`. Keeping the full name stops ep_1.json and ep_1.parquet sharing a tmp file."""
    return path.with_name(path.name + ".tmp")


def atomic_write_text(path: Path, text: str) -> None:
    tmp = tmp_path(path)
    tmp.write_text(text)
    tmp.replace(path)


# ----- episodes -----
class EpisodeMetaData(BaseModel):
    """Everything about one episode except the transitions themselves."""

    episode_id: int
    policy_version: int  # -1 = warmup with random actions
    session_id: str  # one tunnel session; lets the learner drop a session if the rig drifted
    evaluate: bool  # deterministic policy (no exploration), for measuring Cm suppression
    n_envs: int  # 1 in the tunnel
    n_steps: int
    loop_dt_mean: float  # control loop period actually achieved [s]
    loop_dt_max: float  # worst period: a deadline miss shows up here
    turbulence_level: float | None = None  # turbulence device setting, if the env has one
    created: datetime


def transitions_to_frame(
    obs: Tensor, priv: Tensor, act: Tensor, rew: Tensor, next_obs: Tensor, next_priv: Tensor, done: Tensor
) -> pl.DataFrame:
    """One row per transition. Vector fields become fixed-width Array columns. Inputs are [M, ...] on the CPU."""
    values = (obs, priv, act, rew, next_obs, next_priv, done)
    return pl.DataFrame([pl.Series(name, v.float().numpy()) for name, v in zip(TRANSITION_COLUMNS, values)])


def frame_to_transitions(frame: pl.DataFrame) -> dict[str, Tensor]:
    """Inverse of `transitions_to_frame`: column name -> tensor, ready for `ReplayBuffer.add`."""
    return {name: torch.from_numpy(frame[name].to_numpy().copy()) for name in TRANSITION_COLUMNS}


def write_episode(shared: Path, meta: EpisodeMetaData, frame: pl.DataFrame) -> None:
    """Write metadata, then transitions. The parquet appearing is what marks the episode complete. Raises on error."""
    stem = outbox_dir(shared) / f"ep_{meta.episode_id:06d}"
    atomic_write_text(stem.with_suffix(".json"), meta.model_dump_json(indent=2))
    parquet = stem.with_suffix(".parquet")
    frame.write_parquet(tmp_path(parquet), compression="zstd")
    tmp_path(parquet).replace(parquet)


def new_episodes(shared: Path, seen: set[str]) -> list[Path]:
    """Completed episodes (by parquet path) not in `seen`, oldest first. Half-written `.tmp` files are ignored."""
    return sorted(p for p in outbox_dir(shared).glob("ep_*.parquet") if p.name not in seen)


def read_episode(parquet: Path) -> tuple[EpisodeMetaData, pl.DataFrame]:
    meta = EpisodeMetaData.model_validate_json(parquet.with_suffix(".json").read_text())
    return meta, pl.read_parquet(parquet)


# ----- weights -----
def write_weights(shared: Path, version: int, policy: SquashedGaussianPolicy) -> Path:
    """Save CPU copies of the weights, so the tunnel PC (no GPU) can load them."""
    path = weights_dir(shared) / f"policy_v{version:06d}.pt"
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, tmp_path(path))
    tmp_path(path).replace(path)
    return path


def write_learner_state(shared: Path, state: dict[str, Any]) -> None:
    """Only the learner reads this back, so tensors stay on its device."""
    path = learner_state_path(shared)
    torch.save(state, tmp_path(path))
    tmp_path(path).replace(path)


def read_learner_state(shared: Path, device: torch.device) -> dict[str, Any] | None:
    path = learner_state_path(shared)
    return torch.load(path, map_location=device) if path.exists() else None


def weights_version(name: str) -> int | None:
    """`policy_v000007.pt` -> 7. None for anything else, including half-written `.tmp` files."""
    m = _WEIGHTS_RE.fullmatch(name)
    return int(m.group(1)) if m else None


def latest_weights(shared: Path) -> tuple[int, Path] | None:
    """(version, path) of the newest weights, or None if the learner has not published any yet."""
    found = [(v, p) for p in weights_dir(shared).glob("policy_v*.pt") if (v := weights_version(p.name)) is not None]
    return max(found) if found else None
