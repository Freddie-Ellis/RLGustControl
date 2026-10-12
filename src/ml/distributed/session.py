"""Reads a shared folder into what you want to see during a session: per-episode results, the latest episode in
detail, and the learner's progress. No Qt here; the Session tab only draws what this returns.

Reading is incremental: `poll()` only opens episodes it has not seen, so it stays cheap over a long session.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from torch import Tensor

from src.ml.distributed.files import (
    EpisodeMetaData,
    config_path,
    frame_to_transitions,
    latest_weights,
    learner_run_path,
    new_episodes,
    outbox_dir,
    read_config,
    read_episode,
    stop_path,
)
from src.ml.envs.wind_tunnel import WindTunnelParams
from src.ml.run_config import RunConfig, SACConfig
from src.ml.runs import RunDir, RunStatus, make_env

type EpisodeKind = Literal["warmup", "train", "eval"]


@dataclass(frozen=True)
class EpisodeSummary:
    """One point per episode on the session plots."""
    episode_id: int
    kind: EpisodeKind  # warmup = random actions, train = exploring policy, eval = deterministic policy
    version: int  # weights the actor used (-1 = none yet)
    ret: float  # return per env
    cm_rms: float | None  # None if the env has no Cm
    turbulence: float | None
    dt_mean: float  # control period achieved [s]
    dt_max: float
    n_transitions: int


@dataclass(frozen=True)
class EpisodeDetail:
    """Time series of one episode (the first env, if it ran several)."""
    meta: EpisodeMetaData
    t: Tensor  # [T] seconds from the start of the episode
    cm: Tensor | None  # [T]
    taps: Tensor | None  # [T, n_taps] pressures [Pa], None if the env has no taps
    act: Tensor  # [T, act_dim] commands in [-1, 1]


class SessionReader:
    def __init__(self, shared: Path) -> None:
        self.shared = shared
        self.cfg: RunConfig | None = None
        self.episodes: list[EpisodeSummary] = []
        self.latest: EpisodeDetail | None = None
        self._seen: set[str] = set()
        self._cm_col: int | None = None
        self._n_taps = 0  # tunnel obs = [taps / tap_scale_pa, previous commands]
        self._tap_scale = 1.0

    # ----- episodes -----
    def poll(self) -> int:
        """Read episodes written since the last call. Returns how many arrived."""
        if not self._load_config():
            return 0
        newest: tuple[EpisodeMetaData, pl.DataFrame] | None = None
        n_new = 0
        for path in new_episodes(self.shared, self._seen):
            try:
                meta, frame = read_episode(path)
            except (OSError, ValueError, pl.exceptions.PolarsError):
                continue  # caught mid-sync on a network share: try again next poll
            self._seen.add(path.name)
            self.episodes.append(self._summarise(meta, frame))
            newest = (meta, frame)
            n_new += 1
        if newest is not None:
            self.latest = self._detail(*newest)
        return n_new

    def _load_config(self) -> bool:
        if self.cfg is None and config_path(self.shared).exists():
            cfg = read_config(self.shared)
            env = make_env(cfg, n_envs=1)  # dimensions only: a tunnel env opens no hardware here
            self._cm_col = env.priv_names.index("cm") if "cm" in env.priv_names else None
            if isinstance(cfg.env.params, WindTunnelParams):
                self._n_taps = env.obs_dim - env.act_dim
                self._tap_scale = cfg.env.params.tap_scale_pa
            self.cfg = cfg
        return self.cfg is not None

    def _cm(self, next_priv: Tensor, n_envs: int) -> Tensor | None:
        """Cm measured during each step (priv after the step), first env only."""
        return None if self._cm_col is None else next_priv[::n_envs, self._cm_col]

    def _summarise(self, meta: EpisodeMetaData, frame: pl.DataFrame) -> EpisodeSummary:
        tr = frame_to_transitions(frame)
        cm = None if self._cm_col is None else tr["next_priv"][:, self._cm_col]
        return EpisodeSummary(
            episode_id=meta.episode_id,
            kind="eval" if meta.evaluate else ("warmup" if meta.policy_version < 0 else "train"),
            version=meta.policy_version,
            ret=float(tr["rew"].sum()) / meta.n_envs,
            cm_rms=None if cm is None else float(cm.pow(2).mean().sqrt()),
            turbulence=meta.turbulence_level,
            dt_mean=meta.loop_dt_mean,
            dt_max=meta.loop_dt_max,
            n_transitions=frame.height,
        )

    def _detail(self, meta: EpisodeMetaData, frame: pl.DataFrame) -> EpisodeDetail:
        tr = frame_to_transitions(frame)
        act = tr["act"][:: meta.n_envs]
        dt = self.cfg.env.spec.dt if self.cfg else 1.0
        t = (1 + torch.arange(act.shape[0])) * dt  # measured at the end of each step
        # next_obs, like Cm from next_priv: what was measured during each step, after its command went out.
        taps = tr["next_obs"][:: meta.n_envs, : self._n_taps] * self._tap_scale if self._n_taps else None
        return EpisodeDetail(meta=meta, t=t, cm=self._cm(tr["next_priv"], meta.n_envs), taps=taps, act=act)

    # ----- session facts -----
    @property
    def transitions(self) -> int:
        return sum(e.n_transitions for e in self.episodes)

    @property
    def warmup_transitions(self) -> int | None:
        return self.cfg.algo.warmup_transitions if self.cfg and isinstance(self.cfg.algo, SACConfig) else None

    @property
    def target_dt(self) -> float | None:
        return self.cfg.env.spec.dt if self.cfg else None

    @property
    def cm_trip(self) -> float | None:
        params = self.cfg.env.params if self.cfg else None
        return params.safety.cm_trip if isinstance(params, WindTunnelParams) else None

    def episode_seconds(self) -> float | None:
        """How long one episode should take, including the settle time (tunnel envs only)."""
        if self.cfg is None:
            return None
        params = self.cfg.env.params
        settle = params.settle_s if isinstance(params, WindTunnelParams) else 0.0
        return self.cfg.env.spec.horizon * self.cfg.env.spec.dt + settle

    def seconds_since_last_episode(self) -> float | None:
        files = list(outbox_dir(self.shared).glob("ep_*.parquet"))
        return time.time() - max(f.stat().st_mtime for f in files) if files else None

    def latest_version(self) -> int | None:
        found = latest_weights(self.shared)
        return found[0] if found else None

    # ----- learner -----
    def learner_run(self) -> RunDir | None:
        path = learner_run_path(self.shared)
        return RunDir(Path(path.read_text().strip())) if path.exists() else None

    def learner_status(self) -> RunStatus | None:
        run = self.learner_run()
        try:
            return run.read_status() if run else None
        except (OSError, ValueError):
            return None

    def learner_metrics(self) -> pl.DataFrame:
        run = self.learner_run()
        return run.read_metrics() if run else pl.DataFrame()

    # ----- stop -----
    def stop_requested(self) -> bool:
        return stop_path(self.shared).exists()

    def request_stop(self) -> None:
        stop_path(self.shared).touch()

    def clear_stop(self) -> None:
        stop_path(self.shared).unlink(missing_ok=True)
