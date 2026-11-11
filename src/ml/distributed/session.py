"""Reads a shared folder into what you want to see during a session: per-episode results, the latest episode in
detail, the rig's health, and the learner's progress. No Qt here; the Session tab only draws what this returns.

Reading is incremental: `poll()` only opens episodes it has not seen, so it stays cheap over a long session.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from torch import Tensor

from src.ml.distributed.files import (
    EpisodeMetaData,
    SyncStatus,
    config_path,
    frame_extras,
    frame_to_transitions,
    latest_weights,
    learner_is_elsewhere,
    learner_run_dir,
    new_episodes,
    outbox_dir,
    read_config,
    read_episode,
    read_sync_status,
    stop_path,
)
from src.ml.envs.wind_tunnel import WindTunnelParams
from src.ml.run_config import RunConfig
from src.ml.runs import RunDir, RunStatus, env_dims

type EpisodeKind = Literal["warmup", "train", "eval", "baseline"]
type ActorState = Literal["stopping", "none", "running", "idle"]
type LearnerState = Literal["not found", "running", "silent", "done", "stopped", "failed"]

ACTOR_IDLE_MARGIN_S = 5.0  # the actor counts as idle once two episodes' worth of time, plus this, has passed
HEARTBEAT_SILENT_S = 30.0  # a "running" learner whose last heartbeat is older than this has probably been killed


@dataclass(frozen=True)
class EpisodeSummary:
    """One point per episode on the session plots."""
    episode_id: int
    kind: EpisodeKind  # warmup = random actions, train = exploring policy, eval = mean action, baseline = neutral
    version: int  # weights the actor used (-1 = none yet)
    ret: float  # return per env
    cm_rms: float | None  # None if the env has no Cm
    dt_mean: float  # control period achieved [s]
    dt_max: float
    n_transitions: int
    terminated: bool  # ended early on a safety trip
    info: dict[str, float]  # Env.episode_info: missed ticks, lateness, DAQ samples, limiter share, ...


@dataclass(frozen=True)
class EpisodeDetail:
    """Time series of one episode (the first env, if it ran several)."""
    meta: EpisodeMetaData
    t: Tensor  # [T] seconds from the start of the episode
    cm: Tensor | None  # [T]
    taps: Tensor | None  # [T, n_taps] pressures [Pa], None if the env has no taps
    act: Tensor  # [T, act_dim] what the policy asked for, in [-1, 1]
    applied: Tensor | None  # [T, act_dim] what the surfaces were sent after the safety layer; None if not known


@dataclass(frozen=True)
class Health:
    """The rig's health over the episodes read so far. `last_*` are the latest episode's; None = not recorded."""
    trips: int  # episodes ended by the safety layer
    sensor_faults: int  # of those, the ones tripped by a reading that was not a number
    missed_ticks: int  # control ticks skipped because a step overran, over the whole session
    last_late_ms: float | None  # worst tick lateness
    last_samples_min: float | None  # fewest DAQ samples in one tick (0 = the DAQ stalled)
    last_limited_frac: float | None  # share of surface commands the safety layer changed
    last_limit_gap: float | None  # mean |applied - requested| over all surface commands, in action units (range 2)


@dataclass(frozen=True)
class LearnerHealth:
    state: LearnerState  # "silent" = says running, but no heartbeat lately
    elsewhere: bool  # on another machine (the HPC): what is known of it is as old as the last sync
    heartbeat_age_s: float | None  # None if it has never written one
    backlog: float | None  # updates the data has earned that are not done yet
    buffer: float | None  # transitions in the replay buffer
    buffer_capacity: float | None
    skipped_episodes: float | None  # episodes it refused (values that are not numbers)
    error: str | None


def _age_s(when: datetime) -> float:
    """Seconds since `when`, whether it was written with a time zone or without (then: this machine's)."""
    now = datetime.now().astimezone() if when.tzinfo else datetime.now()
    return (now - when).total_seconds()


class SessionReader:
    def __init__(self, shared: Path) -> None:
        self.shared = shared
        self.cfg: RunConfig | None = None
        self.episodes: list[EpisodeSummary] = []
        self.latest: EpisodeDetail | None = None
        self._seen: set[str] = set()
        self._cm_col: int | None = None
        # Tunnel obs frame = [taps / tap_scale_pa, commands in effect]; with history, the newest frame is last.
        self._n_taps = 0
        self._n_surfaces = 0
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
            env = env_dims(cfg)
            self._cm_col = env.priv_names.index("cm") if "cm" in env.priv_names else None
            if isinstance(cfg.env.params, WindTunnelParams):
                self._n_taps = cfg.env.params.rig.n_taps
                self._n_surfaces = cfg.env.params.rig.n_surfaces
                self._tap_scale = cfg.env.params.tap_scale_pa
            self.cfg = cfg
        return self.cfg is not None

    def _cm(self, frame: pl.DataFrame, next_priv: Tensor) -> Tensor | None:
        """Raw Cm measured during each step, every row. The env records it as an extra; episodes from before it did
        only have the critic's copy (priv after the step), which was not scaled then."""
        extras = frame_extras(frame)
        if "cm" in extras:
            return extras["cm"]
        return None if self._cm_col is None else next_priv[:, self._cm_col]

    def _summarise(self, meta: EpisodeMetaData, frame: pl.DataFrame) -> EpisodeSummary:
        tr = frame_to_transitions(frame)
        cm = self._cm(frame, tr["next_priv"])
        kind: EpisodeKind = (
            "baseline" if meta.baseline else "eval" if meta.evaluate else "warmup" if meta.policy_version < 0 else "train"
        )
        return EpisodeSummary(
            episode_id=meta.episode_id,
            kind=kind,
            version=meta.policy_version,
            ret=float(tr["rew"].sum()) / meta.n_envs,
            cm_rms=None if cm is None else float(cm.pow(2).mean().sqrt()),
            dt_mean=meta.loop_dt_mean,
            dt_max=meta.loop_dt_max,
            n_transitions=frame.height,
            terminated=meta.terminated,
            info=meta.info,
        )

    def _detail(self, meta: EpisodeMetaData, frame: pl.DataFrame) -> EpisodeDetail:
        tr = frame_to_transitions(frame)
        n = meta.n_envs
        act = tr["act"][::n]
        dt = self.cfg.env.spec.dt if self.cfg else 1.0
        t = (1 + torch.arange(act.shape[0])) * dt  # measured at the end of each step
        cm = self._cm(frame, tr["next_priv"])
        taps: Tensor | None = None
        applied: Tensor | None = None
        if self._n_taps:
            # next_obs, like Cm: what was measured during each step, after its command went out. Its newest frame
            # also holds the command then in effect, i.e. what the safety layer let through.
            newest = tr["next_obs"][::n, -(self._n_taps + self._n_surfaces):]
            taps = newest[:, : self._n_taps] * self._tap_scale
            applied = newest[:, self._n_taps:]
        return EpisodeDetail(meta=meta, t=t, cm=None if cm is None else cm[::n], taps=taps, act=act, applied=applied)

    # ----- session facts -----
    @property
    def transitions(self) -> int:
        return sum(e.n_transitions for e in self.episodes)

    @property
    def warmup_transitions(self) -> int | None:
        return self.cfg.algo.warmup_transitions if self.cfg else None

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

    def actor_state(self) -> tuple[ActorState, float | None]:
        """What the actor seems to be doing, judged from its episode files, and the seconds since its last one."""
        since = self.seconds_since_last_episode()
        if self.stop_requested():
            return "stopping", since
        if since is None:
            return "none", None
        expected = self.episode_seconds() or 0.0
        return ("running" if since < 2 * expected + ACTOR_IDLE_MARGIN_S else "idle"), since

    def latest_version(self) -> int | None:
        found = latest_weights(self.shared)
        return found[0] if found else None

    def health(self) -> Health:
        last = self.episodes[-1].info if self.episodes else {}
        return Health(
            trips=sum(e.terminated for e in self.episodes),
            sensor_faults=sum(bool(e.info.get("sensor_fault")) for e in self.episodes),
            missed_ticks=int(sum(e.info.get("missed_ticks", 0.0) for e in self.episodes)),
            last_late_ms=last.get("late_ms_max"),
            last_samples_min=last.get("samples_min"),
            last_limited_frac=last.get("limited_frac"),
            last_limit_gap=last.get("limit_gap"),
        )

    def expected_samples_per_tick(self) -> float | None:
        """DAQ samples one control period should hold (mock rig only: a real DAQ's rate is its backend's to say)."""
        params = self.cfg.env.params if self.cfg else None
        if not isinstance(params, WindTunnelParams) or self.cfg is None:
            return None
        return params.mock.daq_rate_hz * self.cfg.env.spec.dt if params.io == "mock" else None

    # ----- learner -----
    def learner_run(self) -> RunDir | None:
        path = learner_run_dir(self.shared)
        return RunDir(path) if path else None

    def learner_status(self) -> RunStatus | None:
        run = self.learner_run()
        try:
            return run.read_status() if run else None
        except (OSError, ValueError):
            return None

    def learner_health(self) -> LearnerHealth:
        """The learner's state and progress, from its run's status. A learner that says it is running but has not
        written a heartbeat for HEARTBEAT_SILENT_S is "silent": killed, or (on the HPC) out of reach of the sync."""
        status = self.learner_status()
        elsewhere = learner_is_elsewhere(self.shared)
        if status is None:
            return LearnerHealth("not found", elsewhere, None, None, None, None, None, None)
        age = _age_s(status.heartbeat) if status.heartbeat else None
        state: LearnerState = status.state
        if state == "running" and age is not None and age > HEARTBEAT_SILENT_S:
            state = "silent"
        info = status.info
        return LearnerHealth(
            state=state,
            elsewhere=elsewhere,
            heartbeat_age_s=age,
            backlog=info.get("backlog"),
            buffer=info.get("buffer"),
            buffer_capacity=info.get("buffer_capacity"),
            skipped_episodes=info.get("skipped_episodes"),
            error=status.error,
        )

    def learner_metrics(self) -> pl.DataFrame:
        run = self.learner_run()
        return run.read_metrics() if run else pl.DataFrame()

    # ----- sync -----
    def sync_status(self) -> SyncStatus | None:
        """How the link to the HPC is doing; None if no sync has run on this folder."""
        return read_sync_status(self.shared)

    def sync_age_s(self) -> float | None:
        """Seconds since the sync last reached the HPC; None if it never has."""
        status = self.sync_status()
        return _age_s(status.last_ok) if status and status.last_ok else None

    # ----- stop -----
    def stop_requested(self) -> bool:
        return stop_path(self.shared).exists()

    def request_stop(self) -> None:
        stop_path(self.shared).touch()

    def clear_stop(self) -> None:
        stop_path(self.shared).unlink(missing_ok=True)
