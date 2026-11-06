"""The learner: trains SAC on episodes the actor writes to the shared folder and publishes new weights.

    uv run python -m src.ml.distributed.learner [session] [config]     e.g. ... learner test sac_tunnel_mock
    on the HPC:   sbatch slurm/learner.slurm <session> [config]

The config belongs to the session (see `session_config`): a new session needs one (a name from configs/ or a .toml
path), and an existing session carries on with the config.json it started with.

Runs on the laptop (GPU if available). Never talks to the actor directly, only through the shared folder:

    outbox/ep_*.parquet  --ingest-->  replay buffer  --SAC.update-->  policy  --publish-->  weights/policy_v*.pt

Learning is tied to data: each env step received earns `updates_per_step` SAC updates. While the tunnel resets and
nothing arrives, the learner waits instead of re-fitting the same buffer over and over (which overfits).

Restart-safe: on start it re-reads every episode already in the outbox and carries on from the last saved learner
state (policy, critics, alpha, optimisers, update count), so stopping the laptop loses no tunnel data and no learning
beyond the updates since the last publish. Its run directory is runs/<session> (the shared folder's name), so a
resumed session keeps one run directory and one metrics.csv.
"""

import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from src.config import CONFIG
from src.ml.algos.replay import ReplayBuffer
from src.ml.algos.sac import SAC
from src.ml.distributed.files import (
    SESSION,
    atomic_write_text,
    config_path,
    frame_to_transitions,
    init_shared,
    latest_weights,
    learner_is_elsewhere,
    learner_run_dir,
    learner_run_path,
    new_episodes,
    read_config,
    read_episode,
    read_learner_state,
    shared,
    stop_path,
    write_learner_state,
    write_weights,
)
from src.ml.run_config import RunConfig, SACConfig
from src.ml.runs import RunDir, make_env, make_policy

UPDATE_CHUNK = 50  # updates between checks for new episodes / STOP
UPDATE_METRICS = ("critic_loss", "actor_loss", "alpha", "entropy", "q_mean")  # what SAC.update reports
IDLE_SLEEP_S = 0.2  # wait when there is no data and no update budget


class Learner:
    def __init__(self, cfg: RunConfig, shared: Path, run: RunDir) -> None:
        self.cfg = cfg
        self.algo_cfg: SACConfig = cfg.algo
        self.shared = shared
        self.run_dir = run

        # Building the env only reads its dimensions; a wind tunnel env opens no hardware until reset().
        env = make_env(cfg, n_envs=1)
        self.priv_dim = env.priv_dim if self.algo_cfg.critic_privileged else 0
        self.policy = make_policy(cfg)
        self.sac = SAC(self.algo_cfg)
        self.sac.setup(self.policy, self.priv_dim)
        self.version = 0
        self.updates = 0
        # Resume a session: run_learner only refuses a config.json that differs from this config, and
        # session_config hands an existing session its own config, so the same session always resumes here.
        if (state := read_learner_state(shared, self.sac.device)) is not None:
            self.sac.load_state_dict(state["sac"])
            self.version, self.updates = state["version"], state["updates"]
            print(f"resuming from v{self.version} after {self.updates} updates (critics and optimisers restored)",
                  flush=True)
        elif (latest := latest_weights(shared)) is not None:  # a session from before learner state was saved
            self.version, path = latest
            self.policy.load(path)
            print(f"resuming policy from weights v{self.version}; no learner state, so the critics start fresh",
                  flush=True)
        self.sac.policy.train()
        self.buffer = ReplayBuffer(
            self.algo_cfg.buffer_size, env.obs_dim, self.priv_dim, env.act_dim, self.sac.device
        )

        self.seen: set[str] = set()
        self.env_steps = 0  # control steps received (each gives n_envs transitions); earns update budget
        self.transitions = 0  # rows received; warmup is counted in these, as in SAC.train
        self._since_log: dict[str, list[float]] = {}  # metrics accumulated since the last log row
        # Data the run's metrics already cover: a resumed session's run directory has rows up to here, so re-reading
        # its outbox on start only logs what came after (see ingest).
        logged = run.read_metrics()
        has_rows = "env_steps" in logged.columns and logged.height > 0
        self.logged_env_steps = int(logged["env_steps"].to_numpy().max()) if has_rows else 0

    # ----- data -----
    def ingest(self, parquets: list[Path], backfill: bool = False) -> int:
        """Add episodes to the replay buffer. Returns how many new env steps arrived.

        Every episode gets a metrics row of its own (its return, never averaged with others), unless the run's metrics
        already have it: a resumed session re-reads its whole outbox on start. `backfill` = data that was there before
        this learner started, whose staleness means nothing now."""
        new_steps = 0
        for path in parquets:
            meta, frame = read_episode(path)
            tr = frame_to_transitions(frame)
            priv, next_priv = tr["priv"][:, : self.priv_dim], tr["next_priv"][:, : self.priv_dim]
            self.buffer.add(tr["obs"], priv, tr["act"], tr["rew"], tr["next_obs"], next_priv, tr["done"])
            self.seen.add(path.name)
            new_steps += meta.n_steps
            self.env_steps += meta.n_steps
            self.transitions += frame.height

            if self.env_steps <= self.logged_env_steps:
                continue  # already in the metrics
            ret = float(tr["rew"].sum()) / meta.n_envs
            stale = meta.policy_version >= 0 and not backfill  # versions behind when ingested; old data: meaningless
            self._write_row({
                "eval_return" if meta.evaluate else "train_return": ret,
                "staleness": float(self.version - meta.policy_version) if stale else math.nan,
            })
            self.logged_env_steps = self.env_steps
        return new_steps

    # ----- learning -----
    def in_warmup(self) -> bool:
        """Still collecting the warmup data: no SAC updates yet."""
        return self.transitions < self.algo_cfg.warmup_transitions or len(self.buffer) < self.algo_cfg.batch_size

    def update_budget(self) -> int:
        """Updates still allowed for the data received so far (zero until the warmup data is in)."""
        if self.in_warmup():
            return 0
        return self.algo_cfg.updates_per_step * self.env_steps - self.updates

    def learn(self, n: int) -> None:
        for _ in range(n):
            for k, v in self.sac.update(self.buffer.sample(self.algo_cfg.batch_size)).items():
                self._note(k, v)
            self.updates += 1
            if self.updates % self.cfg.distributed.publish_every == 0:
                self.publish()
            if self.updates % self.cfg.distributed.log_every == 0:
                self.log_updates()

    def publish(self) -> None:
        """New weights for the actor, which picks them up before its next episode."""
        self.version += 1
        # State before weights: if we die in between, the resume just skips this version number, which is harmless.
        # Restoring `updates` too stops a resume re-spending the whole outbox's update budget on old data.
        write_learner_state(self.shared, {"version": self.version, "updates": self.updates, "sac": self.sac.state_dict()})
        write_weights(self.shared, self.version, self.policy)
        self.policy.save(self.run_dir.policy_path)
        print(f"published v{self.version} after {self.updates} updates, {self.env_steps} env steps", flush=True)

    # ----- logging -----
    def _note(self, key: str, value: float) -> None:
        self._since_log.setdefault(key, []).append(value)

    def log_updates(self) -> None:
        """A row of SAC's own metrics: the mean over the `log_every` updates since the last one (one row per update
        would be millions of rows). Returns are in the episode rows, not here."""
        means = {k: sum(v) / len(v) for k, v in self._since_log.items() if v}
        self._write_row({k: means.get(k, math.nan) for k in UPDATE_METRICS})
        self._since_log = {}

    def _write_row(self, values: dict[str, float]) -> None:
        """One metrics row, keyed by update count. There are two kinds: an episode row (its return) and an update row
        (SAC's metrics); each leaves the other kind's columns NaN. Columns are fixed so the CSV stays valid."""
        row = {
            "train_return": math.nan,
            "eval_return": math.nan,
            "version": float(self.version),
            "env_steps": float(self.env_steps),
            "updates": float(self.updates),
            "staleness": math.nan,
            **{k: math.nan for k in UPDATE_METRICS},
        } | values
        self.run_dir.append_metrics(self.updates, row)
        latest_return = row["train_return"] if math.isnan(row["eval_return"]) else row["eval_return"]
        if not math.isnan(latest_return):  # an episode row: keep the run's status up to date
            status = self.run_dir.read_status()
            status.iterations_done = self.version
            status.final_return = latest_return
            self.run_dir.write_status(status)

    # ----- main loop -----
    def stop_requested(self) -> bool:
        return stop_path(self.shared).exists() or self.run_dir.stop_requested()

    def run(self, max_updates: int | None = None) -> None:
        while not self.stop_requested() and (max_updates is None or self.updates < max_updates):
            self.ingest(new_episodes(self.shared, self.seen))
            budget = self.update_budget()
            if max_updates is not None:
                budget = min(budget, max_updates - self.updates)
            if budget > 0:
                self.learn(min(budget, UPDATE_CHUNK))
            else:
                time.sleep(IDLE_SLEEP_S)  # waiting for the actor


@dataclass(frozen=True)
class LearnerElsewhere:
    """The session's learner, on another machine (e.g. the HPC), as seen through the copy of its run the sync keeps."""

    where: str  # its run directory on that machine
    log: Path  # the copy of its output.log here
    state: str  # its status as far as the copy knows: running / done / stopped / failed, or "unknown" before it arrives

    @property
    def running(self) -> bool:
        return self.state in ("running", "unknown")  # unknown: assume running rather than risk two learners


def learner_elsewhere(shared: Path) -> LearnerElsewhere | None:
    """The session's learner if it is on another machine, else None (no learner yet, or one on this machine)."""
    if not learner_is_elsewhere(shared) or (path := learner_run_dir(shared)) is None:
        return None
    run = RunDir(path)
    try:
        state: str = run.read_status().state
    except (OSError, ValueError):
        state = "unknown"
    return LearnerElsewhere(learner_run_path(shared).read_text().strip(), run.log_path, state)


def find_config(name: str) -> Path:
    """A config by name from configs/ (`sac_tunnel_mock`) or by path (`configs/sac_tunnel_mock.toml`)."""
    path = Path(name)
    return path if path.suffix == ".toml" else CONFIG.configs_dir / f"{name}.toml"


def session_config(shared: Path, config_file: Path | None) -> RunConfig:
    """The config a session runs on. A session keeps the config it started with (its config.json), so only a new
    session needs `config_file`; one that differs from an existing session's is refused. Raises ValueError."""
    existing = read_config(shared) if config_path(shared).exists() else None
    if config_file is None:
        if existing is None:
            raise ValueError(f"session {shared.name!r} is new: choose a config for it (e.g. sac_tunnel_mock)")
        return existing
    if not config_file.exists():
        raise ValueError(f"no config file {config_file}")
    cfg = RunConfig.from_toml(config_file)
    if existing is not None and existing.model_dump() != cfg.model_dump():
        raise ValueError(
            f"session {shared.name!r} runs {existing.name}, not {config_file.stem}: leave the config out to carry on, "
            "or start a new session"
        )
    return cfg


def run_learner(
    config: RunConfig,
    shared: Path,
    seed_from: list[Path] | None = None,
    max_updates: int | None = None,
) -> RunDir:
    """Set up the shared folder and a run directory, then learn until STOP (or `max_updates`). Everything it prints
    also goes to the run's output.log, which the sync copies, so a learner on the HPC can be followed from the UI."""
    if (other := learner_elsewhere(shared)) is not None and other.running:
        raise SystemExit(f"session {shared.name!r} already has a learner running on another machine ({other.where}); "
                         "one session must have one learner, or their weights and metrics get mixed")
    if config_path(shared).exists() and read_config(shared).model_dump() != config.model_dump():
        # The episodes already in this folder were made for another config (maybe other obs/act sizes).
        raise SystemExit(f"{config_path(shared)} holds a different config; use a fresh shared folder")
    init_shared(shared, config)

    torch.manual_seed(config.seed)
    try:
        run = RunDir.open_named(config, shared.name, CONFIG.runs_dir)  # runs/<session>, the same name as the session
    except ValueError as e:
        raise SystemExit(str(e)) from e
    run.tee_output()
    print(f"RUN_DIR {run.path}", flush=True)
    print(f"session {shared.name}: {config.describe()}", flush=True)
    atomic_write_text(learner_run_path(shared), str(run.path))  # lets the Session tab find the learner's metrics
    learner = Learner(config, shared, run)
    for folder in seed_from or []:
        n = learner.ingest(sorted(folder.glob("ep_*.parquet")), backfill=True)
        print(f"seeded {n} env steps from {folder}", flush=True)
    if n := learner.ingest(new_episodes(shared, learner.seen), backfill=True):  # a session carried on: its data so far
        print(f"read {n} env steps already in the outbox", flush=True)

    state, error = "done", None
    try:
        learner.run(max_updates=max_updates)
        if learner.stop_requested():
            state = "stopped"
    except KeyboardInterrupt:
        state = "stopped"
    except Exception as e:
        state, error = "failed", repr(e)
        raise
    finally:
        learner.policy.save(run.policy_path)
        status = run.read_status()  # re-read: log() has been updating it during the run
        status.state, status.error, status.finished = state, error, datetime.now().astimezone()
        run.write_status(status)
        print(f"FINISHED {state}", flush=True)
    return run


if __name__ == "__main__":
    folder = shared(sys.argv[1] if len(sys.argv) > 1 else SESSION)
    try:
        cfg = session_config(folder, find_config(sys.argv[2]) if len(sys.argv) > 2 else None)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    run_learner(cfg, folder)
