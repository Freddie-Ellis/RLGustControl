"""The learner: trains SAC on episodes the actor writes to the shared folder and publishes new weights.

    uv run python -m src.ml.distributed.learner      (edit the settings below first)

Runs on the laptop (GPU if available). Never talks to the actor directly, only through the shared folder:

    outbox/ep_*.parquet  --ingest-->  replay buffer  --SAC.update-->  policy  --publish-->  weights/policy_v*.pt

Learning is tied to data: each env step received earns `updates_per_step` SAC updates. While the tunnel resets and
nothing arrives, the learner waits instead of re-fitting the same buffer over and over (which overfits).

Restart-safe: on start it re-reads every episode already in the outbox and carries on from the last saved learner
state (policy, critics, alpha, optimisers, update count), so stopping the laptop loses no tunnel data and no learning
beyond the updates since the last publish.
"""

import math
import time
from datetime import datetime
from pathlib import Path

import torch

from src.config import CONFIG
from src.ml.algos.replay import ReplayBuffer
from src.ml.algos.sac import SAC
from src.ml.distributed.files import (
    atomic_write_text,
    config_path,
    frame_to_transitions,
    init_shared,
    latest_weights,
    learner_run_path,
    new_episodes,
    read_config,
    read_episode,
    read_learner_state,
    stop_path,
    write_learner_state,
    write_weights,
)
from src.ml.policies.base import SACPolicy
from src.ml.run_config import RunConfig, SACConfig
from src.ml.runs import RunDir, make_env, make_policy

# ----- settings for running this file directly -----
CONFIG_FILE = Path("configs/sac_tunnel_mock.toml")  # becomes SHARED/config.json for the actor
SHARED = Path(r"C:\rl_shared\session1")  # in the tunnel: the tunnel PC's share. Never inside OneDrive.
RUN_NAME: str | None = None  # None -> the name in the config
SEED_FROM: list[Path] = []  # extra outbox folders to pre-load, e.g. open-loop tunnel runs
MAX_UPDATES: int | None = None  # None -> run until SHARED/STOP appears

UPDATE_CHUNK = 50  # updates between checks for new episodes / STOP
LOG_EVERY = 50  # updates per metrics row (the Session tab plots these), independent of publishing
IDLE_SLEEP_S = 0.2  # wait when there is no data and no update budget


class Learner:
    def __init__(self, cfg: RunConfig, shared: Path, run: RunDir) -> None:
        if not isinstance(cfg.algo, SACConfig):
            raise TypeError(f"the learner trains SAC, but the config's algo is {cfg.algo.kind!r}")
        self.cfg = cfg
        self.algo_cfg: SACConfig = cfg.algo
        self.shared = shared
        self.run_dir = run

        # Building the env only reads its dimensions; a wind tunnel env opens no hardware until reset().
        env = make_env(cfg, n_envs=1)
        self.priv_dim = env.priv_dim if self.algo_cfg.critic_privileged else 0
        policy = make_policy(cfg)
        assert isinstance(policy, SACPolicy)  # guaranteed by RunConfig's policy/algo check
        self.policy = policy
        self.sac = SAC(self.algo_cfg)
        self.sac.setup(self.policy, self.priv_dim)
        self.version = 0
        self.updates = 0
        # Resume from the last session. TODO: dont think this works cos if a config exists run learner exits
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

    # ----- data -----
    def ingest(self, parquets: list[Path]) -> int:
        """Add episodes to the replay buffer. Returns how many new env steps arrived."""
        new_steps = 0
        for path in parquets:
            meta, frame = read_episode(path)
            tr = frame_to_transitions(frame)
            priv, next_priv = tr["priv"][:, : self.priv_dim], tr["next_priv"][:, : self.priv_dim]
            self.buffer.add(tr["obs"], priv, tr["act"], tr["rew"], tr["next_obs"], next_priv, tr["done"])
            self.seen.add(path.name)
            new_steps += meta.n_steps
            self.transitions += frame.height

            ret = float(tr["rew"].sum()) / meta.n_envs
            self._note("eval_return" if meta.evaluate else "train_return", ret)
            if meta.policy_version >= 0:
                self._note("staleness", float(self.version - meta.policy_version))  # versions behind when ingested
        self.env_steps += new_steps
        return new_steps

    # ----- learning -----
    def update_budget(self) -> int:
        """Updates still allowed for the data received so far (zero until the warmup data is in)."""
        if self.transitions < self.algo_cfg.warmup_transitions or len(self.buffer) < self.algo_cfg.batch_size:
            return 0
        return self.algo_cfg.updates_per_step * self.env_steps - self.updates

    def learn(self, n: int) -> None:
        for _ in range(n):
            for k, v in self.sac.update(self.buffer.sample(self.algo_cfg.batch_size)).items():
                self._note(k, v)
            self.updates += 1
            if self.updates % self.cfg.distributed.publish_every == 0:
                self.publish()
            if self.updates % LOG_EVERY == 0:
                self.log()

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

    def log(self) -> None:
        """One row per LOG_EVERY updates, keyed by update count. Columns are fixed (NaN when nothing arrived, e.g.
        returns between episodes) so the CSV stays valid."""
        def mean(key: str) -> float:
            vals = self._since_log.get(key, [])
            return sum(vals) / len(vals) if vals else float("nan")

        row = {
            "train_return": mean("train_return"),
            "eval_return": mean("eval_return"),
            "version": float(self.version),
            "env_steps": float(self.env_steps),
            "updates": float(self.updates),
            "staleness": mean("staleness"),
            **{k: mean(k) for k in ("critic_loss", "actor_loss", "alpha", "entropy", "q_mean")},
        }
        self.run_dir.append_metrics(self.updates, row)
        latest_return = row["train_return"] if math.isnan(row["eval_return"]) else row["eval_return"]
        if not math.isnan(latest_return):  # most rows fall between episodes; keep the last real return
            status = self.run_dir.read_status()
            status.iterations_done = self.version
            status.final_return = latest_return
            self.run_dir.write_status(status)
        self._since_log = {}

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


def run_learner(
    config: RunConfig,
    shared: Path,
    seed_from: list[Path] | None = None,
    max_updates: int | None = None,
) -> RunDir:
    """Set up the shared folder and a run directory, then learn until STOP (or `max_updates`)."""
    if config_path(shared).exists() and read_config(shared).model_dump() != config.model_dump():
        # The episodes already in this folder were made for another config (maybe other obs/act sizes).
        raise SystemExit(f"{config_path(shared)} holds a different config; use a fresh shared folder")
    init_shared(shared, config)

    torch.manual_seed(config.seed)
    run = RunDir.create(config, CONFIG.runs_dir)
    print(f"RUN_DIR {run.path}", flush=True)
    atomic_write_text(learner_run_path(shared), str(run.path))  # lets the Session tab find the learner's metrics
    learner = Learner(config, shared, run)
    for folder in seed_from or []:
        n = learner.ingest(sorted(folder.glob("ep_*.parquet")))
        print(f"seeded {n} env steps from {folder}", flush=True)

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
    cfg = RunConfig.from_toml(CONFIG_FILE)
    if RUN_NAME:
        cfg = cfg.model_copy(update={"name": RUN_NAME})
    run_learner(cfg, SHARED, seed_from=SEED_FROM, max_updates=MAX_UPDATES)
