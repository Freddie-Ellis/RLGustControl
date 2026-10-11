"""The actor: runs episodes with frozen policy weights and writes each one to the shared folder.

    uv run python -m src.ml.distributed.actor      (edit the settings below first; start the learner before it)

It never learns and never waits on the learner. Between episodes it picks up the newest weights, so one episode
always runs on one policy version. In the tunnel it runs on the tunnel PC with n_envs = 1; in simulation it can run
anywhere, with any n_envs, to test the decoupled setup.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
from torch import Tensor

from src.ml.distributed.files import (
    EpisodeMetaData,
    latest_weights,
    read_config,
    stop_path,
    transitions_to_frame,
    write_episode,
)
from src.ml.run_config import RunConfig
from src.ml.runs import make_env, make_policy

# ----- settings for running this file directly -----
SHARED = Path(r"C:\rl_shared\session1")  # the same folder the learner uses (config.json comes from it)
N_ENVS = 1  # parallel envs: 1 in the tunnel, more only for sim envs
EVAL_EVERY: int | None = None  # every k-th episode deterministic; None -> from the config
MAX_EPISODES: int | None = None  # None -> run until SHARED/STOP appears

type Episode = tuple[EpisodeMetaData, pl.DataFrame]


class Actor:
    def __init__(self, cfg: RunConfig, shared: Path, n_envs: int = 1, session_id: str | None = None) -> None:
        self.cfg = cfg
        self.shared = shared
        self.env = make_env(cfg, n_envs=n_envs, seed=cfg.seed)
        self.policy = make_policy(cfg)  # CPU, eval mode: acting only
        self.policy.eval()
        self.version = -1  # -1 = no weights yet -> warmup with random actions
        self.episode_id = self._next_episode_id()
        self.session_id = session_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        self._unsent: list[Episode] = []  # episodes whose write failed, retried before the next one

    def _next_episode_id(self) -> int:
        """Continue numbering after any episodes already in the outbox, so a restarted actor never overwrites."""
        existing = [int(p.stem.removeprefix("ep_")) for p in (self.shared / "outbox").glob("ep_*.parquet")]
        return max(existing, default=-1) + 1

    def maybe_load_weights(self) -> None:
        """Load the newest weights if they are newer than ours. Only called between episodes."""
        latest = latest_weights(self.shared)
        if latest is not None and latest[0] > self.version:
            self.version, path = latest
            self.policy.load(path)
            print(f"loaded weights v{self.version}", flush=True)

    def choose(self, obs: Tensor, evaluate: bool) -> Tensor:
        if self.version < 0:
            # Warmup: the learner needs data before it can publish any weights. Uniform random actions also show
            # the critic the whole deflection range.
            return 2.0 * torch.rand(obs.shape[0], self.env.act_dim) - 1.0
        return self.policy.act(obs, deterministic=evaluate)[0]

    @torch.no_grad()
    def run_episode(self, evaluate: bool) -> Episode:
        """One episode in every env. Returns its metadata and one row per transition (T * N rows)."""
        env = self.env
        obs, priv = env.reset(), env.privileged()
        steps: list[tuple[Tensor, ...]] = []
        dts: list[float] = []
        t_prev = time.perf_counter()
        for _ in range(env.spec.horizon):
            act = self.choose(obs, evaluate)
            next_obs, rew = env.step(act)
            next_priv = env.privileged()
            # Episodes end on the clock (truncation), so done = 0. A safety trip in the tunnel would set it to 1.
            steps.append((obs, priv, act, rew, next_obs, next_priv, torch.zeros(env.n_envs)))
            obs, priv = next_obs, next_priv
            now = time.perf_counter()
            dts.append(now - t_prev)
            t_prev = now

        # [T][7] of [N, ...] -> 7 tensors of [T * N, ...]. Every row is a complete transition, so the learner can
        # put rows straight into its replay buffer without reassembling episodes.
        columns = [torch.cat(field) for field in zip(*steps)]
        meta = EpisodeMetaData(
            episode_id=self.episode_id,
            policy_version=self.version,
            session_id=self.session_id,
            evaluate=evaluate,
            n_envs=env.n_envs,
            n_steps=len(steps),
            loop_dt_mean=sum(dts) / len(dts),
            loop_dt_max=max(dts),
            created=datetime.now().astimezone(),
        )
        return meta, transitions_to_frame(*columns)

    def _send(self, episode: Episode) -> None:
        """Write an episode. If the share is unreachable, keep it and retry later instead of stopping the tunnel."""
        self._unsent.append(episode)
        while self._unsent:
            meta, frame = self._unsent[0]
            try:
                write_episode(self.shared, meta, frame)
            except OSError as e:
                print(f"episode {meta.episode_id} not written ({e}); {len(self._unsent)} waiting", file=sys.stderr)
                return
            self._unsent.pop(0)

    def run(self, eval_every: int = 10, max_episodes: int | None = None) -> None:
        """Run episodes until STOP appears in the shared folder (or `max_episodes` is reached)."""
        done = 0
        while not stop_path(self.shared).exists() and (max_episodes is None or done < max_episodes):
            self.maybe_load_weights()
            evaluate = self.version >= 0 and self.episode_id % eval_every == 0
            meta, frame = self.run_episode(evaluate)
            self._send((meta, frame))
            ret = float(frame["rew"].sum()) / meta.n_envs
            kind = "eval " if evaluate else ("warm " if self.version < 0 else "train")
            print(f"ep {meta.episode_id:6d} {kind} v{self.version:<4d} return {ret:9.4f}", flush=True)
            self.episode_id += 1
            done += 1


def run_actor(
    shared: Path,
    n_envs: int = 1,
    eval_every: int | None = None,
    max_episodes: int | None = None,
    session_id: str | None = None,
) -> None:
    """Build the env and policy from SHARED/config.json and run episodes until STOP (or `max_episodes`)."""
    cfg = read_config(shared)
    actor = Actor(cfg, shared, n_envs=n_envs, session_id=session_id)
    try:
        actor.run(eval_every=eval_every or cfg.distributed.eval_every, max_episodes=max_episodes)
    finally:
        actor.env.close()  # surfaces to neutral and release the rig, however the actor stops


if __name__ == "__main__":
    run_actor(SHARED, n_envs=N_ENVS, eval_every=EVAL_EVERY, max_episodes=MAX_EPISODES)
