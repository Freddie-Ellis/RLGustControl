from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

type ActFn = Callable[[Tensor], Tensor]
"""Maps a batch of observations [N, obs_dim] to actions [N, act_dim]. Any policy or controller fits."""


class EnvSpec(BaseModel):
    """Class to hold shared environment specifications."""
    model_config = ConfigDict(extra="forbid")

    dt: float = 0.01  # Time step size
    horizon: int = 100  # Number of time steps in an episode


@dataclass
class Trace:
    """Everything recorded during one rollout. Tensors are stacked over time: [T, N, ...]."""
    t: Tensor  # [T] time in seconds
    obs: Tensor  # [T, N, obs_dim] observation the action was chosen from
    action: Tensor  # [T, N, act_dim]
    reward: Tensor  # [T, N]
    extras: dict[str, Tensor] = field(default_factory=dict)  # env-specific signals, each [T, N]7


class Env(ABC):
    """Base class for all environments."""

    obs_dim: int  # set by each subclass
    act_dim: int
    priv_dim: int = 0  # size of `privileged()`; 0 if the env has none
    priv_names: tuple[str, ...] = ()  # one name per privileged column, so tools can find e.g. "cm"
    history_len: int = 0  # how many past steps the env keeps in its state; 0 = Markov

    def __init__(self, spec: EnvSpec, n_envs: int = 1, seed: int | None = None) -> None:
        """Initialize the environment with the given number of parallel environments and an optional random seed."""
        self.spec = spec
        self.n_envs = n_envs
        self.rng = torch.Generator()
        self.t = 0
        self.seed(seed)

    def seed(self, seed: int | None = None) -> None:
        """Set the random seed for reproducibility."""
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            self.rng.seed()

    @abstractmethod
    def reset(self) -> Tensor:
        """Reset the environment to an initial state and return the initial observation."""

    @abstractmethod
    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """Take an action in the environment and return the next observation and reward."""

    def privileged(self) -> Tensor:
        """Signals of the CURRENT state that only the critic may see, [N, priv_dim]. Call after reset/step.

        The actor is deployed with the observation alone (pressure taps), but the critic only exists during
        training, so it can use anything the rig measures, e.g. balance loads or the turbulence setting. A better
        informed critic gives the actor better gradients. Override in envs that have such signals.
        """
        return torch.zeros(self.n_envs, 0)

    def episode_info(self) -> dict[str, float]:
        """Scalars describing the current episode (e.g. its turbulence level), stored in the episode metadata."""
        return {}

    def _extras(self) -> dict[str, Tensor]:
        """Env-specific signals to record after each step, each [N]. Override to add some."""
        return {}

    def trace_metrics(self, trace: Trace) -> dict[str, float]:
        """Env-specific summary metrics of a rollout (e.g. lift error). Override to add some."""
        return {}

    def close(self) -> None:
        """Release anything the env holds (e.g. the tunnel rig). Override if needed; safe to call twice."""

    def zero_action(self, obs: Tensor) -> Tensor:
        """Uncontrolled baseline: always act with zeros."""
        return torch.zeros(obs.shape[0], self.act_dim)

    @torch.no_grad()
    def run(self, act_fn: ActFn | None = None, steps: int | None = None) -> Trace:
        """Roll out one episode in every parallel env with `act_fn` (zero action if None) and record it."""
        act_fn = act_fn or self.zero_action
        obs = self.reset()
        obs_l: list[Tensor] = []
        act_l: list[Tensor] = []
        rew_l: list[Tensor] = []
        ext_l: dict[str, list[Tensor]] = {}
        for _ in range(steps or self.spec.horizon):
            action = act_fn(obs)
            next_obs, reward = self.step(action)
            obs_l.append(obs)
            act_l.append(action)
            rew_l.append(reward)
            for k, v in self._extras().items():
                ext_l.setdefault(k, []).append(v.clone())
            obs = next_obs
        t = torch.arange(len(rew_l)) * self.spec.dt
        return Trace(
            t=t,
            obs=torch.stack(obs_l),
            action=torch.stack(act_l),
            reward=torch.stack(rew_l),
            extras={k: torch.stack(v) for k, v in ext_l.items()},
        )


class HistoryWrapper(Env):
    """Wraps an env to add a history of past states to the observation, so the actor sees a Markov state."""

    def __init__(self, env: Env, history_len: int) -> None:
        self.env = env
        self.history_len = history_len
        self.obs_dim = env.obs_dim * history_len
        self.act_dim = env.act_dim
        self.priv_dim = env.priv_dim

    def reset(self) -> Tensor:
        obs = self.env.reset()
        self._obs_hist = [obs] * self.history_len
        return torch.cat(self._obs_hist, dim=-1)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        next_obs, reward = self.env.step(action)
        self._obs_hist.append(next_obs)
        if len(self._obs_hist) > self.history_len:
            self._obs_hist.pop(0)
        return torch.cat(self._obs_hist, dim=-1), reward