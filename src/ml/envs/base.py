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
    extras: dict[str, Tensor] = field(default_factory=dict)  # env-specific signals, each [T, N]


class Env(ABC):
    """Base class for all environments."""

    obs_dim: int  # set by each subclass
    act_dim: int

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

    def _extras(self) -> dict[str, Tensor]:
        """Env-specific signals to record after each step, each [N]. Override to add some."""
        return {}

    def trace_metrics(self, trace: Trace) -> dict[str, float]:
        """Env-specific summary metrics of a rollout (e.g. lift error). Override to add some."""
        return {}

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
