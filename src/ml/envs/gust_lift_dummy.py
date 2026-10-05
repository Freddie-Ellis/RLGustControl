import math

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from src.config import CONFIG
from src.ml.envs.base import Env, EnvSpec, Trace


class GustLiftParams(BaseModel):
    """Physics and reward parameters specific to the dummy gust-lift environment."""
    model_config = ConfigDict(extra="forbid")

    tau: float = 0.3
    slope: float = 1.0
    cl_ref: float = 0.5
    gust_len: float = 1.0
    act_limit: float = 1.5
    action_cost: float = 0.01


class GustLiftDummyEnv(Env):
    """A dummy environment for testing purposes."""

    obs_dim = 3  # [cl error, scaled cl rate, previous action]
    act_dim = 1

    def __init__(self, spec: EnvSpec, params: GustLiftParams, n_envs: int, seed: int | None = None) -> None:
        super().__init__(spec, n_envs, seed)
        self.params = params
        self.cl = torch.zeros(n_envs)
        self.prev_u = torch.zeros(n_envs)
        self.gust_amp = torch.zeros(n_envs)
        self.gust_t0 = torch.zeros(n_envs)

    def _rand(self, *shape: int) -> Tensor:
        return torch.rand(*shape, generator=self.rng, device=CONFIG.torch_device)

    def _gust(self) -> Tensor:
        """1-cosine gust velocity at the current time, for every env."""
        s = (self.t * self.spec.dt - self.gust_t0) / self.params.gust_len  # 0..1 while the gust is active
        active = (s >= 0.0) & (s <= 1.0)
        shape = 0.5 * (1.0 - torch.cos(2.0 * math.pi * s))
        return torch.where(active, self.gust_amp * shape, torch.zeros_like(s))

    def _obs(self, cl_dot: Tensor) -> Tensor:
        return torch.stack([self.cl - self.params.cl_ref, 0.1 * cl_dot, self.prev_u], dim=-1)

    def reset(self) -> Tensor:
        """Reset the environment to an initial state and return the initial observation."""
        n = self.n_envs
        self.t = 0
        self.cl = self.params.cl_ref + 0.05 * (2 * self._rand(n) - 1.0)
        sign = torch.where(self._rand(n) < 0.5, -1.0, 1.0)
        self.gust_amp = sign * (0.1 + 0.4 * self._rand(n))
        self.gust_t0 = 0.5 + 1.5 * self._rand(n)
        self.prev_u = torch.zeros(n)
        return self._obs(torch.zeros(n))

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """Take an action in the environment and return the next observation and reward."""
        u = action.squeeze(-1).clamp(-self.params.act_limit, self.params.act_limit)
        cl_dot = (self.params.slope * (u + self._gust()) - self.cl) / self.params.tau
        self.cl = self.cl + cl_dot * self.spec.dt
        self.prev_u = u
        self.t += 1
        err = self.cl - self.params.cl_ref
        reward = -(err**2) - self.params.action_cost * u**2
        return self._obs(cl_dot), reward

    def trace_metrics(self, trace: Trace) -> dict[str, float]:
        err = trace.extras["cl"] - self.params.cl_ref
        return {"cl_err_rms": float(err.pow(2).mean().sqrt()), "cl_err_peak": float(err.abs().max())}

    def _extras(self) -> dict[str, Tensor]:
        return {"cl": self.cl, "cl_ref": torch.full_like(self.cl, self.params.cl_ref), "gust": self._gust()}

