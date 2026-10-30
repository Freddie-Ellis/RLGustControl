"""Tanh-squashed Gaussian policy: the SAC actor, and what the tunnel PC runs.

    u ~ Normal(mean(s), std(s))      unbounded sample
    a = tanh(u)                      squashed into (-1, 1) = the control surface limits

Why squash instead of clip: clipping gives zero gradient at the limits and piles probability onto them, so the
actor cannot learn near full deflection. tanh is smooth, and its effect on the probability can be corrected exactly
(change of variables):

    log pi(a | s) = log Normal(u; mean, std) - sum log(1 - tanh(u)^2)

Envs map a in (-1, 1) to physical deflection, so set their `act_limit` to 1.
"""

import math
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal

LOG_STD_MIN = -20.0  # std ~ 2e-9: effectively deterministic, also prevents bang bang control
LOG_STD_MAX = 2.0  # std ~ 7.4: already saturates tanh, more is pointless, good to clamp so the actor doesn't blow up and NaN the critic when given garbage gradients in early training.


def tanh_log_det(u: Tensor) -> Tensor:
    """log(1 - tanh(u)^2), written so it stays finite for large |u|.

    The naive form fails because float32 tanh(u) rounds to exactly +-1 once |u| > ~9, giving log(0).
    """
    return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))


def mlp(sizes: list[int]) -> nn.Sequential:
    """Linear layers of the given sizes with tanh between them (none after the last)."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class SquashedGaussianPolicy(nn.Module):
    """pi(a | s) = tanh(Normal(mean(s), std(s))).

    std depends on the state: the actor can explore more where the critic is still unsure (e.g. mid-gust) and stay
    precise where it already knows the answer (steady flow).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, init_log_std: float = -0.5) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.trunk = nn.Sequential(mlp([obs_dim, hidden, hidden]), nn.Tanh())  # shared features of the pressures
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

        # Start with deflection ~0 and the same std everywhere. In the tunnel the first episodes then begin from
        # a centred surface with moderate, uniform exploration rather than a random bias from the init.
        for head in (self.mean_head, self.log_std_head):
            nn.init.uniform_(head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.log_std_head.bias, init_log_std)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Mean and log std of the unsquashed Gaussian, each [N, act_dim]. Sample with `act` or `rsample`."""
        h = self.trunk(obs)
        return self.mean_head(h), self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    @staticmethod
    def _log_prob_u(d: Normal, u: Tensor) -> Tensor:
        """log pi(tanh(u) | s) from the pre-squash sample u, summed over action dims -> [N]."""
        return (d.log_prob(u) - tanh_log_det(u)).sum(-1)

    def rsample(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Reparameterised sample WITH gradient: u = mean + std * eps, eps ~ N(0, 1).

        The randomness is in eps only, so gradients flow from the critic, through a, into mean and std. This is
        what the SAC actor loss differentiates. Returns (action [N, act_dim], log_prob [N]).
        """
        mean, log_std = self(obs)
        d = Normal(mean, log_std.exp())
        u = d.rsample() # u = mean + std * eps, eps ~ N(0, 1)  -- the reparameterisation trick linked to the actor via mean and std
        return torch.tanh(u), self._log_prob_u(d, u)

    def act(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        """Choose an action for data collection. No gradient.

        `deterministic=True` returns tanh(mean): the policy's best guess with no exploration, used for
        evaluation episodes where we want to measure Cm suppression rather than explore.
        """
        with torch.no_grad():
            if not deterministic:
                return self.rsample(obs) # Samples an action with some exploration noise, and returns the action and its log probability.
            mean, log_std = self(obs)
            return torch.tanh(mean), self._log_prob_u(Normal(mean, log_std.exp()), mean)

    def as_act_fn(self, deterministic: bool = True) -> Callable[[Tensor], Tensor]:
        """Wrap the policy as `obs -> action`, the form `Env.run` and other controllers use."""
        return lambda obs: self.act(obs, deterministic)[0]

    def save(self, path: Path) -> None:
        """Save the weights. Rebuild the same architecture first, then call `load`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        """Load weights saved by `save` into this (already constructed) policy."""
        self.load_state_dict(torch.load(path, map_location="cpu"))  # weights may come from a GPU learner
        self.eval()
