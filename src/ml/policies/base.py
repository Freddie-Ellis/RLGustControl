from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn


class Policy(nn.Module, ABC):
    """Base class for all stochastic policies pi(a | s).

    Subclasses define how an observation becomes an action distribution. Everything else
    (trainers, evaluation, plotting) only relies on the methods declared here.
    """

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @abstractmethod
    def act(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        """Choose an action for data collection. No gradient.

        Returns (action, log_prob) with shapes [N, act_dim] and [N]. `deterministic=True`
        returns the mean/mode action (no exploration noise), for evaluation.
        """
        raise NotImplementedError("Subclasses must implement act.")

    @abstractmethod
    def log_prob_entropy(self, obs: Tensor, act: Tensor) -> tuple[Tensor, Tensor]:
        """Re-evaluate `act` under the current weights WITH gradient. Returns (log_prob, entropy), each [N]."""
        raise NotImplementedError("Subclasses must implement log_prob_entropy.")
    
    def as_act_fn(self, deterministic: bool = True) -> Callable[[Tensor], Tensor]:
        """Wrap the policy as `obs -> action`, the form `Env.run` and other controllers use."""
        return lambda obs: self.act(obs, deterministic)[0]

    def save(self, path: Path) -> None:
        """Save the weights. Rebuild the same architecture first, then call `load`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        """Load weights saved by `save` into this (already constructed) policy."""
        self.load_state_dict(torch.load(path))
        self.eval()
