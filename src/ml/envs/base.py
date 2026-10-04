from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel
from torch import Tensor


class EnvSpec(BaseModel):
    """Class to hold shared environment specifications."""
    dt: float = 0.01  # Time step size
    horizon: int = 100  # Number of time steps in an episode


class Env(ABC):
    """Base class for all environments."""

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

