from abc import ABC, abstractmethod
from collections.abc import Callable

from src.ml.envs.base import Env
from src.ml.policies.base import Policy

type Callback = Callable[[int, dict[str, float]], bool]
"""Called after every iteration with (iteration, metrics). Return False to stop training early."""


class Algorithm(ABC):
    """Base class for training algorithms. Trains `policy` in place on `env`."""

    @abstractmethod
    def train(self, env: Env, policy: Policy, callback: Callback | None = None) -> None:
        """Run the full training loop, reporting metrics through `callback` each iteration."""
