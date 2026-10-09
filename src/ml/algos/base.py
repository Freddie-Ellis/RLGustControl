from abc import ABC, abstractmethod
from collections.abc import Callable

from src.ml.envs.base import Env
from src.ml.policies.base import Policy

type Callback = Callable[[int, dict[str, float]], bool]
"""Called after every iteration with (iteration, metrics). Return False to stop training early."""


class Algorithm[P: Policy](ABC):
    """Base class for training algorithms. Trains `policy` in place on `env`.

    `P` is the kind of policy the algorithm can train (e.g. PPO needs a `PPOPolicy`), so pairing an algorithm with
    a policy it cannot use is a type error. `RunConfig` also checks this when a config is loaded.
    """

    @abstractmethod
    def train(self, env: Env, policy: P, callback: Callback | None = None) -> None:
        """Run the full training loop, reporting metrics through `callback` each iteration."""
