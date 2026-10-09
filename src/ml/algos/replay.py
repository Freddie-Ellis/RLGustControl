"""Replay buffer: SAC's memory of every transition it has seen.

Every control step adds (s, a, r, s', done). Updates sample random minibatches from all of it, so each expensive
tunnel sample is reused many times, and data from older policies stays useful (the critic learns facts about the
wing, not about whoever chose the action).
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Batch:
    """A random minibatch of transitions. Leading dim is the batch size B."""
    obs: Tensor  # [B, obs_dim]
    act: Tensor  # [B, act_dim]  squashed action in (-1, 1), as the policy output it
    rew: Tensor  # [B]
    next_obs: Tensor  # [B, obs_dim]
    done: Tensor  # [B]  1.0 only for a true terminal state (e.g. a safety trip); time-limit ends are NOT done


class ReplayBuffer:
    """Fixed-size circular buffer of transitions, stored as preallocated tensors on `device`.

    When full, the oldest transitions are overwritten. For the tunnel that also limits how stale the data gets if
    conditions drift between sessions.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.act = torch.zeros(capacity, act_dim, device=device)
        self.rew = torch.zeros(capacity, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.done = torch.zeros(capacity, device=device)
        self._next = 0  # where the next transition is written
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs: Tensor, act: Tensor, rew: Tensor, next_obs: Tensor, done: Tensor) -> None:
        """Add one step from N parallel envs at once: obs [N, obs_dim], act [N, act_dim], rew [N], done [N]."""
        n = obs.shape[0]
        idx = (self._next + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx] = obs.to(self.device)
        self.act[idx] = act.to(self.device)
        self.rew[idx] = rew.to(self.device)
        self.next_obs[idx] = next_obs.to(self.device)
        self.done[idx] = done.to(self.device)
        self._next = (self._next + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Uniformly random minibatch, with replacement."""
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return Batch(self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx])
