import torch
from torch import Tensor, nn
from torch.distributions import Normal

from src.config import CONFIG
from src.ml.policies.base import Policy


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())  # tanh is the RL default; ReLU also fine
    return nn.Sequential(*layers)


OBS_DIM = CONFIG.obs_dim
ACT_DIM = CONFIG.act_dim


class GaussianPolicy(Policy):
    """pi_theta(a | s) = Normal(mean_theta(s), std).

    For CONTINUOUS actions the network outputs the parameters of a distribution and we sample.
    The sampling is what makes the policy *stochastic*, which is how the agent explores.
    (For discrete actions you would output logits and use torch.distributions.Categorical.)

    std is a free parameter (one per action dim), independent of the state. Starting std is
    large (lots of exploration) and training shrinks it as the policy becomes confident.
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__(OBS_DIM, ACT_DIM)
        self.mean_net = mlp([OBS_DIM, hidden, hidden, ACT_DIM])
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

    def dist(self, obs: Tensor) -> Normal:
        return Normal(self.mean_net(obs), self.log_std.exp())

    def act(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        """Sample an action and return (action, log-prob). No gradient: this is data collection."""
        with torch.no_grad():
            d = self.dist(obs)
            if deterministic:
                a = d.mean
            else:
                a = d.sample()
            return a, d.log_prob(a).sum(-1)  # sum over action dims -> log-prob of the joint action

    def log_prob_entropy(
        self, obs: Tensor, act: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Re-evaluate actions under the *current* weights WITH gradient. This is the thing we differentiate."""
        d = self.dist(obs)
        return d.log_prob(act).sum(-1), d.entropy().sum(-1)