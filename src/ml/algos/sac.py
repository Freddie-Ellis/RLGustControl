"""Soft Actor-Critic (Haarnoja et al. 2018) with automatic entropy tuning.

Three things learn, each with its own optimiser:
    critics  Q1, Q2(s, a)   "how good is this deflection here?"  learned from real data via the Bellman equation
    actor    pi(a | s)      "which deflection should I use?"     moves towards actions the critics rate highly
    alpha                   "how random should I stay?"           tuned so the policy keeps a target entropy

`SAC.update` is one gradient step on a batch and knows nothing about where the batch came from. `SAC.train` is the
synchronous simulation loop (step env -> store -> update). The decoupled tunnel learner will call `update` on
batches from tunnel episode files instead.
"""

import copy
import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import CONFIG
from src.ml.algos.replay import Batch, ReplayBuffer
from src.ml.envs.base import Env
from src.ml.policy import SquashedGaussianPolicy
from src.ml.run_config import SACConfig

type Callback = Callable[[int, dict[str, float]], bool]
"""Called after every iteration with (iteration, metrics). Return False to stop training early."""


def critic_mlp(in_dim: int, hidden: int, dropout: float, layer_norm: bool) -> nn.Sequential:
    """Two hidden layers -> scalar. LayerNorm + a little dropout (DroQ, Hiraoka et al. 2021) keep the critic stable
    when it is updated many times per sample, which is how we squeeze the most out of scarce tunnel data."""
    layers: list[nn.Module] = []
    for d_in in (in_dim, hidden):
        layers.append(nn.Linear(d_in, hidden))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


class TwinQ(nn.Module):
    """Two independent critics Q(obs, priv, act). Using the smaller of the two counters Q overestimation: a single
    critic's errors get exploited by the actor, which then chases deflections that only look good because the
    critic is wrong.

    `priv` is the asymmetric part: the critic also sees e.g. the balance loads, so it predicts future Cm more
    easily than from the pressure taps alone. The actor still acts from the taps alone.
    """

    def __init__(
        self, obs_dim: int, priv_dim: int, act_dim: int, hidden: int, dropout: float, layer_norm: bool
    ) -> None:
        super().__init__()
        in_dim = obs_dim + priv_dim + act_dim
        self.q1 = critic_mlp(in_dim, hidden, dropout, layer_norm)
        self.q2 = critic_mlp(in_dim, hidden, dropout, layer_norm)

    def forward(self, obs: Tensor, priv: Tensor, act: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([obs, priv, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, obs: Tensor, priv: Tensor, act: Tensor) -> Tensor:
        q1, q2 = self(obs, priv, act)
        return torch.min(q1, q2)


class SAC:
    def __init__(self, cfg: SACConfig) -> None:
        self.cfg = cfg
        self.device = CONFIG.torch_device
        # Built in `setup` once the env's dimensions are known.
        self.policy: SquashedGaussianPolicy
        self.critic: TwinQ
        self.critic_target: TwinQ
        self.log_alpha: Tensor
        self.target_entropy: float

    def setup(self, policy: SquashedGaussianPolicy, priv_dim: int) -> None:
        """Create the critics, target critics, alpha and the three optimisers around `policy`.

        `priv_dim` is the size of the critic-only input (0 gives the critic the same view as the actor).
        """
        c = self.cfg
        self.policy = policy.to(self.device)
        self.critic = TwinQ(
            policy.obs_dim, priv_dim, policy.act_dim, c.critic_hidden, c.critic_dropout, c.critic_layer_norm
        )
        self.critic = self.critic.to(self.device)
        # Slow-moving copy used for the Bellman target, so the target does not jump every time the critic does.
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)

        # Optimise log(alpha) so alpha stays positive. Default target entropy -act_dim is the standard heuristic.
        self.log_alpha = torch.tensor(math.log(c.init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = c.target_entropy if c.target_entropy is not None else -float(policy.act_dim)

        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=c.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=c.alpha_lr)

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to carry on learning where this left off: the policy, both critics and their targets,
        alpha, and all three optimisers' moments."""
        return {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a `state_dict` into this SAC after `setup` (same config and dimensions)."""
        self.policy.load_state_dict(state["policy"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])  # in place: alpha_opt holds this tensor
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp().detach()

    def update(self, batch: Batch) -> dict[str, float]:
        """One SAC gradient step on `batch`. Returns metrics for logging."""
        c = self.cfg

        # 1. Critic: make Q(s, a_old) match r + gamma * soft value of s' under the CURRENT policy.
        #    a_old can come from any older policy; only the next action a' is re-sampled, which is why old data works.
        with torch.no_grad():
            next_act, next_logp = self.policy.rsample(batch.next_obs)
            next_v = self.critic_target.min_q(batch.next_obs, batch.next_priv, next_act) - self.alpha * next_logp
            target = batch.rew + c.gamma * (1.0 - batch.done) * next_v
        q1, q2 = self.critic(batch.obs, batch.priv, batch.act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. Actor: propose fresh actions for the stored states and move uphill on the critic. Gradients flow
        #    Q -> a -> rsample -> actor weights. Only actor_opt steps, so the critic is used here but not changed.
        act, logp = self.policy.rsample(batch.obs) # Points to the actors weights via the reparameterisation trick, so gradients flow from the critic into the actor
        # The actor saw obs only; priv goes to the critic, which judges the actor's action with the extra info.
        actor_loss = (self.alpha * logp - self.critic.min_q(batch.obs, batch.priv, act)).mean()
        self.actor_opt.zero_grad() # Actor optimiser points to the actors weights via its creation
        actor_loss.backward()
        self.actor_opt.step()

        # 3. Alpha: raise it if the policy is less random than the target entropy, lower it if more.
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4. Target critics drift slowly towards the critics (Polyak averaging).
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.lerp_(p, c.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": -logp.mean().item(),
            "q_mean": q1.mean().item(),
        }

    def train(self, env: Env, policy: SquashedGaussianPolicy, callback: Callback | None = None) -> None:
        """Synchronous simulation loop. One iteration = one episode in every parallel env, the same unit as a tunnel
        episode (where new weights are swapped in between episodes)."""
        c = self.cfg
        priv_dim = env.priv_dim if c.critic_privileged else 0
        self.setup(policy, priv_dim)
        self.policy.train()
        buffer = ReplayBuffer(c.buffer_size, env.obs_dim, priv_dim, env.act_dim, self.device)
        n_transitions = 0

        def privileged() -> Tensor:
            return env.privileged() if c.critic_privileged else torch.zeros(env.n_envs, 0)

        for it in range(c.iterations):
            obs = env.reset()
            priv = privileged()
            ep_return = torch.zeros(env.n_envs)
            sums: dict[str, float] = {}
            n_updates = 0

            for _ in range(env.spec.horizon):
                if n_transitions < c.warmup_transitions:
                    # Uniform random deflections first, so the critic sees the whole action range before the actor
                    # starts trusting it.
                    act = 2.0 * torch.rand(env.n_envs, env.act_dim) - 1.0
                else:
                    act = self.policy.act(obs.to(self.device))[0].cpu()
                next_obs, rew = env.step(act)
                next_priv = privileged()
                # Our episodes only end on the clock (truncation), so done = 0 and the target still bootstraps.
                buffer.add(obs, priv, act, rew, next_obs, next_priv, torch.zeros(env.n_envs))
                ep_return += rew
                obs, priv = next_obs, next_priv
                n_transitions += env.n_envs

                if n_transitions >= c.warmup_transitions and len(buffer) >= c.batch_size:
                    for _ in range(c.updates_per_step):
                        for k, v in self.update(buffer.sample(c.batch_size)).items():
                            sums[k] = sums.get(k, 0.0) + v
                        n_updates += 1

            metrics = {"train_return": float(ep_return.mean()), "transitions": float(n_transitions)}
            metrics |= {k: v / n_updates for k, v in sums.items()} if n_updates else {
                k: float("nan") for k in ("critic_loss", "actor_loss", "alpha", "entropy", "q_mean")
            }
            if callback is not None and not callback(it, metrics):
                break

        # Hand the policy back on the CPU in eval mode: that is how it is saved and how the tunnel PC runs it.
        self.policy.to("cpu").eval()
