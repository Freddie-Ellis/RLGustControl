"""Proximal Policy Optimisation (clipped), ported from the tutorial. See tutorial.py Part 4 for the derivation."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.ml.algos.base import Algorithm, Callback
from src.ml.envs.base import Env
from src.ml.policies.base import Policy
from src.ml.policies.gaussian import mlp
from src.ml.run_config import PPOConfig


class ValueNet(nn.Module):
    """V_phi(s): predicts the expected return from state s."""

    def __init__(self, obs_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 1])

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).squeeze(-1)


@dataclass(frozen=True)
class Rollout:
    """A batch of trajectories. Leading dims are [T (time), N (parallel envs)]."""
    obs: Tensor  # [T, N, obs]
    act: Tensor  # [T, N, act]  raw (unclipped) samples - log-probs refer to these
    logp: Tensor  # [T, N]  log pi_old(a|s) at collection time
    rew: Tensor  # [T, N]
    last_obs: Tensor  # [N, obs]  state after the final step, used to bootstrap the value


def collect(env: Env, policy: Policy) -> Rollout:
    """Run one full episode in every parallel env with the current (stochastic) policy."""
    obs = env.reset()
    obs_l: list[Tensor] = []
    act_l: list[Tensor] = []
    logp_l: list[Tensor] = []
    rew_l: list[Tensor] = []
    for _ in range(env.spec.horizon):
        a, logp = policy.act(obs)
        next_obs, r = env.step(a)
        obs_l.append(obs)
        act_l.append(a)
        logp_l.append(logp)
        rew_l.append(r)
        obs = next_obs
    return Rollout(torch.stack(obs_l), torch.stack(act_l), torch.stack(logp_l), torch.stack(rew_l), obs)


def compute_gae(rew: Tensor, values: Tensor, last_value: Tensor, gamma: float, lam: float) -> tuple[Tensor, Tensor]:
    """Generalised Advantage Estimation. Returns (advantages, value_targets), both [T, N].

    Episodes end by truncation (the clock), so we bootstrap with last_value rather than zero.
    """
    adv = torch.zeros_like(rew)
    gae = torch.zeros_like(last_value)
    next_v = last_value
    for t in reversed(range(rew.shape[0])):
        delta = rew[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_v = values[t]
    return adv, adv + values


class PPO(Algorithm):
    def __init__(self, cfg: PPOConfig) -> None:
        self.cfg = cfg

    def train(self, env: Env, policy: Policy, callback: Callback | None = None) -> None:
        c = self.cfg
        critic = ValueNet(env.obs_dim)
        params = [*policy.parameters(), *critic.parameters()]
        opt = torch.optim.Adam(params, lr=c.lr)
        policy.train()

        for it in range(c.iterations):
            ro = collect(env, policy)
            with torch.no_grad():
                values = critic(ro.obs)
                adv, v_target = compute_gae(ro.rew, values, critic(ro.last_obs), c.gamma, c.lam)

            # Flatten [T, N, ...] -> [T*N, ...]: after GAE, time ordering no longer matters.
            b_obs = ro.obs.reshape(-1, env.obs_dim)
            b_act = ro.act.reshape(-1, env.act_dim)
            b_logp_old = ro.logp.reshape(-1)
            b_adv = adv.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            b_vt = v_target.reshape(-1)

            pg_losses: list[float] = []
            v_losses: list[float] = []
            entropies: list[float] = []
            n = b_obs.shape[0]
            for _ in range(c.epochs):
                perm = torch.randperm(n)
                for start in range(0, n, c.minibatch):
                    idx = perm[start : start + c.minibatch]
                    logp, ent = policy.log_prob_entropy(b_obs[idx], b_act[idx])
                    ratio = (logp - b_logp_old[idx]).exp()
                    a = b_adv[idx]
                    pg_loss = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                    v_loss = (critic(b_obs[idx]) - b_vt[idx]).pow(2).mean()
                    loss = pg_loss + c.vf_coef * v_loss - c.ent_coef * ent.mean()

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, c.max_grad_norm)
                    opt.step()

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    entropies.append(ent.mean().item())

            metrics = {
                "train_return": float(ro.rew.sum(0).mean()),
                "pg_loss": sum(pg_losses) / len(pg_losses),
                "v_loss": sum(v_losses) / len(v_losses),
                "entropy": sum(entropies) / len(entropies),
            }
            if callback is not None and not callback(it, metrics):
                break

        policy.eval()
