from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.distributions import Normal

DEVICE = torch.device("cpu")  # these nets are tiny; CPU is usually faster than GPU for RL
PPO_POLICY_PATH = Path(__file__).with_name("ppo_policy.pt")  # where main() saves the trained policy
OBS_DIM = 3
ACT_DIM = 1

# Shape conventions used in comments: N = parallel envs, T = time steps, obs / act = observation / action dims.
ActFn = Callable[[Tensor], Tensor]


# =============================================================================================
# PART 0b - THE ENVIRONMENT
# =============================================================================================
class GustLiftEnv:
    """Toy 'hold the lift through a gust' environment, batched over N parallel copies.

    Physics (deliberately simple): lift coefficient relaxes to a quasi-steady value with a lag,

        d cl / dt = ( slope * (u + w_gust(t)) - cl ) / tau

    where u is the agent's control input (think: flap/pitch command, in "effective incidence"
    units) and w_gust is a 1-cosine gust the agent does NOT observe directly. It can only see
    its effect on the lift. Gust amplitude, sign and start time are randomised every episode.

    Observation (3):  [ cl - cl_ref,  d(cl)/dt (scaled),  previous action ]
    Action (1):       u, clipped to [-act_limit, act_limit]
    Reward:           -(cl - cl_ref)^2  -  action_cost * u^2        (track the lift, be cheap)

    API mirrors gymnasium (reset / step) but everything is a batched torch tensor, so one call
    steps N environments at once. This is the standard trick to make on-policy RL fast:
    "vectorised environments" give you many independent trajectories per policy forward pass.
    Episodes are a fixed length, so there is no 'done' handling (see truncation note in Part 4).
    """

    dt: float = 0.05
    horizon: int = 100  # 5 s per episode
    tau: float = 0.3
    slope: float = 1.0
    cl_ref: float = 0.5
    gust_len: float = 1.0
    act_limit: float = 1.5
    action_cost: float = 0.01

    def __init__(self, n_envs: int, seed: int | None = None) -> None:
        self.n_envs = n_envs
        self.gen = torch.Generator(device=DEVICE)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.t = 0
        self.cl = torch.zeros(n_envs)
        self.prev_u = torch.zeros(n_envs)
        self.gust_amp = torch.zeros(n_envs)
        self.gust_t0 = torch.zeros(n_envs)

    def _rand(self, *shape: int) -> Tensor:
        return torch.rand(*shape, generator=self.gen, device=DEVICE)

    def _gust(self) -> Tensor:
        """1-cosine gust velocity at the current time, for every env."""
        s = (self.t * self.dt - self.gust_t0) / self.gust_len  # 0..1 while the gust is active
        active = (s >= 0.0) & (s <= 1.0)
        shape = 0.5 * (1.0 - torch.cos(2.0 * math.pi * s))
        return torch.where(active, self.gust_amp * shape, torch.zeros_like(s))

    def _obs(self, cl_dot: Tensor) -> Tensor:
        return torch.stack([self.cl - self.cl_ref, 0.1 * cl_dot, self.prev_u], dim=-1)

    def reset(self) -> Tensor:
        n = self.n_envs
        self.t = 0
        self.cl = self.cl_ref + 0.05 * (2 * self._rand(n) - 1)
        self.prev_u = torch.zeros(n)
        sign = torch.where(self._rand(n) < 0.5, -1.0, 1.0)
        self.gust_amp = sign * (0.3 + 0.5 * self._rand(n))
        self.gust_t0 = 0.5 + 1.5 * self._rand(n)
        return self._obs(torch.zeros(n))

    def step(self, action: Tensor) -> tuple[Tensor, Tensor]:
        """action: [N, 1] (unclipped is fine). Returns (next_obs [N, OBS_DIM], reward [N])."""
        u = action.squeeze(-1).clamp(-self.act_limit, self.act_limit)
        cl_dot = (self.slope * (u + self._gust()) - self.cl) / self.tau
        self.cl = self.cl + self.dt * cl_dot  # explicit Euler
        self.prev_u = u
        self.t += 1
        err = self.cl - self.cl_ref
        reward = -(err**2) - self.action_cost * u**2
        return self._obs(cl_dot), reward


# =============================================================================================
# PART 1 - POLICY AND VALUE NETWORKS
# =============================================================================================
def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())  # tanh is the RL default; ReLU also fine
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """pi_theta(a | s) = Normal(mean_theta(s), std).

    For CONTINUOUS actions the network outputs the parameters of a distribution and we sample.
    The sampling is what makes the policy *stochastic*, which is how the agent explores.
    (For discrete actions you would output logits and use torch.distributions.Categorical.)

    std is a free parameter (one per action dim), independent of the state. Starting std is
    large (lots of exploration) and training shrinks it as the policy becomes confident.
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.mean_net = mlp([OBS_DIM, hidden, hidden, ACT_DIM])
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

    def dist(self, obs: Tensor) -> Normal:
        return Normal(self.mean_net(obs), self.log_std.exp())

    def act(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Sample an action and return (action, log-prob). No gradient: this is data collection."""
        with torch.no_grad():
            d = self.dist(obs)
            a = d.sample()
            return a, d.log_prob(a).sum(-1)  # sum over action dims -> log-prob of the joint action

    def log_prob_entropy(
        self, obs: Tensor, act: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Re-evaluate actions under the *current* weights WITH gradient. This is the thing we differentiate."""
        d = self.dist(obs)
        return d.log_prob(act).sum(-1), d.entropy().sum(-1)


class ValueNet(nn.Module):
    """V_phi(s): predicts the expected return from state s. A plain regression network."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = mlp([OBS_DIM, hidden, hidden, 1])

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).squeeze(-1)


# =============================================================================================
# PART 2 - COLLECTING EXPERIENCE
# =============================================================================================
@dataclass(frozen=True)
class Rollout:
    """A batch of trajectories. Leading dims are [T (time), N (parallel envs)]."""

    obs: Tensor
    act: Tensor  # [T, N, act]  raw (unclipped) samples - log-probs refer to these
    logp: Tensor  # [T, N]  log pi_old(a|s) at collection time (PPO needs this)
    rew: Tensor
    last_obs: Tensor  # [N, obs]  state after the final step, used to bootstrap the value


def collect(env: GustLiftEnv, policy: GaussianPolicy) -> Rollout:
    """Run one full episode in every parallel env with the current policy."""
    obs = env.reset()
    obs_l: list[Tensor] = []
    act_l: list[Tensor] = []
    logp_l: list[Tensor] = []
    rew_l: list[Tensor] = []
    for _ in range(env.horizon):
        a, logp = policy.act(obs)
        next_obs, r = env.step(a)
        obs_l.append(obs)
        act_l.append(a)
        logp_l.append(logp)
        rew_l.append(r)
        obs = next_obs
    return Rollout(torch.stack(obs_l), torch.stack(act_l), torch.stack(logp_l), torch.stack(rew_l), obs)


def discounted_returns(rew: Tensor, gamma: float) -> Tensor:
    """G_t = r_t + gamma * G_{t+1}, computed backwards. rew: [T, N] -> [T, N].

    Why discount? (1) keeps infinite-horizon sums finite, (2) says 'reward now matters more than
    reward later', (3) reduces variance: distant rewards are mostly noise w.r.t. this action.
    gamma ~ 0.95-0.99 is typical; the effective horizon is about 1 / (1 - gamma) steps.
    This loop is T=100 long and vectorised over N, so it is cheap. If it ever shows up in a
    profile it belongs in ruststuff.
    """
    out = torch.zeros_like(rew)
    running = torch.zeros_like(rew[0])
    for t in reversed(range(rew.shape[0])):
        running = rew[t] + gamma * running
        out[t] = running
    return out


@torch.no_grad()
def evaluate(env: GustLiftEnv, act_fn: ActFn) -> float:
    """Mean undiscounted episode return. act_fn maps obs -> action (use the MEAN for evaluation)."""
    obs = env.reset()
    total = torch.zeros(env.n_envs)
    for _ in range(env.horizon):
        obs, r = env.step(act_fn(obs))
        total += r
    return float(total.mean())



# =============================================================================================
# PART 3 - REINFORCE
# =============================================================================================
def train_reinforce(iterations: int = 150, n_envs: int = 64, gamma: float = 0.98, lr: float = 3e-3) -> GaussianPolicy:
    """The simplest policy-gradient algorithm.

        loss = - mean_t [ log pi(a_t | s_t) * (G_t - b) ]

    * G_t is the observed return from step t onward (a Monte-Carlo estimate: no learned model).
    * b is a *baseline*. Subtracting any b that does not depend on a_t leaves the expected
      gradient unchanged but can slash the variance. Here b is just the batch mean (and we also
      divide by the std). Without it, if all rewards are negative (as ours are!), every action
      gets pushed DOWN and learning is hopelessly noisy.

    Shapes: log-probs are [T, N], advantage is [T, N]; detach the advantage - it is a weight,
    not something we differentiate through.

    Weakness: G_t sums up to T random rewards, so the gradient is very noisy and we throw the
    data away after ONE update. Part 4 fixes both.
    """
    env = GustLiftEnv(n_envs, seed=0)
    policy = GaussianPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    for it in range(iterations):
        ro = collect(env, policy)
        returns = discounted_returns(ro.rew, gamma)
        adv = (returns - returns.mean()) / (returns.std() + 1e-8)

        logp, _ = policy.log_prob_entropy(ro.obs, ro.act)  # [T, N] with grad
        loss = -(logp * adv.detach()).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if it % 25 == 0 or it == iterations - 1:
            print(f"[REINFORCE] iter {it:3d}  train return {float(ro.rew.sum(0).mean()):8.3f}  std {policy.log_std.exp().item():.3f}")
    return policy


# =============================================================================================
# PART 4 - ACTOR-CRITIC, GAE, PPO
# =============================================================================================
def compute_gae(
    rew: Tensor, values: Tensor, last_value: Tensor, gamma: float, lam: float
) -> tuple[Tensor, Tensor]:
    """Generalised Advantage Estimation. Returns (advantages, value_targets), both [T, N].

    One-step TD error:   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        -> low variance (only one random reward) but biased (V is imperfect)
    Monte-Carlo return:  G_t - V(s_t)
        -> unbiased but high variance (what REINFORCE used)
    GAE blends them:     A_t = sum_k (gamma*lam)^k * delta_{t+k}
        lam=0 -> pure TD, lam=1 -> pure Monte-Carlo, lam~0.95 is the sweet spot.

    Truncation vs termination: our episodes end because the clock ran out, not because the
    system 'died'. So the future is NOT worth zero at the end - we bootstrap with last_value.
    If the episode could genuinely terminate (a crash), you would multiply by (1 - done) there.
    """
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    gae = torch.zeros_like(last_value)
    next_v = last_value
    for t in reversed(range(T)):
        delta = rew[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_v = values[t]
    return adv, adv + values  # value target = advantage + baseline = an estimate of the return


def train_ppo(
    iterations: int = 60,
    n_envs: int = 64,
    gamma: float = 0.98,
    lam: float = 0.95,
    clip: float = 0.2,
    epochs: int = 8,
    minibatch: int = 1024,
    lr: float = 3e-4,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
) -> tuple[GaussianPolicy, ValueNet, list[float]]:
    """Proximal Policy Optimisation (clipped). The workhorse on-policy algorithm.

    Two ideas on top of REINFORCE:

    1. CRITIC. A second network V(s) gives a learned baseline, and GAE turns it into a much
       lower-variance advantage. The critic is trained by ordinary MSE regression onto the
       value targets - this part is just supervised learning.

    2. REUSE DATA SAFELY. We want several gradient steps per rollout (data is expensive), but
       after the first step the data no longer comes from the current policy. Importance
       sampling fixes the bias with the ratio

            r(theta) = pi_theta(a|s) / pi_old(a|s) = exp(logp_new - logp_old)

       and unconstrained, maximising r * A would let the policy move wildly. PPO clips it:

            L_clip = - mean[ min( r * A,  clip(r, 1-eps, 1+eps) * A ) ]

       Once r leaves [1-eps, 1+eps] in the direction that helps, the gradient is zero, so the
       policy cannot be pushed far from pi_old in one iteration. That is a cheap trust region.

    Total loss = L_clip + vf_coef * MSE(V, value_target) - ent_coef * entropy.
    Returns the policy, the critic, and the per-iteration training return (for plotting).
    """
    env = GustLiftEnv(n_envs, seed=1)
    policy, critic = GaussianPolicy(), ValueNet()
    opt = torch.optim.Adam([*policy.parameters(), *critic.parameters()], lr=lr)
    history: list[float] = []

    for it in range(iterations):
        ro = collect(env, policy)
        with torch.no_grad():
            values = critic(ro.obs)  # [T, N]
            adv, v_target = compute_gae(ro.rew, values, critic(ro.last_obs), gamma, lam)

        # Flatten [T, N, ...] -> [T*N, ...]: after GAE, time ordering no longer matters.
        b_obs = ro.obs.reshape(-1, OBS_DIM)
        b_act = ro.act.reshape(-1, ACT_DIM)
        b_logp_old = ro.logp.reshape(-1)
        b_adv = adv.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)  # per-batch normalisation: always do this
        b_vt = v_target.reshape(-1)

        n = b_obs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, minibatch):
                idx = perm[start : start + minibatch]
                logp, ent = policy.log_prob_entropy(b_obs[idx], b_act[idx])
                ratio = (logp - b_logp_old[idx]).exp()
                a = b_adv[idx]
                pg_loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                v_loss = (critic(b_obs[idx]) - b_vt[idx]).pow(2).mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent.mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*policy.parameters(), *critic.parameters()], 0.5)
                opt.step()

        history.append(float(ro.rew.sum(0).mean()))
        if it % 10 == 0 or it == iterations - 1:
            print(f"[PPO]       iter {it:3d}  train return {history[-1]:8.3f}  std {policy.log_std.exp().item():.3f}  v_loss {v_loss.item():.4f}")
    return policy, critic, history


# =============================================================================================
# PART 5 - EVALUATE AND COMPARE
# =============================================================================================
def rollout_trace(act_fn: ActFn, seed: int = 123) -> tuple[Tensor, Tensor]:
    """One deterministic-policy episode in a single env -> (cl [T], gust [T]) for plotting."""
    env = GustLiftEnv(1, seed=seed)
    obs = env.reset()
    cl: list[float] = []
    gust: list[float] = []
    with torch.no_grad():
        for _ in range(env.horizon):
            gust.append(float(env._gust()[0]))
            obs, _ = env.step(act_fn(obs))
            cl.append(float(env.cl[0]))
    return torch.tensor(cl), torch.tensor(gust)


def main() -> None:
    torch.manual_seed(0)

    # Baselines. Always know what 'dumb' scores before believing a learning curve.
    eval_env = GustLiftEnv(512, seed=999)
    trim = lambda obs: torch.full((obs.shape[0], 1), GustLiftEnv.cl_ref / GustLiftEnv.slope) 
    print(f"baseline  open-loop trim (u = cl_ref, ignores gust): {evaluate(eval_env, trim):8.3f}\n")

    reinforce_policy = train_reinforce()
    ppo_policy, _, ppo_history = train_ppo()

    # Save the weights only (state_dict), not the pickled class: robust to code refactors.
    # To reload: policy = GaussianPolicy(); policy.load_state_dict(torch.load(PPO_POLICY_PATH))
    torch.save(ppo_policy.state_dict(), PPO_POLICY_PATH)

    # Evaluate with the distribution MEAN (no exploration noise) on identical gust seeds.
    mean_act = lambda pol: (lambda obs: pol.mean_net(obs))
    print()
    print(f"REINFORCE  eval return: {evaluate(GustLiftEnv(512, seed=999), mean_act(reinforce_policy)):8.3f}")
    print(f"PPO        eval return: {evaluate(GustLiftEnv(512, seed=999), mean_act(ppo_policy)):8.3f}")

    import matplotlib.pyplot as plt

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ppo_history)
    ax1.set(xlabel="PPO iteration", ylabel="mean training return", title="Learning curve")
    t = torch.arange(GustLiftEnv.horizon) * GustLiftEnv.dt
    for name, fn in [("open-loop trim", trim), ("REINFORCE", mean_act(reinforce_policy)), ("PPO", mean_act(ppo_policy))]:
        cl, gust = rollout_trace(fn)
        ax2.plot(t, cl, label=name)
    ax2.axhline(GustLiftEnv.cl_ref, color="k", ls="--", lw=0.8, label="target")
    ax2.plot(t, 0.5 + gust, color="grey", lw=0.8, label="gust (offset)")
    ax2.set(xlabel="time [s]", ylabel="$C_L$", title="Gust encounter"); ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# =============================================================================================
# WHERE TO GO NEXT  (relevant to the gust-control project)
# =============================================================================================
# 1. PARTIAL OBSERVABILITY. Real pressure taps do not reveal the full flow state, and the gust
#    is invisible until it has already acted. The policy then needs memory: replace the MLP
#    with an LSTM/GRU that carries a hidden state through the rollout, and train on whole
#    sequences (do not shuffle individual timesteps in the PPO minibatches - shuffle whole
#    environments instead). This is where your LSTM / PA-AE work plugs in: a pre-trained
#    estimator's latent state (e.g. estimated forces) can be fed to the policy as its observation.
# 2. OFF-POLICY, SAMPLE-EFFICIENT: SAC / TD3 with a replay buffer. Worth it when each
#    environment step is an expensive CFD or wind-tunnel sample; PPO needs millions of steps.
# 3. REWARD SHAPING: the reward IS your spec. Add a penalty on action rate (|u_t - u_{t-1}|) to
#    stop bang-bang actuation, and test for reward hacking.
# 4. SPEED: a real simulator will dominate runtime. Put stepping in ruststuff, keep the
#    batched-tensor interface used here, and the algorithms above do not change.
# 5. PRACTICAL PPO CHECKLIST: normalise observations (running mean/std), tune ent_coef, watch the
#    approximate KL between pi_old and pi_new, log several seeds - RL results vary a lot
#    between runs.
