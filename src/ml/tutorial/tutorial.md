# Reinforcement Learning in PyTorch: A Tutorial

Companion to [tutorial.py](tutorial.py). The Python file is the runnable code; this document explains the ideas behind it and tells you what to look at and what to try. It assumes you already know deep learning (backprop, optimisers, MLPs, LSTMs) but not RL.

## 0. Running it

```powershell
uv run python src/ml/RL/tutorial.py          # trains REINFORCE then PPO (~30 s, CPU)
uv run python src/ml/RL/tutorial.py --plot   # also shows learning curve and a gust trace
```

Only PyTorch is needed. The environment is written in the file itself, so there is no gymnasium dependency.

Expected output (numbers vary slightly): the open-loop baseline scores about -1.9, and both REINFORCE and PPO reach about -0.3 (higher is better).

---

## 1. What is different about RL?

In supervised learning you have labelled pairs `(x, y)` and a loss that is differentiable in the network output. In RL you have:

| Term | Meaning | In the gust problem |
|---|---|---|
| Observation $s_t$ | what the agent sees | lift error, its rate, last action (later: pressure taps) |
| Action $a_t$ | what the agent does | flap / pitch command |
| Reward $r_t$ | scalar feedback per step | $-(C_L - C_{L,\mathrm{ref}})^2 - 0.01\,u^2$ |
| Policy $\pi_\theta(a \mid s)$ | network: observation to distribution over actions | what we train |
| Episode | one trajectory from reset to end | one gust encounter |
| Return $G_t$ | discounted sum of future reward | $r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$ |

The goal is to maximise the expected return:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t} \gamma^t r_t \right]
$$

**The central problem.** The reward comes from the environment (a simulator, a wind tunnel), which is not differentiable with respect to $\theta$. You cannot call `loss.backward()` through it.

**The way round it: the policy gradient theorem.**

$$
\nabla_\theta J = \mathbb{E}_{\pi_\theta}\left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A_t \right]
$$

$A_t$ (the *advantage*) measures how much better than average the outcome was after taking $a_t$. The only thing we differentiate is $\log \pi_\theta$, which is an ordinary network output. The effect: actions that led to good outcomes become more likely, bad ones less likely.

Everything else in RL algorithms (baselines, critics, GAE, PPO clipping) is a way of getting a lower-variance, safer estimate of $A_t$.

**Rewards are the spec.** The reward function defines the task. The agent will optimise exactly what you write, including loopholes. Keep that in mind when moving to the real project.

**Exploration.** The policy is stochastic: it samples actions from a distribution, and that noise is how it discovers better behaviour. Training gradually shrinks the noise (watch `std` in the printed output fall).

---

## 2. The environment (`GustLiftEnv`)

A deliberately simple model of your problem. The lift coefficient relaxes toward a quasi-steady value with a lag:

$$
\frac{dC_L}{dt} = \frac{k\,\big(u + w_\mathrm{gust}(t)\big) - C_L}{\tau}
$$

- $u$ is the agent's control input and $k$ is the lift slope (`slope` in the code).
- $w_\mathrm{gust}$ is a 1-cosine gust with random sign, amplitude and start time, **which the agent cannot observe directly**. It only sees its effect on $C_L$. This is a first taste of partial observability.
- The agent must output the right trim ($u = C_{L,\mathrm{ref}}$ at rest) and also counteract the gust.

**Vectorised environments.** `step` takes an `[N, 1]` action tensor and advances `N` independent copies at once. On-policy RL needs a lot of trajectories per update, and this gives you a batch of them for one policy forward pass. All environment state is torch tensors. This is also the interface a Rust-backed simulator in `ruststuff` should expose later, so the algorithms don't change.

**Fixed-length episodes.** Every episode is 100 steps. That means no `done` flags, which keeps the first examples simple (see the truncation note in section 6).

---

## 3. Policy and value networks

`GaussianPolicy` outputs the mean of a Normal distribution from an MLP, with a learnable state-independent `log_std`. Three methods matter:

- `dist(obs)` builds the distribution.
- `act(obs)` samples an action and returns its log-probability, under `torch.no_grad()`. This is data collection, so no graph is kept.
- `log_prob_entropy(obs, act)` re-evaluates stored actions under the *current* weights, *with* gradient. This is what the loss differentiates.

Useful habit: keep collection (no grad) and learning (grad) as separate code paths.

For discrete actions you would output logits and use `torch.distributions.Categorical` instead. Everything else is the same.

`ValueNet` is a plain regression MLP predicting $V(s)$, the expected return from state `s`. It appears in Part 4.

---

## 4. Collecting experience

`collect(env, policy)` runs one full episode in every parallel environment and stacks the results into a `Rollout` with leading dimensions `[T, N]` (time, parallel env).

Note that the *raw* sampled action is stored (before the environment clips it) because the log-probability refers to that sample. `logp` is stored too, since PPO needs $\log \pi_\mathrm{old}$.

`discounted_returns` computes $G_t = r_t + \gamma\, G_{t+1}$ backwards through time. Why discount?

1. It keeps sums finite for long or infinite horizons.
2. It says near-term reward matters more.
3. It cuts variance: distant rewards are mostly noise with respect to *this* action.

Rule of thumb: the effective horizon is $1/(1-\gamma)$ steps. With $\gamma = 0.98$ that is about 50 steps (2.5 s here).

---

## 5. REINFORCE: the simplest policy gradient

```python
returns = discounted_returns(ro.rew, gamma)
adv = (returns - returns.mean()) / (returns.std() + 1e-8)
logp, _ = policy.log_prob_entropy(ro.obs, ro.act)
loss = -(logp * adv.detach()).mean()
```

Read it as weighted maximum likelihood: ordinary log-likelihood of the actions the agent took, but each sample is weighted by how good its outcome was.

Points to notice:

- **The baseline.** Subtracting the batch mean from the returns does not change the expected gradient (it doesn't depend on `a_t`) but greatly reduces variance. In this problem all rewards are negative, so without a baseline *every* action would be pushed down. Try removing it and see.
- **`adv.detach()`.** The advantage is a weight, not something to differentiate through.
- **Weaknesses.** $G_t$ sums many random rewards, so gradients are noisy, and each batch is used for exactly one update and thrown away. Both are fixed in the next part.

---

## 6. Actor-critic, GAE and PPO

### 6.1 The critic and GAE

A second network $V(s)$ gives a *learned* baseline. The one-step TD error is

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Low variance (one random reward) but biased (V is imperfect). The Monte-Carlo return used by REINFORCE is the opposite: unbiased but high variance. **Generalised Advantage Estimation** interpolates:

$$
A_t = \sum_{k=0}^{\infty} (\gamma\lambda)^k\, \delta_{t+k}
$$

$\lambda = 0$ is pure TD, $\lambda = 1$ is pure Monte-Carlo, and 0.95 is the usual choice. `compute_gae` also returns `adv + values` as the regression target for the critic. Training the critic is ordinary supervised learning (MSE).

**Truncation vs termination.** Our episodes end because time ran out, not because the system failed. The future is not worth zero at that point, so we bootstrap with `V(last_obs)`. If an episode could genuinely terminate (a crash, leaving a safe envelope), multiply by `(1 - done)` there instead. Mixing these up is one of the most common RL bugs.

### 6.2 The PPO objective

We want several gradient steps per batch, because data is expensive. After the first step the data no longer comes from the current policy, so we correct with the importance ratio

$$
r(\theta) = \frac{\pi_\theta(a \mid s)}{\pi_\mathrm{old}(a \mid s)} = \exp\big(\log\pi_\theta - \log\pi_\mathrm{old}\big)
$$

Maximising $r\,A$ unconstrained would let the policy move wildly. PPO clips it:

$$
L_\mathrm{clip} = -\,\mathbb{E}\Big[ \min\big( r\,A,\ \mathrm{clip}(r,\,1-\epsilon,\,1+\epsilon)\,A \big) \Big]
$$

Once $r$ leaves $[1-\epsilon,\ 1+\epsilon]$ in the direction that helps, the gradient is zero, so the policy can't move far from $\pi_\mathrm{old}$ in one iteration. It is a cheap trust region.

Total loss:

$$
L = L_\mathrm{clip} + c_v\, \mathrm{MSE}\big(V,\ \hat{V}^\mathrm{target}\big) - c_e\, \mathcal{H}[\pi_\theta]
$$

with $c_v$ = `vf_coef`, $c_e$ = `ent_coef` and $\mathcal{H}$ the policy entropy.

### 6.3 Practical details in the code

- Advantages are normalised per batch. Always do this.
- Data is flattened `[T, N] -> [T*N]` and shuffled into minibatches. This is fine for an MLP policy. It is **not** fine for a recurrent one (see section 8).
- Gradient clipping (`clip_grad_norm_`) keeps updates stable.
- `v_loss` should fall quickly. If it doesn't, the critic isn't learning and the advantages will be noise.

---

## 7. Reading the results

- **Always have a baseline.** The open-loop trim controller ignores the gust and scores about -1.9. Both learned policies beat it by a wide margin.
- **Evaluate with the mean action**, with no exploration noise, on identical gust seeds so comparisons are fair.
- **Compare training return to evaluation return.** Training return is lower because it includes exploration noise.
- **Sample efficiency.** REINFORCE needed 150 iterations to get roughly what PPO gets in 60. The gap grows on harder problems.
- **Seeds matter.** RL results vary run to run. Before claiming that algorithm A beats B, run several seeds.

---

## 8. Exercises

Roughly in order of increasing difficulty. Each is a small edit to `tutorial.py`.

1. **Remove the baseline** in REINFORCE (use raw `returns`). Observe what happens to learning.
2. **Vary $\gamma$** (0.9, 0.99, 0.999). The reward here is immediate, so what changes and why?
3. **Vary `clip`** (0.05, 0.2, 0.5) and `epochs` (1, 8, 30). Print the approximate KL, `(logp_old - logp_new).mean()`, to see how far each update moves the policy.
4. **Set `lam`** to 0 and 1 and compare the learning curves.
5. **Add `ent_coef = 0.01`** and compare the final `std`.
6. **Action-rate penalty.** Add `-0.1 * (u - prev_u)^2` to the reward and look at how smooth the control becomes in `--plot`.
7. **Make it harder.** Add sensor noise to the observation, or hide the rate component from it. Where does the MLP policy start to struggle?
8. **Reward hacking.** Set `action_cost = 0` and `act_limit` large. Does the policy do anything you didn't intend?

---

## 9. Connecting this to the gust-control project

1. **Partial observability.** Pressure taps give an incomplete picture, and the gust is only visible after it has acted. The policy needs memory. Replace the MLP with an LSTM/GRU that carries a hidden state through the rollout and train on whole sequences. In PPO minibatches, shuffle whole *environments*, not individual timesteps, and carry (or recompute) hidden states.
2. **Use your estimator.** A pre-trained PA-AE/LSTM estimator can supply a latent state or estimated forces as the policy's observation. The policy then solves a much easier problem.
3. **Off-policy algorithms (SAC, TD3).** These reuse old data from a replay buffer. They are worth it when each environment step is an expensive CFD run or wind-tunnel sample, because PPO typically needs millions of steps.
4. **Reward design.** Constant lift is the obvious objective. Think about control effort, actuation rate, and safety limits, and check for hacking (exercise 8).
5. **Speed.** A real simulator will dominate runtime. Put stepping in `ruststuff` (per the project rules, no slow Python loops), keep the batched-tensor interface, and update `ruststuff.pyi`. The training code stays unchanged.
6. **Practical PPO checklist.** Normalise observations with a running mean/std, tune `ent_coef`, monitor approximate KL, and log several seeds.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| On-policy | Data must come from the current policy (REINFORCE, PPO) |
| Off-policy | Old data can be reused (SAC, DDPG, TD3) |
| Advantage | How much better an action was than average in that state |
| Baseline | A term subtracted from the return to cut variance without adding bias |
| Critic | The value network $V(s)$ |
| Actor | The policy network |
| Bootstrapping | Using a learned value estimate in place of the remaining sum of rewards |
| GAE | Generalised Advantage Estimation, a bias/variance blend controlled by $\lambda$ |
| Trust region | A limit on how far one update can move the policy |
| Truncation | The episode ends for an external reason (time limit), so bootstrap |
| Termination | The episode ends because the task ended (crash), so future value is zero |

## Further reading

- Sutton and Barto, *Reinforcement Learning: An Introduction* (free online), chapters 3, 6 and 13.
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) and *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015).
- OpenAI *Spinning Up in Deep RL*: a short, clear treatment of policy gradients and PPO.
- Haarnoja et al., *Soft Actor-Critic* (2018), for the off-policy route.
