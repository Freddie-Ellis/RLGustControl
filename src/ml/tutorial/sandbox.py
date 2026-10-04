import matplotlib.pyplot as plt
import torch

from src.ml.tutorial.tutorial import PPO_POLICY_PATH, GaussianPolicy, GustLiftEnv

# Load the trained policy: build the same architecture, then load the saved weights.
policy = GaussianPolicy()
policy.load_state_dict(torch.load(PPO_POLICY_PATH))
policy.eval()

env = GustLiftEnv(n_envs=1, seed=2024)
t = torch.arange(env.horizon) * env.dt

tests = []
for i in range(10):
    seed = 2024 + i
    env = GustLiftEnv(n_envs=1, seed=seed)
    obs = env.reset()
    print(f"gust amplitude {env.gust_amp.item():+.2f}, starts at {env.gust_t0.item():.2f} s")
    gust, cl, u = [], [], []
    with torch.no_grad():  # inference only: no graph needed
        for _ in range(env.horizon):
            gust.append(env._gust().item())
            action = policy.mean_net(obs)  # mean action = deterministic, no exploration noise
            obs, _ = env.step(action)
            cl.append(env.cl.item())
            u.append(action.item())
    tests.append((gust, cl, u))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for gust, cl, u in tests:
    ax1.plot(t, cl, label="$C_L$")
    # ax1.plot(t, gust, color="grey", label="gust $w$")
    ax2.plot(t, u)
ax1.axhline(env.cl_ref, color="k", ls="--", lw=0.8, label="target")
ax1.set_ylabel("$C_L$ / gust")
ax2.set(xlabel="time [s]", ylabel="control $u$")
plt.tight_layout()
plt.show()
