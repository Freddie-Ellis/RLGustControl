"""Draw a run's networks and the SAC loop they sit in as a Neurolume system schematic.

    uv run python -m src.ml.architecture                              # configs/sac_tunnel_mock.toml
    uv run python -m src.ml.architecture configs/sac_gust_dummy.toml

Dimensions come from the config's env, so the schematic matches what training would build. Only the networks'
shapes are drawn (untrained weights); building the env never opens the rig (see `WindTunnelEnv.__init__`).
"""

import sys
from pathlib import Path

import neurolume

from src.ml.algos.sac import TwinQ
from src.ml.run_config import RunConfig
from src.ml.runs import make_env, make_policy

DEFAULT_CONFIG = Path("configs/sac_tunnel_mock.toml")


def build_system(cfg: RunConfig) -> neurolume.System:
    """The control loop of `cfg`: env -> observation -> actor -> action -> env, plus the critic that trains the actor."""
    env = make_env(cfg, n_envs=1)
    policy = make_policy(cfg)
    sac = cfg.algo
    obs_dim, act_dim = env.obs_dim, env.act_dim
    priv_dim = env.priv_dim if sac.critic_privileged else 0

    system = neurolume.System(f"{cfg.name}")
    plant = system.block(
        "env",
        label=f"Env: {cfg.env.kind}",
        description=f"dt = {cfg.env.spec.dt} s, horizon = {cfg.env.spec.horizon} steps",
        input_shape=[act_dim],
        output_shape=[obs_dim],
        notes=f"""
            One `step(action)` is one control period of {cfg.env.spec.dt} s.
            The env returns the next observation and the reward.
        """,
    )
    obs = system.block(
        "obs",
        label="Observation",
        description="what the deployed actor sees",
        input_shape=[obs_dim],
        output_shape=[obs_dim],
    )
    actor = system.add("actor", policy, (obs_dim,), label="Actor: squashed Gaussian")
    sample = system.block(
        "sample",
        label="Squashed sample",
        description="a = tanh(mean + std * eps)",
        output_shape=[act_dim],
        notes=r"""
            Reparameterised sample, so the critic's gradient reaches the actor through $a$:

            $$u = \mu(s) + \sigma(s)\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I), \quad a = \tanh(u)$$

            $$\log \pi(a \mid s) = \log \mathcal{N}(u; \mu, \sigma) - \sum \log(1 - \tanh^2 u)$$
        """,
    )
    system.connect(plant, obs, label="s")
    system.connect(obs, actor)
    system.connect(actor, sample, label="mean, log std")
    system.connect(sample, plant, label="a")

    critic = TwinQ(obs_dim, priv_dim, act_dim, sac.critic_hidden, sac.critic_dropout, sac.critic_layer_norm)
    # One sample shape per `TwinQ.forward(obs, priv, act)` argument; each link below feeds the input of that name.
    critic_net = system.add("critic", critic, ((obs_dim,), (priv_dim,), (act_dim,)), label="Critic: twin Q")
    if priv_dim:
        priv = system.block(
            "priv",
            label="Privileged",
            description="critic only, never the deployed actor",
            output_shape=[priv_dim],
        )
        system.connect(plant, priv, shape=[priv_dim])
        system.connect(priv, critic_net, label="priv", to="priv")
    system.connect(obs, critic_net, label="s", to="obs")
    system.connect(sample, critic_net, label="a", to="act")

    losses = system.block(
        "losses",
        label="SAC losses",
        description=f"gamma = {sac.gamma}, tau = {sac.tau}, auto alpha",
        notes=r"""
            Critic: $\; (Q_i(s, a) - y)^2, \quad y = r + \gamma\,[\min_j \bar Q_j(s', a') - \alpha \log \pi(a' \mid s')]$

            Actor: $\; \alpha \log \pi(a \mid s) - \min_j Q_j(s, a)$

            Alpha: $\; -\log\alpha\,(\log \pi(a \mid s) + \bar{\mathcal{H}})$

            Target critics $\bar Q$ follow $Q$ by Polyak averaging with rate $\tau$.
        """,
    )
    system.connect(critic_net, losses, label="Q1, Q2")
    return system


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    neurolume.show(build_system(RunConfig.from_toml(path)))


if __name__ == "__main__":
    main()
