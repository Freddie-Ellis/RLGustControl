"""Draw a run's networks and the SAC loop they sit in as Neurolume system schematics, in one workspace.

    uv run python -m src.ml.architecture                              # configs/sac_tunnel_mock.toml
    uv run python -m src.ml.architecture configs/sac_gust_dummy.toml

Dimensions come from the config's env, so the schematic matches what training would build. Only the networks'
shapes are drawn (untrained weights); building the env never opens the rig (see `WindTunnelEnv.__init__`).

The workspace holds two systems: the run as configured, and the same loop with a PA-AE encoding the observation
before the actor. The PA-AE is drawn as a plain block, since no torch PA-AE exists yet.
"""

import sys
from pathlib import Path

import neurolume

from src.ml.ae.pa_ae import LATENT_DIM
from src.ml.algos.sac import TwinQ
from src.ml.policy import SquashedGaussianPolicy
from src.ml.run_config import RunConfig
from src.ml.runs import make_env, make_policy

DEFAULT_CONFIG = Path("configs/sac_tunnel_mock.toml")


def build_system(cfg: RunConfig, pa_ae: bool = False) -> neurolume.System:
    """The control loop of `cfg`: env -> observation -> actor -> action -> env, plus the critic that trains the actor.

    With `pa_ae`, a PA-AE encodes the observation into its latent state and the actor acts on that instead. The critic
    still sees the full observation: it only trains the actor, so the deployed controller is unchanged by that choice.
    """
    env = make_env(cfg, n_envs=1)
    sac = cfg.algo
    obs_dim, act_dim = env.obs_dim, env.act_dim
    priv_dim = env.priv_dim if sac.critic_privileged else 0
    actor_in_dim = LATENT_DIM if pa_ae else obs_dim
    if pa_ae:
        policy = SquashedGaussianPolicy(actor_in_dim, act_dim, hidden=cfg.policy.hidden, init_log_std=cfg.policy.init_log_std)
    else:
        policy = make_policy(cfg)

    system = neurolume.System(f"{cfg.name} + PA-AE" if pa_ae else cfg.name)
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
        description="what the deployed actor sees" if not pa_ae else "what the PA-AE encodes",
        input_shape=[obs_dim],
        output_shape=[obs_dim],
    )
    actor = system.add("actor", policy, (actor_in_dim,), label="Actor: squashed Gaussian")
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
    if pa_ae:
        encoder = system.block(
            "pa_ae",
            label="PA-AE encoder",
            description=f"pressures -> latent ({LATENT_DIM})",
            input_shape=[obs_dim],
            output_shape=[LATENT_DIM],
            notes="""
                Physics Augmented Autoencoder, pre-trained on CFD data and frozen during SAC.
                Only its encoder runs in the loop: the actor acts on the latent state $z$, not the raw pressures.
            """,
        )
        system.connect(obs, encoder, label="s")
        system.connect(encoder, actor, label="z")
    else:
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


def build_workspace(cfg: RunConfig) -> neurolume.Workspace:
    """The run as configured, and the same loop with a PA-AE in front of the actor."""
    workspace = neurolume.Workspace()
    workspace.add(cfg.name, build_system(cfg))
    workspace.add(f"{cfg.name} + PA-AE", build_system(cfg, pa_ae=True))
    return workspace


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    build_workspace(RunConfig.from_toml(path)).show()


if __name__ == "__main__":
    main()
