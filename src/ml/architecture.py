"""Draw a run's networks and the SAC loop they sit in as Neurolume system schematics, in one workspace.

    uv run python -m src.ml.architecture                              # configs/sac_tunnel_mock.toml
    uv run python -m src.ml.architecture configs/sac_gust_dummy.toml

Dimensions come from the config's env, so the schematic matches what training would build. Only the networks'
shapes are drawn (untrained weights); building the env never opens the rig (see `WindTunnelEnv.__init__`).

The workspace holds the run as configured and, for a wind tunnel run, the sparse-tap latent model
(src/ml/ae/sparse_latent.py):
    1. autoencoder training: the whole autoencoder (encoder E, decoder D, load heads), trained before SAC
    2. SAC training: the actor acts on z_hat = g(k taps); the encoder E alone gives z = E(all taps), to the critic
    3. deployment: k taps, a fault mask, g and the actor; the decoder D alone, inside fault detection
Every network drawn is the real (untrained) torch module.
"""

import sys
from pathlib import Path

import neurolume

from src.ml.ae.sparse_latent import (
    HISTORY_FRAMES,
    K_TAPS,
    LATENT_DIM,
    LOAD_HEADS,
    LatentRegressor,
    TapAutoencoder,
)
from src.ml.algos.sac import TwinQ
from src.ml.envs.base import Env
from src.ml.policy import SquashedGaussianPolicy
from src.ml.run_config import RunConfig
from src.ml.runs import make_env, make_policy

DEFAULT_CONFIG = Path("configs/sac_tunnel_mock.toml")
SAMPLE_NOTES = r"""
    Reparameterised sample, so the critic's gradient reaches the actor through $a$:

    $$u = \mu(s) + \sigma(s)\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I), \quad a = \tanh(u)$$

    $$\log \pi(a \mid s) = \log \mathcal{N}(u; \mu, \sigma) - \sum \log(1 - \tanh^2 u)$$
"""
LOSSES_NOTES = r"""
    Critic: $\; (Q_i(s, a) - y)^2, \quad y = r + \gamma\,[\min_j \bar Q_j(s', a') - \alpha \log \pi(a' \mid s')]$

    Actor: $\; \alpha \log \pi(a \mid s) - \min_j Q_j(s, a)$

    Alpha: $\; -\log\alpha\,(\log \pi(a \mid s) + \bar{\mathcal{H}})$

    Target critics $\bar Q$ follow $Q$ by Polyak averaging with rate $\tau$.
"""


def _plant(system: neurolume.System, cfg: RunConfig, act_dim: int, obs_dim: int) -> neurolume.Component:
    return system.block("env", label=f"Env: {cfg.env.kind}", input_shape=[act_dim], output_shape=[obs_dim])


def _sample(system: neurolume.System, act_dim: int) -> neurolume.Component:
    return system.block(
        "sample", label="Squashed sample", description="a = tanh(mean + std * eps)", output_shape=[act_dim],
        notes=SAMPLE_NOTES,
    )


def _losses(system: neurolume.System, cfg: RunConfig) -> neurolume.Component:
    sac = cfg.algo
    return system.block(
        "losses", label="SAC losses", description=f"gamma = {sac.gamma}, tau = {sac.tau}, auto alpha",
        notes=LOSSES_NOTES,
    )


def _critic(cfg: RunConfig, obs_dim: int, priv_dim: int, act_dim: int) -> TwinQ:
    sac = cfg.algo
    return TwinQ(obs_dim, priv_dim, act_dim, sac.critic_hidden, sac.critic_dropout, sac.critic_layer_norm)


def build_system(cfg: RunConfig) -> neurolume.System:
    """The control loop of `cfg`: env -> observation -> actor -> action -> env, plus the critic that trains the actor."""
    env = make_env(cfg, n_envs=1)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    priv_dim = env.priv_dim if cfg.algo.critic_privileged else 0

    system = neurolume.System(cfg.name)
    plant = _plant(system, cfg, act_dim, obs_dim)
    obs = system.block(
        "obs", label="Observation", description="what the deployed actor sees",
        input_shape=[obs_dim], output_shape=[obs_dim],
    )
    actor = system.add("actor", make_policy(cfg), (obs_dim,), label="Actor: squashed Gaussian")
    sample = _sample(system, act_dim)
    system.connect(plant, obs, label="s")
    system.connect(obs, actor, label="s")
    system.connect(actor, sample, label="mean, log std")
    system.connect(sample, plant, label="a")

    # One sample shape per `TwinQ.forward(obs, priv, act)` argument; each link below feeds the input of that name.
    critic = system.add("critic", _critic(cfg, obs_dim, priv_dim, act_dim), ((obs_dim,), (priv_dim,), (act_dim,)),
                        label="Critic: twin Q")
    if priv_dim:
        priv = system.block(
            "priv", label="Privileged", description="critic only, never the deployed actor", output_shape=[priv_dim],
        )
        system.connect(plant, priv, label=", ".join(env.priv_names), shape=[priv_dim])
        system.connect(priv, critic, label="priv", to="priv")
    system.connect(obs, critic, label="s", to="obs")
    system.connect(sample, critic, label="a", to="act")
    system.connect(critic, _losses(system, cfg), label="Q1, Q2")
    return system


def _tap_counts(env: Env) -> tuple[int, int]:
    """(all taps, deployed k). The tunnel observation is [taps, previous action]."""
    n_taps = env.obs_dim - env.act_dim
    return n_taps, min(K_TAPS, n_taps)


def build_autoencoder_training(cfg: RunConfig) -> neurolume.System:
    """The full autoencoder, trained before SAC: encoder E, decoder D and the load heads. After this only E (for the
    critic) and D (for fault detection) are kept, both frozen."""
    n_taps, _ = _tap_counts(make_env(cfg, n_envs=1))
    heads = ", ".join(LOAD_HEADS)
    system = neurolume.System("Sparse-tap latent: autoencoder training")
    data = system.block(
        "data", label="Tap pressures", description="CFD at the tap positions, then tunnel runs",
        output_shape=[n_taps],
        notes="A random validity mask is applied to the input only.",
    )
    loads = system.block("loads", label=f"Loads: {heads}", description="CFD forces, then the tunnel balance",
                         output_shape=[len(LOAD_HEADS)])
    # Three networks, not one: the viewer lays a network's layers in a row, so heads branching off z would be drawn
    # after the decoder. Linked here, E -> D reads as the autoencoder and the heads hang off its bottleneck.
    tap_ae = TapAutoencoder(n_taps)
    encoder = system.add("E", tap_ae.encoder, (n_taps,), label="Encoder E")
    decoder = system.add("D", tap_ae.decoder, (LATENT_DIM,), label="Decoder D")
    head_net = system.add("heads", tap_ae.heads, (LATENT_DIM,), label="Load heads (training only)")
    losses = system.block(
        "ae_losses", label="Autoencoder losses", description="reconstruction + loads",
        notes=r"""
            $$\lVert D(E(x)) - x \rVert^2 + \lambda\, \lVert h(E(x)) - y \rVert^2$$

            $x$: all the taps, $y$: the measured loads, $h$: the load heads. The heads make the latent carry the
            forces and moments; they are dropped after training.
        """,
    )
    system.connect(data, encoder, label=f"{n_taps} taps")
    system.connect(encoder, decoder, label="z")
    system.connect(encoder, head_net, label="z")
    system.connect(decoder, losses, label=f"{n_taps} taps (reconstructed)")
    system.connect(head_net, losses, label="loads")
    system.connect(data, losses, label=f"{n_taps} taps (target)")
    system.connect(loads, losses, label="loads (target)")
    return system


def build_latent_training(cfg: RunConfig) -> neurolume.System:
    """SAC in the tunnel on the sparse-tap latent: the actor acts on z_hat = g(k taps); the autoencoder's z = E(taps)
    only reaches the critic, as a privileged input. The autoencoder and g stay frozen."""
    env = make_env(cfg, n_envs=1)
    n_taps, k = _tap_counts(env)
    act_dim = env.act_dim
    actor_in = LATENT_DIM + act_dim  # z_hat plus the previous action
    priv_dim = env.priv_dim + LATENT_DIM  # balance (and the env's other privileged signals) plus the true z

    system = neurolume.System("Sparse-tap latent: SAC training")
    plant = _plant(system, cfg, act_dim, env.obs_dim)
    taps = system.block(
        "taps", label="Tap pressures", description="stored raw in the replay buffer",
        input_shape=[n_taps], output_shape=[n_taps],
        notes="The buffer stores raw taps, not latents: with the autoencoder and g frozen, z and z_hat are recomputed "
              "when a batch is sampled, so refitting g needs no new data.",
    )
    encoder = system.add("E", TapAutoencoder(n_taps).encoder, (n_taps,), label="Encoder E (frozen, privileged)")
    subsample = system.block(
        "subsample", label=f"Subsample k = {k}", description="the deployed taps",
        input_shape=[n_taps], output_shape=[k],
    )
    history = system.block(
        "history", label="Concat history", description=f"last {HISTORY_FRAMES} frames",
        input_shape=[k], output_shape=[HISTORY_FRAMES * k],
        notes=f"The last {HISTORY_FRAMES} frames of the k taps, oldest first, concatenated into one vector: pressure "
              "depends on history (hysteresis, unsteady effects), so one frame is not enough.",
    )
    g = system.add("g", LatentRegressor(k), (HISTORY_FRAMES * k,), label="g: MLP (frozen)")
    policy = SquashedGaussianPolicy(actor_in, act_dim, hidden=cfg.policy.hidden, init_log_std=cfg.policy.init_log_std)
    actor = system.add("actor", policy, (actor_in,), label="Actor: squashed Gaussian")
    sample = _sample(system, act_dim)
    prev = system.block(
        "prev", label="Previous action", description="command in effect", input_shape=[act_dim],
        output_shape=[act_dim],
    )
    priv = system.block(
        "priv", label="Privileged", description="balance + true z, critic only",
        input_shape=[env.priv_dim + LATENT_DIM], output_shape=[priv_dim],
        notes=f"The env's privileged signals ({', '.join(env.priv_names)}) and z = E(all {n_taps} taps). The critic "
              "is training-only, so the autoencoder never runs on the wing.",
    )

    system.connect(plant, taps, label=f"{n_taps} taps", shape=[n_taps])
    system.connect(taps, encoder, label=f"{n_taps} taps")
    system.connect(taps, subsample, label=f"{n_taps} taps")
    system.connect(subsample, history, label=f"{k} taps")
    system.connect(history, g, label=f"{HISTORY_FRAMES} x {k} taps")
    system.connect(g, actor, label="z_hat", shape=[LATENT_DIM])
    system.connect(plant, prev, label="prev a", shape=[act_dim])
    system.connect(prev, actor, label="prev a")
    system.connect(actor, sample, label="mean, log std")
    system.connect(sample, plant, label="a")

    critic = system.add("critic", _critic(cfg, actor_in, priv_dim, act_dim), ((actor_in,), (priv_dim,), (act_dim,)),
                        label="Critic: twin Q")
    system.connect(plant, priv, label=", ".join(env.priv_names), shape=[env.priv_dim])
    system.connect(encoder, priv, label="z")
    system.connect(priv, critic, label="priv", to="priv")
    system.connect(g, critic, label="z_hat", to="obs", shape=[LATENT_DIM])
    system.connect(prev, critic, label="prev a", to="obs")
    system.connect(sample, critic, label="a", to="act")
    system.connect(critic, _losses(system, cfg), label="Q1, Q2")
    return system


def build_deployment(cfg: RunConfig) -> neurolume.System:
    """What runs on the wing: k taps, a fault mask, g and the actor. No balance, no encoder, no critic. The decoder
    only runs inside fault detection."""
    env = make_env(cfg, n_envs=1)
    n_taps, k = _tap_counts(env)
    act_dim = env.act_dim
    system = neurolume.System("Sparse-tap latent: deployment")
    wing = system.block("wing", label=f"Wing, {k} taps only", description="no balance", input_shape=[act_dim],
                        output_shape=[k])
    inversion = system.block(
        "inversion", label="Latent inversion", description="z* = argmin_z |mask * (D(z) - taps)|",
        input_shape=[k], output_shape=[k],
        notes=r"""
            No encoder: start from g's $\hat z$ (or the last $z^*$) and take gradient steps on $z$ through the frozen
            decoder until the fit to the working taps stops improving, with a robust loss (Huber or Cauchy):

            $$z^* = \arg\min_z \sum_i m_i\, \rho\big(D(z)_i - x_i\big)$$

            A broken tap pulls the fit, so the output is each tap's residual $D(z^*)_i - x_i$, and the drop in total
            residual when that tap is left out. Needs more working taps than the latent size. Too slow for the
            control loop (one optimisation per check), so it runs at a reduced rate. A large gap between $z^*$ and
            $\hat z$ means g is outside its training range.
        """,
    )
    decoder = system.add("D", TapAutoencoder(n_taps).decoder, (LATENT_DIM,),
                         label="Decoder D (frozen, fault detection only)")
    detector = system.block(
        "detector", label="Fault detector", description="0/1 validity mask", input_shape=[k], output_shape=[k],
        notes="""
            Per-tap checks (range, flatline, rate of change, excess noise) against healthy tunnel data, plus the
            inversion's residuals. Rank the residuals and remove the worst first; require N bad frames before
            masking and a longer healthy stretch before unmasking.
        """,
    )
    history = system.block(
        "history", label="Concat masked history", description=f"last {HISTORY_FRAMES} frames of concat(taps * mask, mask)",
        input_shape=[k], output_shape=[HISTORY_FRAMES * 2 * k],
        notes="Failed taps are zeroed so a faulty reading cannot leak through; the mask tells g which are missing.",
    )
    g = system.add("g", LatentRegressor(2 * k), (HISTORY_FRAMES * 2 * k,), label="g: MLP (mask-aware)")
    prev = system.block("prev", label="Previous action", output_shape=[act_dim])
    policy = SquashedGaussianPolicy(LATENT_DIM + act_dim, act_dim, hidden=cfg.policy.hidden,
                                    init_log_std=cfg.policy.init_log_std)
    actor = system.add("actor", policy, (LATENT_DIM + act_dim,), label="Actor (deterministic mean)")
    fail_safe = system.block(
        "fail_safe", label="Fail-safe", description="too many taps flagged", output_shape=[act_dim],
        notes="Hold the last action, fall back to a conservative policy, or end the run: still to decide.",
    )

    system.connect(wing, detector, label=f"{k} taps")
    system.connect(wing, inversion, label=f"{k} taps")
    system.connect(inversion, decoder, label="z", shape=[LATENT_DIM])
    system.connect(decoder, inversion, label=f"{n_taps} taps (reconstructed)")
    system.connect(g, inversion, label="z_hat", shape=[LATENT_DIM])
    system.connect(inversion, detector, label=f"{k} residuals")
    system.connect(wing, history, label=f"{k} taps")
    system.connect(detector, history, label="mask")
    system.connect(history, g, label=f"{HISTORY_FRAMES} x {k} taps + mask")
    system.connect(g, actor, label="z_hat", shape=[LATENT_DIM])
    system.connect(prev, actor, label="prev a")
    system.connect(detector, fail_safe, label="mask")
    system.connect(actor, fail_safe, label="a")
    system.connect(fail_safe, wing, label="a")
    return system


def build_workspace(cfg: RunConfig) -> neurolume.Workspace:
    """The run as configured and, for a wind tunnel run, the sparse-tap latent workflow."""
    workspace = neurolume.Workspace()
    if cfg.env.kind == "wind_tunnel":  # the latent workflow is built on the tunnel's pressure taps
        for system in (build_autoencoder_training(cfg), build_latent_training(cfg), build_deployment(cfg)):
            workspace.add(system.name, system)
    return workspace


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    build_workspace(RunConfig.from_toml(path)).show()


if __name__ == "__main__":
    main()
