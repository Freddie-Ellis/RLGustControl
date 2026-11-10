"""The sparse-tap latent state the deployed SAC actor sees (see the delta wing sparse-tap latent workflow plan).

    TapAutoencoder   all the tunnel's taps -> latent z -> taps again, plus force and moment heads for training. Its
                     encoder E defines the latent; frozen during SAC, where z = E(taps) is privileged (critic only).
                     Its decoder D is kept for fault detection
    LatentRegressor  g: the k deployed taps over a short history -> z_hat, regressed to E(taps). The actor's input

At deployment only the k taps, g and the actor run. The sizes below are the plan's starting guesses, to be fixed by its
experiments; neither network is trained yet.
"""

from torch import Tensor, nn

LATENT_DIM = 6  # size of E's latent; plan's starting guess 3 to 8
K_TAPS = 10  # taps kept at deployment; plan's starting guess 8 to 12, chosen at the knee of error against k
HISTORY_FRAMES = 4  # recent frames g sees, because pressure depends on history (hysteresis, unsteady effects)
LOAD_HEADS = ("C_l", "C_m")  # coefficients the autoencoder's auxiliary heads predict from the latent
AE_HIDDEN = 64  # width of the autoencoder's encoder and decoder layers
G_HIDDEN = 128  # width of g's layers


def _mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, n_out))


class TapAutoencoder(nn.Module):
    """Taps -> z -> taps, with heads predicting the loads in LOAD_HEADS from z, so the latent carries the forces."""

    def __init__(self, n_taps: int, latent_dim: int = LATENT_DIM, hidden: int = AE_HIDDEN) -> None:
        super().__init__()
        self.encoder = _mlp(n_taps, hidden, latent_dim)
        self.decoder = _mlp(latent_dim, hidden, n_taps)
        self.heads = nn.Linear(latent_dim, len(LOAD_HEADS))

    def forward(self, taps: Tensor) -> Tensor:
        """E: [N, n_taps] -> z [N, latent_dim]. The decoder and heads are only for training and fault detection."""
        return self.encoder(taps)

    def decode(self, z: Tensor) -> Tensor:
        """D: z -> all the taps. Trained with the encoder; afterwards only fault detection uses it (latent inversion:
        the z that best fits the working taps, then each tap's residual against D(z))."""
        return self.decoder(z)

    def loads(self, z: Tensor) -> Tensor:
        """The coefficients in LOAD_HEADS from z: a training loss only, so the latent carries the forces."""
        return self.heads(z)


class LatentRegressor(nn.Module):
    """g: the last `history_frames` frames of the deployed taps, concatenated oldest first, -> z_hat.

    `inputs_per_frame` is k, or 2k for the mask-aware g, whose frames are concat(taps * mask, mask)."""

    def __init__(self, inputs_per_frame: int, history_frames: int = HISTORY_FRAMES, latent_dim: int = LATENT_DIM,
                 hidden: int = G_HIDDEN) -> None:
        super().__init__()
        self.net = _mlp(inputs_per_frame * history_frames, hidden, latent_dim)

    def forward(self, history: Tensor) -> Tensor:
        """[N, history_frames * inputs_per_frame] -> z_hat [N, latent_dim]."""
        return self.net(history)

