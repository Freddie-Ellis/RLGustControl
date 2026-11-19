"""Train the latent regressor g on CFD (Stage 3 of the sparse-tap latent plan) and save the run.

    python -m src.ml.ae.train_regressor configs/latent_regressor/g_cfd.toml   settings from a TOML config
    python -m src.ml.ae.train_regressor                                       the same file, the default

g maps the k deployed taps over the last `history_frames` frames to z_hat, regressed onto z = E(all taps) from a
trained autoencoder run (`autoencoder_run`), whose encoder is frozen. The deployed taps are the first k of that run's
tap layout (the layout is nested: the first k are the best k). The data, the standardisation of the k taps and the
training/validation split are all the autoencoder run's own, so g is validated on runs neither network was trained on.

CFD is steady, so a history window is one snapshot repeated, with independent sensor noise on each frame: g learns
the steady map, and the window only carries information once g is fine-tuned on tunnel time series. Training inputs
are augmented as the plan asks: a per-tap offset and gain error held over the window, and noise on every frame. With
`mask_aware`, each sample also loses up to `max_dropped` taps, and g sees concat(taps * mask, mask) per frame.

Run folder, REGRESSOR_RUNS_DIR/<YYYYmmdd-HHMMSS>_<name>/:
    config.json      the RegressorTrainConfig used
    history.csv      one row per epoch: loss, latent error and R^2, and the loads and taps recovered from z_hat
    model.pt         the weights at the best validation loss, with what `load_regressor` needs to use them
    summary.json     the best epoch and its errors, next to the autoencoder's own (z from all taps) for comparison
    latents.parquet  per run: its split, z and z_hat, and the loads from each
    taps.parquet     the k deployed taps; with mask_aware, the validation latent error when each is lost alone
    plots/           figures from src/ml/ae/plots.py
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from torch import Tensor, nn

from src.config import CONFIG
from src.ml.ae.common import Standardiser, TomlConfig, config_path, new_run_dir, r_squared
from src.ml.ae.sparse_latent import G_HIDDEN, HISTORY_FRAMES, K_TAPS, LOAD_HEADS, LatentRegressor
from src.ml.ae.train_ae import LOG_EVERY, AETrainConfig, TrainedAutoencoder, load_autoencoder, load_data
from src.ml.atomic import atomic_write_text
from src.ml.taps.tap_selection import SENSOR_NOISE_CP

REGRESSOR_RUNS_DIR: Path = CONFIG.runs_dir / "latent_regressor"
DEFAULT_CONFIG: Path = CONFIG.configs_dir / "latent_regressor" / "g_cfd.toml"


class RegressorTrainConfig(TomlConfig):
    """Every setting of one training run. Saved as the run's config.json."""

    name: str = "g"
    autoencoder_run: Path  # the trained autoencoder run whose frozen encoder gives the targets
    k_taps: int = K_TAPS
    history_frames: int = HISTORY_FRAMES
    hidden: int = G_HIDDEN
    mask_aware: bool = False  # input concat(taps * mask, mask) per frame, trained with random tap loss
    max_dropped: int = 2  # mask_aware only: the most taps a training sample loses
    epochs: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    noise_cp: float = SENSOR_NOISE_CP  # sensor noise, independent on every frame, in Cp
    bias_cp: float = 0.01  # placeholder: std of each tap's offset, held over the window, in Cp
    gain_error: float = 0.02  # placeholder: std of each tap's fractional gain error, held over the window
    seed: int = 0


# ---- Data -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RegressorData:
    runs: pl.DataFrame  # N runs, as in the autoencoder run
    taps: pl.DataFrame  # the k deployed taps
    cp: Tensor  # (N, n_taps) Cp at all the autoencoder's taps; the deployed ones are the first k columns
    z: Tensor  # (N, latent_dim) targets: E(all taps)
    loads: Tensor  # (N, len(LOAD_HEADS)) from CFD
    train_rows: Tensor
    val_rows: Tensor


def load_regressor_data(autoencoder_run: Path, k: int) -> tuple[RegressorData, TrainedAutoencoder]:
    """The autoencoder run's data, split and encoder, with z = E(all taps) for every run."""
    ae = load_autoencoder(autoencoder_run)
    ae_cfg = AETrainConfig.model_validate_json((autoencoder_run / "config.json").read_text())
    data = load_data(ae_cfg.data_dir, autoencoder_run / "taps.parquet")
    if not 0 < k <= data.cp.shape[1]:
        raise ValueError(f"k_taps = {k}, but the autoencoder has {data.cp.shape[1]} taps")
    ae_split = pl.read_parquet(autoencoder_run / "latents.parquet").select("run", "split")
    split = data.runs.select("run").join(ae_split, on="run", how="left", maintain_order="left")["split"]
    if unknown := split.null_count():
        raise ValueError(f"{unknown} runs are newer than the autoencoder run {autoencoder_run.name}; retrain it first, "
                         "so every run has a training or validation label")
    is_val = torch.from_numpy((split == "val").to_numpy())
    return RegressorData(
        runs=data.runs,
        taps=data.taps.head(k),
        cp=data.cp,
        z=ae.encode(data.cp).double(),
        loads=data.loads,
        train_rows=(~is_val).nonzero().squeeze(1),
        val_rows=is_val.nonzero().squeeze(1),
    ), ae


# ---- Inputs ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class InputBuilder:
    """Turns Cp at the k taps into g's input: standardised frames, oldest first, flattened."""

    cp_scale: Standardiser  # of the k taps, the autoencoder's own
    history_frames: int
    mask_aware: bool

    @property
    def inputs_per_frame(self) -> int:
        return 2 * len(self.cp_scale.mean) if self.mask_aware else len(self.cp_scale.mean)

    def __call__(self, cp: Tensor, mask: Tensor | None = None) -> Tensor:
        """cp (N, history_frames, k) in Cp, mask (N, k) of 1 = valid (None: all valid) -> (N, history_frames * inputs)."""
        x = self.cp_scale(cp)
        if self.mask_aware:
            m = torch.ones(cp.shape[0], 1, cp.shape[2], dtype=x.dtype) if mask is None else mask[:, None, :].to(x.dtype)
            x = torch.cat([x * m, m.expand_as(x)], dim=2)  # a lost tap reads 0 and is flagged, never leaks through
        return x.flatten(start_dim=1).float()


def steady_window(cp: Tensor, frames: int) -> Tensor:
    """A steady snapshot (N, k) as a window of identical frames (N, frames, k)."""
    return cp[:, None, :].expand(-1, frames, -1)


def augmented_window(cp: Tensor, cfg: RegressorTrainConfig) -> Tensor:
    """A steady snapshot (N, k) as a window of noisy frames: one offset and gain error per tap, held over the window,
    and independent noise on every frame."""
    n, k = cp.shape
    gain = 1.0 + cfg.gain_error * torch.randn(n, 1, k, dtype=cp.dtype)
    bias = cfg.bias_cp * torch.randn(n, 1, k, dtype=cp.dtype)
    noise = cfg.noise_cp * torch.randn(n, cfg.history_frames, k, dtype=cp.dtype)
    return steady_window(cp, cfg.history_frames) * gain + bias + noise


def random_mask(n: int, k: int, max_dropped: int) -> Tensor:
    """(n, k) of 1 = valid: each row loses a random number of taps, 0 to max_dropped, chosen at random."""
    dropped = torch.randint(0, max_dropped + 1, (n, 1))
    rank = torch.rand(n, k).argsort(dim=1).argsort(dim=1)  # a random order of the taps in each row
    return (rank >= dropped).double()


# ---- Using a trained regressor --------------------------------------------------------------------------------------

@dataclass
class TrainedRegressor:
    """A saved g with its input and output scaling: Cp at the k deployed taps in, z_hat in the latent's units."""

    model: LatentRegressor
    inputs: InputBuilder
    z_scale: Standardiser

    def estimate(self, cp: Tensor, mask: Tensor | None = None) -> Tensor:
        """cp (N, history_frames, k), oldest frame first; mask (N, k) of 1 = valid, mask-aware g only -> z_hat."""
        with torch.no_grad():
            return self.z_scale.inverse(self.model(self.inputs(cp, mask)).double())


def load_regressor(run_dir: Path) -> TrainedRegressor:
    saved = torch.load(run_dir / "model.pt", weights_only=True)
    inputs = InputBuilder(Standardiser(saved["cp_mean"], saved["cp_std"]), saved["history_frames"], saved["mask_aware"])
    model = LatentRegressor(inputs.inputs_per_frame, saved["history_frames"], saved["latent_dim"], saved["hidden"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return TrainedRegressor(model, inputs, Standardiser(saved["z_mean"], saved["z_std"]))


# ---- Metrics --------------------------------------------------------------------------------------------------------

def errors(z_hat: Tensor, data: RegressorData, ae: TrainedAutoencoder, rows: Tensor, prefix: str) -> dict[str, float]:
    """How well z_hat (len(rows), latent_dim), in the latent's units, stands in for z on these runs: its own error,
    and the loads and all-tap pressures the autoencoder recovers from it."""
    if len(rows) == 0:
        return {}
    z = data.z[rows]
    z_r2 = r_squared(z_hat, z) if len(rows) > 1 else torch.full((z.shape[1],), math.nan)
    out = {f"{prefix}z_rms": (z_hat - z).pow(2).mean().sqrt().item(), f"{prefix}z_r2": z_r2.mean().item()}
    loads, true = ae.loads(z_hat.float()), data.loads[rows]
    loads_r2 = r_squared(loads, true) if len(rows) > 1 else torch.full((len(LOAD_HEADS),), math.nan)
    for k, name in enumerate(LOAD_HEADS):
        out[f"{prefix}{name}_rms"] = (loads[:, k] - true[:, k]).pow(2).mean().sqrt().item()
        out[f"{prefix}{name}_r2"] = loads_r2[k].item()
    out[f"{prefix}recon_rms_cp"] = (ae.decode(z_hat.float()) - data.cp[rows]).pow(2).mean().sqrt().item()
    return out


# ---- Training -------------------------------------------------------------------------------------------------------

def train(cfg: RegressorTrainConfig, root: Path = REGRESSOR_RUNS_DIR) -> Path:
    """Train g against the autoencoder run in `cfg` and save the run. Returns the run folder."""
    data, ae = load_regressor_data(cfg.autoencoder_run, cfg.k_taps)
    run_dir = new_run_dir(root, cfg.name)
    atomic_write_text(run_dir / "config.json", cfg.model_dump_json(indent=2))
    torch.manual_seed(cfg.seed)

    k, train_rows, val_rows = cfg.k_taps, data.train_rows, data.val_rows
    cp_k = data.cp[:, :k]
    inputs = InputBuilder(Standardiser(ae.cp_scale.mean[:k], ae.cp_scale.std[:k]), cfg.history_frames, cfg.mask_aware)
    z_scale = Standardiser.fit(data.z[train_rows])
    target = z_scale(data.z).float()
    clean = inputs(steady_window(cp_k, cfg.history_frames))  # every run, no noise and no lost taps

    model = LatentRegressor(inputs.inputs_per_frame, cfg.history_frames, data.z.shape[1], cfg.hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"RUN_DIR {run_dir}")
    print(f"{data.runs.height} runs ({len(train_rows)} training, {len(val_rows)} validation), {k} of "
          f"{data.cp.shape[1]} taps, {cfg.history_frames} frames, {'mask-aware, ' if cfg.mask_aware else ''}"
          f"{sum(p.numel() for p in model.parameters())} parameters")

    def estimate(rows: Tensor) -> Tensor:
        with torch.no_grad():
            return z_scale.inverse(model(clean[rows]).double())

    history: list[dict[str, float]] = []
    best_loss, best_epoch, best_state = math.inf, -1, model.state_dict()
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        mask = random_mask(len(train_rows), k, cfg.max_dropped) if cfg.mask_aware else None
        x = inputs(augmented_window(cp_k[train_rows], cfg), mask)
        loss = nn.functional.mse_loss(model(x), target[train_rows])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        model.eval()
        row = {"epoch": float(epoch), "train_loss": loss.item()}
        row |= errors(estimate(train_rows), data, ae, train_rows, "train_")
        row |= errors(estimate(val_rows), data, ae, val_rows, "val_")
        history.append(row)
        score = row["val_z_rms"] if len(val_rows) else -epoch  # no validation runs: keep the last epoch
        if score < best_loss:
            best_loss, best_epoch = score, epoch
            best_state = {name: v.detach().clone() for name, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == cfg.epochs:
            pl.DataFrame(history).write_csv(run_dir / "history.csv")
            prefix = "val_" if len(val_rows) else "train_"
            print(f"epoch {epoch:6d}  loss {loss.item():.4f}  "
                  + "  ".join(f"{k_} {v:.4g}" for k_, v in row.items()
                              if k_.startswith(prefix) and k_.endswith(("z_rms", "_r2", "recon_rms_cp")))
                  + f"  ({time.perf_counter() - started:.0f} s)")

    model.load_state_dict(best_state)
    model.eval()
    torch.save({
        "state_dict": {name: v.detach().cpu() for name, v in model.state_dict().items()},
        "k_taps": k, "history_frames": cfg.history_frames, "mask_aware": cfg.mask_aware, "hidden": cfg.hidden,
        "latent_dim": data.z.shape[1],
        "cp_mean": inputs.cp_scale.mean, "cp_std": inputs.cp_scale.std,
        "z_mean": z_scale.mean, "z_std": z_scale.std,
    }, run_dir / "model.pt")
    _save_results(run_dir, cfg, data, ae, inputs, model, z_scale, clean, best_epoch)
    return run_dir


def _save_results(run_dir: Path, cfg: RegressorTrainConfig, data: RegressorData, ae: TrainedAutoencoder,
                  inputs: InputBuilder, model: LatentRegressor, z_scale: Standardiser, clean: Tensor,
                  best_epoch: int) -> None:
    """summary.json, latents.parquet and taps.parquet for the best weights."""
    with torch.no_grad():
        z_hat = z_scale.inverse(model(clean).double())
    rows = data.val_rows if len(data.val_rows) else data.train_rows
    summary: dict[str, object] = {"best_epoch": best_epoch, "n_runs": data.runs.height,
                                  "n_train": len(data.train_rows), "n_val": len(data.val_rows)}
    summary |= errors(z_hat[data.train_rows], data, ae, data.train_rows, "train_")
    summary |= errors(z_hat[data.val_rows], data, ae, data.val_rows, "val_")
    summary |= errors(data.z[rows], data, ae, rows, "autoencoder_")  # the same, from z = E(all taps): g's ceiling

    taps = data.taps
    if cfg.mask_aware:  # the latent error on the validation runs when one tap is lost, for each tap in turn
        lost = []
        for j in range(cfg.k_taps):
            mask = torch.ones(len(rows), cfg.k_taps, dtype=torch.float64)
            mask[:, j] = 0.0
            with torch.no_grad():
                z_lost = z_scale.inverse(model(inputs(steady_window(data.cp[rows, :cfg.k_taps], cfg.history_frames),
                                                      mask)).double())
            lost.append((z_lost - data.z[rows]).pow(2).mean().sqrt().item())
        taps = taps.with_columns(lost_z_rms=pl.Series(lost))
        summary["worst_single_lost_z_rms"] = max(lost)
    atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))
    taps.write_parquet(run_dir / "taps.parquet")

    split = ["train"] * data.runs.height
    for r in data.val_rows.tolist():
        split[r] = "val"
    loads_z, loads_hat = ae.loads(data.z.float()), ae.loads(z_hat.float())
    latent_dim = data.z.shape[1]
    data.runs.select("run", "alpha_deg", "beta_deg").with_columns(
        split=pl.Series(split),
        **{f"z{i}": pl.Series(data.z[:, i].numpy()) for i in range(latent_dim)},
        **{f"z{i}_hat": pl.Series(z_hat[:, i].numpy()) for i in range(latent_dim)},
        **{name: pl.Series(data.loads[:, j].numpy()) for j, name in enumerate(LOAD_HEADS)},
        **{f"{name}_from_z": pl.Series(loads_z[:, j].numpy()) for j, name in enumerate(LOAD_HEADS)},
        **{f"{name}_from_z_hat": pl.Series(loads_hat[:, j].numpy()) for j, name in enumerate(LOAD_HEADS)},
    ).write_parquet(run_dir / "latents.parquet")

    print(f"best epoch {best_epoch}: " + ", ".join(
        f"{k} {summary[k]:.4g}" for k in summary
        if k.startswith(("val_", "autoencoder_", "worst_")) and k.endswith(("z_rms", "_r2", "recon_rms_cp"))
    ))


def main(config: Path = DEFAULT_CONFIG) -> None:
    """Train with the settings in a TOML config file, then draw the run's figures."""
    from src.ml.ae.plots import plot_regressor_run  # matplotlib is only needed here

    run_dir = train(RegressorTrainConfig.from_toml(config))
    for path in plot_regressor_run(run_dir):
        print(f"saved {path}")


if __name__ == "__main__":
    main(config_path(DEFAULT_CONFIG))
