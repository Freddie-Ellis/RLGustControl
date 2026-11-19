"""Train the tap autoencoder on CFD (Stage 1 of the sparse-tap latent plan) and save the run.

    python -m src.ml.ae.train_ae configs/autoencoder/ae_cfd.toml    settings from a TOML config (AETrainConfig)
    python -m src.ml.ae.train_ae                                    the same file, the default

Data: every converged CFD run (src/data/cfd.py), with Cp interpolated onto the physical taps chosen by
src/ml/taps/cfd_taps.py, and the loads in LOAD_HEADS from the solver. Taps and loads are standardised with the mean
and std of the training runs; the validation runs are held out of everything, the standardisation included.

Run folder, AE_RUNS_DIR/<YYYYmmdd-HHMMSS>_<name>/:
    config.json      the AETrainConfig used
    history.csv      one row per epoch: training loss, and reconstruction and load errors on both splits
    model.pt         the weights at the best validation loss, with what `load_autoencoder` needs to use them
    summary.json     the best epoch and its errors
    latents.parquet  per run: its split, alpha, beta, z, the loads and their predictions, its reconstruction error
    taps.parquet     the tap layout, with each tap's reconstruction error on the validation runs
    plots/           figures from src/ml/ae/plots.py, made at the end of training

Errors are reported in physical units (Cp, coefficients) and as R^2, so runs with different settings compare
directly; the losses are in standardised units.
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
from src.data.cfd import CFD_DATA_DIR
from src.ml.ae.common import Standardiser, TomlConfig, config_path, new_run_dir, r_squared, split_rows
from src.ml.ae.sparse_latent import AE_HIDDEN, LATENT_DIM, LOAD_HEADS, TapAutoencoder
from src.ml.atomic import atomic_write_text
from src.ml.taps.cfd_taps import OUTPUT_DIR as TAP_SELECTIONS_DIR
from src.ml.taps.cfd_taps import candidate_cp, load_runs
from src.ml.taps.tap_selection import SENSOR_NOISE_CP

AE_RUNS_DIR: Path = CONFIG.runs_dir / "autoencoder"
DEFAULT_CONFIG: Path = CONFIG.configs_dir / "autoencoder" / "ae_cfd.toml"
TAPS_FILE: Path = TAP_SELECTIONS_DIR / "taps.parquet"
LOG_EVERY: int = 250  # epochs between progress lines and history.csv writes


class AETrainConfig(TomlConfig):
    """Every setting of one training run. Saved as the run's config.json."""

    name: str = "ae"
    latent_dim: int = LATENT_DIM
    hidden: int = AE_HIDDEN
    epochs: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-4  # with ~100 runs, some shrinkage keeps the fit from chasing individual runs
    load_weight: float = 1.0  # weight of the load heads' loss against the reconstruction loss
    input_noise_cp: float = SENSOR_NOISE_CP  # Gaussian noise added to the input taps each epoch, in Cp
    val_fraction: float = 0.2  # runs held out for validation; 0 trains on all and keeps the last epoch
    seed: int = 0
    data_dir: Path = CFD_DATA_DIR
    taps_file: Path = TAPS_FILE


# ---- Data -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AEData:
    runs: pl.DataFrame  # N converged runs of runs.parquet
    taps: pl.DataFrame  # n_taps rows of the tap layout, in the order chosen
    cp: Tensor  # (N, n_taps) Cp at the taps
    loads: Tensor  # (N, len(LOAD_HEADS))


def load_data(data_dir: Path = CFD_DATA_DIR, taps_file: Path = TAPS_FILE) -> AEData:
    """Cp at the taps and the loads of every converged run. Raises if a tap is off some run's mesh."""
    runs = load_runs(data_dir)
    taps = pl.read_parquet(taps_file).sort("order")
    cp, covered = candidate_cp(runs, taps, data_dir)
    if not bool(covered.all()):
        raise ValueError(f"taps {(~covered).nonzero().squeeze(1).tolist()} are not on every run's surface")
    return AEData(runs=runs, taps=taps, cp=cp, loads=runs.select(LOAD_HEADS).to_torch(dtype=pl.Float64))


# ---- Using a trained autoencoder ------------------------------------------------------------------------------------

@dataclass
class TrainedAutoencoder:
    """A saved autoencoder with its standardisation: everything in physical units (Cp, coefficients)."""

    model: TapAutoencoder
    cp_scale: Standardiser
    load_scale: Standardiser
    load_heads: tuple[str, ...]

    def encode(self, cp: Tensor) -> Tensor:
        """Cp at the taps (N, n_taps) -> z (N, latent_dim)."""
        with torch.no_grad():
            return self.model(self.cp_scale(cp).float())

    def decode(self, z: Tensor) -> Tensor:
        """z -> Cp at every tap."""
        with torch.no_grad():
            return self.cp_scale.inverse(self.model.decode(z).double())

    def loads(self, z: Tensor) -> Tensor:
        """z -> the coefficients in `load_heads`."""
        with torch.no_grad():
            return self.load_scale.inverse(self.model.loads(z).double())


def save_autoencoder(path: Path, model: TapAutoencoder, cfg: AETrainConfig, cp_scale: Standardiser,
                     load_scale: Standardiser) -> None:
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_taps": len(cp_scale.mean),
        "latent_dim": cfg.latent_dim,
        "hidden": cfg.hidden,
        "load_heads": list(LOAD_HEADS),
        "cp_mean": cp_scale.mean, "cp_std": cp_scale.std,
        "load_mean": load_scale.mean, "load_std": load_scale.std,
    }, path)


def load_autoencoder(run_dir: Path) -> TrainedAutoencoder:
    """The best weights of a training run, ready to encode tap pressures."""
    saved = torch.load(run_dir / "model.pt", weights_only=True)
    model = TapAutoencoder(saved["n_taps"], saved["latent_dim"], saved["hidden"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return TrainedAutoencoder(
        model=model,
        cp_scale=Standardiser(saved["cp_mean"], saved["cp_std"]),
        load_scale=Standardiser(saved["load_mean"], saved["load_std"]),
        load_heads=tuple(saved["load_heads"]),
    )


# ---- Metrics --------------------------------------------------------------------------------------------------------

def errors(model: TapAutoencoder, x: Tensor, y: Tensor, cp_scale: Standardiser, load_scale: Standardiser,
           prefix: str) -> dict[str, float]:
    """Reconstruction and load errors of clean (noise-free) inputs, standardised and in physical units."""
    if len(x) == 0:
        return {}
    with torch.no_grad():
        z = model(x)
        recon, loads = model.decode(z), model.loads(z)
    out = {
        f"{prefix}recon": nn.functional.mse_loss(recon, x).item(),
        f"{prefix}load": nn.functional.mse_loss(loads, y).item(),
        f"{prefix}recon_rms_cp": ((recon - x).double() * cp_scale.std).pow(2).mean().sqrt().item(),
    }
    loads_phys, y_phys = load_scale.inverse(loads.double()), load_scale.inverse(y.double())
    r2 = r_squared(loads_phys, y_phys) if len(x) > 1 else torch.full((len(LOAD_HEADS),), math.nan)
    for k, name in enumerate(LOAD_HEADS):
        out[f"{prefix}{name}_rms"] = (loads_phys[:, k] - y_phys[:, k]).pow(2).mean().sqrt().item()
        out[f"{prefix}{name}_r2"] = r2[k].item()
    return out


# ---- Training -------------------------------------------------------------------------------------------------------

def train(cfg: AETrainConfig, data: AEData, root: Path = AE_RUNS_DIR) -> Path:
    """Train an autoencoder on `data` and save the run. Returns the run folder."""
    run_dir = new_run_dir(root, cfg.name)
    atomic_write_text(run_dir / "config.json", cfg.model_dump_json(indent=2))
    torch.manual_seed(cfg.seed)

    train_rows, val_rows = split_rows(data.runs.height, cfg.val_fraction, cfg.seed)
    cp_scale, load_scale = Standardiser.fit(data.cp[train_rows]), Standardiser.fit(data.loads[train_rows])
    x, y = cp_scale(data.cp).float(), load_scale(data.loads).float()
    x_train, y_train, x_val, y_val = x[train_rows], y[train_rows], x[val_rows], y[val_rows]
    noise = (cfg.input_noise_cp / cp_scale.std).float()  # the sensor noise in each tap's standardised units

    model = TapAutoencoder(x.shape[1], cfg.latent_dim, cfg.hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"RUN_DIR {run_dir}")
    print(f"{data.runs.height} runs ({len(train_rows)} training, {len(val_rows)} validation), {x.shape[1]} taps, "
          f"latent {cfg.latent_dim}, {sum(p.numel() for p in model.parameters())} parameters")

    history: list[dict[str, float]] = []
    best_loss, best_epoch, best_state = math.inf, -1, model.state_dict()
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        z = model(x_train + noise * torch.randn_like(x_train))
        recon_loss = nn.functional.mse_loss(model.decode(z), x_train)
        load_loss = nn.functional.mse_loss(model.loads(z), y_train)
        loss = recon_loss + cfg.load_weight * load_loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        model.eval()
        row = {"epoch": float(epoch), "train_loss": loss.item()}
        row |= errors(model, x_train, y_train, cp_scale, load_scale, "train_")
        row |= errors(model, x_val, y_val, cp_scale, load_scale, "val_")
        history.append(row)
        # Model selection on the validation loss; with no validation runs, the last epoch is kept.
        score = row["val_recon"] + cfg.load_weight * row["val_load"] if len(val_rows) else -epoch
        if score < best_loss:
            best_loss, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == cfg.epochs:
            pl.DataFrame(history).write_csv(run_dir / "history.csv")
            shown = row if not len(val_rows) else {k: v for k, v in row.items() if k.startswith("val_")}
            print(f"epoch {epoch:6d}  loss {loss.item():.4f}  "
                  + "  ".join(f"{k} {v:.4g}" for k, v in shown.items() if k.endswith(("recon_rms_cp", "_r2")))
                  + f"  ({time.perf_counter() - started:.0f} s)")

    model.load_state_dict(best_state)
    model.eval()
    save_autoencoder(run_dir / "model.pt", model, cfg, cp_scale, load_scale)
    _save_results(run_dir, cfg, data, model, x, y, train_rows, val_rows, cp_scale, load_scale, best_epoch)
    return run_dir


def _save_results(run_dir: Path, cfg: AETrainConfig, data: AEData, model: TapAutoencoder, x: Tensor, y: Tensor,
                  train_rows: Tensor, val_rows: Tensor, cp_scale: Standardiser, load_scale: Standardiser,
                  best_epoch: int) -> None:
    """summary.json, latents.parquet and taps.parquet for the best weights."""
    summary: dict[str, object] = {"best_epoch": best_epoch, "n_runs": data.runs.height,
                                  "n_train": len(train_rows), "n_val": len(val_rows)}
    summary |= errors(model, x[train_rows], y[train_rows], cp_scale, load_scale, "train_")
    summary |= errors(model, x[val_rows], y[val_rows], cp_scale, load_scale, "val_")
    atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))

    with torch.no_grad():
        z = model(x)
        recon, loads = model.decode(z).double(), load_scale.inverse(model.loads(z).double())
    split = ["train"] * data.runs.height
    for row in val_rows.tolist():
        split[row] = "val"
    tap_error = (recon - x.double()) * cp_scale.std  # (N, n_taps) in Cp
    data.runs.select("run", "alpha_deg", "beta_deg").with_columns(
        split=pl.Series(split),
        recon_rms_cp=pl.Series(tap_error.pow(2).mean(dim=1).sqrt().numpy()),
        **{f"z{i}": pl.Series(z[:, i].numpy()) for i in range(cfg.latent_dim)},
        **{name: pl.Series(data.loads[:, k].numpy()) for k, name in enumerate(LOAD_HEADS)},
        **{f"{name}_pred": pl.Series(loads[:, k].numpy()) for k, name in enumerate(LOAD_HEADS)},
    ).write_parquet(run_dir / "latents.parquet")

    rows = val_rows if len(val_rows) else train_rows
    data.taps.with_columns(
        recon_rms_cp=pl.Series(tap_error[rows].pow(2).mean(dim=0).sqrt().numpy())
    ).write_parquet(run_dir / "taps.parquet")

    print(f"best epoch {best_epoch}: " + ", ".join(
        f"{k} {summary[k]:.4g}" for k in summary if k.startswith("val_") and k.endswith(("rms_cp", "r2"))
    ))


def main(config: Path = DEFAULT_CONFIG) -> None:
    """Train with the settings in a TOML config file, then draw the run's figures."""
    from src.ml.ae.plots import plot_run  # only training needs matplotlib, not loading a trained model

    cfg = AETrainConfig.from_toml(config)
    run_dir = train(cfg, load_data(cfg.data_dir, cfg.taps_file))
    for path in plot_run(run_dir):
        print(f"saved {path}")


if __name__ == "__main__":
    main(config_path(DEFAULT_CONFIG))
