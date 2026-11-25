"""Train the tap autoencoder on CFD (Stage 1 of the sparse-tap latent plan) and save the run.

    python -m src.ml.ae.train_ae configs/autoencoder/ae_cfd.toml    settings from a TOML config (AETrainConfig)
    python -m src.ml.ae.train_ae                                    the same file, the default

Data: every CFD run (src/data/cfd.py), with Cp interpolated onto the physical taps chosen by src/ml/taps/cfd_taps.py,
and the loads in LOAD_HEADS from the solver. Taps and loads are standardised with the mean and std of the runs a model
is trained on.

The saved model is trained on every run. With `folds` > 1, k-fold cross-validation comes first and decides how long
to train it, and gives the validation errors:
    1. The runs are cut into `folds` groups. One model per group is trained on the other groups for `epochs` epochs and
       validated on its own group, so every run is validated once, by a model that never saw it.
    2. The validation curves are averaged over the folds; the epoch where the average validation loss is lowest is the
       best epoch.
    3. Each fold's model is trained again to the best epoch, giving every run an out-of-fold ("cross-validated")
       reconstruction and load prediction: the validation errors in summary.json.
    4. The saved model is trained on every run for the best epoch's number of epochs.
With `folds` <= 1 there is no validation: the saved model is trained on every run for `epochs` epochs.

Run folder, AE_RUNS_DIR/<YYYYmmdd-HHMMSS>_<name>/:
    config.json      the AETrainConfig used
    history.csv      one row per epoch: training loss, and reconstruction and load errors on both splits, averaged
                     over the folds (with no folds: the saved model's training errors). Rewritten after each fold.
    model.pt         the saved model's weights, with what `load_autoencoder` needs to use them
    summary.json     the best epoch; the saved model's errors on every run (train_*), and the cross-validated errors
                     (val_*) pooled over every run's out-of-fold prediction
    latents.parquet  per run: alpha, beta, z, the loads and their predictions, its reconstruction error, from the
                     saved model; with folds, also its fold and its cross-validated predictions and error (*_cv)
    taps.parquet     the tap layout, with each tap's reconstruction error (cross-validated, if there are folds)
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
from src.ml.ae.common import (
    Standardiser,
    TomlConfig,
    config_path,
    fold_rows,
    new_run_dir,
    r_squared,
)
from src.ml.ae.sparse_latent import AE_HIDDEN, LATENT_DIM, LOAD_HEADS, TapAutoencoder
from src.ml.atomic import atomic_write_text
from src.ml.taps.cfd_taps import OUTPUT_DIR as TAP_SELECTIONS_DIR
from src.ml.taps.cfd_taps import candidate_cp, load_runs
from src.ml.taps.tap_selection import SENSOR_NOISE_CP

AE_RUNS_DIR: Path = CONFIG.runs_dir / "autoencoder"
DEFAULT_CONFIG: Path = CONFIG.configs_dir / "autoencoder" / "ae_cfd.toml"
TAPS_FILE: Path = TAP_SELECTIONS_DIR / "taps.parquet"
LOG_EVERY: int = 250  # epochs between progress lines


class AETrainConfig(TomlConfig):
    """Every setting of one training run. Saved as the run's config.json."""

    name: str = "ae"
    latent_dim: int = LATENT_DIM
    hidden: int = AE_HIDDEN
    epochs: int = 5000  # with folds: the most epochs cross-validation tries; without: the epochs trained
    lr: float = 1e-3
    weight_decay: float = 1e-4  # with ~150 runs, some shrinkage keeps the fit from chasing individual runs
    load_weight: float = 1.0  # weight of the load heads' loss against the reconstruction loss
    input_noise_cp: float = SENSOR_NOISE_CP  # Gaussian noise added to the input taps each epoch, in Cp
    folds: int = 5  # cross-validation folds (see the module docstring); <= 1 trains on every run with no validation
    seed: int = 0
    data_dir: Path = CFD_DATA_DIR
    taps_file: Path = TAPS_FILE


# ---- Data -----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AEData:
    runs: pl.DataFrame  # N runs of runs.parquet
    taps: pl.DataFrame  # n_taps rows of the tap layout, in the order chosen
    cp: Tensor  # (N, n_taps) Cp at the taps
    loads: Tensor  # (N, len(LOAD_HEADS))


def load_data(data_dir: Path = CFD_DATA_DIR, taps_file: Path = TAPS_FILE) -> AEData:
    """Cp at the taps and the loads of every run. Raises if a tap is off some run's mesh."""
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
    """The saved model of a training run, ready to encode tap pressures."""
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
    out |= load_errors(load_scale.inverse(loads.double()), load_scale.inverse(y.double()), prefix)
    return out


def load_errors(pred: Tensor, true: Tensor, prefix: str) -> dict[str, float]:
    """RMS error and R^2 of each load in LOAD_HEADS, in physical units."""
    r2 = r_squared(pred, true) if len(true) > 1 else torch.full((len(LOAD_HEADS),), math.nan)
    out: dict[str, float] = {}
    for k, name in enumerate(LOAD_HEADS):
        out[f"{prefix}{name}_rms"] = (pred[:, k] - true[:, k]).pow(2).mean().sqrt().item()
        out[f"{prefix}{name}_r2"] = r2[k].item()
    return out


# ---- Training -------------------------------------------------------------------------------------------------------

@dataclass
class Fitted:
    """A model trained on some of the runs, with the standardisation fitted on those runs."""

    model: TapAutoencoder
    cp_scale: Standardiser
    load_scale: Standardiser
    history: list[dict[str, float]]  # one row per epoch

    def predict(self, cp: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Cp at the taps (N, n_taps) -> (z, reconstructed Cp, loads), in physical units."""
        with torch.no_grad():
            z = self.model(self.cp_scale(cp).float())
            return (z, self.cp_scale.inverse(self.model.decode(z).double()),
                    self.load_scale.inverse(self.model.loads(z).double()))


def fit(cfg: AETrainConfig, data: AEData, train_rows: Tensor, val_rows: Tensor, epochs: int, label: str) -> Fitted:
    """Train a fresh model on `train_rows` for `epochs` epochs, recording its errors on both sets every epoch.

    Seeded from cfg.seed, so the same rows and settings give the same model: training a fold again to fewer epochs
    gives the weights it had at that epoch."""
    torch.manual_seed(cfg.seed)
    cp_scale, load_scale = Standardiser.fit(data.cp[train_rows]), Standardiser.fit(data.loads[train_rows])
    x, y = cp_scale(data.cp).float(), load_scale(data.loads).float()
    x_train, y_train, x_val, y_val = x[train_rows], y[train_rows], x[val_rows], y[val_rows]
    noise = (cfg.input_noise_cp / cp_scale.std).float()  # the sensor noise in each tap's standardised units

    model = TapAutoencoder(x.shape[1], cfg.latent_dim, cfg.hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        z = model(x_train + noise * torch.randn_like(x_train))
        loss = nn.functional.mse_loss(model.decode(z), x_train) + cfg.load_weight * nn.functional.mse_loss(
            model.loads(z), y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        model.eval()
        row = {"epoch": float(epoch), "train_loss": loss.item()}
        row |= errors(model, x_train, y_train, cp_scale, load_scale, "train_")
        row |= errors(model, x_val, y_val, cp_scale, load_scale, "val_")
        history.append(row)
        if epoch % LOG_EVERY == 0 or epoch == epochs:
            shown = {k: v for k, v in row.items() if k.startswith("val_" if len(val_rows) else "train_")}
            print(f"{label} epoch {epoch:6d}  loss {loss.item():.4f}  "
                  + "  ".join(f"{k} {v:.4g}" for k, v in shown.items() if k.endswith(("recon_rms_cp", "_r2")))
                  + f"  ({time.perf_counter() - started:.0f} s)")
    model.eval()
    return Fitted(model, cp_scale, load_scale, history)


@dataclass(frozen=True)
class CrossValidation:
    best_epoch: int
    history: pl.DataFrame  # one row per epoch, every column averaged over the folds
    fold: Tensor  # (N,) the fold that held each run out
    cp: Tensor  # (N, n_taps) each run's out-of-fold reconstruction, Cp
    loads: Tensor  # (N, len(LOAD_HEADS)) each run's out-of-fold load prediction


def score(history: pl.DataFrame, load_weight: float) -> pl.Series:
    """The validation loss each epoch: what the best epoch minimises."""
    return history["val_recon"] + load_weight * history["val_load"]


def cross_validate(cfg: AETrainConfig, data: AEData, run_dir: Path) -> CrossValidation:
    """Steps 1 to 3 of the module docstring."""
    folds = fold_rows(data.runs.height, cfg.folds, cfg.seed)
    histories: list[pl.DataFrame] = []
    mean = pl.DataFrame()
    for k, (train_rows, val_rows) in enumerate(folds):
        histories.append(pl.DataFrame(fit(cfg, data, train_rows, val_rows, cfg.epochs, f"fold {k + 1}/{cfg.folds}").history))
        mean = pl.concat(histories).group_by("epoch", maintain_order=True).mean()
        mean.write_csv(run_dir / "history.csv")  # so far: watch it while the later folds train
    best_epoch = int(mean["epoch"][int(score(mean, cfg.load_weight).arg_min() or 0)])
    print(f"best epoch {best_epoch} of {cfg.epochs} (lowest validation loss averaged over {cfg.folds} folds)")

    fold = torch.empty(data.runs.height, dtype=torch.int64)
    cp, loads = torch.empty_like(data.cp), torch.empty_like(data.loads)
    for k, (train_rows, val_rows) in enumerate(folds):
        fitted = fit(cfg, data, train_rows, val_rows, best_epoch, f"fold {k + 1}/{cfg.folds} to the best epoch")
        _, cp[val_rows], loads[val_rows] = fitted.predict(data.cp[val_rows])
        fold[val_rows] = k
    return CrossValidation(best_epoch, mean, fold, cp, loads)


def train(cfg: AETrainConfig, data: AEData, root: Path = AE_RUNS_DIR) -> Path:
    """Cross-validate (if cfg.folds > 1), train the saved model on every run and save the run. Returns its folder."""
    run_dir = new_run_dir(root, cfg.name)
    atomic_write_text(run_dir / "config.json", cfg.model_dump_json(indent=2))
    n_params = sum(p.numel() for p in TapAutoencoder(data.cp.shape[1], cfg.latent_dim, cfg.hidden).parameters())
    print(f"RUN_DIR {run_dir}")
    print(f"{data.runs.height} runs, {data.cp.shape[1]} taps, latent {cfg.latent_dim}, {n_params} parameters, "
          + (f"{cfg.folds}-fold cross-validation" if cfg.folds > 1 else "no validation"))

    cv = cross_validate(cfg, data, run_dir) if cfg.folds > 1 else None
    every_run = torch.arange(data.runs.height)
    final = fit(cfg, data, every_run, every_run[:0], cv.best_epoch if cv else cfg.epochs, "all runs")
    if cv is None:
        pl.DataFrame(final.history).write_csv(run_dir / "history.csv")
    save_autoencoder(run_dir / "model.pt", final.model, cfg, final.cp_scale, final.load_scale)
    _save_results(run_dir, data, final, cv)
    return run_dir


def _save_results(run_dir: Path, data: AEData, final: Fitted, cv: CrossValidation | None) -> None:
    """summary.json, latents.parquet and taps.parquet."""
    z, cp, loads = final.predict(data.cp)
    summary: dict[str, object] = {"best_epoch": cv.best_epoch if cv else len(final.history), "n_runs": data.runs.height,
                                  "folds": int(cv.fold.max()) + 1 if cv else 0}
    summary["train_recon_rms_cp"] = (cp - data.cp).pow(2).mean().sqrt().item()
    summary |= load_errors(loads, data.loads, "train_")
    if cv:
        summary["val_recon_rms_cp"] = (cv.cp - data.cp).pow(2).mean().sqrt().item()
        summary |= load_errors(cv.loads, data.loads, "val_")
    atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))

    def rms_per_run(pred: Tensor) -> pl.Series:
        return pl.Series((pred - data.cp).pow(2).mean(dim=1).sqrt().numpy())

    latents = data.runs.select("run", "alpha_deg", "beta_deg").with_columns(
        split=pl.lit("train"),  # the saved model is trained on every run
        recon_rms_cp=rms_per_run(cp),
        **{f"z{i}": pl.Series(z[:, i].numpy()) for i in range(z.shape[1])},
        **{name: pl.Series(data.loads[:, k].numpy()) for k, name in enumerate(LOAD_HEADS)},
        **{f"{name}_pred": pl.Series(loads[:, k].numpy()) for k, name in enumerate(LOAD_HEADS)},
    )
    if cv:
        latents = latents.with_columns(
            fold=pl.Series(cv.fold.numpy()),
            recon_rms_cp_cv=rms_per_run(cv.cp),
            **{f"{name}_cv": pl.Series(cv.loads[:, k].numpy()) for k, name in enumerate(LOAD_HEADS)},
        )
    latents.write_parquet(run_dir / "latents.parquet")

    tap_error = (cv.cp if cv else cp) - data.cp
    data.taps.with_columns(
        recon_rms_cp=pl.Series(tap_error.pow(2).mean(dim=0).sqrt().numpy())
    ).write_parquet(run_dir / "taps.parquet")

    print(f"best epoch {summary['best_epoch']}: " + ", ".join(
        f"{k} {v:.4g}" for k, v in summary.items() if isinstance(v, float) and k.endswith(("rms_cp", "r2"))
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
