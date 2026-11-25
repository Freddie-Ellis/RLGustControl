"""Figures for a training run of the autoencoder (train_ae.py) or the latent regressor g (train_regressor.py), drawn
from the run's saved files only.

    python -m src.ml.ae.plots                  the newest run of either
    python -m src.ml.ae.plots <run folder>     a given run (the kind is read from its config)
    python -m src.ml.ae.plots --show           also open the figures

Saved in <run folder>/plots/. Autoencoder run:
    history.png    losses and validation errors against epoch, best epoch marked
    loads.png      predicted against true loads: the saved model's, and the cross-validated (out-of-fold) ones, with
                   the cross-validated R^2
    latent.png     each latent dimension against alpha, coloured by sideslip
    taps.png       each tap's validation (cross-validated) reconstruction error on the planform
Latent regressor run:
    history.png    latent error, R^2 of z and of the loads, and all-tap Cp error, against the autoencoder's ceiling
    latent.png     z_hat = g(k taps) against z = E(all taps), per dimension
    loads.png      the loads recovered from z_hat and from z, on the validation runs
    taps.png       the deployed taps; for a mask-aware g, coloured by the latent error when each is lost
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from src.ml.ae.train_ae import AE_RUNS_DIR
from src.ml.ae.train_regressor import REGRESSOR_RUNS_DIR

SPLIT_COLOURS: dict[str, str] = {"train": "tab:blue", "val": "tab:orange"}


def _history(history: pl.DataFrame, summary: dict[str, float]) -> Figure:
    load_names = [c.removeprefix("train_").removesuffix("_r2") for c in history.columns
                  if c.startswith("train_") and c.endswith("_r2")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    epoch = history["epoch"]
    panels = [
        (axes[0], ["train_recon", "val_recon", "train_load", "val_load"], "Loss (standardised)", True),
        (axes[1], ["train_recon_rms_cp", "val_recon_rms_cp"], "Reconstruction RMS error [Cp]", True),
        (axes[2], [f"{s}_{n}_r2" for n in load_names for s in ("train", "val")], "Load R²", False),
    ]
    for ax, columns, label, log in panels:
        for column in columns:
            if column in history.columns:
                ax.plot(epoch, history[column], "--" if column.startswith("val") else "-", label=column)
        if log:
            ax.set_yscale("log")
        ax.axvline(summary["best_epoch"], color="grey", lw=0.8, ls=":", label="best epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    axes[2].set_ylim(max(-0.5, axes[2].get_ylim()[0]), 1.02)
    fig.tight_layout()
    return fig


def _loads(latents: pl.DataFrame, summary: dict[str, float]) -> Figure:
    """Predicted against true loads: the saved model's (by split) and, for a cross-validated autoencoder run, each
    run's out-of-fold prediction (`<load>_cv`)."""
    names = [c.removesuffix("_pred") for c in latents.columns if c.endswith("_pred")]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5.5), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        shown = [f"{name}_pred"]
        for split, rows in latents.group_by("split"):
            ax.scatter(rows[name], rows[f"{name}_pred"], s=14, color=SPLIT_COLOURS[str(split[0])], label=split[0])
        if f"{name}_cv" in latents.columns:
            shown.append(f"{name}_cv")
            ax.scatter(latents[name], latents[f"{name}_cv"], s=14, color=SPLIT_COLOURS["val"],
                       label="cross-validated (held out)")
        lo = min(latents[name].min(), *(latents[c].min() for c in shown))
        hi = max(latents[name].max(), *(latents[c].max() for c in shown))
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8)
        ax.set_title(f"{name}\nval R² {summary.get(f'val_{name}_r2', float('nan')):.3f}", fontsize=10)
        ax.set_xlabel(f"{name} (CFD)")
        ax.set_ylabel(f"{name} (from latent)")
        ax.set_aspect("equal")
        ax.legend()
    fig.tight_layout()
    return fig


def _latent(latents: pl.DataFrame) -> Figure:
    dims = [c for c in latents.columns if c.startswith("z") and c[1:].isdigit()]
    cols = min(3, len(dims))
    rows = -(-len(dims) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), squeeze=False)
    limit = latents["beta_deg"].abs().max() or 1.0
    for ax, dim in zip(axes.flat, dims, strict=False):
        points = ax.scatter(latents["alpha_deg"], latents[dim], c=latents["beta_deg"], cmap="coolwarm",
                            vmin=-limit, vmax=limit, s=14, edgecolors=["k" if s == "val" else "none" for s in latents["split"]])
        ax.set_xlabel("alpha [deg]")
        ax.set_ylabel(dim)
    for ax in list(axes.flat)[len(dims):]:
        ax.axis("off")
    fig.colorbar(points, ax=axes, label="sideslip [deg]  (black edge: validation run)")
    return fig


def _taps(taps: pl.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    limit = taps["recon_rms_cp"].max()
    for ax, (surface, title) in zip(axes, [(0, "Upper skin"), (1, "Lower skin")], strict=True):
        on = taps.filter(pl.col("surface") == surface)
        points = ax.scatter(on["y"], on["x"], c=on["recon_rms_cp"], cmap="viridis", vmin=0, vmax=limit, s=60)
        for order, y, x in on.select("order", "y", "x").iter_rows():
            ax.annotate(str(order), (y, x), fontsize=6, xytext=(4, 4), textcoords="offset points")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # nose at the top
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.set_title(f"{title}: validation reconstruction RMS error [Cp]")
    fig.colorbar(points, ax=axes)
    return fig


def plot_run(run_dir: Path, show: bool = False) -> list[Path]:
    """Draw every figure for one run into its plots/ folder. Returns the files written."""
    summary = json.loads((run_dir / "summary.json").read_text())
    latents = pl.read_parquet(run_dir / "latents.parquet")
    return _save(run_dir, show, {
        "history": _history(pl.read_csv(run_dir / "history.csv"), summary),
        "loads": _loads(latents, summary),
        "latent": _latent(latents),
        "taps": _taps(pl.read_parquet(run_dir / "taps.parquet")),
    })


def _regressor_history(history: pl.DataFrame, summary: dict[str, float]) -> Figure:
    load_names = [c.removeprefix("val_").removesuffix("_r2") for c in history.columns
                  if c.startswith("val_") and c.endswith("_r2") and c != "val_z_r2"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    epoch = history["epoch"]
    panels = [
        (axes[0], ["train_loss", "train_z_rms", "val_z_rms"], "Latent error", True, None),
        (axes[1], ["train_z_r2", "val_z_r2", *(f"val_{n}_r2" for n in load_names)], "R²", False, None),
        (axes[2], ["train_recon_rms_cp", "val_recon_rms_cp"], "All-tap Cp from z_hat, RMS error", True,
         "autoencoder_recon_rms_cp"),
    ]
    for ax, columns, label, log, ceiling in panels:
        for column in columns:
            if column in history.columns:
                ax.plot(epoch, history[column], "--" if column.startswith("val") else "-", label=column)
        if ceiling and ceiling in summary:  # what the autoencoder gets from all the taps
            ax.axhline(summary[ceiling], color="k", lw=0.8, label="from all taps (autoencoder)")
        if log:
            ax.set_yscale("log")
        ax.axvline(summary["best_epoch"], color="grey", lw=0.8, ls=":", label="best epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    axes[1].set_ylim(max(-0.5, axes[1].get_ylim()[0]), 1.02)
    fig.tight_layout()
    return fig


def _regressor_latent(latents: pl.DataFrame) -> Figure:
    dims = [c for c in latents.columns if c.startswith("z") and c[1:].isdigit()]
    cols = min(3, len(dims))
    rows = -(-len(dims) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows), squeeze=False)
    for ax, dim in zip(axes.flat, dims, strict=False):
        for split, part in latents.group_by("split"):
            ax.scatter(part[dim], part[f"{dim}_hat"], s=14, color=SPLIT_COLOURS[str(split[0])], label=split[0])
        lo, hi = latents[dim].min(), latents[dim].max()
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8)
        ax.set_xlabel(f"{dim} = E(all taps)")
        ax.set_ylabel(f"{dim}_hat = g(k taps)")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(dims):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _regressor_loads(latents: pl.DataFrame, summary: dict[str, float]) -> Figure:
    names = [c.removesuffix("_from_z_hat") for c in latents.columns if c.endswith("_from_z_hat")]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5.5), squeeze=False)
    val = latents.filter(pl.col("split") == "val")
    shown = val if val.height else latents
    for ax, name in zip(axes[0], names, strict=True):
        ax.scatter(shown[name], shown[f"{name}_from_z"], s=14, color="grey", label="from z (all taps)")
        ax.scatter(shown[name], shown[f"{name}_from_z_hat"], s=14, color="tab:orange", label="from z_hat (k taps)")
        lo, hi = latents[name].min(), latents[name].max()
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8)
        ax.set_title(f"{name}, {'validation' if val.height else 'all'} runs\n"
                     f"R² all taps {summary.get(f'autoencoder_{name}_r2', float('nan')):.3f},  "
                     f"k taps {summary.get(f'val_{name}_r2', float('nan')):.3f}", fontsize=10)
        ax.set_xlabel(f"{name} (CFD)")
        ax.set_ylabel(f"{name} (autoencoder head)")
        ax.set_aspect("equal")
        ax.legend()
    fig.tight_layout()
    return fig


def _regressor_taps(taps: pl.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    lost = "lost_z_rms" in taps.columns
    for ax, (surface, title) in zip(axes, [(0, "Upper skin"), (1, "Lower skin")], strict=True):
        on = taps.filter(pl.col("surface") == surface)
        if lost:
            points = ax.scatter(on["y"], on["x"], c=on["lost_z_rms"], cmap="magma_r", s=80, vmin=0,
                                vmax=taps["lost_z_rms"].max())
        else:
            ax.scatter(on["y"], on["x"], color="tab:red", s=80)
        for order, y, x in on.select("order", "y", "x").iter_rows():
            ax.annotate(str(order), (y, x), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # nose at the top
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.set_title(f"{title}: deployed taps" + (" (colour: latent error if lost)" if lost else ""))
    if lost:
        fig.colorbar(points, ax=axes, label="validation z RMS error with this tap lost")
    return fig


def plot_regressor_run(run_dir: Path, show: bool = False) -> list[Path]:
    """Draw every figure for one latent regressor run into its plots/ folder. Returns the files written."""
    summary = json.loads((run_dir / "summary.json").read_text())
    latents = pl.read_parquet(run_dir / "latents.parquet")
    return _save(run_dir, show, {
        "history": _regressor_history(pl.read_csv(run_dir / "history.csv"), summary),
        "latent": _regressor_latent(latents),
        "loads": _regressor_loads(latents, summary),
        "taps": _regressor_taps(pl.read_parquet(run_dir / "taps.parquet")),
    })


def _save(run_dir: Path, show: bool, figures: dict[str, Figure]) -> list[Path]:
    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    written = []
    for name, fig in figures.items():
        fig.suptitle(run_dir.name, fontsize=9, x=0.01, ha="left")
        written.append(out_dir / f"{name}.png")
        fig.savefig(written[-1], dpi=150)
    if show:
        plt.show()
    plt.close("all")
    return written


def is_regressor_run(run_dir: Path) -> bool:
    return "autoencoder_run" in json.loads((run_dir / "config.json").read_text())


def newest_run(roots: tuple[Path, ...] = (AE_RUNS_DIR, REGRESSOR_RUNS_DIR)) -> Path:
    """The most recently started finished run of either network."""
    runs = [p for root in roots if root.exists() for p in root.iterdir() if (p / "summary.json").exists()]
    if not runs:
        raise FileNotFoundError(f"no finished runs in {', '.join(map(str, roots))}")
    return max(runs, key=lambda p: p.name)  # folder names start with the start time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?", help="run folder (default: the newest of either network)")
    parser.add_argument("--show", action="store_true", help="open the figures as well as saving them")
    args = parser.parse_args()
    run_dir = args.run or newest_run()
    plot = plot_regressor_run if is_regressor_run(run_dir) else plot_run
    for path in plot(run_dir, show=args.show):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
