"""How well a tap layout sees each control surface: can the actuators' effects be observed from the chosen taps?

    python -m src.ml.taps.actuator_observability                  the layout in use (selections/taps.parquet)
    python -m src.ml.taps.actuator_observability selections/<run>/taps.parquet   a layout from one selection run

For every actuator varied in the CFD runs (`DEFLECTIONS` columns of runs.parquet), on the runs of `load_runs`:
    seen_alone_r2         cross-validated R^2 of a linear estimate of its deflection from Cp at every tap, with
                          nothing else known: alpha, beta and the other deflections vary freely, as in a gust. Can the
                          taps tell what this surface is doing at all?
    seen_others_known_r2  the same with the other actuators' deflections given (the controller knows its own
                          commands): does this surface leave a signature of its own, apart from the others?
    best_tap, best_tap_r2 the single tap that estimates it best with the others known and the flow (`flow_inputs`)
                          known too (one tap cannot tell a flap from the flow, so the flow is given): the tap that
                          carries its clearest signature, and that R^2
    taps_significant      how many taps its deflection changes by a statistically clear amount: |t| > T_SIGNIFICANT
                          for its slope in a least-squares fit of each tap's Cp on the flow and every deflection
    max_signal_cp         the largest Cp change over its full travel at any of those taps (0 if none)
R^2 near 1: the taps see the surface well; near 0 or below: they cannot see it. The runs vary every deflection
independently (a Latin hypercube), so each deflection can be estimated on its own. Sensor noise (SENSOR_NOISE_CP) is
added to the held-out taps, as in the tap selection.

The all-tap estimates fit 48 taps to ~120 runs, so they are ridge-regularised; each outer fold picks the strength
from RIDGE_GRID_CP by a cross-validation of its own training runs (nested), so the R^2 stays an honest held-out score.
A single tap cannot overfit and uses the sensor noise as its ridge, as the tap selection does.

Writes, next to the layout file: actuator_observability.csv, actuator_observability.png (the R^2 of every actuator)
and actuator_signal.png (per actuator, the Cp change over its travel at each tap it clearly changes, on the planform,
with its hinge line).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import torch
from matplotlib.colors import LogNorm
from torch import Tensor

from src.cfd.geometry import GeometryConfig
from src.config import CONFIG
from src.data.cfd import CFD_DATA_DIR
from src.ml.taps.cfd_taps import DEFLECTIONS, FOLDS, LOWER, OUTPUT_DIR, SEED, UPPER, candidate_cp, load_runs
from src.ml.taps.tap_selection import SENSOR_NOISE_CP, fit_estimator

TAPS_FILE: Path = OUTPUT_DIR / "taps.parquet"
GEOMETRY_FILE: Path = CONFIG.configs_dir.parent / "cad" / "uav" / "geometry.toml"  # for the hinge lines on the plots
# Ridge strengths tried for the all-tap estimates, as an effective tap noise in Cp (fit_estimator's noise_std).
RIDGE_GRID_CP: tuple[float, ...] = (SENSOR_NOISE_CP, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
T_SIGNIFICANT: float = 3.0  # |slope / standard error| above which a tap counts as changed by an actuator


def actuator_name(column: str) -> str:
    return column.removeprefix("deflection_").removesuffix("_deg")


def flow_inputs(runs: pl.DataFrame) -> Tensor:
    """The flow condition as fit inputs (N, 6): alpha, alpha^2, alpha^3, beta, beta^2 and alpha * beta, so the vortex
    lift's curvature in alpha and the alpha-beta coupling are not put down to the flaps."""
    alpha = runs["alpha_deg"].to_torch().to(torch.float64)
    beta = runs["beta_deg"].to_torch().to(torch.float64)
    return torch.stack([alpha, alpha.pow(2), alpha.pow(3), beta, beta.pow(2), alpha * beta], dim=1)


def _folds(n: int, n_folds: int, seed: int) -> list[Tensor]:
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return [order[i::n_folds] for i in range(n_folds)]


def _r2(predicted: Tensor, target: Tensor) -> float:
    return float(1.0 - (predicted - target).pow(2).sum() / (target - target.mean()).pow(2).sum())


def _out_of_fold(tap_cp: Tensor, target: Tensor, taps: Tensor, known: Tensor | None, ridge_cp: float,
                 n_folds: int, seed: int) -> Tensor:
    """Each run's estimate (N,) from a fit on the other folds, with sensor noise on its taps."""
    n = len(target)
    generator = torch.Generator().manual_seed(seed + 1)
    predicted = torch.empty(n, dtype=torch.float64)
    for test in _folds(n, n_folds, seed):
        train = torch.ones(n, dtype=torch.bool)
        train[test] = False
        selection = fit_estimator(tap_cp[train], target[train, None], taps,
                                  None if known is None else known[train], noise_std=ridge_cp)
        noisy = tap_cp[test][:, taps] + SENSOR_NOISE_CP * torch.randn(len(test), len(taps), generator=generator,
                                                                      dtype=torch.float64)
        predicted[test] = selection.estimate(noisy, None if known is None else known[test])[:, 0]
    return predicted


def cross_validated_r2(tap_cp: Tensor, target: Tensor, taps: Tensor, known: Tensor | None,
                       ridge_grid: tuple[float, ...] = RIDGE_GRID_CP, n_folds: int = FOLDS, seed: int = SEED) -> float:
    """Pooled out-of-fold R^2 of the linear estimate of `target` (N,) from Cp at `taps` (columns of tap_cp) and the
    `known` inputs. With more than one ridge strength, each outer fold chooses its own by an inner cross-validation
    on its training runs."""
    if len(ridge_grid) == 1:
        return _r2(_out_of_fold(tap_cp, target, taps, known, ridge_grid[0], n_folds, seed), target)
    n = len(target)
    predicted = torch.empty(n, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed + 2)
    for test in _folds(n, n_folds, seed):
        train = torch.ones(n, dtype=torch.bool)
        train[test] = False
        cp_train, target_train = tap_cp[train], target[train]
        known_train = None if known is None else known[train]
        ridge = max(ridge_grid, key=lambda r: _r2(
            _out_of_fold(cp_train, target_train, taps, known_train, r, n_folds, seed), target_train))
        selection = fit_estimator(cp_train, target_train[:, None], taps, known_train, noise_std=ridge)
        noisy = tap_cp[test][:, taps] + SENSOR_NOISE_CP * torch.randn(len(test), len(taps), generator=generator,
                                                                      dtype=torch.float64)
        predicted[test] = selection.estimate(noisy, None if known is None else known[test])[:, 0]
    return _r2(predicted, target)


def signal(tap_cp: Tensor, runs: pl.DataFrame, deflections: list[str]) -> tuple[Tensor, Tensor]:
    """Per actuator and tap (J, n_taps): the Cp change over the actuator's full travel in the runs, and its t-statistic
    (slope / standard error), from a least-squares fit of every tap's Cp on [1, flow_inputs, deflections]."""
    travel = runs.select(deflections).to_torch(dtype=pl.Float64)
    inputs = torch.cat([torch.ones(runs.height, 1, dtype=torch.float64), flow_inputs(runs), travel], dim=1)
    fit = torch.linalg.lstsq(inputs, tap_cp).solution  # (P, n_taps)
    residual_var = (tap_cp - inputs @ fit).pow(2).sum(dim=0) / (runs.height - inputs.shape[1])  # (n_taps,)
    coefficient_var = torch.linalg.inv(inputs.T @ inputs).diagonal()  # (P,) per unit residual variance
    first = inputs.shape[1] - len(deflections)
    slopes = fit[first:]  # (J, n_taps) Cp per degree
    t = slopes / (coefficient_var[first:, None] * residual_var[None, :]).sqrt()
    return slopes.abs() * (travel.max(dim=0).values - travel.min(dim=0).values)[:, None], t


def observability(taps: pl.DataFrame, runs: pl.DataFrame, tap_cp: Tensor) -> tuple[pl.DataFrame, Tensor, Tensor]:
    """The table in the module docstring, one row per actuator, and the signal and its t-statistic (J, n_taps)."""
    deflections = [c for c in runs.select(pl.col(DEFLECTIONS)).columns if runs[c].n_unique() > 1]
    every_tap = torch.arange(tap_cp.shape[1])
    change, t = signal(tap_cp, runs, deflections)
    significant = t.abs() > T_SIGNIFICANT
    flow = flow_inputs(runs)
    rows = []
    for j, column in enumerate(deflections):
        target = runs[column].to_torch().to(torch.float64)
        others = runs.select([c for c in deflections if c != column]).to_torch(dtype=pl.Float64)
        others_and_flow = torch.cat([others, flow], dim=1)
        single = [cross_validated_r2(tap_cp, target, every_tap[i:i + 1], others_and_flow, (SENSOR_NOISE_CP,))
                  for i in range(len(every_tap))]
        best = max(range(len(single)), key=single.__getitem__)
        rows.append({
            "actuator": actuator_name(column),
            "seen_alone_r2": cross_validated_r2(tap_cp, target, every_tap, None),
            "seen_others_known_r2": cross_validated_r2(tap_cp, target, every_tap, others),
            "best_tap": int(taps["order"][best]),
            "best_tap_r2": single[best],
            "taps_significant": int(significant[j].sum()),
            "max_signal_cp": float((change[j] * significant[j]).max()),
        })
        print(f"{rows[-1]['actuator']}: done")
    return pl.DataFrame(rows), change, significant


def plot_r2(table: pl.DataFrame, folder: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = torch.arange(table.height).numpy()
    for offset, column, label in [(-0.27, "seen_alone_r2", "all taps, nothing else known"),
                                  (0.0, "seen_others_known_r2", "all taps, other deflections known"),
                                  (0.27, "best_tap_r2", "best single tap, other deflections and flow known")]:
        ax.bar(x + offset, table[column].clip(lower_bound=0).to_numpy(), width=0.27, label=label)
    ax.set_xticks(x, table["actuator"].to_list(), rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cross-validated R² of the deflection")
    ax.set_title("How well the taps see each actuator (estimating its deflection from tap Cp, with sensor noise)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = folder / "actuator_observability.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_signal(taps: pl.DataFrame, table: pl.DataFrame, change: Tensor, significant: Tensor,
                geometry: GeometryConfig, folder: Path) -> Path:
    """Per actuator: the Cp change over its travel at each tap it clearly changes (filled, log colour scale from the
    sensor noise up), the other taps hollow; upper skin circles, lower skin squares; its hinge line and best tap."""
    cols = 4
    rows = -(-table.height // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    # Log scale: one tap next to a flap can change by far more than the rest, which would wash them out.
    norm = LogNorm(vmin=SENSOR_NOISE_CP, vmax=max(float((change * significant).max()), 10 * SENSOR_NOISE_CP))
    scale = geometry.scale_to_m
    y, x = taps["y"].to_numpy(), taps["x"].to_numpy()
    points = None
    for ax, (j, row) in zip(axes.flat, enumerate(table.iter_rows(named=True)), strict=False):
        for surface, marker in [(UPPER, "o"), (LOWER, "s")]:
            on = (taps["surface"] == surface).to_numpy()
            seen = on & significant[j].numpy()
            ax.scatter(y[on & ~seen], x[on & ~seen], marker=marker, s=22, facecolors="none", edgecolors="0.7",
                       linewidths=0.6)
            points = ax.scatter(y[seen], x[seen], c=change[j].numpy()[seen].clip(SENSOR_NOISE_CP), marker=marker,
                                cmap="magma_r", norm=norm, s=30, edgecolors="0.4", linewidths=0.4)
        hinge = geometry.control_surface(row["actuator"]).hinge
        ax.plot([hinge.p1[1] * scale, hinge.p2[1] * scale], [hinge.p1[0] * scale, hinge.p2[0] * scale],
                color="tab:blue", lw=2.5)
        best = taps["order"].to_list().index(row["best_tap"])
        ax.scatter(y[best], x[best], s=150, facecolors="none", edgecolors="tab:green", linewidths=1.5)
        ax.set_title(f"{row['actuator']}: R² {row['seen_alone_r2']:.2f} alone, {row['seen_others_known_r2']:.2f} "
                     f"others known,\nbest tap {row['best_tap_r2']:.2f}, {row['taps_significant']} taps changed",
                     fontsize=8)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # nose at the top
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[table.height:]:
        ax.axis("off")
    if points is not None:
        fig.colorbar(points, ax=axes, shrink=0.6, label="Cp change over the travel (from the sensor noise up)")
    fig.suptitle("What each actuator does at each tap. Filled: clearly changed (hollow: not), circle: upper skin, "
                 "square: lower skin, blue: hinge line, green ring: best single tap")
    path = folder / "actuator_signal.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(taps_file: Path = TAPS_FILE, data_dir: Path = CFD_DATA_DIR) -> None:
    runs = load_runs(data_dir)
    taps = pl.read_parquet(taps_file).sort("order")
    tap_cp, covered = candidate_cp(runs, taps, data_dir)
    if not bool(covered.all()):
        raise ValueError(f"taps {(~covered).nonzero().squeeze(1).tolist()} are not on every run's surface")
    print(f"{runs.height} runs, {taps.height} taps from {taps_file}, sensor noise {SENSOR_NOISE_CP} Cp")

    table, change, significant = observability(taps, runs, tap_cp)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=200):
        print(table)
    folder = taps_file.parent  # next to the layout it describes, e.g. that selection run's folder
    table.write_csv(folder / "actuator_observability.csv")
    print(f"saved {folder / 'actuator_observability.csv'}")
    print(f"saved {plot_r2(table, folder)}")
    print(f"saved {plot_signal(taps, table, change, significant, GeometryConfig.from_toml(GEOMETRY_FILE), folder)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else TAPS_FILE)
