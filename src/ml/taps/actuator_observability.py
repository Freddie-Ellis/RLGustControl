"""How well a tap layout sees each control surface: can the actuators' effects be observed from the chosen taps?

    python -m src.ml.taps.actuator_observability                  the saved layout (selections/taps.parquet)
    python -m src.ml.taps.actuator_observability <taps.parquet>   another layout

For every actuator varied in the CFD runs (`DEFLECTIONS` columns of runs.parquet), on the runs of `load_runs`:
    seen_alone_r2         cross-validated R^2 of a linear estimate of its deflection from Cp at every tap, with
                          nothing else known: alpha, beta and the other deflections vary freely, as in a gust. Can the
                          taps tell what this surface is doing at all?
    seen_others_known_r2  the same with the other actuators' deflections given (the controller knows its own
                          commands): does this surface leave a signature of its own, apart from the others?
    best_tap, best_tap_r2 the single tap that estimates it best with the others known and alpha and beta known too
                          (one tap cannot tell a flap from the flow, so the flow is given): the tap that carries its
                          clearest signature, and that R^2
    max_signal_cp         the largest Cp change any tap sees over the surface's full travel in the runs, from a
                          linear fit of each tap's Cp on alpha, alpha^2, beta and every deflection
    taps_above_noise      how many taps see a change over the travel of more than SIGNAL_NOISE_RATIO x sensor noise
R^2 near 1: the taps see the surface well; near 0: they cannot see it. The runs vary every deflection independently
(a Latin hypercube), so each deflection can be estimated on its own. Sensor noise (SENSOR_NOISE_CP) is added to the
held-out taps, as in the tap selection.

Writes selections/actuator_observability.csv, plots/actuator_observability.png (the R^2 of every actuator) and
plots/actuator_signal.png (per actuator, each tap's Cp change over its travel on the planform, with its hinge line).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.colors import LogNorm
import torch
from torch import Tensor

from src.cfd.geometry import GeometryConfig
from src.config import CONFIG
from src.data.cfd import CFD_DATA_DIR
from src.ml.taps.cfd_taps import DEFLECTIONS, FOLDS, LOWER, OUTPUT_DIR, PLOTS_DIR, SEED, UPPER, candidate_cp, load_runs
from src.ml.taps.tap_selection import SENSOR_NOISE_CP, fit_estimator

TAPS_FILE: Path = OUTPUT_DIR / "taps.parquet"
GEOMETRY_FILE: Path = CONFIG.configs_dir.parent / "cad" / "uav" / "geometry.toml"  # for the hinge lines on the plots
SIGNAL_NOISE_RATIO: float = 3.0  # a tap "sees" a surface when its Cp change over the travel exceeds this x noise
SIGNAL_FLOOR: float = 0.1  # bottom of the signal plot's log colour scale, in units of sensor noise


def actuator_name(column: str) -> str:
    return column.removeprefix("deflection_").removesuffix("_deg")


def cross_validated_r2(tap_cp: Tensor, target: Tensor, taps: Tensor, known: Tensor | None,
                       n_folds: int = FOLDS, seed: int = SEED) -> float:
    """Pooled out-of-fold R^2 of the linear estimate of `target` (N,) from Cp at `taps` (columns of tap_cp) and the
    `known` inputs, with sensor noise on the held-out taps."""
    n = len(target)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    predicted = torch.empty(n, dtype=torch.float64)
    for i in range(n_folds):
        test = order[i::n_folds]
        train = torch.ones(n, dtype=torch.bool)
        train[test] = False
        selection = fit_estimator(tap_cp[train], target[train, None], taps, None if known is None else known[train])
        noisy = tap_cp[test][:, taps] + SENSOR_NOISE_CP * torch.randn(len(test), len(taps), generator=generator,
                                                                      dtype=torch.float64)
        predicted[test] = selection.estimate(noisy, None if known is None else known[test])[:, 0]
    residual = (predicted - target).pow(2).sum()
    return float(1.0 - residual / (target - target.mean()).pow(2).sum())


def signal(tap_cp: Tensor, runs: pl.DataFrame, deflections: list[str]) -> Tensor:
    """Cp change at each tap over each actuator's full travel in the runs (J, n_taps), from a least-squares fit of
    every tap's Cp on [1, alpha, alpha^2, beta, deflections]."""
    alpha = runs["alpha_deg"].to_torch().to(torch.float64)
    inputs = torch.cat([
        torch.ones(runs.height, 1, dtype=torch.float64),
        torch.stack([alpha, alpha.pow(2), runs["beta_deg"].to_torch().to(torch.float64)], dim=1),
        runs.select(deflections).to_torch(dtype=pl.Float64),
    ], dim=1)
    slopes = torch.linalg.lstsq(inputs, tap_cp).solution[4:]  # (J, n_taps) Cp per degree of each deflection
    travel = runs.select(deflections).to_torch(dtype=pl.Float64)
    return slopes.abs() * (travel.max(dim=0).values - travel.min(dim=0).values)[:, None]


def observability(taps: pl.DataFrame, runs: pl.DataFrame, tap_cp: Tensor) -> tuple[pl.DataFrame, Tensor]:
    """The table in the module docstring, one row per actuator, and the signal (J, n_taps)."""
    deflections = [c for c in runs.select(pl.col(DEFLECTIONS)).columns if runs[c].n_unique() > 1]
    every_tap = torch.arange(tap_cp.shape[1])
    seen = signal(tap_cp, runs, deflections)
    alpha = runs["alpha_deg"].to_torch().to(torch.float64)
    flow = torch.stack([alpha, alpha.pow(2), runs["beta_deg"].to_torch().to(torch.float64)], dim=1)
    rows = []
    for j, column in enumerate(deflections):
        target = runs[column].to_torch().to(torch.float64)
        others = runs.select([c for c in deflections if c != column]).to_torch(dtype=pl.Float64)
        others_and_flow = torch.cat([others, flow], dim=1)
        single = [cross_validated_r2(tap_cp, target, every_tap[t:t + 1], others_and_flow)
                  for t in range(len(every_tap))]
        best = max(range(len(single)), key=single.__getitem__)
        rows.append({
            "actuator": actuator_name(column),
            "seen_alone_r2": cross_validated_r2(tap_cp, target, every_tap, None),
            "seen_others_known_r2": cross_validated_r2(tap_cp, target, every_tap, others),
            "best_tap": int(taps["order"][best]),
            "best_tap_r2": single[best],
            "max_signal_cp": float(seen[j].max()),
            "taps_above_noise": int((seen[j] > SIGNAL_NOISE_RATIO * SENSOR_NOISE_CP).sum()),
        })
    return pl.DataFrame(rows), seen


def plot_r2(table: pl.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = torch.arange(table.height).numpy()
    for offset, column, label in [(-0.27, "seen_alone_r2", "all taps, nothing else known"),
                                  (0.0, "seen_others_known_r2", "all taps, other deflections known"),
                                  (0.27, "best_tap_r2", "best single tap, other deflections and alpha, beta known")]:
        ax.bar(x + offset, table[column].clip(lower_bound=0).to_numpy(), width=0.27, label=label)
    ax.set_xticks(x, table["actuator"].to_list(), rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cross-validated R² of the deflection")
    ax.set_title("How well the taps see each actuator (estimating its deflection from tap Cp, with sensor noise)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "actuator_observability.png"
    fig.savefig(path, dpi=200)
    return path


def plot_signal(taps: pl.DataFrame, table: pl.DataFrame, seen: Tensor, geometry: GeometryConfig) -> Path:
    """Per actuator: each tap's Cp change over the actuator's travel (upper skin circles, lower skin squares), on a
    colour scale in units of sensor noise, with the actuator's hinge line."""
    cols = 4
    rows = -(-table.height // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    # Log scale: one tap next to a flap can see ~100x noise, which would wash out the signals near the noise floor.
    norm = LogNorm(vmin=SIGNAL_FLOOR, vmax=max(float(seen.max()) / SENSOR_NOISE_CP, 10 * SIGNAL_FLOOR))
    ratio = (seen / SENSOR_NOISE_CP).clamp_min(SIGNAL_FLOOR)
    scale = geometry.scale_to_m
    points = None
    for ax, (j, row) in zip(axes.flat, enumerate(table.iter_rows(named=True)), strict=False):
        for surface, marker in [(UPPER, "o"), (LOWER, "s")]:
            on = (taps["surface"] == surface).to_numpy()
            points = ax.scatter(taps["y"].to_numpy()[on], taps["x"].to_numpy()[on], c=ratio[j].numpy()[on],
                                marker=marker, cmap="magma_r", norm=norm, s=28, edgecolors="0.6",
                                linewidths=0.4)
        hinge = geometry.control_surface(row["actuator"]).hinge
        ax.plot([hinge.p1[1] * scale, hinge.p2[1] * scale], [hinge.p1[0] * scale, hinge.p2[0] * scale],
                color="tab:blue", lw=2.5)
        best = taps["order"].to_list().index(row["best_tap"])
        ax.scatter(taps["y"][best], taps["x"][best], s=150, facecolors="none", edgecolors="tab:green", linewidths=1.5)
        ax.set_title(f"{row['actuator']}: R² {row['seen_alone_r2']:.2f} alone, {row['seen_others_known_r2']:.2f} "
                     f"others known", fontsize=8)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # nose at the top
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[table.height:]:
        ax.axis("off")
    if points is not None:
        bar = fig.colorbar(points, ax=axes, shrink=0.6, label="Cp change over travel / sensor noise")
        bar.ax.axhline(SIGNAL_NOISE_RATIO, color="tab:blue", lw=1)  # the "sees it" threshold of taps_above_noise
    fig.suptitle("Cp change at each tap over each actuator's travel. Circle: upper skin, square: lower skin, "
                 "blue: hinge line, green ring: best single tap")
    path = PLOTS_DIR / "actuator_signal.png"
    fig.savefig(path, dpi=200)
    return path


def main(taps_file: Path = TAPS_FILE, data_dir: Path = CFD_DATA_DIR) -> None:
    runs = load_runs(data_dir)
    taps = pl.read_parquet(taps_file).sort("order")
    tap_cp, covered = candidate_cp(runs, taps, data_dir)
    if not bool(covered.all()):
        raise ValueError(f"taps {(~covered).nonzero().squeeze(1).tolist()} are not on every run's surface")
    print(f"{runs.height} runs, {taps.height} taps from {taps_file}, sensor noise {SENSOR_NOISE_CP} Cp")

    table, seen = observability(taps, runs, tap_cp)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=200):
        print(table)
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    table.write_csv(OUTPUT_DIR / "actuator_observability.csv")
    print(f"saved {OUTPUT_DIR / 'actuator_observability.csv'}")
    print(f"saved {plot_r2(table)}")
    print(f"saved {plot_signal(taps, table, seen, GeometryConfig.from_toml(GEOMETRY_FILE))}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else TAPS_FILE)
