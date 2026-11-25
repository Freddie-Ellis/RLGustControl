"""Control surface loads, from the dataset written by src/cfd/dataset.py (setpoints.parquet).

    .venv/Scripts/python.exe -m postpro.surfaces data/Batch2/setpoints.parquet [--out FOLDER] [--show]

Writes to <out> (default <parquet folder>/postpro/surfaces/):
    clean_wing_by_part.png
        clean wing, beta 0: C_L, C_D and C_m against alpha, stacked by part (the body and each surface type,
        left + right), with the total
    clean_wing_sideslip_by_part.png
        clean wing at each beta (one row each): C_Y_body (side force along the model's y axis), C_l and C_n against
        alpha, stacked by part, with the total. Only the alphas solved at every beta (-3, 3, 9, 15, 21, 27)
    surface_loads/<surface>.png
        the right-hand surface's own C_L, C_D, C_m and C_h against alpha at beta 0: the clean wing (black) and one
        line per deflection, both sides deflected
    aircraft_change/<family>.png
        change in the aircraft's C_L, C_D and C_m from the clean wing against deflection (both sides deflected),
        beta 0, one line per alpha. Error bars: deflected min - clean max to deflected max - clean min
    hinge_moments_sideslip/<surface>.png
        right-only runs: the right surface's C_h against beta, one line per deflection (colour) and alpha (symbol)

Coefficients use the aircraft's reference values (see src/cfd/dataset.py). The right-hand axis of each plot gives the
same quantity in N or N m at the solved speed (C_l and C_n keep their flight-mechanics sign). Error bars span min to
max over the averaging window (for the stacked bars, on the total only; for C_Y_body, the worst-case combination of
the coefficients it is made from); hollow markers are unconverged set points.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.cfd.dataset import SURFACE_FAMILIES

SURFACES: tuple[str, ...] = tuple(SURFACE_FAMILIES)  # surface names without their side
FAMILIES: tuple[str, ...] = tuple(dict.fromkeys(SURFACE_FAMILIES.values()))
DIMENSIONAL: dict[str, str] = {
    "C_L": "lift (N)", "C_D": "drag (N)", "C_Y": "side force (N)", "C_Y_body": "side force, model y (N)",
    "C_l": "rolling moment (N m)",
    "C_m": "pitching moment (N m)", "C_n": "yawing moment (N m)", "C_h": "hinge moment (N m)",
}
AIRCRAFT_COEFFICIENTS: tuple[str, ...] = ("C_L", "C_D", "C_m")  # symmetric deflections give no C_Y, C_l or C_n
SIDESLIP_COEFFICIENTS: tuple[str, ...] = ("C_Y_body", "C_l", "C_n")
ALPHA_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")


def factor(df: pl.DataFrame, coefficient: str) -> float:
    """Coefficient -> N or N m: q S for forces, q S c for C_m and C_h, q S b for C_l and C_n."""
    q_s = float(df["q_pa"][0] * df["ref_area_m2"][0])
    if coefficient in ("C_L", "C_D", "C_Y", "C_Y_body"):
        return q_s
    if coefficient in ("C_m", "C_h"):
        return q_s * float(df["ref_chord_m"][0])
    return q_s * float(df["ref_span_m"][0])


def finish(ax: plt.Axes, df: pl.DataFrame, xlabel: str, coefficient: str, ylabel: str | None = None,
           per: str = "") -> None:
    """Axis labels, grid, and a right-hand axis in N or N m (`per` is appended to its label, e.g. ' per rad')."""
    k = factor(df, coefficient)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or coefficient)
    ax.grid(alpha=0.3)
    ax.secondary_yaxis("right", functions=(lambda c: c * k, lambda v: v / k)).set_ylabel(DIMENSIONAL[coefficient] + per)


def plot_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, converged: np.ndarray, colour: str, style: str = "-",
              marker: str = "o", label: str | None = None) -> None:
    """A line through (x, y) with markers (so the legend shows them), hollow where not converged."""
    ax.plot(x, y, style, marker=marker, color=colour, lw=1, ms=4, label=label)
    ax.plot(x[~converged], y[~converged], marker, color=colour, mfc="white", ms=4, ls="none")


def plot_series(ax: plt.Axes, rows: pl.DataFrame, x: str, column: str, colour: str, style: str = "-",
                marker: str = "o", label: str | None = None) -> None:
    """<column>_mean against x, with min-max error bars and hollow markers where not converged."""
    rows = rows.sort(x)
    xs = rows[x].to_numpy()
    mean = rows[f"{column}_mean"].to_numpy()
    error = [mean - rows[f"{column}_min"].to_numpy(), rows[f"{column}_max"].to_numpy() - mean]
    ax.errorbar(xs, mean, yerr=error, fmt="none", ecolor=colour, capsize=2, lw=1)
    plot_line(ax, xs, mean, rows["converged"].to_numpy(), colour, style, marker, label)


def stacked_bars(ax: plt.Axes, x: np.ndarray, parts: dict[str, np.ndarray], total: np.ndarray, low: np.ndarray,
                 high: np.ndarray) -> None:
    """Bars of each part stacked at each x (positive parts up, negative down), and the total as a line with error
    bars from `low` to `high`."""
    positive, negative = np.zeros_like(x), np.zeros_like(x)
    for i, (name, values) in enumerate(parts.items()):
        ax.bar(x, values, bottom=np.where(values >= 0, positive, negative), width=2.4, color=f"C{i}", label=name)
        positive += np.clip(values, 0, None)
        negative += np.clip(values, None, 0)
    ax.errorbar(x, total, yerr=[total - low, high - total], fmt="k-o", ms=4, capsize=2, label="total")


def mean(rows: pl.DataFrame, part: str, coefficient: str) -> np.ndarray:
    """A part's mean coefficient. "C_Y_body" is its side force along the model's y axis (starboard), F_y / (q S),
    rather than along side_dir, which tilts towards the model's z axis when alpha and beta are both nonzero."""
    if coefficient == "C_Y_body":
        return (rows[f"{part}__C_L_mean"] * rows["lift_dir_y"] + rows[f"{part}__C_D_mean"] * rows["drag_dir_y"]
                + rows[f"{part}__C_Y_mean"] * rows["side_dir_y"]).to_numpy()
    return rows[f"{part}__{coefficient}_mean"].to_numpy()


def bounds(rows: pl.DataFrame, part: str, coefficient: str) -> tuple[np.ndarray, np.ndarray]:
    """A part's min and max coefficient over the window. For "C_Y_body", the worst-case combination of the min and
    max of the C_L, C_D and C_Y it is made from."""
    if coefficient != "C_Y_body":
        return rows[f"{part}__{coefficient}_min"].to_numpy(), rows[f"{part}__{coefficient}_max"].to_numpy()
    low, high = np.zeros(rows.height), np.zeros(rows.height)
    for c, d in (("C_L", "lift"), ("C_D", "drag"), ("C_Y", "side")):
        weight = rows[f"{d}_dir_y"].to_numpy()
        a, b = weight * rows[f"{part}__{c}_min"].to_numpy(), weight * rows[f"{part}__{c}_max"].to_numpy()
        low += np.minimum(a, b)
        high += np.maximum(a, b)
    return low, high


def part_values(rows: pl.DataFrame, coefficient: str) -> dict[str, np.ndarray]:
    """The body's and each surface type's (left + right) mean coefficient."""
    return {"body": mean(rows, "body", coefficient)} | {
        s: mean(rows, f"{s}_r", coefficient) + mean(rows, f"{s}_l", coefficient) for s in SURFACES}


def save(fig: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if not show:
        plt.close(fig)


def plot_clean_breakdown(df: pl.DataFrame, path: Path, show: bool) -> None:
    clean = df.filter(pl.col("mode") == "clean", pl.col("beta_deg") == 0).sort("alpha_deg")
    alpha = clean["alpha_deg"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, c in zip(axes, ("C_L", "C_D", "C_m")):
        stacked_bars(ax, alpha, part_values(clean, c), mean(clean, "total", c), *bounds(clean, "total", c))
        finish(ax, df, "α (deg)", c)
    axes[0].legend(fontsize=8)
    fig.suptitle("Clean wing by part, β = 0°")
    save(fig, path, show)


def plot_clean_sideslip(df: pl.DataFrame, path: Path, show: bool) -> None:
    clean = df.filter(pl.col("mode") == "clean")
    betas = sorted(clean["beta_deg"].unique().to_list())
    # Only the alphas solved at every sideslip, so each row has the same bars.
    alphas = set.intersection(*(set(clean.filter(pl.col("beta_deg") == b)["alpha_deg"].to_list()) for b in betas))
    clean = clean.filter(pl.col("alpha_deg").is_in(sorted(alphas)))
    fig, axes = plt.subplots(len(betas), len(SIDESLIP_COEFFICIENTS), figsize=(18, 5 * len(betas)), sharey="col",
                             squeeze=False)
    for row, beta in zip(axes, betas):
        rows = clean.filter(pl.col("beta_deg") == beta).sort("alpha_deg")
        alpha = rows["alpha_deg"].to_numpy()
        for ax, c in zip(row, SIDESLIP_COEFFICIENTS):
            stacked_bars(ax, alpha, part_values(rows, c), mean(rows, "total", c), *bounds(rows, "total", c))
            ax.set_title(f"β = {beta:g}°")
            finish(ax, df, "α (deg)", c)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Clean wing by part for sideslip")
    save(fig, path, show)


def plot_surface_loads(df: pl.DataFrame, surface: str, path: Path, show: bool) -> None:
    at_zero = df.filter(pl.col("beta_deg") == 0)
    clean = at_zero.filter(pl.col("mode") == "clean")
    deflected = at_zero.filter(pl.col("family") == SURFACE_FAMILIES[surface], pl.col("mode") == "sym")
    angle = f"{surface}_r_deg"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, quantity in zip(axes.flat, ("C_L", "C_D", "C_m", "C_h")):
        column = f"{surface}_r__{quantity}"
        plot_series(ax, clean, "alpha_deg", column, "black", label="clean")
        for i, delta in enumerate(sorted(deflected[angle].unique().to_list())):
            plot_series(ax, deflected.filter(pl.col(angle) == delta), "alpha_deg", column, f"C{i}",
                        label=f"δ {delta:g}°")
        finish(ax, df, "α (deg)", quantity)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{surface}_r loads, β = 0°")
    save(fig, path, show)


def aircraft_changes(df: pl.DataFrame, coefficients: tuple[str, ...] = AIRCRAFT_COEFFICIENTS) -> pl.DataFrame:
    """Each deflected set point's change in the aircraft's coefficients (d<C>) from the clean set point at the same
    alpha and beta, with worst-case bounds d<C>_lo (deflected min - clean max) and d<C>_hi (deflected max - clean
    min). Set points without a clean one (beta -10) are left out."""
    clean = df.filter(pl.col("mode") == "clean").select(
        "alpha_deg", "beta_deg",
        *(pl.col(f"total__{c}_{stat}").alias(f"{c}_clean_{stat}") for c in coefficients
          for stat in ("mean", "min", "max")))
    changes: list[pl.Expr] = []
    for c in coefficients:
        changes += [(pl.col(f"total__{c}_mean") - pl.col(f"{c}_clean_mean")).alias(f"d{c}"),
                    (pl.col(f"total__{c}_min") - pl.col(f"{c}_clean_max")).alias(f"d{c}_lo"),
                    (pl.col(f"total__{c}_max") - pl.col(f"{c}_clean_min")).alias(f"d{c}_hi")]
    return (df.filter(pl.col("mode") != "clean").join(clean, on=["alpha_deg", "beta_deg"])
            .select("family", "mode", "delta_deg", "alpha_deg", "beta_deg", "converged", *changes))


def plot_aircraft_change(df: pl.DataFrame, changes: pl.DataFrame, family: str, path: Path, show: bool) -> None:
    rows = changes.filter(pl.col("family") == family, pl.col("mode") == "sym", pl.col("beta_deg") == 0)
    fig, axes = plt.subplots(1, len(AIRCRAFT_COEFFICIENTS), figsize=(18, 5.5))
    for i, alpha in enumerate(sorted(rows["alpha_deg"].unique().to_list())):
        sub = rows.filter(pl.col("alpha_deg") == alpha)
        # The clean wing is the point (0, 0); sort it in with the deflections.
        delta = np.append(sub["delta_deg"].to_numpy(), 0.0)
        order = np.argsort(delta)
        converged = np.append(sub["converged"].to_numpy(), True)[order]
        for ax, c in zip(axes, AIRCRAFT_COEFFICIENTS):
            values, low, high = (np.append(sub[f"d{c}{end}"].to_numpy(), 0.0)[order] for end in ("", "_lo", "_hi"))
            ax.errorbar(delta[order], values, yerr=[values - low, high - values], fmt="none", ecolor=f"C{i}",
                        capsize=2, lw=1)
            plot_line(ax, delta[order], values, converged, f"C{i}", label=f"α {alpha:g}°")
    for ax, c in zip(axes, AIRCRAFT_COEFFICIENTS):
        finish(ax, df, "deflection (deg)", c, f"Δ{c}")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{family}: change from clean wing, β = 0°")
    save(fig, path, show)


def plot_hinge_sideslip(df: pl.DataFrame, surface: str, path: Path, show: bool) -> None:
    rows = df.filter(pl.col("mode") == "right", pl.col("family") == SURFACE_FAMILIES[surface])
    angle = f"{surface}_r_deg"
    alphas = sorted(rows["alpha_deg"].unique().to_list())
    deltas = sorted(rows[angle].unique().to_list())
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, delta in enumerate(deltas):
        for j, alpha in enumerate(alphas):
            sub = rows.filter(pl.col("alpha_deg") == alpha, pl.col(angle) == delta)
            plot_series(ax, sub, "beta_deg", f"{surface}_r__C_h", f"C{i}", marker=ALPHA_MARKERS[j % len(ALPHA_MARKERS)],
                        label=f"δ {delta:g}°, α {alpha:g}°")
    finish(ax, df, "β (deg)", "C_h")
    ax.legend(fontsize=8, ncol=len(deltas))
    ax.set_title(f"{surface}_r hinge moment for sideslip")
    save(fig, path, show)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Control surface loads from setpoints.parquet.")
    parser.add_argument("setpoints", type=Path, help="setpoints.parquet from src/cfd/dataset.py")
    parser.add_argument("--out", type=Path, help="output folder (default: <parquet folder>/postpro/surfaces)")
    parser.add_argument("--show", action="store_true", help="also open the figures")
    args = parser.parse_args(argv)

    df = pl.read_parquet(args.setpoints)
    out: Path = args.out or args.setpoints.parent / "postpro" / "surfaces"

    plot_clean_breakdown(df, out / "clean_wing_by_part.png", args.show)
    plot_clean_sideslip(df, out / "clean_wing_sideslip_by_part.png", args.show)
    for surface in SURFACES:
        plot_surface_loads(df, surface, out / "surface_loads" / f"{surface}.png", args.show)
        plot_hinge_sideslip(df, surface, out / "hinge_moments_sideslip" / f"{surface}.png", args.show)
    changes = aircraft_changes(df)
    for family in FAMILIES:
        plot_aircraft_change(df, changes, family, out / "aircraft_change" / f"{family}.png", args.show)
    print(f"Written to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
