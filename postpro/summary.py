"""Summary of solved set points: coefficients, derivatives and control surface hinge moments.

    .venv/Scripts/python.exe -m postpro.summary <run folder | folder of runs> ... [--fit-alpha MIN MAX]
        [--clean CASE] [--out FOLDER] [--show]

Writes to <out> (default <first path>/postpro/summary/):
    setpoints.csv          one row per set point (case and flow condition): deflections, alpha, beta,
                           convergence, the six coefficients and their ranges over the final window, and every
                           surface's hinge moment (C_h_<surface>, and M_h_<surface> in N m)
    derivatives_alpha.csv  dC/d(alpha) per radian at every set point, for each coefficient (central
                           differences along alpha within one case and sideslip)
    slopes.csv             straight-line fits over --fit-alpha (the linear part of the lift curve), per case and
                           sideslip: C_L_alpha, C_L at alpha = 0, C_m_alpha and the static margin -C_m_alpha /
                           C_L_alpha (neutral point behind the moment reference, in reference chords)
    derivatives_beta.csv   C_Y, C_l and C_n per radian of sideslip at each case and alpha (straight-line fit
                           over the sideslips solved there). Statically stable: C_n_beta > 0, C_l_beta < 0
    control.csv            each deflected case's change in the six coefficients from the clean (undeflected)
                           case at the same alpha and beta, and per degree of deflection where every deflected
                           surface is at the same angle. Needs a clean case among the runs (--clean, or the
                           only run with no deflections)
    coefficients.png       the six coefficients against alpha and the drag polar, one line per case and beta
    hinge_moments.png      every surface's C_h against alpha, one line per case and beta
Hollow markers are unconverged set points. Tables of the main values are also printed.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from postpro.common import COEFFICIENTS, find_runs, load_results, output_dir, print_table, run_parser, surfaces_of

MAX_LEGEND_ENTRIES: int = 16  # beyond this the legend would hide the plot


def alpha_derivatives(results: pl.DataFrame) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for _, group in results.group_by("case", "beta", maintain_order=True):
        group = group.sort("alpha")
        if group.height < 2:
            continue
        alpha = np.radians(group["alpha"].to_numpy())
        rows.append(group.select("case", "condition", "alpha", "beta").with_columns(
            **{f"{n}_alpha": pl.Series(np.gradient(group[n].to_numpy(), alpha)) for n in COEFFICIENTS}))
    return pl.concat(rows) if rows else pl.DataFrame()


def slopes(results: pl.DataFrame, alpha_min: float, alpha_max: float) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    fitted = results.filter(pl.col("alpha").is_between(alpha_min, alpha_max))
    for (case, beta), group in fitted.group_by("case", "beta", maintain_order=True):
        if group["alpha"].n_unique() < 2:
            continue
        alpha = np.radians(group["alpha"].to_numpy())
        lift_slope, lift_zero = np.polyfit(alpha, group["C_L"].to_numpy(), 1)
        moment_slope = np.polyfit(alpha, group["C_m"].to_numpy(), 1)[0]
        rows.append({"case": case, "beta": beta, "points": group.height, "C_L_alpha": lift_slope,
                     "C_L_0": lift_zero, "C_m_alpha": moment_slope, "static_margin": -moment_slope / lift_slope})
    return pl.DataFrame(rows)


def beta_derivatives(results: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for (case, alpha), group in results.group_by("case", "alpha", maintain_order=True):
        if group["beta"].n_unique() < 2:
            continue
        beta = np.radians(group["beta"].to_numpy())
        rows.append({"case": case, "alpha": alpha, "betas": group.height,
                     **{f"{n}_beta": np.polyfit(beta, group[n].to_numpy(), 1)[0] for n in ("C_Y", "C_l", "C_n")}})
    return pl.DataFrame(rows).sort("case", "alpha") if rows else pl.DataFrame()


def clean_case(results: pl.DataFrame, chosen: str | None) -> str | None:
    """The undeflected reference case: `chosen`, or the only case with every deflection 0 (if any are deflected)."""
    if chosen:
        return chosen
    deflection_columns = [c for c in results.columns if c.endswith("_deg")]
    per_case = results.group_by("case").agg(pl.max_horizontal(pl.col(deflection_columns).abs()).max().alias("max"))
    clean = per_case.filter(pl.col("max") == 0)["case"].to_list()
    return clean[0] if len(clean) == 1 and per_case.height > 1 else None


def control_effects(results: pl.DataFrame, clean: str) -> pl.DataFrame:
    deflection_columns = [c for c in results.columns if c.endswith("_deg")]
    reference = results.filter(pl.col("case") == clean).select("alpha", "beta", *COEFFICIENTS)
    deflected = results.filter(pl.col("case") != clean)
    rows: list[dict[str, object]] = []
    for row in deflected.join(reference, on=["alpha", "beta"], suffix="_clean").iter_rows(named=True):
        angles = {c.removesuffix("_deg"): row[c] for c in deflection_columns if row[c] != 0}
        same_angle = len({abs(a) for a in angles.values()}) == 1
        deflection = next(iter(angles.values())) if same_angle else None
        rows.append({
            "case": row["case"], "condition": row["condition"], "alpha": row["alpha"], "beta": row["beta"],
            "surfaces": " ".join(f"{s}={a:g}" for s, a in angles.items()), "deflection_deg": deflection,
            **{f"d{n}": row[n] - row[f"{n}_clean"] for n in COEFFICIENTS},
            **{f"d{n}_per_deg": ((row[n] - row[f"{n}_clean"]) / deflection if deflection else None)
               for n in COEFFICIENTS},
        })
    return pl.DataFrame(rows)


SIDESLIP_STYLES: tuple[str, ...] = ("--", ":", "-.")  # for each nonzero sideslip, in order; beta = 0 is solid


def _series(results: pl.DataFrame) -> list[tuple[str, str, str, pl.DataFrame]]:
    """(label, colour, line style, rows) for each case and sideslip: one colour per case, one line style per
    sideslip, the same in every figure."""
    colours = {case: f"C{i % 10}" for i, case in enumerate(results["case"].unique(maintain_order=True))}
    sideslips = sorted(b for b in results["beta"].unique() if b != 0)
    styles = {0.0: "-"} | {b: SIDESLIP_STYLES[i % len(SIDESLIP_STYLES)] for i, b in enumerate(sideslips)}
    return [(f"{case}, β {beta:g}°", colours[case], styles[beta], group.sort("alpha"))
            for (case, beta), group in results.group_by("case", "beta", maintain_order=True)]


def _plot_points(ax: plt.Axes, group: pl.DataFrame, x: str, y: str, label: str, colour: str, style: str) -> None:
    ax.plot(group[x], group[y], style, color=colour, label=label)
    converged = group.filter(pl.col("converged"))
    unconverged = group.filter(~pl.col("converged"))
    ax.plot(converged[x], converged[y], "o", color=colour, ms=4)
    ax.plot(unconverged[x], unconverged[y], "o", color=colour, mfc="white", ms=4)


def plot_coefficients(results: pl.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    panels = [("alpha", "C_L"), ("alpha", "C_D"), ("alpha", "C_m"), ("C_D", "C_L"),
              ("alpha", "C_Y"), ("alpha", "C_l"), ("alpha", "C_n")]
    series = _series(results)
    for label, colour, style, group in series:
        for ax, (x, y) in zip(axes.flat, panels):
            _plot_points(ax, group, x, y, label, colour, style)
    for ax, (x, y) in zip(axes.flat, panels):
        ax.set_xlabel("α (deg)" if x == "alpha" else x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
    axes[0, 3].set_title("drag polar")
    axes[1, 3].set_visible(False)
    if len(series) <= MAX_LEGEND_ENTRIES:
        axes[0, 0].legend(fontsize=7)
    fig.suptitle("Coefficients (colour = case, line style = sideslip, hollow = not converged)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_hinge_moments(results: pl.DataFrame, path: Path) -> None:
    surfaces = surfaces_of(results)
    columns = 4
    rows = math.ceil(len(surfaces) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(18, 3.6 * rows), squeeze=False)
    series = _series(results)
    for label, colour, style, group in series:
        for ax, surface in zip(axes.flat, surfaces):
            _plot_points(ax, group, "alpha", f"C_h_{surface}", label, colour, style)
    for ax, surface in zip(axes.flat, surfaces):
        ax.set_title(surface, fontsize=10)
        ax.set_xlabel("α (deg)")
        ax.set_ylabel("C_h")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(surfaces):]:
        ax.set_visible(False)
    if len(series) <= MAX_LEGEND_ENTRIES:
        axes[0, 0].legend(fontsize=7)
    fig.suptitle("Hinge moment coefficients, positive in the direction of positive deflection "
                 "(colour = case, line style = sideslip, hollow = not converged)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv: list[str] | None = None) -> None:
    parser = run_parser("Tables of coefficients, derivatives and hinge moments for solved set points.")
    parser.add_argument("--fit-alpha", type=float, nargs=2, default=(0.0, 10.0), metavar=("MIN", "MAX"),
                        help="alpha range (deg) for the straight-line slopes")
    parser.add_argument("--clean", help="name of the undeflected case to measure control effects against")
    args = parser.parse_args(argv)

    runs = find_runs(args.paths)
    out = output_dir(args.out, args.paths, runs, "summary")
    results = load_results(runs)

    results.drop("folder").write_csv(out / "setpoints.csv")
    derivatives = alpha_derivatives(results)
    derivatives.write_csv(out / "derivatives_alpha.csv")
    fits = slopes(results, *args.fit_alpha)
    fits.write_csv(out / "slopes.csv")
    lateral = beta_derivatives(results)
    lateral.write_csv(out / "derivatives_beta.csv")
    clean = clean_case(results, args.clean)
    control = control_effects(results, clean) if clean else pl.DataFrame()
    if clean:
        control.write_csv(out / "control.csv")

    plot_coefficients(results, out / "coefficients.png")
    plot_hinge_moments(results, out / "hinge_moments.png")

    print_table(results.select("case", "alpha", "beta", "converged", *COEFFICIENTS), "Coefficients:")
    print_table(fits, f"Slopes (per rad) over alpha {args.fit_alpha[0]:g} to {args.fit_alpha[1]:g} deg:")
    if not lateral.is_empty():
        print_table(lateral, "Sideslip derivatives (per rad):")
    if clean:
        print_table(control.select("case", "alpha", "beta", "surfaces", *[f"d{n}" for n in COEFFICIENTS]),
                    f"Control effects (change from {clean}):")
    else:
        print("\nNo control effects: no single undeflected case among the runs (choose one with --clean).")
    print(f"\nTables (including hinge moments for every surface) and plots written to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
