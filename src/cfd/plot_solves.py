"""Plot solved flow conditions: convergence for each case, and coefficients against alpha across cases.

    uv run python -m src.cfd.plot_solves <run folder> [<run folder> ...] [--fit-alpha MIN MAX] [--show]

A run folder is a downloaded hpc/runs/<case name>/, holding solve/<condition>/
folders with results.json. Several run folders (e.g. the mesh study's three) are overlaid. Writes to
<first run folder>/plots/:
    convergence_<case>.png   C_L, C_D, C_m and C_Y against iteration, one line per condition
    coefficients.png         all six coefficients against alpha, and the drag polar; one line per case and beta
and prints the coefficients, the lift-curve and pitching-moment slopes, and the sideslip derivatives.
Unconverged conditions are drawn hollow.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.cfd.solve_report import SolveResult, coefficient_history

COEFFICIENTS = ("C_L", "C_D", "C_Y", "C_l", "C_m", "C_n")


def load_results(run_dirs: list[Path]) -> pl.DataFrame:
    """One row per solved condition: case, folder, alpha, beta, converged and the six coefficients."""
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        for path in (run_dir / "solve").glob("*/results.json"):
            if path.parent.name.endswith(".previous"):
                continue
            result = SolveResult.model_validate_json(path.read_text())
            rows.append({"case": run_dir.name, "folder": str(path.parent), "alpha": result.alpha_deg,
                         "beta": result.beta_deg, "converged": result.converged,
                         **{name: result.coefficients[name].mean for name in COEFFICIENTS}})
    return pl.DataFrame(rows).sort("case", "beta", "alpha") if rows else pl.DataFrame()


def plot_convergence(case: pl.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row in case.iter_rows(named=True):
        iterations, coefficients = coefficient_history(Path(row["folder"]))
        label = f"α {row['alpha']:g}°, β {row['beta']:g}°" + ("" if row["converged"] else " (not converged)")
        for ax, name in zip(axes.flat, ("C_L", "C_D", "C_m", "C_Y")):
            # Skip the start-up transient, which would flatten the rest of the history.
            ax.plot(iterations[50:], coefficients[name][50:], "-" if row["beta"] == 0 else "--", label=label)
            ax.set_ylabel(name)
            ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("iteration")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{case['case'][0]}: coefficient histories from iteration 50 (dashed = sideslip)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_coefficients(results: pl.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    panels = [("alpha", "C_L"), ("alpha", "C_D"), ("alpha", "C_m"), ("C_D", "C_L"),
              ("alpha", "C_Y"), ("alpha", "C_l"), ("alpha", "C_n")]
    for i, ((case, beta), group) in enumerate(results.group_by("case", "beta", maintain_order=True)):
        label, colour = f"{case}, β {beta:g}°", f"C{i}"  # one colour per case and beta, in every panel
        converged, unconverged = group.filter(pl.col("converged")), group.filter(~pl.col("converged"))
        for ax, (x, y) in zip(axes.flat, panels):
            ax.plot(group[x], group[y], "-" if beta == 0 else "--", color=colour, label=label)
            ax.plot(converged[x], converged[y], "o", color=colour)
            ax.plot(unconverged[x], unconverged[y], "o", color=colour, mfc="white")
    for ax, (x, y) in zip(axes.flat, panels):
        ax.set_xlabel("α (deg)" if x == "alpha" else x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
    axes[0, 3].set_title("drag polar")
    axes[1, 3].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Coefficients (hollow = not converged)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def slopes(results: pl.DataFrame, alpha_min: float, alpha_max: float) -> pl.DataFrame:
    """Lift-curve and pitching-moment slopes per case and beta, fitted over alpha_min..alpha_max.

    The static margin -dC_m/dC_L is how far the neutral point is behind the moment reference point, in
    reference chords (positive = stable in pitch).
    """
    rows: list[dict[str, object]] = []
    fitted = results.filter(pl.col("alpha").is_between(alpha_min, alpha_max))
    for (case, beta), group in fitted.group_by("case", "beta", maintain_order=True):
        if group.height < 2:
            continue
        lift_slope, lift_zero = np.polyfit(group["alpha"], group["C_L"], 1)
        moment_slope = np.polyfit(group["alpha"], group["C_m"], 1)[0]
        rows.append({"case": case, "beta": beta, "points": group.height,
                     "C_L_alpha_per_rad": lift_slope * 180 / math.pi, "C_L_0": lift_zero,
                     "C_m_alpha_per_rad": moment_slope * 180 / math.pi,
                     "static_margin": -moment_slope / lift_slope})
    return pl.DataFrame(rows)


def sideslip_derivatives(results: pl.DataFrame) -> pl.DataFrame:
    """C_Y, C_l and C_n per degree of sideslip at each alpha, from the beta = 0 condition.

    Statically stable: C_n_beta > 0 (weathercock) and C_l_beta < 0 (dihedral effect).
    """
    lateral = ("C_Y", "C_l", "C_n")
    zero = results.filter(pl.col("beta") == 0).select("case", "alpha", *lateral)
    return (results.filter(pl.col("beta") != 0)
            .join(zero, on=["case", "alpha"], suffix="_0")
            .select("case", "alpha", "beta",
                    *[((pl.col(n) - pl.col(f"{n}_0")) / pl.col("beta")).alias(f"{n}_beta") for n in lateral]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot convergence and coefficients of solved flow conditions.")
    parser.add_argument("run_dirs", type=Path, nargs="+", help="run folders, each holding solve/<condition>/")
    parser.add_argument("--fit-alpha", type=float, nargs=2, default=(0.0, 10.0), metavar=("MIN", "MAX"),
                        help="alpha range (deg) for the slopes: the linear part of the lift curve")
    parser.add_argument("--show", action="store_true", help="also open the figures")
    args = parser.parse_args(argv)

    results = load_results(args.run_dirs)
    if results.is_empty():
        raise SystemExit("no solve/<condition>/results.json in " + ", ".join(map(str, args.run_dirs)))
    out = args.run_dirs[0] / "plots"
    out.mkdir(exist_ok=True)
    for (case,), group in results.group_by("case", maintain_order=True):
        plot_convergence(group, out / f"convergence_{case}.png")
    plot_coefficients(results, out / "coefficients.png")
    print(f"plots written to {out}")

    # ASCII tables: the Windows console can't print Polars' default box-drawing characters.
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_formatting="ASCII_MARKDOWN",
                   tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
        print(results.drop("folder"))
        print(f"\nSlopes over alpha {args.fit_alpha[0]:g} to {args.fit_alpha[1]:g} deg:")
        print(slopes(results, *args.fit_alpha))
        print("\nSideslip derivatives (per deg):")
        print(sideslip_derivatives(results))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
