"""Convergence of every solve: a table, and coefficient and residual histories.

    .venv/Scripts/python.exe -m postpro.convergence <run folder | folder of runs> ... [--out FOLDER] [--show]

A solve has converged when C_L, C_D and C_m are each within the tolerance of their running mean over the
window (the test OpenFOAM uses to stop the run; see src/cfd/solve_report.py). Writes to <out> (default
<first path>/postpro/convergence/):
    convergence.csv            per set point: iterations, stopped early, converged, each monitored coefficient's
                               final distance from its running mean ("gap", against the tolerance) and its range
                               over the final window (large for an oscillating solve), and the final initial
                               residuals
    coefficients_<case>.png    C_L, C_D, C_m and C_Y against iteration, one line per condition
    residuals_<case>.png       initial residuals of Ux, Uy, Uz, p, k and omega against iteration (log scale)
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from postpro.common import find_runs, output_dir, print_table, run_parser, solve_case, solve_dirs
from src.cfd.solve_report import MONITORED, SolveResult, coefficient_history, gap_from_running_mean, read_table

RESIDUALS: tuple[str, ...] = ("Ux", "Uy", "Uz", "p", "k", "omega")
SKIP_ITERATIONS: int = 50  # start-up transient, left out of the coefficient plots so it doesn't flatten them


def residual_history(solve_dir: Path) -> tuple[list[float], dict[str, list[float]]]:
    """Iterations, and each field's initial residual at each (from postProcessing/solverInfo)."""
    columns, rows = read_table("solverInfo/*/solverInfo.dat", solve_dir / "postProcessing")
    found = {name: columns.index(f"{name}_initial") for name in RESIDUALS if f"{name}_initial" in columns}
    return [r[0] for r in rows], {name: [r[i] for r in rows] for name, i in found.items()}


def convergence_row(case: str, solve_dir: Path) -> dict[str, object]:
    result = SolveResult.model_validate_json((solve_dir / "results.json").read_text())
    settings = solve_case(solve_dir)["settings"]
    _, history = coefficient_history(solve_dir)
    return {
        "case": case, "condition": solve_dir.name, "alpha": result.alpha_deg, "beta": result.beta_deg,
        "iterations": result.iterations, "stopped_early": result.stopped_early, "converged": result.converged,
        "tolerance": settings["convergence_tolerance"],
        **{f"{n}_gap": gap_from_running_mean(history[n], int(settings["convergence_window"])) for n in MONITORED},
        **{f"{n}_range": result.coefficients[n].range for n in MONITORED},
        **{f"res_{k}": v for k, v in result.final_residuals.items()},
    }


def plot_coefficients(case: str, folders: list[Path], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for folder in folders:
        iterations, coefficients = coefficient_history(folder)
        for ax, name in zip(axes.flat, ("C_L", "C_D", "C_m", "C_Y")):
            ax.plot(iterations[SKIP_ITERATIONS:], coefficients[name][SKIP_ITERATIONS:],
                    "--" if "_b+00.0" not in folder.name else "-", label=folder.name)
    for ax, name in zip(axes.flat, ("C_L", "C_D", "C_m", "C_Y")):
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("iteration")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{case}: coefficient histories from iteration {SKIP_ITERATIONS} (dashed = sideslip)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_residuals(case: str, folders: list[Path], path: Path) -> None:
    columns = 3
    fig, axes = plt.subplots(math.ceil(len(RESIDUALS) / columns), columns, figsize=(15, 8), sharex=True)
    for folder in folders:
        iterations, residuals = residual_history(folder)
        for ax, name in zip(axes.flat, RESIDUALS):
            if name in residuals:
                ax.semilogy(iterations, residuals[name], label=folder.name)
    for ax, name in zip(axes.flat, RESIDUALS):
        ax.set_title(f"{name} initial residual", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    for ax in axes[-1]:
        ax.set_xlabel("iteration")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"{case}: residuals")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv: list[str] | None = None) -> None:
    args = run_parser("Convergence table and histories of solved set points.").parse_args(argv)
    runs = find_runs(args.paths)
    out = output_dir(args.out, args.paths, runs, "convergence")

    rows: list[dict[str, object]] = []
    for run in runs:
        folders = solve_dirs(run)
        if not folders:
            continue
        rows.extend(convergence_row(run.name, folder) for folder in folders)
        plot_coefficients(run.name, folders, out / f"coefficients_{run.name}.png")
        plot_residuals(run.name, folders, out / f"residuals_{run.name}.png")
        if not args.show:
            plt.close("all")
    table = pl.from_dicts(rows, infer_schema_length=None).sort("case", "beta", "alpha")
    table.write_csv(out / "convergence.csv")

    print_table(table.select("case", "alpha", "beta", "iterations", "converged",
                             *[f"{n}_gap" for n in MONITORED], *[f"{n}_range" for n in MONITORED]),
                "Convergence (gap: final distance from the running mean, against the tolerance):")
    unconverged = table.filter(~pl.col("converged"))
    print(f"\n{table.height - unconverged.height} of {table.height} set points converged."
          + (" Not converged: " + ", ".join(f"{c} {d}" for c, d in unconverged.select("case", "condition").rows())
             if unconverged.height else ""))
    print(f"Table and plots written to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
