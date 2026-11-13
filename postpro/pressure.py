"""Surface pressure along chordwise cuts of the wing, compared between runs (e.g. the mesh study's meshes).

    .venv/Scripts/python.exe -m postpro.pressure <run folder> ... [--conditions a+10.0_b+00.0 ...]
        [--stations 60 120 180 240 300] [--out FOLDER] [--show]

Cp = p / (1/2 V^2): simpleFoam's p is kinematic (pressure / density) and the freestream pressure is 0. The
values come from postProcessing/wallSurfaces/<iteration>/walls.vtp (one value per wall face), so those files
must be downloaded. Each spanwise station (mm, starboard side) cuts the walls; the faces it crosses that look
up or down give the upper and lower surface, and x/c runs from the cut's leading edge (0) to its trailing
edge (1).

Writes to <out> (default <first path>/postpro/pressure/), for every condition solved in all the runs (or those
given with --conditions):
    cp_<condition>.png     -Cp against x/c at each station, upper (solid) and lower (dashed), one colour per run
    cp_differences.csv     with two or more runs: the RMS difference in Cp from the last run listed (put the
                           finest mesh last), per run, condition, station and surface, over x/c 0.02-0.98
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from postpro.common import (DEFAULT_STATIONS_MM, chordwise, common_conditions, find_runs, output_dir, print_table,
                            read_vtp, run_parser, solve_case, wall_surface_path)

COMPARE_X_C: np.ndarray = np.linspace(0.02, 0.98, 97)  # leaves out the leading and trailing edges themselves


def pressure_cuts(solve_dir: Path, stations_mm: list[float]) -> dict[float, pl.DataFrame] | None:
    """Cp along each station's cut, or None if the solve's wall surface wasn't downloaded."""
    path = wall_surface_path(solve_dir)
    if path is None:
        return None
    surface = read_vtp(path)
    speed = solve_case(solve_dir)["settings"]["speed"]
    cp = surface.fields["p"] / (0.5 * speed**2)
    return {y: chordwise(surface, y / 1000, cp) for y in stations_mm}


def rms_difference(cuts: pl.DataFrame, reference: pl.DataFrame, side: str) -> float:
    a, b = cuts.filter(pl.col("side") == side), reference.filter(pl.col("side") == side)
    if a.height < 2 or b.height < 2:
        return math.nan
    ya = np.interp(COMPARE_X_C, a["x_c"].to_numpy(), a["value"].to_numpy())
    yb = np.interp(COMPARE_X_C, b["x_c"].to_numpy(), b["value"].to_numpy())
    return float(np.sqrt(np.mean((ya - yb) ** 2)))


def plot_condition(condition: str, by_run: dict[str, dict[float, pl.DataFrame]], stations_mm: list[float],
                   path: Path) -> None:
    columns = min(3, len(stations_mm))
    rows = math.ceil(len(stations_mm) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.2 * rows), squeeze=False)
    for i, (run, cuts) in enumerate(by_run.items()):
        for ax, y in zip(axes.flat, stations_mm):
            for side, style in (("upper", "-"), ("lower", "--")):
                line = cuts[y].filter(pl.col("side") == side)
                ax.plot(line["x_c"], -line["value"], style, color=f"C{i % 10}", lw=1.2,
                        label=f"{run} {side}")
    for ax, y in zip(axes.flat, stations_mm):
        ax.set_title(f"y = {y:g} mm", fontsize=10)
        ax.set_xlabel("x/c")
        ax.set_ylabel("-Cp")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(stations_mm):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Surface pressure, {condition} (solid = upper, dashed = lower)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv: list[str] | None = None) -> None:
    parser = run_parser("Compare surface pressure along chordwise cuts between runs.")
    parser.add_argument("--conditions", nargs="+", help="condition folders to compare, e.g. a+10.0_b+00.0")
    parser.add_argument("--stations", type=float, nargs="+", default=list(DEFAULT_STATIONS_MM),
                        help="spanwise stations (mm, starboard side)")
    args = parser.parse_args(argv)

    runs = find_runs(args.paths)
    out = output_dir(args.out, args.paths, runs, "pressure")
    differences: list[dict[str, object]] = []
    for condition in common_conditions(runs, args.conditions):
        by_run: dict[str, dict[float, pl.DataFrame]] = {}
        for run in runs:
            cuts = pressure_cuts(run / "solve" / condition, args.stations)
            if cuts is None:
                print(f"{run.name} {condition}: no postProcessing/wallSurfaces/*/walls.vtp, skipped")
                continue
            by_run[run.name] = cuts
        if not by_run:
            continue
        plot_condition(condition, by_run, args.stations, out / f"cp_{condition}.png")
        if not args.show:
            plt.close("all")

        reference_run = list(by_run)[-1]
        for run, cuts in list(by_run.items())[:-1]:
            for y in args.stations:
                differences.append({
                    "run": run, "reference": reference_run, "condition": condition, "station_mm": y,
                    **{f"rms_dCp_{side}": rms_difference(cuts[y], by_run[reference_run][y], side)
                       for side in ("upper", "lower")},
                })

    if differences:
        table = pl.DataFrame(differences)
        table.write_csv(out / "cp_differences.csv")
        print_table(table, "RMS difference in Cp from the reference run (last listed):")
    print(f"\nPlots written to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
