"""Boundary-layer checks: y+ on the walls, prism-layer coverage, and skin friction against flat-plate values.

    .venv/Scripts/python.exe -m postpro.boundary_layer <run folder> ... [--conditions a+00.0_b+00.0 ...]
        [--stations 60 120 180 240 300] [--out FOLDER] [--show]

y+ is the height of the first cell centre off the wall in wall units. The walls use Spalding's wall function,
which covers the viscous sublayer, buffer layer and log layer, but wall treatments are least accurate in the
buffer layer (y+ about 5-30), so the share of the wetted area there is worth watching. Writes to <out>
(default <first path>/postpro/boundary_layer/):
    yplus_patches.csv     min / mean / max y+ per patch (postProcessing/yPlus), per run and condition
    yplus_bands.csv       share of the wetted area (all walls) in the y+ bands below, per run and condition
    yplus_<condition>.png area-weighted y+ distribution, one line per run
    layers.csv            prism layers per patch from each run's mesh/mesh_report.json, if downloaded
    cf_<condition>.png    skin friction Cf = |tau_w| / (1/2 V^2) along chordwise cuts, upper (solid) and lower
                          (dashed), with the turbulent (0.0592 Re_x^-1/5) and laminar (Blasius, 0.664 Re_x^-1/2)
                          flat-plate values, Re_x from the cut's leading edge
The flat-plate comparison is a sanity check, not a validation: the wing is a flat plate at zero incidence only
at alpha = -3 deg (alpha = 0 is the nearest in the mesh study), and its swept leading edge makes the boundary
layer three-dimensional. The solver is fully turbulent, so Cf should sit near the turbulent curve, well above
the laminar one. The wall surfaces (postProcessing/wallSurfaces/*/walls.vtp) must be downloaded for the y+
bands and the Cf plots.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from postpro.common import (DEFAULT_STATIONS_MM, WallSurface, chordwise, common_conditions, find_runs, output_dir,
                            print_table, read_vtp, run_parser, solve_case, wall_surface_path)

YPLUS_BANDS: tuple[float, ...] = (0.0, 1.0, 5.0, 30.0, 100.0, 300.0, math.inf)


def yplus_patches(solve_dir: Path) -> list[dict[str, object]]:
    """min / mean / max y+ per patch at the last iteration (postProcessing/yPlus/*/yPlus.dat)."""
    files = sorted(solve_dir.glob("postProcessing/yPlus/*/yPlus.dat"), key=lambda p: float(p.parent.name))
    rows: dict[str, dict[str, object]] = {}
    last_time = -math.inf
    for file in files:
        for line in file.read_text().splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            time = float(parts[0])
            if time > last_time:
                last_time, rows = time, {}
            rows[parts[1]] = {"patch": parts[1], "min": float(parts[2]), "max": float(parts[3]),
                              "mean": float(parts[4])}
    return list(rows.values())


def yplus_bands(surface: WallSurface) -> dict[str, float]:
    """Share of the wetted area in each y+ band."""
    yplus, area = surface.fields["yPlus"], surface.areas
    total = area.sum()
    return {f"y+ {lo:g}-{hi:g}": float(area[(yplus >= lo) & (yplus < hi)].sum() / total)
            for lo, hi in zip(YPLUS_BANDS[:-1], YPLUS_BANDS[1:])}


def layers(run: Path) -> list[dict[str, object]]:
    path = run / "mesh" / "mesh_report.json"
    if not path.is_file():
        return []
    report = json.loads(path.read_text())
    return [{"case": run.name, **layer} for layer in report.get("layers", [])]


def plot_yplus(condition: str, surfaces: dict[str, WallSurface], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(-1, 3, 81)
    for i, (run, surface) in enumerate(surfaces.items()):
        weights = surface.areas / surface.areas.sum() * 100
        ax.hist(np.clip(surface.fields["yPlus"], bins[0], bins[-1]), bins=bins, weights=weights,
                histtype="step", color=f"C{i % 10}", label=run, lw=1.4)
    for edge in YPLUS_BANDS[1:-1]:
        ax.axvline(edge, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("y+")
    ax.set_ylabel("share of wetted area (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"y+ distribution over all walls, {condition}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_skin_friction(condition: str, surfaces: dict[str, WallSurface], speeds: dict[str, float], nu: float,
                       stations_mm: list[float], path: Path) -> None:
    columns = min(3, len(stations_mm))
    rows = math.ceil(len(stations_mm) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.2 * rows), squeeze=False)
    for i, (run, surface) in enumerate(surfaces.items()):
        cf = np.linalg.norm(surface.fields["wallShearStress"], axis=1) / (0.5 * speeds[run] ** 2)
        for ax, y in zip(axes.flat, stations_mm):
            line = chordwise(surface, y / 1000, cf)
            for side, style in (("upper", "-"), ("lower", "--")):
                part = line.filter(pl.col("side") == side)
                ax.plot(part["x_c"], part["value"], style, color=f"C{i % 10}", lw=1.2, label=f"{run} {side}")
            if i == 0 and not line.is_empty():
                # Flat-plate values along this cut, with x from its leading edge.
                x_le, x_te = line["x"].min(), line["x"].max()
                x = np.linspace(x_le, x_te, 201)[1:]
                re_x = speeds[run] * (x - x_le) / nu
                x_c = (x - x_le) / (x_te - x_le)
                ax.plot(x_c, 0.0592 * re_x ** -0.2, color="black", lw=1, label="turbulent flat plate")
                ax.plot(x_c, 0.664 * re_x ** -0.5, color="black", lw=1, ls=":", label="laminar flat plate")
    for ax, y in zip(axes.flat, stations_mm):
        ax.set_title(f"y = {y:g} mm", fontsize=10)
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cf")
        ax.set_ylim(0, 0.02)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(stations_mm):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Skin friction, {condition} (solid = upper, dashed = lower)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv: list[str] | None = None) -> None:
    parser = run_parser("y+, prism layers and skin friction of solved set points.")
    parser.add_argument("--conditions", nargs="+", help="condition folders to compare, e.g. a+00.0_b+00.0")
    parser.add_argument("--stations", type=float, nargs="+", default=list(DEFAULT_STATIONS_MM),
                        help="spanwise stations for the skin friction cuts (mm, starboard side)")
    args = parser.parse_args(argv)

    runs = find_runs(args.paths)
    out = output_dir(args.out, args.paths, runs, "boundary_layer")
    patch_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    for condition in common_conditions(runs, args.conditions):
        surfaces: dict[str, WallSurface] = {}
        speeds: dict[str, float] = {}
        nu = math.nan
        for run in runs:
            solve_dir = run / "solve" / condition
            patch_rows += [{"case": run.name, "condition": condition, **row} for row in yplus_patches(solve_dir)]
            path = wall_surface_path(solve_dir)
            if path is None:
                print(f"{run.name} {condition}: no postProcessing/wallSurfaces/*/walls.vtp, y+ bands and Cf skipped")
                continue
            surfaces[run.name] = read_vtp(path)
            settings = solve_case(solve_dir)["settings"]
            speeds[run.name], nu = settings["speed"], settings["kinematic_viscosity"]
            band_rows.append({"case": run.name, "condition": condition, **yplus_bands(surfaces[run.name])})
        if surfaces:
            plot_yplus(condition, surfaces, out / f"yplus_{condition}.png")
            plot_skin_friction(condition, surfaces, speeds, nu, args.stations, out / f"cf_{condition}.png")
            if not args.show:
                plt.close("all")

    if patch_rows:
        patches = pl.DataFrame(patch_rows)
        patches.write_csv(out / "yplus_patches.csv")
        print_table(patches.group_by("case", "patch", maintain_order=True)
                    .agg(pl.col("min").min(), pl.col("mean").mean(), pl.col("max").max()),
                    "y+ per patch (min / mean / max over the conditions):")
    if band_rows:
        bands = pl.DataFrame(band_rows)
        bands.write_csv(out / "yplus_bands.csv")
        print_table(bands, "Share of wetted area in each y+ band:")
    layer_rows = [row for run in runs for row in layers(run)]
    if layer_rows:
        layer_table = pl.DataFrame(layer_rows)
        layer_table.write_csv(out / "layers.csv")
        print_table(layer_table, "Prism layers (from mesh_report.json):")
    print(f"\nTables and plots written to {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
