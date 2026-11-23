"""Collect every solved set point into one table, and save each set point's coefficient histories.

Run on Iridis, where the run folders are, then download the output folder:

    source hpc/env.sh && load_python
    python -m src.cfd.dataset [runs dir] [--cases PATTERN ...] [--out FOLDER] [--window N]

    runs dir    default RUNS_DIR (src/cfd/case.py)
    --cases     only cases whose name matches one of these glob patterns, e.g. "batch2_*" (default: every case)
    --out       output folder (default: <runs dir>/dataset)
    --window    iterations the statistics are taken over (default: each solve's convergence_window, 200)

Writes to <out>/:
    setpoints.parquet, setpoints.csv     one row per solved set point (case and flow condition)
    history/<case>/<condition>.parquet   that set point's coefficients at every iteration

Parts: `total` (all walls), `body` (total minus the control surfaces, i.e. the fixed bodies) and each control
surface. The body is taken iteration by iteration, so the parts add up exactly to the total at every iteration.
Every part uses the aircraft's reference values and moment reference point, with the definitions of
solve_report.py:
    C_L, C_D, C_Y   force along lift_dir, drag_dir and side_dir, / (q S)
    C_l, C_m, C_n   -M_x / (q S b), M_y / (q S c), -M_z / (q S b): moments about the reference point, model frame
    C_h             control surfaces only: moment about the hinge axis / (q S c), positive along positive deflection

Columns of setpoints:
    identity      case, condition, description, mesh_job_id, solve_job_id
    deflections   <surface>_deg for every control surface (0 where the case leaves it out)
    labels        family (clean, wingtip, le, obte, ibte, splitter, combined), mode (clean, sym, right, left, mixed,
                  combined), delta_deg (the deflection, when every deflected surface is at the same angle)
    mesh          cells, max_non_orthogonality, min_layer_percent
    reference     ref_area_m2 (S), ref_chord_m (c), ref_span_m (b), ref_x_m, ref_y_m, ref_z_m (moment point)
    hinges        <surface>_hinge_x_m, _y_m, _z_m (point p1, which hinge moments are taken about) and
                  <surface>_hinge_dx, _dy, _dz (unit hinge axis, p1 -> p2)
    flow          alpha_deg, beta_deg, speed_m_s, density_kg_m3, kinematic_viscosity_m2_s, q_pa, reynolds (on c),
                  turbulence_intensity, viscosity_ratio
    axes          drag_dir_x/y/z (the freestream), lift_dir_x/y/z, side_dir_x/y/z: unit vectors, model frame
    convergence   iterations, max_iterations, stopped_early, converged (as in results.json), window_start and
                  window_end (first and last iteration of the window), gap_C_L, gap_C_D, gap_C_m (the stopping
                  test's |last value - running mean| for the total), warnings
    residuals     residual_<field>: the final initial residual of Ux, Uy, Uz, k, omega and p
    loads         <part>__<coefficient>_<stat>, and <surface>__C_h_<stat>, where stat is mean, min, max or std
                  over the window. The window is the same iterations for every part and coefficient in a row.

Back to dimensional loads for any part (q, S, c, b, the directions and the points are all in the same row):
    force (N, model frame)          F = q S (C_L lift_dir + C_D drag_dir + C_Y side_dir)
    moment (N m, about ref point)   M = q S (-b C_l, c C_m, -b C_n); about another point P, M + (r_ref - r_P) x F
    hinge moment (N m)              q S c C_h
Convert means, or the histories: different coefficients reach their min and max at different iterations, so the
min and max converted separately are not a load that actually occurred.

A history file has `iteration`, then <part>__<coefficient> and <surface>__C_h: the same quantities, unwindowed.
"""

import argparse
import fnmatch
import json
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.cfd.case import RUNS_DIR
from src.cfd.runs import RunSummary, summarise
from src.cfd.solve_report import (MONITORED, SolveResult, force_coefficients, gap_from_running_mean, read_table,
                                  vectors, wind_axes)

DEFAULT_OUT = "dataset"  # output folder inside the runs dir
HISTORY_DIR = "history"
SETPOINTS = "setpoints"
SIDES: tuple[str, ...] = ("_r", "_l")
# Surface name without its side -> the family it is labelled with. The upper and lower splitters move together.
SURFACE_FAMILIES: dict[str, str] = {
    "wingtip": "wingtip",
    "le_flap": "le",
    "obte_flap": "obte",
    "ibte_flap": "ibte",
    "upr_splitter_flap": "splitter",
    "lwr_splitter_flap": "splitter",
}

type Loads = tuple[np.ndarray, np.ndarray]  # force and moment at each iteration, each (iterations, 3)


def _split_side(surface: str) -> tuple[str, str]:
    """('wingtip', 'r') for 'wingtip_r'; (surface, '') for a surface without a side."""
    return (surface[:-2], surface[-1]) if surface.endswith(SIDES) else (surface, "")


def labels(deflections: dict[str, float]) -> dict[str, object]:
    """family, mode and delta_deg for a set of deflections (see the module docstring)."""
    moved = {s: d for s, d in deflections.items() if d != 0.0}
    angles = set(moved.values())
    delta = next(iter(angles)) if len(angles) == 1 else None
    split = [_split_side(s) for s in moved]
    families = {SURFACE_FAMILIES.get(base, base) for base, _ in split}
    sides = {side for _, side in split}
    if not moved:
        family = mode = "clean"
    elif len(families) > 1:
        family = mode = "combined"
    else:
        family = families.pop()
        if sides == {"r", "l"} and delta is not None:
            mode = "sym"
        elif sides == {"r"}:
            mode = "right"
        elif sides == {"l"}:
            mode = "left"
        else:
            mode = "mixed"
    return {"family": family, "mode": mode, "delta_deg": delta}


def _read_loads(post: Path, function_object: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterations, and the force and moment (model frame) at each, from one forces function object's output."""
    force_cols, force_rows = read_table(f"{function_object}/*/force.dat", post)
    moment_cols, moment_rows = read_table(f"{function_object}/*/moment.dat", post)
    if not force_rows or not moment_rows:
        raise FileNotFoundError(f"{post}: no {function_object} output")
    n = min(len(force_rows), len(moment_rows))
    return (np.array([r[0] for r in force_rows[:n]]), np.array(vectors(force_cols, force_rows)[:n]),
            np.array(vectors(moment_cols, moment_rows)[:n]))


def part_histories(solve_dir: Path, info: dict[str, Any]) -> pl.DataFrame:
    """Every part's coefficients, and every surface's hinge moment coefficient, at each iteration recorded by
    all the forces function objects. `info` is the solve's solve_case.json."""
    post = solve_dir / "postProcessing"
    hinges: dict[str, dict[str, Any]] = info["hinges"]
    raw = {"total": _read_loads(post, "forces_aircraft")} | {s: _read_loads(post, f"forces_{s}") for s in hinges}
    iterations: np.ndarray = reduce(np.intersect1d, [it for it, _, _ in raw.values()])
    loads: dict[str, Loads] = {}
    for part, (it, force, moment) in raw.items():
        keep = np.isin(it, iterations)
        loads[part] = (force[keep], moment[keep])

    settings, reference = info["settings"], info["reference"]
    q = 0.5 * settings["density"] * settings["speed"] ** 2
    reference_point = np.array(reference["point_m"])
    # The surfaces' moments are about their hinge point p1; move them to the reference point.
    about_reference: dict[str, Loads] = {}
    hinge_moments: dict[str, np.ndarray] = {}
    for surface, hinge in hinges.items():
        force, about_hinge = loads[surface]
        about_reference[surface] = (force, about_hinge + np.cross(np.array(hinge["point_m"]) - reference_point, force))
        hinge_moments[surface] = about_hinge @ np.array(hinge["direction"])
    total_force, total_moment = loads["total"]
    body: Loads = (total_force - sum((f for f, _ in about_reference.values()), np.zeros_like(total_force)),
                   total_moment - sum((m for _, m in about_reference.values()), np.zeros_like(total_moment)))

    columns: dict[str, np.ndarray] = {"iteration": iterations}
    for part, (force, moment) in {"total": loads["total"], "body": body, **about_reference}.items():
        columns |= {f"{part}__{name}": values for name, values in force_coefficients(info, force, moment).items()}
    columns |= {f"{s}__C_h": m / (q * reference["area_m2"] * reference["length_m"]) for s, m in hinge_moments.items()}
    return pl.DataFrame(columns)


def window_stats(history: pl.DataFrame, window: int) -> dict[str, float]:
    """mean, min, max and std of every history column over its last `window` rows."""
    last = history.tail(window)
    stats: dict[str, float] = {}
    for column in history.columns[1:]:  # after `iteration`
        values = last[column].to_numpy()
        stats |= {f"{column}_mean": float(values.mean()), f"{column}_min": float(values.min()),
                  f"{column}_max": float(values.max()), f"{column}_std": float(values.std())}
    return stats


def setpoint_row(summary: RunSummary, solve_dir: Path, info: dict[str, Any], history: pl.DataFrame,
                 window: int | None) -> dict[str, object]:
    """One row of the setpoints table (see the module docstring)."""
    result = SolveResult.model_validate_json((solve_dir / "results.json").read_text())
    settings, reference = info["settings"], info["reference"]
    deflections = {s: summary.deflections_deg.get(s, 0.0) for s in info["hinges"]}
    n = min(window or int(settings["convergence_window"]), history.height)
    iterations = history["iteration"]
    drag_dir, lift_dir, side_dir = wind_axes(info)

    row: dict[str, object] = {
        "case": summary.case,
        "condition": solve_dir.name,
        "description": summary.description,
        "mesh_job_id": summary.job_id,
        "solve_job_id": None if info.get("slurm_job_id") is None else str(info["slurm_job_id"]),
        **{f"{s}_deg": d for s, d in deflections.items()},
        **labels(deflections),
        "cells": summary.cells,
        "max_non_orthogonality": summary.max_non_orthogonality,
        "min_layer_percent": summary.min_layer_percent,
        "ref_area_m2": reference["area_m2"],
        "ref_chord_m": reference["length_m"],
        "ref_span_m": reference["span_m"],
        **{f"ref_{axis}_m": v for axis, v in zip("xyz", reference["point_m"])},
    }
    for surface, hinge in info["hinges"].items():
        row |= {f"{surface}_hinge_{axis}_m": v for axis, v in zip("xyz", hinge["point_m"])}
        row |= {f"{surface}_hinge_d{axis}": v for axis, v in zip("xyz", hinge["direction"])}
    row |= {
        "alpha_deg": info["condition"]["alpha_deg"],
        "beta_deg": info["condition"]["beta_deg"],
        "speed_m_s": settings["speed"],
        "density_kg_m3": settings["density"],
        "kinematic_viscosity_m2_s": settings["kinematic_viscosity"],
        "q_pa": 0.5 * settings["density"] * settings["speed"] ** 2,
        "reynolds": settings["speed"] * reference["length_m"] / settings["kinematic_viscosity"],
        "turbulence_intensity": settings["turbulence_intensity"],
        "viscosity_ratio": settings["viscosity_ratio"],
        **{f"drag_dir_{axis}": v for axis, v in zip("xyz", drag_dir)},
        **{f"lift_dir_{axis}": v for axis, v in zip("xyz", lift_dir)},
        **{f"side_dir_{axis}": v for axis, v in zip("xyz", side_dir)},
        "iterations": result.iterations,
        "max_iterations": result.max_iterations,
        "stopped_early": result.stopped_early,
        "converged": result.converged,
        "window_start": int(iterations[-n]),
        "window_end": int(iterations[-1]),
        **{f"gap_{name}": gap_from_running_mean(history[f"total__{name}"].to_numpy(),
                                                int(settings["convergence_window"])) for name in MONITORED},
        "warnings": "; ".join(result.warnings),
        **{f"residual_{field}": value for field, value in result.final_residuals.items()},
    }
    return row | window_stats(history, n)


def case_dirs(runs_dir: Path, patterns: list[str] | None) -> list[Path]:
    """Run folders (with case.toml and solve/) in `runs_dir` whose name matches one of `patterns` (all if None)."""
    return [d for d in sorted(runs_dir.iterdir())
            if (d / "case.toml").is_file() and (d / "solve").is_dir()
            and (not patterns or any(fnmatch.fnmatchcase(d.name, p) for p in patterns))]


def solve_dirs(run_dir: Path) -> list[Path]:
    """The run's finished solve folders (with results.json), skipping kept-aside failed attempts."""
    return sorted(p.parent for p in (run_dir / "solve").glob("*/results.json")
                  if not p.parent.name.endswith(".previous"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect solved set points into setpoints.parquet / .csv, and "
                                                 "save each one's coefficient histories.")
    parser.add_argument("runs_dir", type=Path, nargs="?", default=RUNS_DIR)
    parser.add_argument("--cases", nargs="+", metavar="PATTERN",
                        help='only cases whose name matches one of these glob patterns, e.g. "batch2_*"')
    parser.add_argument("--out", type=Path, help=f"output folder (default: <runs dir>/{DEFAULT_OUT})")
    parser.add_argument("--window", type=int,
                        help="iterations the statistics are taken over (default: each solve's convergence_window)")
    args = parser.parse_args(argv)

    out: Path = args.out or args.runs_dir / DEFAULT_OUT
    rows: list[dict[str, object]] = []
    failed: list[str] = []
    for run_dir in case_dirs(args.runs_dir, args.cases):
        summary = summarise(run_dir)
        if summary is None:
            continue
        found = solve_dirs(run_dir)
        for solve_dir in found:
            try:
                info = json.loads((solve_dir / "solve_case.json").read_text())
                history = part_histories(solve_dir, info)
                rows.append(setpoint_row(summary, solve_dir, info, history, args.window))
            except (OSError, KeyError, ValueError) as e:
                failed.append(f"{run_dir.name}/{solve_dir.name}: {e}")
                continue
            path = out / HISTORY_DIR / run_dir.name / f"{solve_dir.name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            history.write_parquet(path)
        print(f"{run_dir.name}: {len(found)} set points")

    if not rows:
        raise SystemExit(f"no solved set points in {args.runs_dir}")
    table = pl.from_dicts(rows, infer_schema_length=None).sort("case", "beta_deg", "alpha_deg")
    out.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out / f"{SETPOINTS}.parquet")
    table.write_csv(out / f"{SETPOINTS}.csv")
    print()
    print(f"{table.height} set points x {table.width} columns written to {out / SETPOINTS}.parquet and .csv; "
          f"histories in {out / HISTORY_DIR}/")
    print(f"{table['converged'].sum()} converged, {table.height - table['converged'].sum()} not converged")
    if failed:
        print(f"\n{len(failed)} set points skipped:")
        print("\n".join(f"  {f}" for f in failed))


if __name__ == "__main__":
    main()
