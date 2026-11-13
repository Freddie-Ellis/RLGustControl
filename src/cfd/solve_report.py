"""Turn a finished solve into results: aerodynamic coefficients, hinge moments and a convergence verdict.

The Slurm job runs this after simpleFoam. It reads the function object output in <solve dir>/postProcessing/
and the settings in solve_case.json, writes <solve dir>/results.json and prints a summary.

Coefficients (q = 1/2 rho V^2, S reference area, c reference chord, b span):
    C_L, C_D, C_Y   lift, drag and side force: the total force along the lift direction, along the freestream,
                    and along lift x freestream (to starboard)
    C_l, C_m, C_n   rolling, pitching and yawing moments about the reference point, in the usual flight
                    mechanics sense (positive right wing down, nose up, nose right): C_l = -M_x / (q S b),
                    C_m = M_y / (q S c), C_n = -M_z / (q S b) in the model frame (x aft, y starboard, z up)
    C_h             each control surface's hinge moment, M_h / (q S c), where M_h is its moment about the
                    hinge axis, positive in the direction of positive deflection

Every value is the mean over the last convergence_window iterations, with its range (max - min) over that
window. A run is converged if, at its last iteration, each of C_L, C_D and C_m is within convergence_tolerance
of its running mean: the test OpenFOAM's runTimeControl uses to stop the run (see solve.py). An unconverged run
(e.g. unsteady separated flow at high alpha) is still reported, with its ranges showing how much it was
oscillating, so it can be judged or rerun.

Run by hand on a solve folder:
    python -m src.cfd.solve_report hpc/runs/<case name>/solve/a+08.0_b+00.0
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from src.cfd.geometry import StrictModel

MONITORED = ("C_L", "C_D", "C_m")  # coefficients that decide convergence


class WindowValue(StrictModel):
    """A quantity's mean over the final window of iterations, and its range (max - min) over it."""
    mean: float
    range: float


class SolveResult(StrictModel):
    alpha_deg: float
    beta_deg: float
    speed_m_s: float
    iterations: int
    max_iterations: int
    stopped_early: bool          # before max_iterations, by the stopping rule in solve.py
    converged: bool
    final_residuals: dict[str, float]
    coefficients: dict[str, WindowValue]
    force_n: dict[str, WindowValue]       # total force in the model frame: x, y, z
    moment_n_m: dict[str, WindowValue]    # about the reference point, model frame
    hinge_moment_n_m: dict[str, WindowValue]
    hinge_moment_coefficient: dict[str, WindowValue]
    warnings: list[str]


def read_table(path_pattern: str, folder: Path) -> tuple[list[str], list[list[float]]]:
    """Read an OpenFOAM function object table (e.g. forces_aircraft/*/force.dat) into column names and rows.

    A restarted run writes a new time folder, so every matching file is read in time order and a repeated
    iteration keeps its latest value. Non-numeric entries (solver names) become NaN.
    """
    files = sorted(folder.glob(path_pattern), key=lambda p: float(p.parent.name))
    columns: list[str] = []
    by_time: dict[float, list[float]] = {}
    for file in files:
        for line in file.read_text().splitlines():
            if line.startswith("#"):
                names = line.lstrip("#").split()
                if names and names[0] == "Time":
                    columns = names
                continue
            values: list[float] = []
            for item in line.replace("(", " ").replace(")", " ").split():
                try:
                    values.append(float(item))
                except ValueError:
                    values.append(math.nan)
            if values:
                by_time[values[0]] = values
    return columns, [by_time[t] for t in sorted(by_time)]


def _vectors(columns: list[str], rows: list[list[float]], prefix: str = "total") -> list[tuple[float, float, float]]:
    """The (x, y, z) columns named <prefix>_x/y/z, or, in older OpenFOAM formats without names, the sum of the
    pressure and viscous parts that follow the time column."""
    if f"{prefix}_x" in columns:
        i = columns.index(f"{prefix}_x")
        return [(r[i], r[i + 1], r[i + 2]) for r in rows]
    return [(r[1] + r[4], r[2] + r[5], r[3] + r[6]) for r in rows]


def _window(values: list[float]) -> WindowValue:
    return WindowValue(mean=sum(values) / len(values), range=max(values) - min(values))


def _dot(a: tuple[float, float, float] | list[float], b: tuple[float, float, float] | list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: tuple[float, ...] | list[float], b: tuple[float, ...] | list[float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


type Vec3 = tuple[float, float, float]


def _aircraft_loads(solve_dir: Path) -> tuple[list[float], list[Vec3], list[Vec3]]:
    """Iterations, and the aircraft's total force and moment (about the reference point) at each."""
    post = solve_dir / "postProcessing"
    force_cols, force_rows = read_table("forces_aircraft/*/force.dat", post)
    moment_cols, moment_rows = read_table("forces_aircraft/*/moment.dat", post)
    if not force_rows or not moment_rows:
        raise FileNotFoundError(f"{post}: no forces_aircraft output (did simpleFoam run?)")
    n = min(len(force_rows), len(moment_rows))
    return ([r[0] for r in force_rows[:n]], _vectors(force_cols, force_rows)[:n],
            _vectors(moment_cols, moment_rows)[:n])


def _coefficients(info: dict[str, Any], forces: list[Vec3], moments: list[Vec3]) -> dict[str, list[float]]:
    """C_L, C_D, C_Y, C_l, C_m, C_n for each force and moment (see the module docstring)."""
    settings, reference = info["settings"], info["reference"]
    q = 0.5 * settings["density"] * settings["speed"] ** 2
    area, chord, span = reference["area_m2"], reference["length_m"], reference["span_m"]
    drag_dir = [c / settings["speed"] for c in info["freestream_velocity_m_s"]]
    lift_dir = info["lift_direction"]
    side_dir = _cross(lift_dir, drag_dir)
    return {
        "C_L": [_dot(f, lift_dir) / (q * area) for f in forces],
        "C_D": [_dot(f, drag_dir) / (q * area) for f in forces],
        "C_Y": [_dot(f, side_dir) / (q * area) for f in forces],
        "C_l": [-m[0] / (q * area * span) for m in moments],
        "C_m": [m[1] / (q * area * chord) for m in moments],
        "C_n": [-m[2] / (q * area * span) for m in moments],
    }


def coefficient_history(solve_dir: Path) -> tuple[list[float], dict[str, list[float]]]:
    """Iterations, and the aircraft's coefficients at each, as averaged for results.json (for plotting)."""
    info = json.loads((solve_dir / "solve_case.json").read_text())
    iterations, forces, moments = _aircraft_loads(solve_dir)
    return iterations, _coefficients(info, forces, moments)


def gap_from_running_mean(values: list[float], window: int) -> float:
    """|last value - running mean|, the mean as OpenFOAM's runTimeControl `average` condition computes it
    (windowType approximate): the plain mean of the first `window` values, after which each new value has
    weight 1 / window."""
    mean = values[0]
    for n, value in enumerate(values[1:], start=2):
        mean += (value - mean) / min(n, window)
    return abs(values[-1] - mean)


def build_result(solve_dir: Path) -> SolveResult:
    info = json.loads((solve_dir / "solve_case.json").read_text())
    settings = info["settings"]
    reference = info["reference"]
    post = solve_dir / "postProcessing"
    warnings: list[str] = []

    steps, all_forces, all_moments = _aircraft_loads(solve_dir)
    iterations = int(steps[-1])
    window = min(int(settings["convergence_window"]), len(steps))
    if window < int(settings["convergence_window"]):
        warnings.append(f"only {window} iterations recorded, fewer than the convergence window")

    forces, moments = all_forces[-window:], all_moments[-window:]
    q = 0.5 * settings["density"] * settings["speed"] ** 2
    area, chord = reference["area_m2"], reference["length_m"]
    history = _coefficients(info, all_forces, all_moments)
    coefficients = {name: _window(values[-window:]) for name, values in history.items()}
    gaps = {name: gap_from_running_mean(history[name], int(settings["convergence_window"])) for name in MONITORED}
    # Like OpenFOAM, only judge once a full window has been averaged.
    full_window = len(steps) > int(settings["convergence_window"])
    unsettled = [name for name, gap in gaps.items() if not (full_window and gap < settings["convergence_tolerance"])]
    if unsettled:
        warnings.append("not converged: " + ", ".join(
            f"{n} is {gaps[n]:.2g} from its running mean (tolerance {settings['convergence_tolerance']:g})"
            for n in unsettled))

    hinge_moment: dict[str, WindowValue] = {}
    hinge_coefficient: dict[str, WindowValue] = {}
    for name, hinge in info["hinges"].items():
        cols, rows = read_table(f"forces_{name}/*/moment.dat", post)
        if not rows:
            warnings.append(f"no forces_{name} output")
            continue
        about_hinge = [_dot(m, hinge["direction"]) for m in _vectors(cols, rows)[-window:]]
        hinge_moment[name] = _window(about_hinge)
        hinge_coefficient[name] = _window([m / (q * area * chord) for m in about_hinge])

    residual_cols, residual_rows = read_table("solverInfo/*/solverInfo.dat", post)
    final_residuals = {
        column.removesuffix("_initial"): residual_rows[-1][i]
        for i, column in enumerate(residual_cols)
        if column.endswith("_initial") and residual_rows and i < len(residual_rows[-1])
    }

    return SolveResult(
        alpha_deg=info["condition"]["alpha_deg"],
        beta_deg=info["condition"]["beta_deg"],
        speed_m_s=settings["speed"],
        iterations=iterations,
        max_iterations=settings["max_iterations"],
        stopped_early=iterations < settings["max_iterations"],
        converged=not unsettled,
        final_residuals=final_residuals,
        coefficients=coefficients,
        force_n={axis: _window([f[i] for f in forces]) for i, axis in enumerate("xyz")},
        moment_n_m={axis: _window([m[i] for m in moments]) for i, axis in enumerate("xyz")},
        hinge_moment_n_m=hinge_moment,
        hinge_moment_coefficient=hinge_coefficient,
        warnings=warnings,
    )


def format_result(result: SolveResult) -> str:
    status = "CONVERGED" if result.converged else "NOT CONVERGED"
    how = "stopped early" if result.stopped_early else f"ran to the {result.max_iterations}-iteration limit"
    lines = [
        f"{status}: alpha {result.alpha_deg:g} deg, beta {result.beta_deg:g} deg, {result.speed_m_s:g} m/s, "
        f"{result.iterations} iterations ({how})",
        "  " + "   ".join(f"{name} {v.mean:.4f} (+-{v.range / 2:.1e})" for name, v in result.coefficients.items()),
        "  final residuals: " + ", ".join(f"{k} {v:.1e}" for k, v in result.final_residuals.items()),
    ]
    if result.hinge_moment_coefficient:
        lines.append("  hinge moment coefficients: " + ", ".join(
            f"{name} {v.mean:.4f}" for name, v in result.hinge_moment_coefficient.items()))
    lines.extend(f"  warn: {w}" for w in result.warnings)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise a finished solve and write results.json.")
    parser.add_argument("solve_dir", type=Path)
    args = parser.parse_args(argv)

    try:
        result = build_result(args.solve_dir)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    (args.solve_dir / "results.json").write_text(result.model_dump_json(indent=2))
    print(format_result(result))

    # The final 3D fields are large; drop them if the settings say so (the results above are kept).
    info = json.loads((args.solve_dir / "solve_case.json").read_text())
    if not info["settings"]["keep_fields"]:
        for folder in args.solve_dir.iterdir():
            if folder.is_dir() and folder.name.replace(".", "", 1).isdigit() and float(folder.name) > 0:
                shutil.rmtree(folder)


if __name__ == "__main__":
    main()
