"""Check a finished mesh: read the checkMesh and snappyHexMesh logs, apply the acceptance limits, report.

The Slurm job runs this after meshing. It writes <case>/mesh_report.json, prints a summary, and exits with
status 1 if the mesh fails, so the job is marked as failed.

What is checked:
    - checkMesh's own verdict ("Mesh OK." or "Failed N mesh checks."), with the reasons it gives. Failures of
      the extra checks that only `-allGeometry` runs (concave cells, small determinant, ...) are warnings;
    - maximum non-orthogonality and skewness against the limits in the mesh settings' [acceptance] table;
    - prism-layer coverage per wall from snappyHexMesh's final layer table (low coverage is a warning, since
      layers often collapse at sharp trailing edges without making the mesh unusable).

Run by hand on a case directory:
    python -m src.cfd.mesh_report hpc/runs/mesh_<jobid>
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.cfd.geometry import StrictModel
from src.cfd.mesh import MeshAcceptance

_NUMBER = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

# Checks that only run with `checkMesh -allGeometry`. They are stricter than the solver needs; in particular
# every snappyHexMesh mesh has "concave" polyhedral cells where refinement levels meet. Reported as warnings.
_STRICT_CHECKS = ("concave cells", "concave faces", "determinant", "twist", "flatness")


class LayerCoverage(StrictModel):
    """One row of snappyHexMesh's final layer table."""
    patch: str
    faces: int
    mean_layers: float          # average number of layers actually added
    thickness_m: float          # overall layer thickness achieved
    thickness_percent: float    # achieved thickness as a share of the requested thickness


class MeshReport(StrictModel):
    passed: bool
    failures: list[str]
    warnings: list[str]
    cells: int | None
    points: int | None
    faces: int | None
    cell_types: dict[str, int]
    max_non_orthogonality: float | None
    mean_non_orthogonality: float | None
    max_skewness: float | None
    max_aspect_ratio: float | None
    min_volume: float | None
    checkmesh_failed_checks: int | None   # None if checkMesh didn't finish
    checkmesh_messages: list[str]         # the lines checkMesh flags with "***"
    layers: list[LayerCoverage]
    snappy_seconds: float | None


def _first(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def _first_int(pattern: str, text: str) -> int | None:
    value = _first(pattern, text)
    return int(value) if value is not None else None


def _read(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def parse_layer_table(snappy_log: str) -> list[LayerCoverage]:
    """Parse the final layer summary table in a snappyHexMesh log: what was actually achieved.

    OpenFOAM v2506 prints the requested and achieved layer counts (6 columns):
        patch       faces    layers        overall thickness
                             target   mesh     [m]       [%]
        -----       -----    -----    ----     ---       ---
        main_body   62353    5        3.88     0.00266   81.6
    Older versions print only the achieved count (5 columns: patch, faces, layers, [m], [%]).

    snappyHexMesh also prints a table of the *requested* layers earlier on (patch, faces, layers, near-wall
    and overall thickness in metres), so the final table is told apart by the "[%]" in its header.
    """
    rows: list[LayerCoverage] = []
    lines = snappy_log.splitlines()
    for start in (i for i, line in enumerate(lines) if line.strip().startswith("-----")):
        if "[%]" not in "".join(lines[max(0, start - 2):start]):
            continue
        table: list[LayerCoverage] = []
        for line in lines[start + 1:]:
            parts = line.split()
            if len(parts) not in (5, 6):
                break
            # The last three columns are always: achieved layers, thickness [m], thickness [%].
            try:
                table.append(LayerCoverage(
                    patch=parts[0], faces=int(parts[1]), mean_layers=float(parts[-3]),
                    thickness_m=float(parts[-2]), thickness_percent=float(parts[-1]),
                ))
            except ValueError:
                break
        if table:
            rows = table  # keep the last one, in case the log holds several runs
    return rows


def build_report(case_dir: Path, acceptance: MeshAcceptance) -> MeshReport:
    check = _read(case_dir / "log.checkMesh")
    snappy = _read(case_dir / "log.snappyHexMesh")
    failures: list[str] = []
    warnings: list[str] = []

    if not check:
        failures.append("log.checkMesh is missing or empty")
    if not snappy:
        failures.append("log.snappyHexMesh is missing or empty")

    # checkMesh's verdict.
    if re.search(r"^\s*Mesh OK\.", check, re.MULTILINE):
        failed_checks: int | None = 0
    else:
        failed_checks = _first_int(r"Failed (\d+) mesh checks", check)
        if failed_checks is None and check:
            failures.append("checkMesh did not finish (no 'Mesh OK.' or 'Failed N mesh checks.')")
    messages = [line.strip().lstrip("*") for line in check.splitlines() if line.strip().startswith("***")]
    for message in messages:
        if any(phrase in message.lower() for phrase in _STRICT_CHECKS):
            warnings.append(f"checkMesh (strict check): {message}")
        else:
            failures.append(f"checkMesh: {message}")
    if failed_checks and not messages:
        failures.append(f"checkMesh failed {failed_checks} check(s)")

    # Quality metrics.
    max_non_ortho = _first(rf"Mesh non-orthogonality Max:\s*{_NUMBER}", check)
    match = re.search(rf"Mesh non-orthogonality Max:\s*{_NUMBER}\s+average:\s*{_NUMBER}", check)
    mean_non_ortho = float(match.group(2)) if match else None
    max_skewness = _first(rf"Max skewness = {_NUMBER}", check)
    if max_non_ortho is not None and max_non_ortho > acceptance.max_non_orthogonality:
        failures.append(f"max non-orthogonality {max_non_ortho:g} > {acceptance.max_non_orthogonality:g}")
    if max_skewness is not None and max_skewness > acceptance.max_skewness:
        failures.append(f"max skewness {max_skewness:g} > {acceptance.max_skewness:g}")

    cell_types = {
        kind: count
        for kind in ("hexahedra", "prisms", "wedges", "pyramids", "tet wedges", "tetrahedra", "polyhedra")
        if (count := _first_int(rf"^\s*{kind}:\s+(\d+)", check)) is not None
    }

    # Prism layers.
    layers = parse_layer_table(snappy)
    if snappy and not layers:
        warnings.append("no layer table found in log.snappyHexMesh")
    for row in layers:
        if row.thickness_percent < acceptance.min_layer_thickness_percent:
            warnings.append(
                f"layers on {row.patch}: {row.thickness_percent:g}% of requested thickness "
                f"({row.mean_layers:g} layers on average)"
            )

    return MeshReport(
        passed=not failures,
        failures=failures,
        warnings=warnings,
        cells=_first_int(r"^\s*cells:\s+(\d+)", check),
        points=_first_int(r"^\s*points:\s+(\d+)", check),
        faces=_first_int(r"^\s*faces:\s+(\d+)", check),
        cell_types=cell_types,
        max_non_orthogonality=max_non_ortho,
        mean_non_orthogonality=mean_non_ortho,
        max_skewness=max_skewness,
        max_aspect_ratio=_first(rf"Max aspect ratio = {_NUMBER}", check),
        min_volume=_first(rf"Min volume = {_NUMBER}", check),
        checkmesh_failed_checks=failed_checks,
        checkmesh_messages=messages,
        layers=layers,
        snappy_seconds=_first(rf"Finished meshing in = {_NUMBER} s", snappy),
    )


def format_report(report: MeshReport) -> str:
    def show(value: float | int | None, fmt: str = "g") -> str:
        return "n/a" if value is None else format(value, fmt)

    lines = [
        f"Mesh {'PASSED' if report.passed else 'FAILED'}",
        f"  cells {show(report.cells, ',')}   points {show(report.points, ',')}   faces {show(report.faces, ',')}",
        "  cell types: " + ", ".join(f"{k} {v:,}" for k, v in report.cell_types.items()),
        f"  non-orthogonality max {show(report.max_non_orthogonality)} (mean {show(report.mean_non_orthogonality)})"
        f"   skewness max {show(report.max_skewness)}   aspect ratio max {show(report.max_aspect_ratio)}",
        f"  min cell volume {show(report.min_volume)} m^3   snappyHexMesh {show(report.snappy_seconds)} s",
    ]
    if report.layers:
        lines.append("  prism layers:   patch                    faces   layers  thickness")
        lines.extend(
            f"                  {r.patch:22s} {r.faces:8,d}   {r.mean_layers:5.2f}   {r.thickness_percent:5.1f}%"
            for r in report.layers
        )
    lines.extend(f"  FAIL: {f}" for f in report.failures)
    lines.extend(f"  warn: {w}" for w in report.warnings)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check a meshed case and write mesh_report.json.")
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args(argv)

    # Acceptance limits come from the settings the case was built with.
    info = json.loads((args.case_dir / "mesh_case.json").read_text())
    acceptance = MeshAcceptance.model_validate(info["settings"]["acceptance"])

    report = build_report(args.case_dir, acceptance)
    (args.case_dir / "mesh_report.json").write_text(report.model_dump_json(indent=2))
    print(format_report(report))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
