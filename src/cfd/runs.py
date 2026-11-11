"""List every CFD run and its state, built from the run folders themselves.

Each case has one folder, hpc/runs/<case name>/ (see src/cfd/submit.py), holding its case file, its mesh and
one folder per solved flow condition. This module reads those folders, so there is no separate index file to
keep in step with them (many jobs writing to one shared file at once could also corrupt it).

    python -m src.cfd.runs                 one line per case: mesh state and how many conditions are solved
    python -m src.cfd.runs --failed        only cases whose mesh failed or has not finished
"""

import argparse
import json
import tomllib
from pathlib import Path

from src.cfd.geometry import StrictModel

RUNS_DIR = Path("hpc/runs")


class RunSummary(StrictModel):
    case: str
    description: str
    deflections_deg: dict[str, float]
    mesh_status: str               # passed, FAILED, or "no report" (still running, or the job died early)
    job_id: str | None
    cells: int | None
    max_non_orthogonality: float | None
    min_layer_percent: float | None
    solve_folders: int    # flow conditions submitted (solve/<condition>/ folders)
    solved: int           # ... with results.json
    converged: int        # ... of which converged


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarise(run_dir: Path) -> RunSummary | None:
    """Summary of one run folder, or None if it is not one (no case.toml)."""
    try:
        case = tomllib.loads((run_dir / "case.toml").read_text())
    except (OSError, ValueError):
        return None
    mesh_dir = run_dir / "mesh"
    report = _read_json(mesh_dir / "mesh_report.json")
    info = _read_json(mesh_dir / "mesh_case.json") or {}

    # The newest Slurm output gives the job id even when the job died before writing anything else.
    outputs = sorted(mesh_dir.glob("slurm_*.out"), key=lambda p: p.stat().st_mtime)
    job_id = info.get("slurm_job_id") or (outputs[-1].stem.removeprefix("slurm_") if outputs else None)

    report = report or {}
    layers = report.get("layers")
    percents: list[float] = []
    for row in layers if isinstance(layers, list) else []:
        percent = _number(row.get("thickness_percent")) if isinstance(row, dict) else None
        if percent is not None:
            percents.append(percent)
    cells = _number(report.get("cells"))
    solve_folders = [d for d in (run_dir / "solve").glob("*") if d.is_dir() and not d.name.endswith(".previous")]
    results = [r for d in solve_folders if (r := _read_json(d / "results.json")) is not None]
    return RunSummary(
        case=run_dir.name,
        description=str(case.get("description", "")),
        deflections_deg={k: float(v) for k, v in case.get("deflections_deg", {}).items()},
        mesh_status="no report" if not report else ("passed" if report.get("passed") is True else "FAILED"),
        job_id=str(job_id) if job_id else None,
        cells=int(cells) if cells is not None else None,
        max_non_orthogonality=_number(report.get("max_non_orthogonality")),
        min_layer_percent=min(percents) if percents else None,
        solve_folders=len(solve_folders),
        solved=len(results),
        converged=sum(r.get("converged") is True for r in results),
    )


def summarise_all(runs_dir: Path = RUNS_DIR) -> list[RunSummary]:
    if not runs_dir.is_dir():
        return []
    summaries = (summarise(d) for d in sorted(runs_dir.iterdir()) if d.is_dir())
    return [s for s in summaries if s is not None]


def format_table(summaries: list[RunSummary]) -> str:
    def show(value: float | int | str | None, fmt: str = "") -> str:
        return "-" if value is None else format(value, fmt)

    header = (f"{'case':24s} {'mesh':10s} {'job':>9s} {'cells':>11s} {'nonOrtho':>8s} {'layers':>7s} "
              f"{'solved':>7s} {'conv.':>5s}  deflections")
    lines = [header, "-" * len(header)]
    for s in summaries:
        deflections = ", ".join(f"{k}={v:g}" for k, v in s.deflections_deg.items() if v) or "all zero"
        solved = f"{s.solved}/{s.solve_folders}" if s.solve_folders else "-"
        lines.append(
            f"{s.case:24s} {s.mesh_status:10s} {show(s.job_id):>9s} {show(s.cells, ','):>11s} "
            f"{show(s.max_non_orthogonality, '.1f'):>8s} {show(s.min_layer_percent, '.0f'):>6s}% "
            f"{solved:>7s} {s.converged if s.solve_folders else '-':>5}  {deflections}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="List CFD runs and their state, from the run folders.")
    parser.add_argument("runs_dir", type=Path, nargs="?", default=RUNS_DIR)
    parser.add_argument("--failed", action="store_true", help="only runs whose mesh failed or has no report")
    args = parser.parse_args(argv)

    summaries = summarise_all(args.runs_dir)
    if args.failed:
        summaries = [s for s in summaries if s.mesh_status != "passed"]
    print(format_table(summaries) if summaries else f"no runs in {args.runs_dir}")


if __name__ == "__main__":
    main()
