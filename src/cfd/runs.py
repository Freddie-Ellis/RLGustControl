"""List every CFD run and the state of each of its flow conditions, built from the run folders themselves.

Each case has one folder, hpc/runs/<case name>/ (see src/cfd/submit.py), holding its case file, its mesh and
one folder per solved flow condition. This module reads those folders (and squeue, for jobs still waiting or
running), so there is no separate index file to keep in step with them (many jobs writing to one shared file at
once could also corrupt it).

    python -m src.cfd.runs                  one line per case, then a count of flow conditions by status
    python -m src.cfd.runs --failed         only cases with a failed or unfinished mesh or a failed condition,
                                            then every failed or never-started condition and why
    python -m src.cfd.runs --csv-dir PATH   write the CSVs somewhere other than the runs folder

Every call rewrites two CSVs in the runs folder, covering every case whatever --failed shows:
    runs.csv        one row per case: mesh state, how many conditions are solved, converged and failed, and one
                    <surface>_deg column per deflectable surface
    conditions.csv  one row per flow condition (each in the case's [flow] table, plus any other solve folders):
                    status, why it failed, job id, iterations, the coefficients and any warnings

Flow condition statuses:
    converged       results.json written and converged
    not converged   results.json written but not converged (still has coefficients; see solve_report.py)
    queued/running  its job is in squeue
    FAILED          a job ran but wrote no results.json; `reason` says why when its Slurm output shows it
    not started     submitted, but never ran and is no longer queued (e.g. cancelled because its mesh failed)
    not submitted   in the case's [flow] table, but never submitted
"""

import argparse
import csv
import getpass
import json
import re
import subprocess
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from src.cfd.case import CaseSpec
from src.cfd.geometry import StrictModel

RUNS_DIR = Path("hpc/runs")
RUNS_CSV = "runs.csv"
CONDITIONS_CSV = "conditions.csv"
STATUSES: tuple[str, ...] = ("converged", "not converged", "running", "queued", "FAILED", "not started",
                             "not submitted")  # the order they are counted in
NEEDS_RESUBMITTING: tuple[str, ...] = ("FAILED", "not started")
CONDITION_NAME = re.compile(r"a([+-]\d+(?:\.\d+)?)_b([+-]\d+(?:\.\d+)?)")  # e.g. a+08.0_b-05.0
SQUEUE_STATES: dict[str, str] = {"PENDING": "queued", "RUNNING": "running"}


class ConditionSummary(StrictModel):
    case: str
    condition: str
    alpha_deg: float | None
    beta_deg: float | None
    status: str                    # one of STATUSES, or another squeue state in lower case
    reason: str                    # why it failed or never started, where known
    job_id: str | None
    iterations: int | None
    max_iterations: int | None
    coefficients: dict[str, float]  # mean over the final window (results.json)
    warnings: str


class RunSummary(StrictModel):
    case: str
    description: str
    deflections_deg: dict[str, float]
    mesh_status: str               # passed, FAILED, queued, running, or "no report" (the job died early)
    job_id: str | None
    cells: int | None
    max_non_orthogonality: float | None
    min_layer_percent: float | None
    conditions_total: int          # in the case's [flow] table, plus any other solve folders
    solved: int                    # ... with results.json
    converged: int                 # ... of which converged
    failed: int                    # ... FAILED or never started, so needing resubmitting
    conditions: list[ConditionSummary]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _slurm_outputs(folder: Path) -> list[Path]:
    """The folder's Slurm outputs, oldest first."""
    return sorted(folder.glob("slurm_*.out"), key=lambda p: p.stat().st_mtime)


def _job_id(folder: Path, info_file: str) -> str | None:
    """The job id from the folder's settings file, or else from its newest Slurm output (written even when the
    job died before anything else)."""
    info = _read_json(folder / info_file) or {}
    if job_id := info.get("slurm_job_id"):
        return str(job_id)
    outputs = _slurm_outputs(folder)
    return outputs[-1].stem.removeprefix("slurm_") if outputs else None


def job_states() -> dict[str, tuple[str, str]]:
    """Job name -> (job id, state) for the user's jobs in squeue, e.g. ("123456", "running"). Empty where there is
    no squeue (off the cluster)."""
    try:
        output = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%j %i %T"],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    jobs: dict[str, tuple[str, str]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 3:
            jobs[parts[0]] = (parts[1], SQUEUE_STATES.get(parts[2], parts[2].lower()))
    return jobs


def failure_reason(log: str) -> str:
    """Why a solve job stopped without results, from its Slurm output (hpc/solve_case.slurm)."""
    lower = log.lower()
    if "due to time limit" in lower:
        return "time limit"
    if "out of memory" in lower or "oom-kill" in lower or "oom_kill" in lower:
        return "out of memory"
    if "cancelled" in lower:
        return "cancelled"
    if "has not passed; nothing to solve on" in log:
        return "mesh not passed"
    if step := re.search(r"^FAILED: (\S+)", log, re.MULTILINE):
        return f"{step.group(1)} failed"
    if "Traceback" in log:
        return "Python error"
    return "stopped without results"


def condition_names(run_dir: Path, case: dict[str, Any]) -> list[str]:
    """The case's [flow] conditions, then any other solve folders (e.g. from an earlier version of the case)."""
    try:
        planned = [c.name for c in CaseSpec.model_validate(case).conditions()]
    except ValueError:
        planned = []
    folders = sorted(d.name for d in (run_dir / "solve").glob("*")
                     if d.is_dir() and not d.name.endswith(".previous"))
    return list(dict.fromkeys(planned + folders))


def summarise_condition(run_dir: Path, name: str, mesh_status: str,
                        jobs: dict[str, tuple[str, str]]) -> ConditionSummary:
    folder = run_dir / "solve" / name
    job = jobs.get(f"solve_{run_dir.name}_{name}")
    result = _read_json(folder / "results.json")
    reason = ""
    if result is not None:
        status = "converged" if result.get("converged") is True else "not converged"
    elif job:
        status = job[1]
    elif not folder.is_dir():
        status = "not submitted"
    elif not (outputs := _slurm_outputs(folder)):
        status = "not started"
        reason = {"FAILED": "mesh FAILED", "no report": "mesh did not finish"}.get(mesh_status, "")
    else:
        status = "FAILED"
        try:
            reason = failure_reason(outputs[-1].read_text(errors="replace"))
        except OSError:
            reason = "Slurm output unreadable"

    result = result or {}
    coefficients = result.get("coefficients")
    means = {key: mean for key, value in coefficients.items()
             if isinstance(value, dict) and (mean := _number(value.get("mean"))) is not None
             } if isinstance(coefficients, dict) else {}
    warnings = result.get("warnings")
    angles = CONDITION_NAME.fullmatch(name)
    return ConditionSummary(
        case=run_dir.name,
        condition=name,
        alpha_deg=float(angles.group(1)) if angles else None,
        beta_deg=float(angles.group(2)) if angles else None,
        status=status,
        reason=reason,
        job_id=job[0] if job and not result else _job_id(folder, "solve_case.json"),
        iterations=_integer(result.get("iterations")),
        max_iterations=_integer(result.get("max_iterations")),
        coefficients=means,
        warnings="; ".join(str(w) for w in warnings) if isinstance(warnings, list) else "",
    )


def summarise(run_dir: Path, jobs: dict[str, tuple[str, str]] | None = None) -> RunSummary | None:
    """Summary of one run folder, or None if it is not one (no case.toml). `jobs` is job_states()."""
    jobs = jobs or {}
    try:
        case = tomllib.loads((run_dir / "case.toml").read_text())
    except (OSError, ValueError):
        return None
    mesh_dir = run_dir / "mesh"
    report = _read_json(mesh_dir / "mesh_report.json") or {}
    mesh_job = jobs.get(f"mesh_{run_dir.name}")
    if report:
        mesh_status = "passed" if report.get("passed") is True else "FAILED"
    else:
        mesh_status = mesh_job[1] if mesh_job else "no report"

    layers = report.get("layers")
    percents: list[float] = []
    for row in layers if isinstance(layers, list) else []:
        percent = _number(row.get("thickness_percent")) if isinstance(row, dict) else None
        if percent is not None:
            percents.append(percent)
    conditions = [summarise_condition(run_dir, name, mesh_status, jobs) for name in condition_names(run_dir, case)]
    return RunSummary(
        case=run_dir.name,
        description=str(case.get("description", "")),
        deflections_deg={k: float(v) for k, v in case.get("deflections_deg", {}).items()},
        mesh_status=mesh_status,
        job_id=mesh_job[0] if mesh_job and not report else _job_id(mesh_dir, "mesh_case.json"),
        cells=_integer(report.get("cells")),
        max_non_orthogonality=_number(report.get("max_non_orthogonality")),
        min_layer_percent=min(percents) if percents else None,
        conditions_total=len(conditions),
        solved=sum(c.status in ("converged", "not converged") for c in conditions),
        converged=sum(c.status == "converged" for c in conditions),
        failed=sum(c.status in NEEDS_RESUBMITTING for c in conditions),
        conditions=conditions,
    )


def summarise_all(runs_dir: Path = RUNS_DIR) -> list[RunSummary]:
    if not runs_dir.is_dir():
        return []
    jobs = job_states()
    summaries = (summarise(d, jobs) for d in sorted(runs_dir.iterdir()) if d.is_dir())
    return [s for s in summaries if s is not None]


def write_csv(columns: list[str], rows: list[dict[str, object]], path: Path) -> None:
    """Rewrite `path`. Written to a temporary file and then renamed over the old one, so anything reading it
    (e.g. a download) never sees a half-written file. Values missing from a row are left blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_runs_csv(summaries: list[RunSummary], path: Path) -> None:
    surfaces = list(dict.fromkeys(name for s in summaries for name in s.deflections_deg))
    columns = ["case", "mesh_status", "job_id", "cells", "max_non_orthogonality", "min_layer_percent",
               "conditions_total", "solved", "converged", "failed", *(f"{name}_deg" for name in surfaces),
               "description"]
    rows: list[dict[str, object]] = []
    for s in summaries:
        row: dict[str, object] = s.model_dump(exclude={"deflections_deg", "conditions"})
        # Surfaces a case file doesn't list stay at 0.
        row |= {f"{name}_deg": s.deflections_deg.get(name, 0.0) for name in surfaces}
        rows.append(row)
    write_csv(columns, rows, path)


def write_conditions_csv(summaries: list[RunSummary], path: Path) -> None:
    conditions = [c for s in summaries for c in s.conditions]
    coefficients = list(dict.fromkeys(name for c in conditions for name in c.coefficients))
    columns = ["case", "condition", "alpha_deg", "beta_deg", "status", "reason", "job_id", "iterations",
               "max_iterations", *coefficients, "warnings"]
    rows = [c.model_dump(exclude={"coefficients"}) | c.coefficients for c in conditions]
    write_csv(columns, rows, path)


def format_table(summaries: list[RunSummary]) -> str:
    def show(value: float | int | str | None, fmt: str = "") -> str:
        return "-" if value is None else format(value, fmt)

    header = (f"{'case':24s} {'mesh':10s} {'job':>9s} {'cells':>11s} {'nonOrtho':>8s} {'layers':>7s} "
              f"{'solved':>7s} {'conv.':>5s} {'failed':>6s}  deflections")
    lines = [header, "-" * len(header)]
    for s in summaries:
        deflections = ", ".join(f"{k}={v:g}" for k, v in s.deflections_deg.items() if v) or "all zero"
        solved = f"{s.solved}/{s.conditions_total}" if s.conditions_total else "-"
        lines.append(
            f"{s.case:24s} {s.mesh_status:10s} {show(s.job_id):>9s} {show(s.cells, ','):>11s} "
            f"{show(s.max_non_orthogonality, '.1f'):>8s} {show(s.min_layer_percent, '.0f'):>6s}% "
            f"{solved:>7s} {s.converged if s.conditions_total else '-':>5} "
            f"{s.failed if s.conditions_total else '-':>6}  {deflections}"
        )
    return "\n".join(lines)


def format_status_counts(conditions: list[ConditionSummary]) -> str:
    counts = Counter(c.status for c in conditions)
    order = [*STATUSES, *sorted(set(counts) - set(STATUSES))]
    return f"Flow conditions ({len(conditions)}): " + ", ".join(f"{counts[s]} {s}" for s in order if counts[s])


def format_failed(conditions: list[ConditionSummary]) -> str:
    failed = [c for c in conditions if c.status in NEEDS_RESUBMITTING]
    if not failed:
        return "No failed or never-started flow conditions."
    lines = [f"{'case':24s} {'condition':14s} {'status':12s} {'job':>9s}  reason"]
    lines += [f"{c.case:24s} {c.condition:14s} {c.status:12s} {c.job_id or '-':>9s}  {c.reason}" for c in failed]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="List CFD runs and their flow conditions, from the run folders.")
    parser.add_argument("runs_dir", type=Path, nargs="?", default=RUNS_DIR)
    parser.add_argument("--failed", action="store_true",
                        help="only cases with a failed or unfinished mesh or a failed condition, then each failed "
                             "or never-started condition and why")
    parser.add_argument("--csv-dir", type=Path,
                        help=f"folder for {RUNS_CSV} and {CONDITIONS_CSV} (default: the runs folder)")
    args = parser.parse_args(argv)

    summaries = summarise_all(args.runs_dir)
    if not summaries:
        print(f"no runs in {args.runs_dir}")
        return
    csv_dir: Path = args.csv_dir or args.runs_dir
    write_runs_csv(summaries, csv_dir / RUNS_CSV)
    write_conditions_csv(summaries, csv_dir / CONDITIONS_CSV)
    print(f"{len(summaries)} cases written to {csv_dir / RUNS_CSV}, their flow conditions to "
          f"{csv_dir / CONDITIONS_CSV}")
    print()

    shown = [s for s in summaries if s.mesh_status != "passed" or s.failed] if args.failed else summaries
    print(format_table(shown) if shown else "No cases with a failed mesh or failed conditions.")
    conditions = [c for s in summaries for c in s.conditions]
    print()
    print(format_status_counts(conditions))
    if args.failed:
        print()
        print(format_failed(conditions))


if __name__ == "__main__":
    main()
