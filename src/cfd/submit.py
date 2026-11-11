"""Submit CFD cases to Slurm: one mesh job per case, then one solve job per flow condition.

Run on the Iridis login node through hpc/submit_cases.sh, which loads the Python environment first:
    bash hpc/submit_cases.sh configs/cases/batch1/batch1_000.toml              one case
    bash hpc/submit_cases.sh configs/cases/mesh_study                  every case file in a folder
    bash hpc/submit_cases.sh configs/cases/mesh_study --mesh-only      mesh, don't solve
    bash hpc/submit_cases.sh configs/cases/batch1/batch1_000.toml --dry-run    show what would be submitted

Everything for a case goes in hpc/runs/<case name>/ (the case file's name):
    case.toml          the case file as last submitted
    mesh/              the mesh job (hpc/mesh_case.slurm)
    solve/<condition>/ one solve job per flow condition (hpc/solve_case.slurm), e.g. solve/a+08.0_b+00.0/

For each case:
    mesh   Skipped if a job for it is queued or running, or if it already has a mesh that passed. A case whose
           mesh inputs (geometry, mesh settings, deflections, overrides) differ from the passed mesh's is
           refused, so a finished mesh is never silently replaced; flow conditions can be added freely. A failed
           or unfinished attempt is kept as mesh.previous.
    solve  One job per flow condition, skipped if it already has results.json or a job queued or running. If
           the mesh job is still to run, the solve jobs wait for it, and are cancelled if it fails. A failed or
           unfinished attempt is kept as <condition>.previous.
Jobs are named mesh_<case> and solve_<case>_<condition>, so squeue shows what each one is.
"""

import argparse
import getpass
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

from src.cfd.case import CaseSpec
from src.cfd.mesh import MeshSettings
from src.cfd.solve import check_condition

RUNS_DIR = Path("hpc/runs")


def queued_jobs() -> dict[str, str]:
    """Job name -> job id for the user's jobs that are queued or running."""
    try:
        output = subprocess.run(
            ["squeue", "-h", "-u", getpass.getuser(), "-o", "%j %i"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    return dict(line.split() for line in output.splitlines() if len(line.split()) == 2)


def mesh_passed(mesh_dir: Path) -> bool:
    try:
        return json.loads((mesh_dir / "mesh_report.json").read_text()).get("passed") is True
    except (OSError, ValueError):
        return False


def stored_mesh_inputs(run_dir: Path) -> dict[str, object] | None:
    """The mesh inputs of the case file a run folder was last submitted from, if it can be read."""
    try:
        stored = CaseSpec.model_validate(tomllib.loads((run_dir / "case.toml").read_text()))
    except (OSError, ValueError):
        return None
    return stored.mesh_inputs()


def keep_previous(folder: Path) -> None:
    """Move a failed or unfinished attempt aside as <folder>.previous, replacing an older one."""
    if folder.exists():
        previous = folder.with_name(folder.name + ".previous")
        if previous.exists():
            shutil.rmtree(previous)
        folder.rename(previous)


class Submitter:
    def __init__(self, dry_run: bool, mesh_args: list[str], solve_args: list[str]) -> None:
        self.dry_run = dry_run
        self.mesh_args = mesh_args
        self.solve_args = solve_args
        self.queued = queued_jobs()

    def sbatch(self, name: str, output_dir: Path, script: str, args: list[str], extra: list[str]) -> str:
        """Submit a job (from the repo root, which the scripts rely on) and return its id."""
        command = ["sbatch", "--parsable", f"--job-name={name}",
                   f"--output={output_dir}/slurm_%j.out", f"--error={output_dir}/slurm_%j.out",
                   *extra, script, *args]
        if self.dry_run:
            print("    would run: " + " ".join(command))
            return "DRYRUN"
        output_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().split(";")[0]

    def submit_case(self, case_file: Path, mesh_only: bool) -> None:
        name = case_file.stem
        case = CaseSpec.from_toml(case_file)
        mesh_settings = MeshSettings.from_toml(case.mesh, case.mesh_overrides)
        conditions = [] if mesh_only else case.conditions()
        for condition in conditions:
            check_condition(condition, mesh_settings)
        run_dir = RUNS_DIR / name
        mesh_dir = run_dir / "mesh"

        # Mesh.
        mesh_job: str | None = None
        if f"mesh_{name}" in self.queued:
            mesh_job = self.queued[f"mesh_{name}"]
            print(f"{name}: mesh job {mesh_job} already queued or running")
        elif mesh_passed(mesh_dir):
            if stored_mesh_inputs(run_dir) != case.mesh_inputs():
                print(f"{name}: REFUSED, {run_dir} holds a finished mesh made with different mesh inputs "
                      "(geometry, mesh settings, deflections or overrides).")
                print(f"    Rename the case file, or delete {mesh_dir} to remesh.")
                return
            print(f"{name}: mesh already passed")
        else:
            if not self.dry_run:
                keep_previous(mesh_dir)
            mesh_job = self.sbatch(f"mesh_{name}", mesh_dir.resolve(), "hpc/mesh_case.slurm",
                                   [str(case_file)], self.mesh_args)
            print(f"{name}: submitted mesh job {mesh_job}")
        if not self.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(case_file, run_dir / "case.toml")

        # Solves, one per flow condition.
        for condition in conditions:
            solve_dir = run_dir / "solve" / condition.name
            job_name = f"solve_{name}_{condition.name}"
            if (solve_dir / "results.json").is_file():
                print(f"{name} {condition.name}: already solved")
                continue
            if job_name in self.queued:
                print(f"{name} {condition.name}: job {self.queued[job_name]} already queued or running")
                continue
            if not self.dry_run:
                keep_previous(solve_dir)
            wait = [f"--dependency=afterok:{mesh_job}", "--kill-on-invalid-dep=yes"] if mesh_job else []
            job = self.sbatch(job_name, solve_dir.resolve(), "hpc/solve_case.slurm",
                              [str(case_file), f"{condition.alpha_deg:g}", f"{condition.beta_deg:g}"],
                              wait + self.solve_args)
            print(f"{name} {condition.name}: submitted solve job {job}" + (f" (after mesh job {mesh_job})"
                                                                          if mesh_job else ""))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Submit CFD cases: mesh jobs, then a solve job per flow condition.")
    parser.add_argument("target", type=Path, help="a case file, or a folder of case files")
    parser.add_argument("--mesh-only", action="store_true", help="submit mesh jobs only")
    parser.add_argument("--dry-run", action="store_true", help="print what would be submitted, submit nothing")
    parser.add_argument("--mesh-time", help="time limit for mesh jobs, e.g. 04:00:00")
    parser.add_argument("--solve-time", help="time limit for solve jobs, e.g. 08:00:00")
    args = parser.parse_args(argv)

    case_files = sorted(args.target.glob("*.toml")) if args.target.is_dir() else [args.target]
    if not case_files:
        raise SystemExit(f"no .toml case files in {args.target}")
    names = [f.stem for f in case_files]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SystemExit(f"case names must be unique: {duplicates}")

    submitter = Submitter(
        dry_run=args.dry_run,
        mesh_args=[f"--time={args.mesh_time}"] if args.mesh_time else [],
        solve_args=[f"--time={args.solve_time}"] if args.solve_time else [],
    )
    for case_file in case_files:
        submitter.submit_case(case_file, args.mesh_only)


if __name__ == "__main__":
    main()
