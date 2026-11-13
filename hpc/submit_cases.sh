#!/bin/bash
# Submit CFD cases on the Iridis login node: a mesh job per case, then a solve job per flow condition.
# All the logic is in src/cfd/submit.py; this loads the Python environment and runs it from the repo root.
#
#     bash hpc/submit_cases.sh configs/cases/batch1/batch1_000.toml              one case
#     bash hpc/submit_cases.sh configs/cases/mesh_study                  every case file in a folder
#     bash hpc/submit_cases.sh configs/cases/mesh_study --mesh-only      mesh only
#     bash hpc/submit_cases.sh configs/cases/batch1/batch1_000.toml --dry-run    show what would be submitted
# Options: --mesh-only, --dry-run, --mesh-time HH:MM:SS, --solve-time HH:MM:SS,
#          --mesh-cores N, --solve-cores N (MPI ranks per job, up to 192 = one node; default 32),
#          --shared (run on any free cores, alongside other users' jobs: starts soonest, speed varies),
#          --whole-node (a node to itself per job). By default jobs share nodes only with your own jobs
#          (sbatch --exclusive=user): steady speed, but the first job on each node waits for an empty one.

set -euo pipefail
source "$(dirname "$(realpath "$0")")/env.sh"
load_python
target="$(realpath "${1:?usage: bash hpc/submit_cases.sh <case file | folder of case files> [options]}")"
shift
cd "${repo_root}"
python -m src.cfd.submit "$(realpath --relative-to="${repo_root}" "${target}")" "$@"
