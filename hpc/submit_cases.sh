#!/bin/bash
# Submit every case file in a folder as one Slurm array job (one task per case file, sorted by name).
#
#     bash hpc/submit_cases.sh configs/cases/sweep_01
# Extra arguments go to sbatch, e.g. more time per case:
#     bash hpc/submit_cases.sh configs/cases/sweep_01 --time=04:00:00

set -euo pipefail

folder="${1:?usage: bash hpc/submit_cases.sh <folder of case files> [sbatch options]}"
shift
n_cases="$(find "${folder}" -maxdepth 1 -name '*.toml' | wc -l)"
if [ "${n_cases}" -eq 0 ]; then
    echo "No .toml case files in ${folder}" >&2
    exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
echo "Submitting ${n_cases} case(s) from ${folder}"
sbatch --array="0-$(( n_cases - 1 ))" "$@" "${repo_root}/hpc/mesh_case.slurm" "${folder}"
