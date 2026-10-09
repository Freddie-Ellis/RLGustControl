#!/bin/bash
# Minimal SLURM smoke test: checks that jobs can be submitted, run across
# several tasks, see the module system and write output back.
#
# Submit from the login node:   sbatch hpc/test_job.slurm
# Check status:                 squeue -u $USER
# Output lands in:              test_job_<jobid>.out

#SBATCH --job-name=rlgust_test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.out
## Uncomment and set if the default partition is rejected (list them with `sinfo -s`):
##SBATCH --partition=batch

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Job name:      ${SLURM_JOB_NAME}"
echo "Partition:     ${SLURM_JOB_PARTITION:-unknown}"
echo "Node list:     ${SLURM_JOB_NODELIST}"
echo "Tasks:         ${SLURM_NTASKS}"
echo "Submit dir:    ${SLURM_SUBMIT_DIR}"
echo "Start time:    $(date)"

echo
echo "=== Node hardware ==="
hostname
lscpu | grep -E '^(Model name|CPU\(s\)|Socket|Core|Thread)' || true
free -h | head -2

echo
echo "=== Parallel tasks (srun) ==="
srun --ntasks="${SLURM_NTASKS}" bash -c 'echo "task ${SLURM_PROCID} of ${SLURM_NTASKS} on $(hostname)"'

echo
echo "=== Software available ==="
command -v module >/dev/null 2>&1 && module -t avail 2>&1 | grep -iE 'openfoam|python|gcc|openmpi|xfoil' | head -30 || echo "module command not found"
command -v python3 >/dev/null 2>&1 && python3 --version || echo "python3 not on PATH"

echo
echo "=== Write test ==="
out_file="${SLURM_SUBMIT_DIR}/test_job_${SLURM_JOB_ID}.touch"
echo "written by job ${SLURM_JOB_ID} on $(hostname)" > "${out_file}"
cat "${out_file}" && rm "${out_file}"

echo
echo "End time:      $(date)"
echo "=== Test job completed OK ==="
