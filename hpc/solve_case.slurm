#!/bin/bash
# Solve one flow condition of a meshed case with steady RANS (simpleFoam, k-omega SST) on Iridis 6.
#
# Submitted by hpc/submit_cases.sh (see src/cfd/submit.py), one job per flow condition in the case's [flow]
# table, each waiting for the case's mesh job if that is still to run. Arguments: case file, alpha, beta (deg).
#
# Writes hpc/runs/<case name>/solve/<condition>/ (e.g. solve/a+08.0_b+00.0/):
#   slurm_<jobid>.out          the job's output (this log)
#   log.*                      one log per step
#   solve_case.json            every setting used, the flow condition and the job id
#   results.json               coefficients, hinge moments, residuals and the convergence verdict
#   postProcessing/            force, moment and residual histories; wall pressure, shear stress and y+ (VTK)
#   <last iteration>/          the final 3D fields, unless keep_fields = false in the solver settings
#
# Steps: Python writes the case (linked to the mesh) -> decomposePar -> potentialFoam (starting field)
#        -> simpleFoam (MPI, stops early once converged) -> reconstructPar -> report.

#SBATCH --job-name=solve
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%x_%j.out   # replaced by submit_cases.sh with <solve folder>/slurm_<jobid>.out
#SBATCH --error=slurm_%x_%j.out

set -eo pipefail
source "${SLURM_SUBMIT_DIR}/hpc/env.sh"   # Slurm runs a copy of this script, so find env.sh via the repo
load_openfoam
load_python

case_file="$(cd "${SLURM_SUBMIT_DIR}" && realpath "${1:?usage: solve_case.slurm <case file> <alpha> <beta>}")"
alpha="${2:?missing alpha}"
beta="${3:?missing beta}"
case_name="$(basename "${case_file}" .toml)"
mesh_dir="${repo_root}/hpc/runs/${case_name}/mesh"

print_environment
echo "Case file:     ${case_file}"
echo "Condition:     alpha ${alpha} deg, beta ${beta} deg"

cd "${repo_root}"
condition="$(python -c "from src.cfd.case import FlowCondition as F; print(F(alpha_deg=${alpha}, beta_deg=${beta}).name)")"
solve_dir="${repo_root}/hpc/runs/${case_name}/solve/${condition}"
echo "Solve dir:     ${solve_dir}"
if [ ! -f "${mesh_dir}/mesh_report.json" ] || ! grep -q '"passed": true' "${mesh_dir}/mesh_report.json"; then
    echo "The case's mesh (${mesh_dir}) has not passed; nothing to solve on."
    exit 1
fi

echo
echo "=== Case setup ==="
python -m src.cfd.solve "${case_file}" "${solve_dir}" --alpha "${alpha}" --beta "${beta}" \
    --mesh-dir "${mesh_dir}" --n-procs "${SLURM_NTASKS}"
cd "${solve_dir}"

echo
echo "=== Solving ==="
run_step decomposePar decomposePar -force
# A potential-flow solution is a better start than uniform flow. If it fails, simpleFoam starts from uniform.
echo "--- potentialFoam"
if run_parallel potentialFoam -writephi > log.potentialFoam 2>&1; then
    echo "    done"
else
    echo "    warning: potentialFoam failed (see log.potentialFoam); starting from uniform flow"
fi
run_step simpleFoam run_parallel simpleFoam
run_step reconstructPar reconstructPar -latestTime
rm -rf processor*

echo
echo "=== Results ==="
cd "${repo_root}"
python -m src.cfd.solve_report "${solve_dir}"
