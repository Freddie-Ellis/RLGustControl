#!/bin/bash
# Mesh one case on Iridis 6, check the mesh, and write small extracts to download.
#
# The only argument is a case file (see configs/cases/), which names the geometry, the mesh settings and the
# control surface deflections:
#     sbatch hpc/mesh_case.slurm configs/cases/uav_wingtip_10.toml
# Many cases at once: pass a folder of case files as an array job, one task per file (sorted by name).
# hpc/submit_cases.sh works out the array size for you:
#     bash hpc/submit_cases.sh configs/cases/sweep_01
# One-off setup first:   bash hpc/setup_python.sh   (on the login node)
#
# Output log:     mesh_<jobid>.out (in the submit directory)
# Case directory: <repo>/hpc/runs/<case name>_<jobid>/
#   mesh_report.json         cell counts, quality metrics, layer coverage, pass/fail
#   case.toml                copy of the case file
#   log.*                    one log per step
#   postProcessing/meshSlices/   cutting planes and wall surfaces (VTK) with quality fields: download these
#   postProcessing/checkMesh/    VTK surfaces of any faces checkMesh flags as bad
#
# Steps: Python writes the case (deflected aircraft STL + dictionaries) -> surfaceFeatureExtract -> blockMesh
#        -> decomposePar -> snappyHexMesh (MPI) -> reconstructParMesh -> checkMesh -> slices -> report.
# The job fails (non-zero exit) if any step fails or the mesh does not meet the acceptance limits.
# Resources below can be overridden at submission, e.g. sbatch --ntasks=64 --time=04:00:00 ...

#SBATCH --job-name=mesh
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=mesh_%j.out
#SBATCH --error=mesh_%j.out

OF_VERSION="v2506"

module purge
# OpenFOAM's bashrc references unset variables, so enable strict mode after sourcing it.
source "/iridisfs/i6software/openfoam/${OF_VERSION}/OpenFOAM-${OF_VERSION}/etc/bashrc"
set -eo pipefail

# Resolve the repo root so the job works whichever directory it was submitted from.
repo_root="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR}")"

# Pick the case file: the argument itself, or this array task's file from a folder.
target="${1:?usage: sbatch hpc/mesh_case.slurm <case file | folder of case files>}"
target="$(cd "${SLURM_SUBMIT_DIR}" && realpath "${target}")"
if [ -d "${target}" ]; then
    mapfile -t case_files < <(find "${target}" -maxdepth 1 -name '*.toml' | sort)
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "A folder of case files must be submitted as an array job: bash hpc/submit_cases.sh ${1}"
        exit 1
    fi
    case_file="${case_files[${SLURM_ARRAY_TASK_ID}]}"
else
    case_file="${target}"
fi
case_name="$(basename "${case_file}" .toml)"
case_dir="${repo_root}/hpc/runs/${case_name}_${SLURM_JOB_ID}"

# Python environment created by hpc/setup_python.sh, with the same Python module it was built with.
venv="${repo_root}/.venv-hpc"
if [ ! -f "${venv}/python_module" ]; then
    echo "No Python environment at ${venv}; run 'bash hpc/setup_python.sh' on the login node first."
    exit 1
fi
module load "$(cat "${venv}/python_module")"
source "${venv}/bin/activate"

echo "=== Environment ==="
echo "Job ID:        ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST} (${SLURM_NTASKS} tasks)"
echo "OpenFOAM:      ${WM_PROJECT_VERSION:?OpenFOAM bashrc did not load}"
echo "Python:        $(command -v python) ($(python --version 2>&1))"
echo "Commit:        $(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Case file:     ${case_file}"
echo "Case dir:      ${case_dir}"

# Run one step, logging to log.<name>; print the log tail and abort on failure.
run_step() {
    local name="$1"
    shift
    echo "--- ${name}"
    local start=${SECONDS}
    if ! "$@" > "log.${name}" 2>&1; then
        echo "FAILED: ${name}"
        tail -n 40 "log.${name}"
        exit 1
    fi
    echo "    done in $(( SECONDS - start )) s"
}

echo
echo "=== Case setup ==="
cd "${repo_root}"
python -m src.cfd.mesh "${case_file}" "${case_dir}" --n-procs "${SLURM_NTASKS}"
cd "${case_dir}"

echo
echo "=== Meshing ==="
run_step surfaceFeatureExtract surfaceFeatureExtract
run_step blockMesh blockMesh
run_step decomposePar decomposePar -force
run_step snappyHexMesh mpirun -np "${SLURM_NTASKS}" snappyHexMesh -parallel -overwrite
run_step reconstructParMesh reconstructParMesh -constant
rm -rf processor*   # the mesh is reconstructed, so the per-processor copies are no longer needed

echo
echo "=== Checking ==="
# checkMesh reports problems in its log rather than its exit status; mesh_report.py judges the result.
# -writeAllFields saves per-cell quality fields; -writeSets vtk saves any bad faces as VTK surfaces.
run_step checkMesh checkMesh -allGeometry -allTopology -writeAllFields -writeSets vtk
# The slices are for viewing only, so a failure here doesn't fail the job.
if ! postProcess -dict system/meshSlices -time 0 > log.meshSlices 2>&1; then
    echo "warning: writing mesh slices failed; see log.meshSlices"
fi

echo
echo "=== Mesh report ==="
cd "${repo_root}"
python -m src.cfd.mesh_report "${case_dir}"
