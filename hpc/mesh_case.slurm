#!/bin/bash
# Mesh one case on Iridis 6, check the mesh, and write small extracts to download.
#
# Submitted by hpc/submit_cases.sh (see src/cfd/submit.py), from the repository root, with this job's output in
# the case's folder:
#     bash hpc/submit_cases.sh configs/cases/uav_clean.toml
# The only argument is a case file (see configs/cases/), which names the geometry, the mesh settings and the
# control surface deflections. One-off setup first: bash hpc/setup_python.sh (on the login node).
#
# Everything for a case goes in hpc/runs/<case name>/ (the case file's name); this job writes mesh/:
#   slurm_<jobid>.out          the job's output (this log)
#   log.*                      one log per step
#   mesh_case.json             every setting used, the job id and when the case was written
#   mesh_report.json           cell counts, quality metrics, layer coverage, pass/fail
#   postProcessing/meshSlices/ cutting planes and wall surfaces (VTK) with quality fields, and layer coverage
#                              on the walls: download these
#   postProcessing/constant/   VTK surfaces of any faces checkMesh flags as bad (e.g. lowQualityTetFaces/)
#
# Steps: Python writes the case (aircraft skin STL, its feature edges, dictionaries) -> blockMesh
#        -> decomposePar -> snappyHexMesh (MPI) -> reconstructParMesh -> checkMesh -> slices -> report.
# The job fails (non-zero exit) if any step fails or the mesh does not meet the acceptance limits.

#SBATCH --job-name=mesh
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G   # the ~3M-cell mesh peaked at 12.8 GB; the fine mesh-study mesh has about twice the cells
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%x_%j.out   # replaced by submit_cases.sh with <case folder>/mesh/slurm_<jobid>.out
#SBATCH --error=slurm_%x_%j.out

set -eo pipefail
source "${SLURM_SUBMIT_DIR}/hpc/env.sh"   # Slurm runs a copy of this script, so find env.sh via the repo
load_openfoam
load_python

case_file="$(cd "${SLURM_SUBMIT_DIR}" && realpath "${1:?usage: bash hpc/submit_cases.sh <case file>}")"
case_name="$(basename "${case_file}" .toml)"
case_dir="${repo_root}/hpc/runs/${case_name}/mesh"
mkdir -p "${case_dir}"

print_environment
echo "Case file:     ${case_file}"
echo "Case dir:      ${case_dir}"

echo
echo "=== Case setup ==="
cd "${repo_root}"
python -m src.cfd.mesh "${case_file}" "${case_dir}" --n-procs "${SLURM_NTASKS}"
cd "${case_dir}"

echo
echo "=== Meshing ==="
run_step blockMesh blockMesh
run_step decomposePar decomposePar -force
run_step snappyHexMesh run_parallel snappyHexMesh -overwrite
run_step reconstructParMesh reconstructParMesh -constant
# snappyHexMesh writes its layer fields (layers added and share of the requested thickness reached, per wall
# face) into the processor folders only. Rebuild them into constant/, next to checkMesh's fields, so the wall
# surface in postProcessing/meshSlices/ carries them. For viewing only, so a failure is just a warning.
ls -R processor0 > log.reconstructLayerFields 2>&1 || true   # records where they were written
if reconstructPar -withZero -constant -fields '(nSurfaceLayers thickness thicknessFraction)' \
        >> log.reconstructLayerFields 2>&1; then
    for field in nSurfaceLayers thickness thicknessFraction; do
        if [ -f "0/${field}" ]; then mv "0/${field}" constant/; fi
    done
    rmdir 0 2>/dev/null || true
else
    echo "warning: rebuilding the layer fields failed; see log.reconstructLayerFields"
fi
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
