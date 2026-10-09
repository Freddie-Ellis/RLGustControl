#!/bin/bash
# OpenFOAM smoke test on Iridis 6: copies the pitzDaily tutorial, meshes and
# decomposes it, runs simpleFoam in parallel with MPI and reconstructs the result.
#
# Submit from anywhere in the repo:   sbatch hpc/openfoam_test.slurm
# Output log:                         of_test_<jobid>.out (in the submit directory)
# Case directory:                     <repo>/hpc/runs/of_test_<jobid>/

#SBATCH --job-name=of_test
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=of_test_%j.out
#SBATCH --error=of_test_%j.out

# OpenFOAM.com (ESI) release; also installed: v2406, v2412.
# The .org releases (10, 11, 12) keep their tutorials elsewhere, so the case path below would change.
OF_VERSION="v2506"
N_ITER=100

module purge
# OpenFOAM's bashrc references unset variables, so enable strict mode after sourcing it.
source "/iridisfs/i6software/openfoam/${OF_VERSION}/OpenFOAM-${OF_VERSION}/etc/bashrc"
set -eo pipefail

echo "=== Environment ==="
echo "Job ID:        ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST} (${SLURM_NTASKS} tasks)"
echo "OpenFOAM:      ${WM_PROJECT_VERSION:?OpenFOAM bashrc did not load}"
echo "simpleFoam:    $(command -v simpleFoam)"
echo "mpirun:        $(command -v mpirun)"
mpirun --version 2>&1 | head -1

# Run one OpenFOAM step, logging to log.<name>; print the log tail and abort on failure.
run_step() {
    local name="$1"
    shift
    echo "--- ${name}"
    if ! "$@" > "log.${name}" 2>&1; then
        echo "FAILED: ${name}"
        tail -n 40 "log.${name}"
        exit 1
    fi
}

echo
echo "=== Case setup ==="
# Resolve the repo root so the job works whichever directory it was submitted from.
repo_root="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR}")"
case_dir="${repo_root}/hpc/runs/of_test_${SLURM_JOB_ID}"
mkdir -p "$(dirname "${case_dir}")"
cp -r "${FOAM_TUTORIALS}/incompressible/simpleFoam/pitzDaily" "${case_dir}"
cd "${case_dir}"
echo "Case dir:      ${case_dir}"
# Newer tutorials keep initial conditions in 0.orig and copy them to 0 at run time.
if [ -d 0.orig ] && [ ! -d 0 ]; then
    cp -r 0.orig 0
fi
echo "Contents:      $(ls -m)"

foamDictionary -entry endTime -set "${N_ITER}" system/controlDict > /dev/null
foamDictionary -entry writeInterval -set "${N_ITER}" system/controlDict > /dev/null

cat > system/decomposeParDict <<EOF
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
numberOfSubdomains ${SLURM_NTASKS};
method          scotch;
EOF

echo
echo "=== Run ==="
if [ ! -d constant/polyMesh ]; then
    run_step blockMesh blockMesh
fi
run_step checkMesh checkMesh
run_step decomposePar decomposePar -force
run_step simpleFoam mpirun -np "${SLURM_NTASKS}" simpleFoam -parallel
run_step reconstructPar reconstructPar -latestTime

echo
echo "=== simpleFoam log tail ==="
tail -n 25 log.simpleFoam

echo
grep -q "^End" log.simpleFoam
echo "=== OpenFOAM test completed OK ==="
