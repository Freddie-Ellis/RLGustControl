#!/bin/bash
# Thin every CFD case down to what the UI's 3D CFD viewer needs (src/cfd/paraview_scripts/thin_cases.py): the last
# flow slices (p, U) and wall surface (p), float32 and compressed. Submit from the repo root:
#   sbatch hpc/slurm/thin_cases.slurm
# Restart-safe: cases already thinned are skipped, so after a timeout just submit it again.
# Then copy the result to the laptop's src/data/cfd_view (the CFD tab's default folder), e.g. from the laptop:
#   scp -r <user>@iridis:/iridisfs/scratch/<user>/RLGustControl/cfd_view/* src/data/cfd_view/

#SBATCH --job-name=thin_cases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%x_%j.out

module load paraview
PV=$(echo /iridisfs/scratch/$USER/ParaView-*-osmesa-*/bin/pvbatch)
RUNS=/iridisfs/scratch/$USER/RLGustControl/runs
OUT=/iridisfs/scratch/$USER/RLGustControl/cfd_view

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
$PV src/cfd/paraview_scripts/thin_cases.py "$RUNS" "$OUT"
du -sh "$OUT"
