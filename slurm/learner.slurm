#!/bin/bash
# The learner on Iridis. Submit from the repo root:   sbatch slurm/learner.slurm
# It trains on the episodes the tunnel PC's sync uploads to HPC_SHARED (src/ml/distributed/files.py) and stops when
# STOP appears there or the time limit is hit. It is restart-safe: resubmit to carry on from the last saved state.

#SBATCH --job-name=rl_learner
# scavenger only accepts the scavenger QOS. To use the default batch partition instead, delete these two lines.
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j-learner.out

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1  # print lines as they happen, so `tail -f` on the .out file is live
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
.venv/bin/python -m src.ml.distributed.learner
