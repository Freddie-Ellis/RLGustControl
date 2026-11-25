#!/bin/bash
# Train the tap autoencoder (src/ml/ae/train_ae.py) on Iridis, with the settings in one TOML config.
# Submit from the repo root:
#   sbatch hpc/slurm/train_ae.slurm configs/autoencoder/ae_cfd.toml
#
# Reads the CFD data in src/data/cfd/ (runs.parquet and surface/), writes the run folder to runs/autoencoder/ and this
# job's log to slurm-<jobid>-train_ae.out in the repo root. To train several models, submit once per config.

#SBATCH --job-name=train_ae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j-train_ae.out

set -euo pipefail

config="${1:?usage: sbatch hpc/slurm/train_ae.slurm <config.toml>}"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f "${config}" ]; then
    echo "no config at ${config} (paths are relative to the repo root)" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1  # print lines as they happen, so `tail -f` on the .out file is live
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg      # no display on the compute nodes: the end-of-training plots are drawn straight to file

echo "train_ae ${config} on ${SLURM_JOB_NODELIST} (job ${SLURM_JOB_ID}, commit $(git rev-parse --short HEAD))"
exec .venv/bin/python -m src.ml.ae.train_ae "${config}"
