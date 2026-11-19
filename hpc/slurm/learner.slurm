#!/bin/bash
# The learner on Iridis. Submit from the repo root:   sbatch slurm/learner.slurm <session> [config]
#   new session:      sbatch slurm/learner.slurm test sac_tunnel_mock      (config: a name from configs/)
#   carry on:         sbatch slurm/learner.slurm test                      (uses the session's own config)
# It trains on the episodes the tunnel PC's sync uploads to HPC_SESSIONS/<session> (src/ml/distributed/files.py), and
# stops when STOP appears there or the time limit is hit. Restart-safe: resubmit with the same session to carry on.
# A minute before the time limit (and on preemption) SLURM sends SIGTERM, which the learner treats as Ctrl+C: it
# saves its policy and writes "stopped", so the tunnel PC does not go on believing it is running.

#SBATCH --job-name=rl_learner
# scavenger only accepts the scavenger QOS. To use the default batch partition instead, delete these two lines.
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%j-learner.out

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1  # print lines as they happen, so `tail -f` on the .out file is live
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# exec: python replaces this shell, so it is the process the signal above is sent to.
exec .venv/bin/python -m src.ml.distributed.learner ${1:+"$1"} ${2:+"$2"}  # no session given -> files.SESSION
