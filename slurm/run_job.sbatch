#!/bin/bash -l
#SBATCH --job-name=entropy-rag
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH -G 1
#SBATCH --mem=32g
#SBATCH --time=2-00:00:00

# Generic wrapper for running this project's compute jobs (dataset onboarding
# builds, entropy/FAISS refreshes, experiment runs) on the Rutgers CS Slurm
# cluster instead of the ilab login node directly. Login-node processes are
# capped at 24 CPU-hours and per-user memory limits (see
# https://resources.cs.rutgers.edu/docs/limitation-enforced-on-cs-linux-machines/);
# those caps do not apply to jobs submitted via sbatch/srun
# (https://resources.cs.rutgers.edu/docs/scheduler-for-gpu-jobs/).
#
# Usage:
#   sbatch slurm/run_job.sbatch python experiment_runner.py --run --dataset hotpotqa \
#       --experiment simple_reordering --number_of_queries 1
#
# Monitor:  squeue -u $USER   |   tail -f slurm/logs/<job-name>-<jobid>.out
# Cancel:   scancel <jobid>

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
export PYTHONUNBUFFERED=1
exec "$@"
