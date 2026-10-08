#!/bin/bash
# Build the environment as a BATCH job instead of an interactive srun.
#
#   ./submit.sh slurm/setup.sbatch
#   tail -f logs/omni-setup-<jobid>.out
#
# submit.sh supplies the account from .env via SBATCH_ACCOUNT. There is deliberately
# no `#SBATCH --account=` line: SLURM parses directives as literal text before any
# shell runs, so they cannot reference $ACCOUNT.
#
# Preferred over `srun --pty` because:
#   - no interactive allocation to negotiate, and no stale SLURM_* env to trip over
#   - SLURM_JOB_ID is set, so setup_env.sh's login-node guard is satisfied
#   - survives your laptop closing / SSH dropping
#   - the full build log is kept, which matters when flash-attn fails obscurely
#
# A GPU is requested because flash-attn detects the target CUDA architecture from a
# visible device. Everything else here is CPU and network work.
#SBATCH --job-name=omni-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

# SLURM starts the job in the submission directory; fall back to this file's parent.
PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
cd "$PROJECT"

echo "=== allocation ==="
echo "job:    $SLURM_JOB_ID"
echo "node:   $(hostname)"
echo "project: $PROJECT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo

export PROJECT
bash setup_env.sh
