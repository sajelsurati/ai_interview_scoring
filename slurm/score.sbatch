#!/bin/bash
# Submit with ./submit.sh slurm/score.sbatch -- it supplies the account from .env via
# SBATCH_ACCOUNT. There is deliberately no `#SBATCH --account=` line: SLURM parses
# directives as literal text, so they cannot reference $ACCOUNT.
#SBATCH --job-name=omni-score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1                   # one H200 (141GB) holds the BF16 model alone
##SBATCH --constraint=h200             # uncomment once you confirm the node feature name
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Confirm what you actually got before anything expensive happens. If this shows an
# L40S (48GB) or an A100 40GB, stop: the BF16 model will not fit and the job will OOM
# after a long weight load. Request 2 GPUs or an H100/H200 explicitly.
nvidia-smi --query-gpu=index,name,memory.total --format=csv

PROJECT=${PROJECT:-/scratch/$USER/ai_interview_scoring}
OVERLAY=${OVERLAY:-/scratch/$USER/envs/omni-env.ext3}
SIF=${SIF:-/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif}

[ -f "$OVERLAY" ] || { echo "error: overlay not found: $OVERLAY (run setup_env.sh)" >&2; exit 1; }

mkdir -p "$PROJECT/logs"

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh

export HF_HOME=/scratch/\$USER/.hugging_face   # must match setup_env.sh
export HF_HUB_OFFLINE=1          # weights are pre-downloaded; fail loudly if not
export TOKENIZERS_PARALLELISM=false
export VIDEO_MAX_PIXELS=\$((360 * 420))   # cap frame resolution
export FPS=1.0                            # 1 fps is plenty for a talking head

cd $PROJECT

echo '=== pre-flight ==='
python src/preflight.py --manifest manifest.csv

echo '=== scoring: video / audio / text x 3 runs ==='
python src/score_run.py \
  --manifest manifest.csv \
  --prompt prompts/score_v1.txt \
  --modes video audio text \
  --runs 3 \
  --temperature 0.7

echo '=== wording-sensitivity arm: same instructions reworded (video only) ==='
python src/score_run.py \
  --manifest manifest.csv \
  --prompt prompts/score_v2.txt \
  --modes video \
  --runs 1

echo '=== aggregate ==='
python src/aggregate.py --runs runs --out reports
"
