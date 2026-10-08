#!/bin/bash
# Rung 1 of the smoke ladder: does the model load on this allocation and emit text?
# No video, no manifest. Run this before anything else.
# Submit with ./submit.sh slurm/smoke.sbatch -- it supplies the account from .env via
# SBATCH_ACCOUNT. There is deliberately no `#SBATCH --account=` line: SLURM parses
# directives as literal text, so they cannot reference $ACCOUNT.
#SBATCH --job-name=omni-smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
##SBATCH --constraint=h200             # uncomment once you confirm the node feature name
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

nvidia-smi --query-gpu=index,name,memory.total --format=csv

PROJECT=${PROJECT:-/scratch/$USER/ai_interview_scoring}
OVERLAY=${OVERLAY:-/scratch/$USER/envs/omni-env.ext3}
SIF=${SIF:-/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif}

[ -f "$OVERLAY" ] || { echo "error: overlay not found: $OVERLAY (run setup_env.sh)" >&2; exit 1; }

singularity exec --nv --overlay "${OVERLAY}:ro" "$SIF" /bin/bash -c "
set -euo pipefail
source /ext3/env.sh
export HF_HOME=/scratch/\$USER/.hugging_face   # must match setup_env.sh
cd $PROJECT/src

python - <<'PY'
import time, torch, omni

print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      'devices', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  gpu{i}: {p.name}, {p.total_memory/2**30:.0f} GiB, sm_{p.major}{p.minor}')

t0 = time.time()
model, processor = omni.load_model()
print(f'loaded in {time.time()-t0:.1f}s')
for i in range(torch.cuda.device_count()):
    print(f'  gpu{i}: {torch.cuda.memory_allocated(i)/2**30:.1f} GiB allocated')

conv = [
    {'role': 'system', 'content': [{'type': 'text', 'text': omni.SYSTEM_PROMPT}]},
    {'role': 'user', 'content': [{'type': 'text', 'text':
        'Reply with exactly this JSON and nothing else: {\"ok\": true, \"score\": 3}'}]},
]
raw, tokens = omni.generate(model, processor, conv, False, max_new_tokens=64)
print('tokens:', omni.format_breakdown(tokens))
print('raw:', raw)
print('parsed:', omni.parse_score_json(raw))

# Text-only prompt, so the modality pad counts must be 0. A warning here means the
# pad token ids could not be resolved and the per-modality split will be unusable.
ids, warnings = omni.resolve_modality_token_ids(processor, model)
print('modality token ids:', ids)
if warnings:
    print('TOKEN-ID WARNINGS:', warnings)
PY
"
