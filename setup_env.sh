#!/bin/bash
# One-time environment build on NYU Torch. Must run on a GPU compute node, because
# flash-attn compiles against CUDA. Easiest route is the batch job:
#
#   ./submit.sh slurm/setup.sbatch
#
# Interactively instead (ACCOUNT comes from .env):
#
#   srun --account=$ACCOUNT --cpus-per-task=8 --mem-per-cpu=8G --gres=gpu:1 \
#        --time=04:00:00 --pty /bin/bash
#   bash setup_env.sh
#
# Conda envs create >100k files and Torch caps you at 1M inodes on /scratch, so the
# env lives inside an Apptainer overlay image rather than on the filesystem directly.
set -euo pipefail

SCRATCH=/scratch/$USER

# The project is wherever this script lives, so the repo can sit anywhere on scratch.
PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}

# Load .env first: the guard below quotes $ACCOUNT in its hint, and step 3 needs
# $HF_TOKEN. Shared parser, also used by submit.sh.
source "$PROJECT/lib/load_env.sh"
load_env "${ENV_FILE:-$PROJECT/.env}"

# Refuse to run on a login node. The flash-attn build is a long multi-core nvcc
# compile that login-node cgroups will kill partway through -- and that leaves the
# overlay half-built, which is worse than failing outright.
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE:-0}" != "1" ]; then
  cat >&2 <<MSG
error: not inside a SLURM allocation (SLURM_JOB_ID is unset).

This script must run on a GPU compute node. Either submit it as a job:

  ./submit.sh slurm/setup.sbatch

or get an interactive allocation first:

  srun --account=${ACCOUNT:-<set ACCOUNT in .env>} --cpus-per-task=8 \\
       --mem-per-cpu=8G --gres=gpu:1 --time=04:00:00 --pty /bin/bash

then re-run \`bash setup_env.sh\` inside that shell.

(To override deliberately -- e.g. to download weights only -- set ALLOW_LOGIN_NODE=1.)
MSG
  exit 1
fi

# Torch paths (differ from Greene -- Greene used /scratch/work/public/...).
OVERLAY_SRC=${OVERLAY_SRC:-/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz}
OVERLAY=$SCRATCH/envs/omni-env.ext3
# MUST be a `-devel` image: only those ship nvcc, which flash-attn needs to compile.
# The runtime-only images (cuda12.6.3, 12.8.1, 12.9.1, 13.x) are NEWER but have no
# nvcc, so flash-attn silently falls back to sdpa there. Newest devel image wins:
#   cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif   <-- newest -devel on Torch
#   cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
# CUDA 12.2 covers sm_90 (H100/H200). B200 is sm_100 and would need 12.8+, but no
# 12.8+ devel image exists here -- another reason to target H200.
SIF=${SIF:-/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif}

# Weight cache. MUST match HF_HOME in slurm/*.sbatch, or the jobs will run with
# HF_HUB_OFFLINE=1 against an empty cache and fail after queueing.
HF_HOME_DIR=${HF_HOME_DIR:-$SCRATCH/.hugging_face}

for f in requirements.txt requirements-flashattn.txt; do
  [ -f "$PROJECT/$f" ] || { echo "error: $PROJECT/$f not found" >&2; exit 1; }
done

echo "Available -devel images (these are the only ones with nvcc):"
ls /share/apps/images/ | grep -i 'cuda.*devel.*\.sif' || true
echo "Using:   $SIF"
echo "Project: $PROJECT"
echo

mkdir -p "$SCRATCH" "$HF_HOME_DIR" "$(dirname "$OVERLAY")"

# --- 1. overlay image for the python env ------------------------------------
if [ ! -f "$OVERLAY" ]; then
  gz="$(dirname "$OVERLAY")/$(basename "$OVERLAY_SRC")"   # ...overlay-15GB-500K.ext3.gz
  cp -rp "$OVERLAY_SRC" "$gz"
  gunzip -f "$gz"                                         # -> ...ext3 (drops .gz)
  mv "${gz%.gz}" "$OVERLAY"
fi

# --- 2. build the env inside the overlay (writable needs --fakeroot on Torch) ---
# PROJECT is passed through --env because the heredoc below is quoted, so the outer
# shell does not expand anything inside it.
singularity exec --fakeroot --nv \
  --env PROJECT="$PROJECT" \
  --overlay "$OVERLAY:rw" "$SIF" /bin/bash <<'INNER'
set -euo pipefail

# Test for the conda BINARY, not the directory. A killed or failed install leaves the
# directory present but unusable, and a directory check would skip the repair.
if [ ! -x /ext3/miniforge3/bin/conda ]; then
  echo "=== installing miniforge (clearing any partial install) ==="
  rm -rf /ext3/miniforge3 /ext3/env.sh
  cd /tmp
  wget --no-check-certificate -q \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
  cat > /ext3/env.sh <<'EOF'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
EOF
  chmod +x /ext3/env.sh
fi

source /ext3/env.sh

# ffmpeg is a conda package, not pip, so it stays out of requirements.txt.
conda install -y -c conda-forge python=3.11 ffmpeg

cd "$PROJECT"

# If a lock file exists from a previous successful build, prefer it -- it pins the
# exact resolution that worked rather than re-solving against whatever is current.
if [ -f requirements.lock.txt ]; then
  echo "=== installing from requirements.lock.txt ==="
  pip install --no-cache-dir -r requirements.lock.txt
else
  echo "=== installing from requirements.txt ==="
  pip install --no-cache-dir -r requirements.txt
fi

# flash-attn last: its setup.py imports torch, so torch must already be present and
# build isolation must be off. MAX_JOBS caps parallel nvcc to avoid exhausting RAM.
# Optional dependency -- src/omni.py falls back to sdpa -- so a failure here warns
# rather than aborting a build that is otherwise complete.
echo "=== building flash-attn (optional, slow) ==="
if ! MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
     -r requirements-flashattn.txt; then
  echo
  echo "WARNING: flash-attn failed to build. The pipeline still works --" >&2
  echo "src/omni.py falls back to PyTorch sdpa (slower, identical outputs)." >&2
  echo "To skip the attempt entirely next time: QWEN_OMNI_ATTN=sdpa" >&2
  echo
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "from transformers import Qwen3OmniMoeForConditionalGeneration; print('transformers OK')"
python -c "import importlib.util as u; \
print('flash-attn:', 'present' if u.find_spec('flash_attn') else 'absent (will use sdpa)')"

# Freeze the resolution that actually worked, for reproducible rebuilds.
# flash-attn is filtered out: it must stay in the separate --no-build-isolation pass,
# and a lock file containing it would fail on reinstall.
# Caveat: if you switch torch to a custom index, freeze emits a local version like
# `torch==2.9.0+cu121`, which plain PyPI cannot resolve -- re-add the --index-url line.
pip freeze | grep -v -i '^flash[-_]attn' > requirements.lock.txt
echo "wrote requirements.lock.txt ($(wc -l < requirements.lock.txt) packages)"
INNER

# --- 3. pull the weights to scratch (~70GB BF16) ----------------------------
# Note: /scratch is flushed after 60 days without access. If this sits idle,
# the weights will vanish and need re-downloading.
#
# The token is handed to the container through APPTAINERENV_*, NOT interpolated into
# the exec string below. /proc/<pid>/cmdline is world-readable on a shared node, so a
# token on the command line would be visible to every other user; /proc/<pid>/environ
# is owner-only. Qwen3-Omni is public, so this is about download rate limits, not
# access -- the download works fine without it.
if [ -n "${HF_TOKEN:-}" ]; then
  export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
  export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"   # older Singularity builds
  echo "HF_TOKEN: set (authenticated download)"
else
  echo "HF_TOKEN: not set (anonymous download -- fine, the model is public)"
fi

singularity exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
source /ext3/env.sh
export HF_HOME=$HF_HOME_DIR
# huggingface_hub[cli] comes from requirements.txt in step 2. Nothing is installed
# here -- the overlay is mounted :ro, so a pip install would fail.
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct
du -sh $HF_HOME_DIR
"

echo
echo "Done."
echo "  overlay:  $OVERLAY"
echo "  weights:  $HF_HOME_DIR"
echo "  next:     sbatch slurm/smoke.sbatch"
