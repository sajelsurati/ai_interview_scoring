#!/bin/bash
# Submit a SLURM job with --account taken from .env.
#
#   ./submit.sh slurm/setup.sbatch
#   ./submit.sh slurm/smoke.sbatch
#   ./submit.sh slurm/score.sbatch
#   ./submit.sh slurm/score.sbatch --time=04:00:00     # extra sbatch args pass through
#
# WHY A WRAPPER: #SBATCH directives cannot reference shell variables. SLURM reads them
# as literal text before any shell runs, so `#SBATCH --account=$ACCOUNT` would submit
# with an account literally named "$ACCOUNT". The sbatch files therefore carry no
# --account line, and the account arrives via SBATCH_ACCOUNT, which sbatch reads from
# the environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/load_env.sh"
load_env "${ENV_FILE:-$HERE/.env}"

if [ -z "${ACCOUNT:-}" ]; then
  cat >&2 <<MSG
error: ACCOUNT is not set.

Add it to $HERE/.env:

  ACCOUNT=your_slurm_account

or export it for this shell:

  export ACCOUNT=your_slurm_account

Your allocations:  sacctmgr show assoc where user=\$USER format=account,qos
MSG
  exit 1
fi

script=${1:-}
if [ -z "$script" ]; then
  echo "usage: ./submit.sh slurm/<script>.sbatch [extra sbatch args...]" >&2
  exit 1
fi
shift
[ -f "$script" ] || { echo "error: no such script: $script" >&2; exit 1; }

# sbatch reads these from the environment; equivalent to passing --account.
export SBATCH_ACCOUNT="$ACCOUNT"
# srun/salloc use different names, exported so an interactive shell inherits them.
export SLURM_ACCOUNT="$ACCOUNT"
export SALLOC_ACCOUNT="$ACCOUNT"

mkdir -p "$HERE/logs"
echo "submitting $script  (account=$ACCOUNT)"
exec sbatch "$@" "$script"
