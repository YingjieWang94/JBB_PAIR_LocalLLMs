#!/bin/bash -l
#SBATCH -J pair-sanity
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:h100:1

set -euo pipefail
set -x

module purge
module load cuda/12.8.0 || true

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Conda (portable)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [ -n "${ENV_PATH:-}" ]; then
    conda activate "$ENV_PATH"
  else
    conda activate "${ENV_NAME:-jbb_pair}"
  fi
fi

echo "Node: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-N/A}"
python -V
which python
nvidia-smi || true

python -c "import sys; print('sys.path[0:3]=', sys.path[0:3])"
python -m py_compile scripts/run.py src/judges/llama3.py

python -u scripts/run.py \
  --profile server \
  --subset harmful \
  --max-behaviors 2 \
  --budget-per-try 2 \
  --dtype bf16 \
  --device cuda \
  --guard-device cuda \
  --use-cache true \
  --log-every 1

echo "Generated files:"
ls -lh data/generated | tail -n 50
