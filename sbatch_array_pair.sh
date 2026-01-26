#!/bin/bash -l
#SBATCH -J pair-gen
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

# Choose partition/GRES based on sinfo:
# #SBATCH -p gpu-h100
# #SBATCH --gres=gpu:h100:1
# OR:
# #SBATCH -p gpu
# #SBATCH --gres=gpu:1

# Array size: set this to the number of GPUs you want to occupy (e.g., 8, 16, 32...)
#SBATCH --array=0-7

set -euo pipefail
module purge
module load cuda/12.8.0 || true

# --- Python/Conda environment (customize as needed) ---
export PYTHONUNBUFFERED=1
# If conda is available, activate an env (set ENV_NAME or ENV_PATH before sbatch)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  if [ -n "${ENV_PATH:-}" ]; then
    conda activate "$ENV_PATH" || true
  else
    conda activate "${ENV_NAME:-jbb_pair}" || true
  fi
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Sharding parameters
SHARD_IDX=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT}

echo "Node: $(hostname)"
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS}"
nvidia-smi || true

# Kill any cluster-injected/poisoned tokens
unset HF_TOKEN
unset HUGGINGFACEHUB_API_TOKEN
unset HUGGINGFACE_HUB_TOKEN

# Use the SAME HF_HOME as your interactive environment
export HF_HOME="/users/yjwang/work/jbb_pair/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"



python scripts/run.py \
  --profile server \
  --subset harmful \
  --num-shards ${NUM_SHARDS} \
  --shard-idx ${SHARD_IDX} \
  --budget-per-try 20 \
  --dtype bf16 \
  --device cuda \
  --guard-device cuda \
  --use-cache true \
  --log-every 50
