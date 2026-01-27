#!/bin/bash
#SBATCH -p gpu-l40s
#SBATCH --job-name=pair-real
#SBATCH --output=logs/pair-real-%A_%a.out
#SBATCH --error=logs/pair-real-%A_%a.err
#SBATCH --array=0-39%4
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

set -eo pipefail
mkdir -p logs

# --- Conda activation without sourcing ~/.bashrc ---
# Adjust this path if your conda lives elsewhere
CONDA_BASE="$HOME/.conda"
if [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniforge3/bin/conda"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  CONDA_EXE="$HOME/miniconda3/bin/conda"
else
  CONDA_EXE="$(command -v conda || true)"
fi

if [ -z "$CONDA_EXE" ]; then
  echo "ERROR: conda not found"
  exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"
conda activate jbb_pair

cd /users/yjwang/repos/JBB_PAIR_LocalLLMs

export PAIR_OUT_DIR="/users/yjwang/scratch/pair_data/generated/real_2k_b20/shard_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$PAIR_OUT_DIR"

python -u scripts/run.py \
  --profile server \
  --config configs/server.json \
  --guard-device cpu \
  --subset harmful \
  --max-behaviors 2000 \
  --budget-per-try 20 \
  --num-shards 40 \
  --shard-idx "${SLURM_ARRAY_TASK_ID}" \
  --max-gpu-mem-util 0.40
