#!/bin/bash
#SBATCH -p gpu-l40s
#SBATCH --job-name=pair-real
#SBATCH --output=logs/pair-real-%A_%a.out
#SBATCH --error=logs/pair-real-%A_%a.err
#SBATCH --array=0-39
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

set -euo pipefail
mkdir -p logs

source ~/.bashrc
conda activate jbb_pair
cd /users/yjwang/repos/JBB_PAIR_LocalLLMs

# Keep outputs separated per shard
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
