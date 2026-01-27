#!/bin/bash
#SBATCH --job-name=pair-real
#SBATCH --output=logs/pair-real-%A_%a.out
#SBATCH --error=logs/pair-real-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-39

set -euo pipefail

mkdir -p logs

# >>> Activate env (adjust to your cluster)
source ~/.bashrc
conda activate jbb_pair

cd /users/yjwang/repos/JBB_PAIR_LocalLLMs

# Put outputs per array task into its own folder (clean + avoids collisions)
export PAIR_OUT_DIR="/users/yjwang/scratch/pair_data/generated/real_2k_b20/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$PAIR_OUT_DIR"

# Use the GPU assigned by Slurm; if your cluster sets CUDA_VISIBLE_DEVICES already, this is fine.
# Otherwise, you can export CUDA_VISIBLE_DEVICES=0 here.
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
