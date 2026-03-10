#!/bin/bash
#SBATCH --job-name=pair-2k-mixed-b20
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/pair2k_mixed_%A_%a.out
#SBATCH --error=logs/pair2k_mixed_%A_%a.err

set -euo pipefail

echo "========== JOB START =========="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
echo "==============================="

source ~/.bashrc
conda activate jbb_pair

export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="/users/yjwang/scratch/hf"
export HF_OFFLOAD_DIR="/users/yjwang/scratch/hf_offload"
mkdir -p "$HF_HOME" "$HF_OFFLOAD_DIR"

REPO="/users/yjwang/repos/JBB_PAIR_LocalLLMs"
cd "$REPO"

mkdir -p logs

OUT_BASE="/users/yjwang/scratch/pair_data/generated/real_2k_mixed_b20"
mkdir -p "$OUT_BASE"
export PAIR_OUT_DIR="$OUT_BASE"

python -u scripts/run_2k_mixed.py \
  --profile server \
  --config configs/server_mixtral_vicuna_5k.json \
  --manifest-relpath data/behaviours/behaviors_2000_mixed.jsonl \
  --subset-filter all \
  --goal-field expanded_goal \
  --budget-per-try 20 \
  --num-shards 100 \
  --shard-idx "${SLURM_ARRAY_TASK_ID}" \
  --seed 0 \
  --max-gpu-mem-util 0.25 \
  --fail-fast \
  2>&1 | tee "$OUT_BASE/shard_${SLURM_ARRAY_TASK_ID}.log"

echo "=========== JOB END ==========="
date
echo "==============================="