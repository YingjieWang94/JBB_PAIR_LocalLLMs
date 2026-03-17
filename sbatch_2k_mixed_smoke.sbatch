#!/bin/bash
#SBATCH --job-name=pair-2k-mixed-smoke
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pair2k_smoke_%j.out
#SBATCH --error=logs/pair2k_smoke_%j.err

set -euo pipefail

echo "========== JOB START =========="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PWD=$(pwd)"

mkdir -p logs
mkdir -p /users/yjwang/scratch/pair_data/generated/real_2k_mixed_b20_smoke

# Activate env
source /users/yjwang/.bashrc
conda activate jbb_pair

# Hugging Face / cache / offload
export HF_HOME=/users/yjwang/scratch/hf_home
export TRANSFORMERS_CACHE=/users/yjwang/scratch/hf_home/transformers
export HF_DATASETS_CACHE=/users/yjwang/scratch/hf_home/datasets
export HF_OFFLOAD_DIR=/users/yjwang/scratch/hf_offload_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_OFFLOAD_DIR"

# Output dir for this smoke test
export PAIR_OUT_DIR=/users/yjwang/scratch/pair_data/generated/real_2k_mixed_b20_smoke

# Optional: reduce tokenizer thread noise
export TOKENIZERS_PARALLELISM=false

echo "========== GPU INFO =========="
nvidia-smi || true
echo "========== MEMORY INFO =========="
free -h || true

cd /users/yjwang/repos/JBB_PAIR_LocalLLMs

python scripts/run_2k_mixed.py \
  --profile server \
  --manifest-relpath data/behaviours/behaviors_2000_mixed.jsonl \
  --subset-filter all \
  --goal-field expanded_goal \
  --budget-per-try 1 \
  --num-shards 100 \
  --shard-idx 0 \
  --max-behaviors 1 \
  --max-gpu-mem-util 0.70 \
  --offload-dir "$HF_OFFLOAD_DIR" \
  --seed 0

status=$?

echo "========== JOB END =========="
date
echo "Exit status: $status"

echo "========== OUTPUT FILES =========="
ls -lh "$PAIR_OUT_DIR" || true