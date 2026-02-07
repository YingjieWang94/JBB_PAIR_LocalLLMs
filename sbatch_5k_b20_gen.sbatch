#!/bin/bash
#SBATCH --job-name=pair-5k-b20
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/pair5k_%A_%a.out
#SBATCH --error=logs/pair5k_%A_%a.err

set -euo pipefail

# --- env ---
source ~/.bashrc
conda activate jbb_pair

export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0

export HF_HOME="/users/yjwang/scratch/hf"
export HF_OFFLOAD_DIR="/users/yjwang/scratch/hf_offload"
mkdir -p "$HF_HOME" "$HF_OFFLOAD_DIR"

REPO="/users/yjwang/repos/JBB_PAIR_LocalLLMs"
cd "$REPO"
export PAIR_DATA_ROOT="$REPO/data"

# Ensure the 5k manifest is in processed/
cp -f "$PAIR_DATA_ROOT/behaviours/jbb_manifest.jsonl" "$PAIR_DATA_ROOT/processed/jbb_manifest.jsonl"

# --- outputs ---
OUT_BASE="/users/yjwang/scratch/pair_data/generated/real_5k_b20"
mkdir -p "$OUT_BASE" logs

# --- run ---
python -u scripts/run.py \
  --profile server \
  --config configs/server_mixtral_vicuna_5k.json \
  --subset harmful \
  --max-behaviors 0 \
  --budget-per-try 20 \
  --num-shards 100 \
  --shard-idx ${SLURM_ARRAY_TASK_ID} \
  --seed 0 \
  --max-gpu-mem-util 0.5 \
  --fail-fast \
  2>&1 | tee "$OUT_BASE/shard_${SLURM_ARRAY_TASK_ID}.log"
