#!/bin/bash
#SBATCH -J pair-demo
#SBATCH -p gpu-a-lowsmall
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o slurm-pair-demo-%j.out
#SBATCH -e slurm-pair-demo-%j.err

set -euo pipefail

# === Storage (compute-node writable) ===
export HOME_BASE=/users/$USER
export SCRATCH_BASE=$HOME_BASE/scratch

# Read-only is fine for models; keep using data2 if it's visible on compute nodes
export MODEL_ROOT=/mnt/data2/users/$USER/models

# Put cache + outputs on scratch (writable everywhere)
export HF_HOME=$SCRATCH_BASE/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PAIR_DATA_ROOT=$SCRATCH_BASE/pair_data

mkdir -p "$PAIR_DATA_ROOT"/generated
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"


# === Offline enforcement ===
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# === Repo ===
REPO=/users/$USER/repos/JBB_PAIR_LocalLLMs
cd "$REPO"
git rev-parse --abbrev-ref HEAD
python -V

# === Output ===
RUN_ID="pair_demo_${SLURM_JOB_ID}"
OUT="$PAIR_DATA_ROOT/generated/${RUN_ID}.jsonl"

python scripts/run.py \
  --profile server \
  --subset harmful \
  --max-behaviors 2 \
  --budget-per-try 20 \
  --num-shards 1 \
  --shard-idx 0 \
  --log-every 1 \
  --ckpt-every-turns 1 \
  --out-jsonl "$OUT" \
  --attacker-model-id "Qwen/Qwen2.5-14B-Instruct" \
  --target-model-id "meta-llama/Llama-3.1-8B-Instruct" \
  --guard-model-id "meta-llama/Llama-Guard-3-8B" \
  --device cuda \
  --guard-device cuda \
  --dtype bf16 \
  --use-cache true

echo "DONE: $OUT"
echo "CKPT: ${OUT%.jsonl}.ckpt.json"
