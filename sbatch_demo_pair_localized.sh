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

# === Storage (data2) ===
export DATA2_BASE=/mnt/data2/users/$USER
export MODEL_ROOT=$DATA2_BASE/models
export HF_HOME=$DATA2_BASE/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PAIR_DATA_ROOT=$DATA2_BASE/pair_data
mkdir -p "$PAIR_DATA_ROOT"/generated

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
