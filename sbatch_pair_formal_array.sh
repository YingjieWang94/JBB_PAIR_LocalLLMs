#!/bin/bash
#SBATCH -J pair-formal
#SBATCH -p gpu-a-lowsmall
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --array=0-3
#SBATCH -o slurm-pair-formal-%A_%a.out
#SBATCH -e slurm-pair-formal-%A_%a.err

set -euo pipefail

echo "=== SLURM JOB START ==="
date
echo "HOST=$(hostname)"
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo

# === Storage (compute-node writable) ===
export HOME_BASE=/users/$USER
export SCRATCH_BASE=$HOME_BASE/scratch

# Models live on data2; compute nodes may mount it read-only, which is fine.
export MODEL_ROOT=/mnt/data2/users/$USER/models

# Cache + outputs on scratch (writable everywhere)
export HF_HOME=$SCRATCH_BASE/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PAIR_DATA_ROOT=$SCRATCH_BASE/pair_data

mkdir -p "$PAIR_DATA_ROOT/generated"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

test -w "$SCRATCH_BASE" || { echo "[FATAL] SCRATCH not writable: $SCRATCH_BASE"; exit 2; }

# === Offline enforcement ===
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# === Repo ===
REPO=/users/$USER/repos/JBB_PAIR_LocalLLMs
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
python -V
echo

# === Model readability checks (fail fast) ===
test -r "$MODEL_ROOT/meta-llama__Llama-3.1-8B-Instruct/config.json" || { echo "[FATAL] Target model not readable at $MODEL_ROOT"; exit 3; }
test -r "$MODEL_ROOT/Qwen__Qwen2.5-14B-Instruct/config.json" || { echo "[FATAL] Attacker model not readable at $MODEL_ROOT"; exit 3; }
test -r "$MODEL_ROOT/meta-llama__Llama-Guard-3-8B/config.json" || { echo "[FATAL] Guard model not readable at $MODEL_ROOT"; exit 3; }

# === Manifest check (your run.py requires it) ===
MANIFEST="$PAIR_DATA_ROOT/processed/jbb_manifest.jsonl"
test -f "$MANIFEST" || { echo "[FATAL] Manifest missing: $MANIFEST"; exit 11; }

# === Sharding ===
NUM_SHARDS=4
SHARD_IDX=${SLURM_ARRAY_TASK_ID}

# === Output naming (stable for resume) ===
RUN_ID="pair_formal_${SLURM_ARRAY_JOB_ID}"
OUT="$PAIR_DATA_ROOT/generated/${RUN_ID}.shard${SHARD_IDX}of${NUM_SHARDS}.jsonl"
CKPT="${OUT%.jsonl}.ckpt.json"

echo "NUM_SHARDS=$NUM_SHARDS"
echo "SHARD_IDX=$SHARD_IDX"
echo "OUT=$OUT"
echo "CKPT=$CKPT"
echo

# === Auto-resume if checkpoint exists ===
RESUME_ARGS=""
if [ -f "$CKPT" ]; then
  echo "[INFO] Checkpoint exists -> resuming shard"
  RESUME_ARGS="--resume"
fi

python -u scripts/run.py \
  --profile server \
  --subset harmful \
  --budget-per-try 20 \
  --num-shards ${NUM_SHARDS} \
  --shard-idx ${SHARD_IDX} \
  --log-every 10 \
  --ckpt-every-turns 1 \
  --out-jsonl "$OUT" \
  $RESUME_ARGS \
  --attacker-model-id "Qwen/Qwen2.5-14B-Instruct" \
  --target-model-id "meta-llama/Llama-3.1-8B-Instruct" \
  --guard-model-id "meta-llama/Llama-Guard-3-8B" \
  --device cuda \
  --guard-device cuda \
  --dtype bf16 \
  --use-cache true

echo
echo "=== DONE ==="
echo "DONE: $OUT"
echo "CKPT: $CKPT"
date
