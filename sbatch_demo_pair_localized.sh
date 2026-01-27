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

echo "=== SLURM JOB START ==="
date
echo "HOST=$(hostname)"
echo "JOBID=${SLURM_JOB_ID}"
echo

# -----------------------------
# Storage: compute-node writable
# -----------------------------
export HOME_BASE=/users/$USER
export SCRATCH_BASE=$HOME_BASE/scratch

# Models can live on data2 even if it is mounted read-only on compute nodes
export MODEL_ROOT=/mnt/data2/users/$USER/models

# HF cache + outputs MUST be on a writable FS on compute nodes
export HF_HOME=$SCRATCH_BASE/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PAIR_DATA_ROOT=$SCRATCH_BASE/pair_data

# Create dirs
mkdir -p "$PAIR_DATA_ROOT/generated"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Sanity checks: writability
test -d "$SCRATCH_BASE" || { echo "[FATAL] SCRATCH_BASE missing: $SCRATCH_BASE"; exit 2; }
test -w "$SCRATCH_BASE" || { echo "[FATAL] SCRATCH not writable: $SCRATCH_BASE"; exit 2; }
test -w "$PAIR_DATA_ROOT" || { echo "[FATAL] PAIR_DATA_ROOT not writable: $PAIR_DATA_ROOT"; exit 2; }

echo "HOME_BASE=$HOME_BASE"
echo "SCRATCH_BASE=$SCRATCH_BASE"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "HF_HOME=$HF_HOME"
echo "PAIR_DATA_ROOT=$PAIR_DATA_ROOT"
echo
echo "Disk free (HOME_BASE):"
df -h "$HOME_BASE" || true
echo "Disk free (SCRATCH_BASE):"
df -h "$SCRATCH_BASE" || true
echo "Disk free (MODEL_ROOT mount):"
df -h "$MODEL_ROOT" || true
echo

# -----------------------------
# Offline enforcement (token-free runtime)
# -----------------------------
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# -----------------------------
# Repo / python path
# -----------------------------
REPO=/users/$USER/repos/JBB_PAIR_LocalLLMs
test -d "$REPO" || { echo "[FATAL] Repo not found: $REPO"; exit 10; }
cd "$REPO"

# set -u safe:
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "REPO=$REPO"
echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
python -V
echo

# -----------------------------
# Model readability checks
# -----------------------------
test -r "$MODEL_ROOT/meta-llama__Llama-3.1-8B-Instruct/config.json" || { echo "[FATAL] Target model not readable at $MODEL_ROOT"; exit 3; }
test -r "$MODEL_ROOT/Qwen__Qwen2.5-14B-Instruct/config.json" || { echo "[FATAL] Attacker model not readable at $MODEL_ROOT"; exit 3; }
test -r "$MODEL_ROOT/meta-llama__Llama-Guard-3-8B/config.json" || { echo "[FATAL] Guard model not readable at $MODEL_ROOT"; exit 3; }

echo "Models readable OK."
echo

# -----------------------------
# Manifest existence check (common early failure)
# -----------------------------
# Your runner expects: $PAIR_DATA_ROOT/processed/jbb_manifest.jsonl by default if you set PAIR_DATA_ROOT
# If your patched run.py uses a different env var, adjust accordingly.
MANIFEST="$PAIR_DATA_ROOT/processed/jbb_manifest.jsonl"
if [ ! -f "$MANIFEST" ]; then
  echo "[FATAL] Manifest not found: $MANIFEST"
  echo "Build/copy it to: $PAIR_DATA_ROOT/processed/jbb_manifest.jsonl"
  exit 11
fi
echo "Manifest OK: $MANIFEST"
echo

# -----------------------------
# Output
# -----------------------------
RUN_ID="pair_demo_${SLURM_JOB_ID}"
OUT="$PAIR_DATA_ROOT/generated/${RUN_ID}.jsonl"

echo "OUT=$OUT"
echo "CKPT=${OUT%.jsonl}.ckpt.json"
echo

# -----------------------------
# Run demo (2 behaviors, budget 20)
# -----------------------------
python -u scripts/run.py \
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

echo
echo "=== DONE ==="
echo "DONE: $OUT"
echo "CKPT: ${OUT%.jsonl}.ckpt.json"
date
