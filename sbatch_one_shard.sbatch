#!/bin/bash
#SBATCH --job-name=pair5k
#SBATCH --partition=gpu-h100 # gpu-l40s gpu-a100-lowbig 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pair5k_%j.out
#SBATCH --error=logs/pair5k_%j.err
#SBATCH --requeue

echo "[START] $(date) host=$(hostname) jobid=${SLURM_JOB_ID:-NA} args: $@"
pwd
whoami

set -eo pipefail
set -x
trap 'echo "[FATAL] line=$LINENO rc=$? cmd=$BASH_COMMAND" >&2' ERR


# ---------- user config ----------
REPO="/users/yjwang/repos/JBB_PAIR_LocalLLMs"
NUM_SHARDS=100
# --------------------------------

SHARD_IDX="${1:?need shard index (0..NUM_SHARDS-1)}"

# Robust conda init for non-interactive batch shells (Barkla2 miniforge)
source /opt/apps/pkg/tools/miniforge3/25.3.0_python3.12.10/etc/profile.d/conda.sh
conda activate /users/yjwang/.conda/envs/jbb_pair

python -V
which python


export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="/users/yjwang/scratch/hf"
export HF_HUB_CACHE="/users/yjwang/scratch/hf/hub"
export TRANSFORMERS_CACHE="/users/yjwang/scratch/hf/transformers"
export HF_OFFLOAD_DIR="/users/yjwang/scratch/hf_offload"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_OFFLOAD_DIR"

cd "$REPO"
export PAIR_DATA_ROOT="$REPO/data"
cp -f "$PAIR_DATA_ROOT/behaviours/jbb_manifest.jsonl" "$PAIR_DATA_ROOT/processed/jbb_manifest.jsonl"

# Log where this shard is starting
echo "[INFO] shard=${SHARD_IDX}/${NUM_SHARDS} host=$(hostname) start=$(date -Is)"

# Run generation for this shard
python -u scripts/run.py \
  --profile server \
  --config configs/server_mixtral_vicuna_5k.json \
  --subset harmful \
  --max-gpu-mem-util 0.5 \
  --budget-per-try 20 \
  --num-shards "$NUM_SHARDS" \
  --shard-idx "$SHARD_IDX" \
  --seed 0 \
  --fail-fast

echo "[INFO] shard=${SHARD_IDX} done end=$(date -Is)"

NEXT=$((SHARD_IDX + 1))
if [ "$NEXT" -lt "$NUM_SHARDS" ]; then
  echo "[INFO] submitting next shard $NEXT"
  sbatch sbatch_one_shard.sbatch "$NEXT" || echo "[WARN] could not submit next shard (QOS limit?)"
fi
