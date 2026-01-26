#!/bin/bash -l
#SBATCH -J pair4
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --array=0-3

set -euo pipefail
set -x

module purge
module load cuda/12.8.0 || true

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"

# Ensure HF token is present for gated Meta repos
if [ -f "$HOME/.hf_env" ]; then
  source "$HOME/.hf_env"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Conda (portable)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [ -n "${ENV_PATH:-}" ]; then
    conda activate "$ENV_PATH"
  else
    conda activate "${ENV_NAME:-jbb_pair}"
  fi
fi

echo "Node: $(hostname)"
echo "Job:  ${SLURM_JOB_ID}  ArrayTask: ${SLURM_ARRAY_TASK_ID}"
nvidia-smi || true

unset HF_TOKEN
unset HUGGINGFACEHUB_API_TOKEN
unset HUGGINGFACE_HUB_TOKEN


export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"


# Optional: quick dep check (keeps failures obvious)
python -c "import orjson, torch; print('deps ok; cuda:', torch.cuda.is_available())"

# Run one shard per array task
python -u scripts/run.py \
  --profile server \
  --subset harmful \
  --dtype bf16 \
  --device cuda \
  --guard-device cuda \
  --use-cache true \
  --max-behaviors 200 \
  --budget-per-try 10 \
  --num-shards 4 \
  --shard-idx ${SLURM_ARRAY_TASK_ID} \
  --log-every 5

echo "Done. Latest generated files:"
ls -lt data/generated | head -n 20
