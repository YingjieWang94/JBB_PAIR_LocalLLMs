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

cd "$SLURM_SUBMIT_DIR"

# Ensure HF token is present for gated Meta repos
if [ -f "$HOME/.hf_env" ]; then
  source "$HOME/.hf_env"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

source /opt/apps/pkg/tools/miniforge3/25.3.0_python3.12.10/etc/profile.d/conda.sh
conda activate /users/yjwang/.conda/envs/jbb_pair

echo "Node: $(hostname)"
echo "Job:  ${SLURM_JOB_ID}  ArrayTask: ${SLURM_ARRAY_TASK_ID}"
nvidia-smi || true

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
