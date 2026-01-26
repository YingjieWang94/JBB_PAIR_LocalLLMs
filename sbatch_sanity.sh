#!/bin/bash -l
#SBATCH -J pair-sanity
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:h100:1

set -euo pipefail
set -x

module purge
module load cuda/12.8.0 || true

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

source /opt/apps/pkg/tools/miniforge3/25.3.0_python3.12.10/etc/profile.d/conda.sh
conda activate /users/yjwang/.conda/envs/jbb_pair

echo "Node: $(hostname)"
echo "Submit dir: $SLURM_SUBMIT_DIR"
python -V
which python
nvidia-smi

python -c "import sys; print('sys.path[0:3]=', sys.path[0:3])"
python -m py_compile scripts/run.py src/judges/llama3.py

python -u scripts/run.py \
  --profile server \
  --subset harmful \
  --max-behaviors 2 \
  --budget-per-try 2 \
  --dtype bf16 \
  --device cuda \
  --guard-device cuda \
  --use-cache true \
  --log-every 1

echo "Generated files:"
ls -lh data/generated | tail -n 50
