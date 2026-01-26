#!/bin/bash
#SBATCH --job-name=gpu-probe
#SBATCH --output=/users/%u/logs/%x-%j.out
#SBATCH --error=/users/%u/logs/%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p /users/$USER/logs

module purge
module load miniforge3/25.3.0-python3.12.10

# Robust conda activation for non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jbb_pair

echo "HOST: $(hostname)"
echo "WHICH PYTHON: $(which python)"
python -V

nvidia-smi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.randn(1024,1024, device="cuda")
    y = x @ x
    print("matmul ok:", float(y.mean()))
PY
