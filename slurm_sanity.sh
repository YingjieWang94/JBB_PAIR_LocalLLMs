#!/bin/bash
#SBATCH --job-name=sanity
#SBATCH --output=/users/%u/logs/%x-%j.out
#SBATCH --error=/users/%u/logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --gres=gpu:1

set -euo pipefail
module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jbb_pair

# inside sbatch_sanity.sh
if [ -f "$HOME/.hf_env" ]; then
  source "$HOME/.hf_env"
fi


cd /users/$USER/repos/JBB_PAIR_LocalLLMs
python -u sanity_check.py
