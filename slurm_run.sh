#!/bin/bash
#SBATCH --job-name=jbb-run
#SBATCH --output=/users/%u/logs/%x-%j.out
#SBATCH --error=/users/%u/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --gres=gpu:1

set -euo pipefail
module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jbb_pair

export HF_HOME=$HOME/work/jbb_pair/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HOME/work/jbb_pair/datasets
export WANDB_DIR=$HOME/work/jbb_pair/wandb_cache
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR"

cd /users/$USER/repos/JBB_PAIR_LocalLLMs

# Example: run your generation entrypoint (adjust to your repo)
python -u scripts/run.py "$@"
