#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --output=/users/%u/logs/%x-%j.out
#SBATCH --error=/users/%u/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --gres=gpu:1

set -euo pipefail
module purge
module load miniforge3/25.3.0-python3.12.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jbb_pair

export HF_HOME=$HOME/work/jbb_pair/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

MODEL="${1:-lmsys/vicuna-13b-v1.5}"
PORT="${2:-8000}"

echo "HOST=$(hostname)"
echo "MODEL=$MODEL"
echo "PORT=$PORT"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --max-model-len 4096
