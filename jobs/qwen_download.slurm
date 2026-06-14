#!/bin/bash
#SBATCH --job-name=qwen_download
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qwen_download_%j.out
#SBATCH --error=logs/qwen_download_%j.err

set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_DIR="/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Model ID: ${MODEL_ID}"
echo "Target directory: ${MODEL_DIR}"

mkdir -p "${MODEL_DIR}"
mkdir -p logs

echo "Python: $(which python)"
python --version

python - <<'PY'
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model_dir = "/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct"

print(f"Downloading {model_id} -> {model_dir}")

snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)

print("Download complete.")
PY

echo "End time: $(date)"
echo "Model saved under: ${MODEL_DIR}"