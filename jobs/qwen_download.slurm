#!/bin/bash
#SBATCH --job-name=qwen_download
#SBATCH --partition=yangGrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qwen_download-%j.out
#SBATCH --error=logs/qwen_download-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Download Qwen2.5-VL-7B-Instruct to group model storage on DEAC.
# Submit from the project root:
#   mkdir -p logs
#   sbatch jobs/qwen_download.slurm
# ---------------------------------------------------------------------------

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_DIR="/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Model ID: ${MODEL_ID}"
echo "Target directory: ${MODEL_DIR}"

mkdir -p "${MODEL_DIR}"
mkdir -p logs

# Activate conda environment with huggingface_hub installed.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate palm-lvm

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
