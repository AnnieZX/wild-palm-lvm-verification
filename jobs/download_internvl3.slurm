#!/bin/bash
#SBATCH --job-name=internvl3_dl
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/internvl3_download_%j.out
#SBATCH --error=logs/slurm/internvl3_download_%j.err

set -euo pipefail

MODEL_ID="OpenGVLab/InternVL3-8B-Instruct"
MODEL_DIR="/deac/csc/yangGrp/luoz23/models/InternVL3-8B-Instruct"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs/slurm "${MODEL_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date)"
echo "Model ID: ${MODEL_ID}"
echo "Target: ${MODEL_DIR}"
python --version

python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

model_id = "OpenGVLab/InternVL3-8B-Instruct"
model_dir = "/deac/csc/yangGrp/luoz23/models/InternVL3-8B-Instruct"

api = HfApi()
info = api.model_info(model_id)
rev = getattr(info, "sha", None) or getattr(info, "id", None)
print(f"HF repo: {model_id}")
print(f"Revision/commit: {rev}")

path = snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"Download complete: {path}")

# record revision
meta = Path(model_dir) / "DOWNLOAD_META.txt"
meta.write_text(
    f"repo_id={model_id}\n"
    f"revision={rev}\n"
    f"local_dir={model_dir}\n"
    f"downloaded_path={path}\n",
    encoding="utf-8",
)
print(meta.read_text())
PY

echo "End: $(date)"
# list key files
ls -lah "${MODEL_DIR}" | head -40
du -sh "${MODEL_DIR}"
