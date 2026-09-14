#!/bin/bash
#SBATCH --job-name=glm46v_dl
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/glm46v_flash_download_%j.out
#SBATCH --error=logs/slurm/glm46v_flash_download_%j.err

set -euo pipefail

MODEL_ID="zai-org/GLM-4.6V-Flash"
MODEL_DIR="/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash"
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

model_id = "zai-org/GLM-4.6V-Flash"
model_dir = "/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash"

api = HfApi()
info = api.model_info(model_id)
rev = getattr(info, "sha", None) or getattr(info, "id", None)
print(f"HF repo: {model_id}")
print(f"Revision/commit: {rev}")
print(f"License: {getattr(info, 'license', None)}")

path = snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"Download complete: {path}")

meta = Path(model_dir) / "DOWNLOAD_META.txt"
meta.write_text(
    f"repo_id={model_id}\n"
    f"revision={rev}\n"
    f"local_dir={model_dir}\n"
    f"downloaded_path={path}\n"
    f"license={getattr(info, 'license', None)}\n",
    encoding="utf-8",
)
print(meta.read_text())

# Completeness check
required = ["config.json", "tokenizer_config.json"]
missing = [name for name in required if not (Path(model_dir) / name).exists()]
weight_files = list(Path(model_dir).glob("*.safetensors")) + list(Path(model_dir).glob("*.bin"))
print(f"weight_files={len(weight_files)}")
if missing:
    raise SystemExit(f"Missing required files: {missing}")
if not weight_files:
    raise SystemExit("No weight files (*.safetensors / *.bin) found")
print("CHECKPOINT_COMPLETE")
PY

echo "End: $(date)"
ls -lah "${MODEL_DIR}" | head -40
du -sh "${MODEL_DIR}"
