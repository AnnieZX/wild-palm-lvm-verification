#!/bin/bash
#SBATCH --job-name=gemma4_dl
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/gemma4_12b_it_download_%j.out
#SBATCH --error=logs/slurm/gemma4_12b_it_download_%j.err

set -euo pipefail

MODEL_ID="google/gemma-4-12B-it"
MODEL_DIR="/deac/csc/yangGrp/luoz23/models/gemma-4-12B-it"
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

model_id = "google/gemma-4-12B-it"
model_dir = "/deac/csc/yangGrp/luoz23/models/gemma-4-12B-it"

api = HfApi()
info = api.model_info(model_id)
rev = getattr(info, "sha", None) or getattr(info, "id", None)
gated = getattr(info, "gated", None)
print(f"HF repo: {model_id}")
print(f"Revision/commit: {rev}")
print(f"gated: {gated}")
print(f"private: {getattr(info, 'private', None)}")

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
    f"gated={gated}\n"
    f"local_dir={model_dir}\n"
    f"downloaded_path={path}\n",
    encoding="utf-8",
)
print(meta.read_text())

cfg = Path(model_dir) / "config.json"
if cfg.exists():
    import json
    data = json.loads(cfg.read_text(encoding="utf-8"))
    print("architectures:", data.get("architectures"))
    print("model_type:", data.get("model_type"))
    print("transformers_version:", data.get("transformers_version"))
PY

echo "End: $(date)"
ls -lah "${MODEL_DIR}" | head -40
du -sh "${MODEL_DIR}"
