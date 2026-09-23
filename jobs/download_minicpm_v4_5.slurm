#!/bin/bash
#SBATCH --job-name=minicpm45_dl
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/minicpm_v4_5_download_%j.out
#SBATCH --error=logs/slurm/minicpm_v4_5_download_%j.err
#
# Download openbmb/MiniCPM-V-4_5 to the canonical local path.
#   sbatch jobs/download_minicpm_v4_5.slurm
#
# Do NOT submit while Qwen3-VL full-scale jobs need cluster priority unless needed.
# Prefer an interactive/login download if the checkpoint is already in progress.

set -euo pipefail

MODEL_ID="openbmb/MiniCPM-V-4_5"
MODEL_DIR="/deac/csc/yangGrp/luoz23/models/MiniCPM-V-4_5"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs/slurm "${MODEL_DIR}"

MINICPM_VENV="${MINICPM_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-minicpm45}"
module load apps/python/3.11.8 2>/dev/null || true
if [[ -x "${MINICPM_VENV}/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${MINICPM_VENV}/bin/activate"
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH || true
  export PATH="${MINICPM_VENV}/bin:${PATH}"
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date)"
echo "Model ID: ${MODEL_ID}"
echo "Target: ${MODEL_DIR}"
which python
python --version

python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

model_id = "openbmb/MiniCPM-V-4_5"
model_dir = Path("/deac/csc/yangGrp/luoz23/models/MiniCPM-V-4_5")
model_dir.mkdir(parents=True, exist_ok=True)

api = HfApi()
info = api.model_info(model_id)
rev = getattr(info, "sha", None) or getattr(info, "id", None)
print(f"HF repo: {model_id}")
print(f"Revision/commit: {rev}")
print(f"License: {getattr(info, 'license', None)}")

path = snapshot_download(
    repo_id=model_id,
    local_dir=str(model_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"Download complete: {path}")

meta = model_dir / "DOWNLOAD_META.txt"
meta.write_text(
    f"repo_id={model_id}\n"
    f"revision={rev}\n"
    f"local_dir={model_dir}\n"
    f"downloaded_path={path}\n"
    f"license={getattr(info, 'license', None)}\n",
    encoding="utf-8",
)
print(meta.read_text())

required = [
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "model.safetensors.index.json",
    "modeling_minicpmv.py",
]
missing = [name for name in required if not (model_dir / name).exists()]
weight_files = list(model_dir.glob("*.safetensors"))
if missing:
    raise SystemExit(f"Missing required files: {missing}")
if not weight_files:
    raise SystemExit("No safetensors weight files found")
print(f"Weight files: {len(weight_files)}")
print("DOWNLOAD_OK")
PY

du -sh "${MODEL_DIR}"
echo "End: $(date)"
