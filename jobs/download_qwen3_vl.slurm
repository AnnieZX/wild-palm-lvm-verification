#!/bin/bash
#SBATCH --job-name=qwen3vl_dl
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/qwen3_vl_download_%j.out
#SBATCH --error=logs/slurm/qwen3_vl_download_%j.err
#
# Download Qwen/Qwen3-VL-8B-Instruct to the canonical local path.
#   sbatch jobs/download_qwen3_vl.slurm

set -euo pipefail

MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
MODEL_DIR="/deac/csc/yangGrp/luoz23/models/Qwen3-VL-8B-Instruct"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs/slurm "${MODEL_DIR}"

QWEN3_VENV="${QWEN3_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-qwen3vl}"
module load apps/python/3.11.8 2>/dev/null || true
if [[ -x "${QWEN3_VENV}/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${QWEN3_VENV}/bin/activate"
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH || true
  export PATH="${QWEN3_VENV}/bin:${PATH}"
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

model_id = "Qwen/Qwen3-VL-8B-Instruct"
model_dir = Path("/deac/csc/yangGrp/luoz23/models/Qwen3-VL-8B-Instruct")
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
]
missing = [name for name in required if not (model_dir / name).exists()]
weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("model-*.safetensors"))
if missing:
    raise SystemExit(f"Missing required files: {missing}")
if not weight_files:
    raise SystemExit("No safetensors weight files found")
print(f"Weight files: {len(weight_files)}")
print("DOWNLOAD_OK")
PY

du -sh "${MODEL_DIR}"
echo "End: $(date)"
