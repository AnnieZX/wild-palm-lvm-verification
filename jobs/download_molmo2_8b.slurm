#!/bin/bash
#SBATCH --job-name=molmo2_dl
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/molmo2_8b_download_%j.out
#SBATCH --error=logs/slurm/molmo2_8b_download_%j.err

# Download allenai/Molmo2-8B to the canonical cluster path.
# Do NOT submit as part of integration-only work unless explicitly requested.

set -euo pipefail

MODEL_ID="allenai/Molmo2-8B"
MODEL_DIR="/deac/csc/yangGrp/luoz23/models/Molmo2-8B"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs/slurm "${MODEL_DIR}"

MOLMO2_VENV="${MOLMO2_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-molmo2}"
if [[ -d "${MOLMO2_VENV}" ]]; then
    module load apps/python/3.11.8 2>/dev/null || true
    # shellcheck disable=SC1091
    source "${MOLMO2_VENV}/bin/activate"
    export PYTHONNOUSERSITE=1
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date)"
echo "Model ID: ${MODEL_ID}"
echo "Target: ${MODEL_DIR}"
python --version

python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

model_id = "allenai/Molmo2-8B"
model_dir = "/deac/csc/yangGrp/luoz23/models/Molmo2-8B"

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

# Completeness check (remote-code models also ship modeling_*.py).
required = ["config.json", "tokenizer_config.json"]
missing = [name for name in required if not (Path(model_dir) / name).exists()]
weight_files = list(Path(model_dir).glob("*.safetensors")) + list(Path(model_dir).glob("*.bin"))
remote_code = list(Path(model_dir).glob("*.py"))
print(f"weight_files={len(weight_files)}")
print(f"remote_code_py={len(remote_code)}")
if missing:
    raise SystemExit(f"Missing required files: {missing}")
if not weight_files:
    raise SystemExit("No weight files (*.safetensors / *.bin) found")
print("CHECKPOINT_COMPLETE")
PY

echo "End: $(date)"
ls -lah "${MODEL_DIR}" | head -40
du -sh "${MODEL_DIR}"
