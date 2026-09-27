#!/bin/bash
#SBATCH --job-name=gemma4_env
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/gemma4_env_setup_%j.out
#SBATCH --error=logs/slurm/gemma4_env_setup_%j.err

set -euo pipefail

# Isolated env for Gemma 4 12B Unified (transformers >= 5.10.1).
# Pins match the known-good multimodal stack on this cluster (GLM uses 5.17.0).

GEMMA4_VENV="${GEMMA4_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-gemma4}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs/slurm

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date)"
echo "Target venv: ${GEMMA4_VENV}"

module load apps/python/3.11.8

if [[ ! -d "${GEMMA4_VENV}" ]]; then
    python -m venv "${GEMMA4_VENV}"
fi

# shellcheck disable=SC1091
source "${GEMMA4_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

python -m pip install --upgrade pip wheel

# Match molmo2/glm CUDA torch pins on this cluster.
python -m pip install \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
    "transformers==5.17.0" \
    "accelerate==1.15.0" \
    "pillow==12.3.0" \
    "huggingface_hub>=0.36.0" \
    "safetensors>=0.8.0" \
    "pandas>=2.0" \
    "pyyaml" \
    "tqdm"

python - <<'PY'
import json
from pathlib import Path
import transformers
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: F401

venv = Path("/deac/csc/yangGrp/luoz23/envs/wild-palm-gemma4")
assert transformers.__version__ == "5.17.0", transformers.__version__
print("GEMMA4_ENV_OK", transformers.__version__, torch.__version__)

(venv / "VERSION_PIN.txt").write_text(
    "transformers==5.17.0\n"
    "torch==2.11.0+cu128\n"
    "torchvision==0.26.0+cu128\n"
    "accelerate==1.15.0\n"
    "pillow==12.3.0\n"
    "min_transformers>=5.10.1 (Gemma 4 / AutoModelForMultimodalLM)\n",
    encoding="utf-8",
)
versions = {
    "python": "3.11.8",
    "executable": str(venv / "bin" / "python"),
    "transformers": transformers.__version__,
    "torch": torch.__version__,
}
(venv / "ENV_VERSIONS.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
print(versions)
PY

echo "End: $(date)"
which python
python --version
