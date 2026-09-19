#!/bin/bash
#SBATCH --job-name=phi4_gate0
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/phi4_gate0_%j.out
#SBATCH --error=logs/slurm/phi4_gate0_%j.err
#
# Gate 0: Phi-4 Multimodal Instruct load + one-image generation smoke.
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 jobs/run_phi4_gate0_load.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

PHI_VENV="${PHI_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-phi4}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/Phi-4-multimodal-instruct}"
IMAGE_PATH="${IMAGE_PATH:-${PROJECT_DIR}/outputs/verification_dataset/images/sample_000001.png}"

module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source "${PHI_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PATH="${PHI_VENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}"

mkdir -p logs/slurm

echo "===== Phi-4 Gate 0 Load Test ====="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PHI_VENV:     ${PHI_VENV}"
echo "Model path:   ${MODEL_PATH}"
echo "Image:        ${IMAGE_PATH}"
echo "Start time:   $(date)"
echo

case "$(which python)" in
    */envs/wild-palm-phi4/*) ;;
    *)
        echo "ERROR: python is not from wild-palm-phi4: $(which python)" >&2
        exit 1
        ;;
esac

which python
python --version
python - <<'PY'
import json
from pathlib import Path
versions = json.loads(Path("/deac/csc/yangGrp/luoz23/envs/wild-palm-phi4/ENV_VERSIONS.json").read_text())
print(json.dumps(versions, indent=2))
PY
nvidia-smi
echo

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "ERROR: missing image: ${IMAGE_PATH}" >&2
    exit 1
fi

python -u - <<PY
import torch
from pathlib import Path
from src.lvm.phi4_multimodal_verifier import Phi4MultimodalVerifier

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model_path = "${MODEL_PATH}"
image_path = Path("${IMAGE_PATH}")
verifier = Phi4MultimodalVerifier(
    model_name=model_path,
    device_map="auto",
    attn_implementation="eager",  # flash_attn not installed in this env
)
prompt = "Reply with a single short sentence describing the image."
text = verifier.generate_response(
    image_path=image_path,
    prompt=prompt,
    max_new_tokens=64,
)
print("GATE0_GENERATION_OK")
print("RAW:", repr(text[:500]))
if not text.strip():
    raise SystemExit("Empty generation")
if torch.cuda.is_available():
    print("peak_allocated_gb:", round(torch.cuda.max_memory_allocated() / (1024**3), 3))
    print("peak_reserved_gb:", round(torch.cuda.max_memory_reserved() / (1024**3), 3))
print("GATE0_PASS")
PY

echo "End time: $(date)"
