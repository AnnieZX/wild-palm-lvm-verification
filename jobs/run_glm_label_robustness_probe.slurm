#!/bin/bash
#SBATCH --job-name=glm46v_labrob
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/glm46v_label_robustness_%j.out
#SBATCH --error=logs/slurm/glm46v_label_robustness_%j.err
#
# DIAGNOSTIC ONLY — 20 samples × 4 conditions = 80 generations.
# Submit:
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     jobs/run_glm_label_robustness_probe.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

GLM_VENV="${GLM_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-glm46v}"
module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source "${GLM_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PATH="${GLM_VENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}"

OUT_ROOT="${PROJECT_DIR}/outputs/diagnostics/model_qualification/glm46v_label_robustness_20"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash}"

mkdir -p logs/slurm

echo "===== GLM-4.6V-Flash Label-Robustness Diagnostic ====="
echo "DIAGNOSTIC ONLY — isolated under outputs/diagnostics/"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Out root:     ${OUT_ROOT}"
echo "Model:        ${MODEL_PATH}"
echo "GLM_VENV:     ${GLM_VENV}"
echo "Start:        $(date)"
echo

echo "===== Isolated runtime evidence ====="
which python
python --version
python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
case "$(which python)" in
    */envs/wild-palm-glm46v/*) ;;
    *)
        echo "ERROR: python is not from wild-palm-glm46v" >&2
        exit 1
        ;;
esac
python - <<'PY'
import transformers
from src.lvm.glm_4_6v_flash_verifier import Glm46vFlashVerifier
assert transformers.__version__ == "5.17.0"
assert not hasattr(Glm46vFlashVerifier, "_ensure_rope_scaling")
print("Glm4v OK; old RoPE patch executed = NO")
PY
nvidia-smi
echo

if [[ ! -f "${OUT_ROOT}/prompt_index.csv" ]]; then
    echo "ERROR: missing prepared prompt_index at ${OUT_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${OUT_ROOT}/prompt_parity_audit/VERDICT.json" ]]; then
    echo "ERROR: missing prompt-parity VERDICT; audit first." >&2
    exit 1
fi
python - <<'PY'
import json, sys
from pathlib import Path
v = json.loads(Path("outputs/diagnostics/model_qualification/glm46v_label_robustness_20/prompt_parity_audit/VERDICT.json").read_text())
if not v.get("prompt_parity_ok"):
    print("PROMPT PARITY FAILED — refusing to run.", file=sys.stderr)
    sys.exit(2)
print("Prompt parity OK")
PY

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing model ${MODEL_PATH}" >&2
    exit 1
fi

case "${OUT_ROOT}" in
    */outputs/diagnostics/*) ;;
    *)
        echo "ERROR: refusing non-diagnostics path: ${OUT_ROOT}" >&2
        exit 1
        ;;
esac

echo "===== Inference (80 cases) ====="
python -u scripts/diagnostics/run_glm_label_robustness_probe.py \
    --out-root "${OUT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --no-skip-existing

echo
echo "===== Evaluation ====="
python -u scripts/diagnostics/evaluate_glm_label_robustness_probe.py \
    --out-root "${OUT_ROOT}"

echo
echo "End: $(date)"
echo "Results: ${OUT_ROOT}"
echo "ROBUSTNESS_DIAGNOSTIC_COMPLETE — hard stop (no A1-1000)"
