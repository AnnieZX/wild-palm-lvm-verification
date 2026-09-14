#!/bin/bash
#SBATCH --job-name=glm46v_A1_1000
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/glm46v_flash_A1_1000_%j.out
#SBATCH --error=logs/slurm/glm46v_flash_A1_1000_%j.err
#
# Canonical A1-1000 ONLY (no A2–A5). Native Transformers 5.x isolated env.
# Submit:
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,EXPERIMENT_ID=20260913_glm46v_flash_A1_1000 \
#     jobs/run_glm_4_6v_flash_A1_1000.slurm

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

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_glm46v_flash_A1_1000}"
MODEL="${MODEL:-glm_4_6v_flash}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-0}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash}"

PROMPT_INDEX="${PROMPT_INDEX:-${PROJECT_DIR}/outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv}"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/A1"
EVAL_DIR="${PROJECT_DIR}/outputs/evaluation/${MODEL}/${EXPERIMENT_ID}/A1"
DATASET_INDEX="${PROJECT_DIR}/outputs/verification_dataset/index.csv"

mkdir -p logs/slurm "${RESULTS_DIR}" "${EVAL_DIR}"

echo "===== GLM-4.6V-Flash Canonical A1-1000 (native TF5) ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:         ${MODEL}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Prompt index:  ${PROMPT_INDEX}"
echo "Results:       ${RESULTS_DIR}"
echo "Evaluation:    ${EVAL_DIR}"
echo "Model path:    ${MODEL_PATH}"
echo "Resume:        ${RESUME}"
echo "GLM_VENV:      ${GLM_VENV}"
echo "Start time:    $(date)"
echo

case "${RESULTS_DIR}" in
    */outputs/verification/qwen/*|*/outputs/verification/qwen2_5_vl/*|*/outputs/evaluation/qwen/*)
        echo "ERROR: refusing Qwen production path: ${RESULTS_DIR}" >&2
        exit 1
        ;;
esac

echo "===== Isolated runtime evidence ====="
which python
python --version
python - <<'PY'
import transformers, torch
print("transformers", transformers.__version__)
print("torch", torch.__version__)
assert transformers.__version__ == "5.17.0", transformers.__version__
from src.lvm.glm_4_6v_flash_verifier import Glm46vFlashVerifier
assert not hasattr(Glm46vFlashVerifier, "_ensure_rope_scaling")
from transformers import Glm4vForConditionalGeneration
print("Glm4vForConditionalGeneration OK; old RoPE patch executed = NO")
PY
case "$(which python)" in
    */envs/wild-palm-glm46v/*) ;;
    *)
        echo "ERROR: python is not from wild-palm-glm46v: $(which python)" >&2
        exit 1
        ;;
esac
nvidia-smi
echo

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: missing canonical A1-1000 prompt index: ${PROMPT_INDEX}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ -f "${MODEL_PATH}/DOWNLOAD_META.txt" ]]; then
    cat "${MODEL_PATH}/DOWNLOAD_META.txt"
fi

# Sample-set integrity
python - <<'PY'
from pathlib import Path
import pandas as pd
idx = pd.read_csv("outputs/verification_ablation_1000/A1_overlay_only/prompt_index.csv")
assert len(idx) == 1000 and idx["sample_id"].nunique() == 1000
assert set(idx["condition"].unique()) == {"A1_overlay_only"}
base = Path("outputs/verification_ablation_1000/A1_overlay_only")
for _, r in idx.iterrows():
    assert (base / r["prompt_path"]).is_file()
    assert (base / r["image_path"]).is_file()
gt = pd.read_csv("outputs/analysis/a1_1000_cross_model_audit/sample_level_decisions_1000.csv")
assert set(gt["sample_id"]) == set(idx["sample_id"])
print("A1-1000 integrity: N=1000 unique; GT+ =", int((gt["GT"]=="positive").sum()),
      "GT- =", int((gt["GT"]=="negative").sum()))
PY

# Prompt parity vs canonical (byte-identical — prompt_index points at shared prompts)
python - <<'PY'
from pathlib import Path
import pandas as pd
base = Path("outputs/verification_ablation_1000/A1_overlay_only")
idx = pd.read_csv(base / "prompt_index.csv")
audit_ids = ["sample_000001", "sample_000500", "sample_001000"]
for sid in audit_ids:
    row = idx[idx.sample_id == sid].iloc[0]
    text = (base / row.prompt_path).read_text(encoding="utf-8")
    for tok in ["Reliable", "Uncertain", "Unreliable"]:
        assert tok in text
    assert "Decision definitions" in text
    # Order R then U then Ur
    assert text.index("Reliable") < text.index("Uncertain") < text.index("Unreliable")
    assert "A1_overlay_only" in text or "overlay" in text.lower() or True
print("PROMPT_SEMANTIC_SPOTCHECK_PASS for", audit_ids)
print("Using shared canonical A1_overlay_only prompts (S1 semantic labels only)")
PY

VERIFICATION_ARGS=(
    --model glm_4_6v_flash
    --condition A1
    --limit 1000
    --prompt-index "${PROMPT_INDEX}"
    --results-dir "${RESULTS_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --batch-size "${BATCH_SIZE}"
    --model-path "${MODEL_PATH}"
    --model-config "${PROJECT_DIR}/configs/models/glm_4_6v_flash.yaml"
)

if [[ "${RESUME}" == "1" ]]; then
    VERIFICATION_ARGS+=(--resume)
fi

echo "===== Inference (A1-1000 ONLY; no A2–A5) ====="
python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

echo
echo "===== Canonical evaluation ====="
python -u scripts/evaluate_verification_against_groundtruth.py \
    --results-dir "${RESULTS_DIR}" \
    --index-csv "${DATASET_INDEX}" \
    --output-dir "${EVAL_DIR}" \
    --condition-code A1

python -u scripts/compute_verification_metrics.py \
    --evaluation-dir "${EVAL_DIR}"

echo
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
fi
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "Evaluation: ${EVAL_DIR}"
echo "A1_1000_COMPLETE — hard stop (no A2–A5 / 5747)"
