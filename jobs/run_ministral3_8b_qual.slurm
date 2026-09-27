#!/bin/bash
#SBATCH --job-name=ministral3_qual
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/ministral3_8b_qual_%j.out
#SBATCH --error=logs/slurm/ministral3_8b_qual_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Ministral 3 8B Instruct qualification (engineering only).
#
#   STAGE=sanity10  fixed Stage 0 set outputs/diagnostics/model_qualification/stage0_A1_10/
#   STAGE=smoke     first SMOKE_LIMIT (default 4) samples of each A1..A5 from
#                   outputs/verification_ablation_1000/<condition>/prompt_index.csv
#
# Submit from project root:
#   STAGE=sanity10 EXPERIMENT_ID=YYYYMMDD_ministral3_8b_qual sbatch \
#     --partition=yangGrp --nodelist=lovelace --gres=gpu:L40S:1 \
#     jobs/run_ministral3_8b_qual.slurm
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

STAGE="${STAGE:?Set STAGE=sanity10|smoke}"
MODEL="ministral3_8b"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_ministral3_8b_qual}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SMOKE_LIMIT="${SMOKE_LIMIT:-4}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/Ministral-3-8B-Instruct-2512-BF16}"
MODEL_CONFIG="${PROJECT_DIR}/configs/models/${MODEL}.yaml"
VENV="${MINISTRAL3_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-gemma4}"
EXP_ROOT="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}"

mkdir -p logs/slurm "${EXP_ROOT}"

module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PYTHONPATH="${PROJECT_DIR}"

echo "===== Ministral 3 8B qualification: ${STAGE} ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Model path:    ${MODEL_PATH}"
echo "Venv:          ${VENV}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Start time:    $(date)"
cat "${MODEL_PATH}/DOWNLOAD_META.txt"
which python
python --version
python - <<'PY'
import torch, transformers
from transformers import AutoProcessor, Mistral3ForConditionalGeneration  # noqa: F401
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
nvidia-smi
echo

MEM_LOG="${EXP_ROOT}/gpu_mem_${STAGE}_${SLURM_JOB_ID:-local}.csv"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader -l 2 > "${MEM_LOG}" &
MEM_PID=$!
trap 'kill ${MEM_PID} 2>/dev/null || true' EXIT

run_one() {
    local condition="$1" prompt_index="$2" results_dir="$3" limit="$4"
    if [[ -f "${results_dir}/results_index.csv" ]]; then
        echo "ERROR: ${results_dir}/results_index.csv exists; choose a new EXPERIMENT_ID." >&2
        exit 1
    fi
    mkdir -p "${results_dir}"
    python -u scripts/run_verification.py \
        --model "${MODEL}" \
        --condition "${condition}" \
        --prompt-index "${prompt_index}" \
        --results-dir "${results_dir}" \
        --experiment-id "${EXPERIMENT_ID}" \
        --batch-size "${BATCH_SIZE}" \
        --model-config "${MODEL_CONFIG}" \
        --model-path "${MODEL_PATH}" \
        --limit "${limit}"
}

case "${STAGE}" in
    sanity10)
        run_one stage0 \
            "${PROJECT_DIR}/outputs/diagnostics/model_qualification/stage0_A1_10/prompt_index.csv" \
            "${EXP_ROOT}/stage0" 10
        ;;
    smoke)
        for cond in A1_overlay_only A2_overlay_confidence A3_overlay_confidence_geometry \
                    A4_overlay_crop_confidence A5_crop_only; do
            code="${cond%%_*}"
            run_one "${code}" \
                "${PROJECT_DIR}/outputs/verification_ablation_1000/${cond}/prompt_index.csv" \
                "${EXP_ROOT}/smoke/${code}" "${SMOKE_LIMIT}"
        done
        ;;
    *)
        echo "ERROR: STAGE must be sanity10|smoke (got ${STAGE})" >&2
        exit 1
        ;;
esac

kill "${MEM_PID}" 2>/dev/null || true
echo
echo "Peak GPU memory (MiB): $(awk -F', ' '{gsub(/ MiB/,"",$3); if ($3>m) m=$3} END {print m}' "${MEM_LOG}")"
echo "End time: $(date)"
echo "MINISTRAL3_${STAGE}_COMPLETE"
