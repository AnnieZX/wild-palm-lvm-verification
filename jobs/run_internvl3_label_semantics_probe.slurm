#!/bin/bash
#SBATCH --job-name=internvl3_arr
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/internvl3_label_semantics_%j.out
#SBATCH --error=logs/slurm/internvl3_label_semantics_%j.err
# Submit with known-good GPU config from Stage 0/1:
#   sbatch --partition=gpu_small --gres=gpu:V100_32:1 \
#     jobs/run_internvl3_label_semantics_probe.slurm

set -euo pipefail

# ---------------------------------------------------------------------------
# InternVL3 label-semantics probe (ACCEPT / REVIEW / REJECT) — DIAGNOSTIC ONLY
# 20 samples from fixed balanced-100. Does NOT overwrite Stage 0/1.
# ---------------------------------------------------------------------------

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

EXPERIMENT_ID="${EXPERIMENT_ID:-20260909_internvl3_qual}"
PROBE_ROOT="${PROJECT_DIR}/outputs/diagnostics/model_qualification/internvl_label_semantics_20"
PROMPT_INDEX="${PROBE_ROOT}/prompt_index.csv"
# Isolated results under diagnostics (not Stage0/Stage1 verification paths)
RESULTS_DIR="${PROBE_ROOT}/results_${EXPERIMENT_ID}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/InternVL3-8B-Instruct}"

mkdir -p logs/slurm "${RESULTS_DIR}"

echo "===== InternVL3 Label-Semantics Probe (ACCEPT/REVIEW/REJECT) ====="
echo "DIAGNOSTIC ONLY — not official qualification"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-unknown}"
echo "Experiment:   ${EXPERIMENT_ID}"
echo "Prompt index: ${PROMPT_INDEX}"
echo "Results:      ${RESULTS_DIR}"
echo "Model:        ${MODEL_PATH}"
echo "Start:        $(date)"
echo

python --version
python -c "import transformers; print('transformers', transformers.__version__)"
nvidia-smi
echo

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: missing ${PROMPT_INDEX}" >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing model ${MODEL_PATH}" >&2
    exit 1
fi

# Guard: never write into Stage0/Stage1 dirs
case "${RESULTS_DIR}" in
    */stage0|*/balanced100|*/stage0/*|*/balanced100/*)
        echo "ERROR: refusing Stage0/Stage1 path: ${RESULTS_DIR}" >&2
        exit 1
        ;;
esac

PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
python -u scripts/diagnostics/run_internvl_label_semantics_probe.py \
    --prompt-index "${PROMPT_INDEX}" \
    --results-dir "${RESULTS_DIR}" \
    --model-path "${MODEL_PATH}" \
    --experiment-id "${EXPERIMENT_ID}_label_semantics_ARR" \
    --max-new-tokens 512

echo
echo "End: $(date)"
echo "Results: ${RESULTS_DIR}"
