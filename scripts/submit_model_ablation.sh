#!/usr/bin/env bash
# Submit (or dry-run) A1–A5 local verification jobs for ONE explicit model.
#
# Usage:
#   ./scripts/submit_model_ablation.sh internvl3_8b
#   ./scripts/submit_model_ablation.sh qwen2_5_vl --conditions A1 --limit 20
#   DRY_RUN=1 ./scripts/submit_model_ablation.sh phi4_multimodal --limit 1000
#   SUBMIT=1 ./scripts/submit_model_ablation.sh glm_4_6v_flash
#
# Defaults:
#   DRY_RUN=1  (prints sbatch commands; does NOT submit)
#   SUBMIT=1   to actually submit (also set DRY_RUN=0)
#
# Does NOT auto-submit every model. Model argument is required.
# Does NOT download checkpoints or run inference locally.
# GPU concurrency is left to Slurm (one L40S per job by default).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/jobs/lib/model_runtime.sh"

usage() {
    cat <<'EOF'
Usage: ./scripts/submit_model_ablation.sh MODEL [options]

Required:
  MODEL                 Registry key or alias (e.g. qwen2_5_vl, internvl3_8b, phi4)

Options:
  --conditions LIST     Comma-separated A1-A5 (default: A1,A2,A3,A4,A5)
  --limit N             Samples per job (default: 1000)
  --experiment-id ID    Shared experiment id (default: timestamp_model_ablation_limit)
  --ablation-size N     Ablation input set under verification_ablation_N (default: 1000)
  --batch-size N        Override default batch size
  --resume              Pass RESUME=1
  --no-eval             Skip evaluation step in the job
  --partition NAME      Slurm partition (default: yangGrp)
  --gres SPEC           Slurm GRES (default: gpu:L40S:1)
  --time HH:MM:SS       Walltime (default: inferred from --limit)
  --mem SIZE            Memory (default: 96G)
  --dry-run             Print commands only (default)
  --submit              Actually sbatch
  -h, --help            Show this help

Environment:
  DRY_RUN=1|0   SUBMIT=1|0   EXPERIMENT_ID=...   LIMIT=...   CONDITIONS=A1,A2
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

MODEL_RAW="$1"
shift

CONDITIONS_CSV="${CONDITIONS:-A1,A2,A3,A4,A5}"
LIMIT="${LIMIT:-1000}"
ABLATION_SAMPLE_SIZE="${ABLATION_SAMPLE_SIZE:-1000}"
PARTITION="${PARTITION:-yangGrp}"
GRES="${GRES:-gpu:L40S:1}"
MEM="${MEM:-96G}"
RESUME="${RESUME:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
BATCH_SIZE="${BATCH_SIZE:-}"
TIME_LIMIT="${TIME_LIMIT:-}"
DRY_RUN="${DRY_RUN:-1}"
SUBMIT="${SUBMIT:-0}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conditions)
            CONDITIONS_CSV="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --experiment-id)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        --ablation-size)
            ABLATION_SAMPLE_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --no-eval)
            RUN_EVAL=0
            shift
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --gres)
            GRES="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            SUBMIT=0
            shift
            ;;
        --submit)
            SUBMIT=1
            DRY_RUN=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

MODEL="$(canonicalize_model_key "${MODEL_RAW}")"
if [[ -z "${BATCH_SIZE}" ]]; then
    BATCH_SIZE="$(model_default_batch_size "${MODEL}")"
fi
if [[ -z "${TIME_LIMIT}" ]]; then
    TIME_LIMIT="$(suggest_time_limit "${LIMIT}")"
fi
if [[ -z "${EXPERIMENT_ID}" ]]; then
    EXPERIMENT_ID="$(date +%Y%m%d_%H%M)_${MODEL}_ablation_${LIMIT}"
fi

IFS=',' read -r -a CONDITIONS <<< "${CONDITIONS_CSV}"
for c in "${CONDITIONS[@]}"; do
    condition_dir_name "${c}" >/dev/null
done

mkdir -p logs/slurm

echo "MODEL=${MODEL} (from ${MODEL_RAW})"
echo "EXPERIMENT_ID=${EXPERIMENT_ID}"
echo "CONDITIONS=${CONDITIONS[*]}"
echo "LIMIT=${LIMIT}"
echo "ABLATION_SAMPLE_SIZE=${ABLATION_SAMPLE_SIZE}"
echo "PARTITION=${PARTITION} GRES=${GRES} MEM=${MEM} TIME=${TIME_LIMIT}"
echo "BATCH_SIZE=${BATCH_SIZE} RESUME=${RESUME} RUN_EVAL=${RUN_EVAL}"
echo "RESULTS_ROOT=outputs/verification/${MODEL}/${EXPERIMENT_ID}"
echo "DRY_RUN=${DRY_RUN} SUBMIT=${SUBMIT}"
echo

if [[ -d "outputs/verification/${MODEL}/${EXPERIMENT_ID}" ]] && [[ "${RESUME}" != "1" ]]; then
    echo "WARNING: experiment dir already exists: outputs/verification/${MODEL}/${EXPERIMENT_ID}" >&2
    echo "         Jobs will refuse non-empty results unless RESUME=1." >&2
fi

for CONDITION in "${CONDITIONS[@]}"; do
    JOB_NAME="${MODEL}_${CONDITION}_${LIMIT}"
    EXPORT_VARS="ALL,MODEL=${MODEL},CONDITION=${CONDITION},LIMIT=${LIMIT},EXPERIMENT_ID=${EXPERIMENT_ID},BATCH_SIZE=${BATCH_SIZE},RESUME=${RESUME},RUN_EVAL=${RUN_EVAL},ABLATION_SAMPLE_SIZE=${ABLATION_SAMPLE_SIZE}"

    cmd=(
        sbatch
        --partition="${PARTITION}"
        --gres="${GRES}"
        --mem="${MEM}"
        --time="${TIME_LIMIT}"
        --account=yanggrp
        --job-name="${JOB_NAME}"
        --output="logs/slurm/${JOB_NAME}_%j.out"
        --error="logs/slurm/${JOB_NAME}_%j.err"
        --export="${EXPORT_VARS}"
        jobs/run_verification.slurm
    )

    echo "${cmd[*]}"
    if [[ "${SUBMIT}" == "1" && "${DRY_RUN}" != "1" ]]; then
        "${cmd[@]}"
    fi
done

if [[ "${SUBMIT}" != "1" || "${DRY_RUN}" == "1" ]]; then
    echo
    echo "DRY_RUN only — no jobs submitted."
    echo "Re-run with --submit (or SUBMIT=1 DRY_RUN=0) to submit."
fi
