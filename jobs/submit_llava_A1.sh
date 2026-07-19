#!/usr/bin/env bash
# Submit LLaVA-OneVision A1 verification (SAMPLE_SIZE=1000) with timestamped logs.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs/slurm

LOG_STAMP="$(date +%Y%m%d_%H%M)"
EXPERIMENT_ID="${EXPERIMENT_ID:-${LOG_STAMP}}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
MODEL="${MODEL:-llava}"
ABLATION="${ABLATION:-A1}"
SKIP_DATASET_PREP="${SKIP_DATASET_PREP:-1}"
RESUME="${RESUME:-0}"

LOG_OUT="logs/slurm/llava_A1_${LOG_STAMP}.out"
LOG_ERR="logs/slurm/llava_A1_${LOG_STAMP}.err"

export EXPERIMENT_ID SAMPLE_SIZE MODEL ABLATION SKIP_DATASET_PREP RESUME

# Forward optional environment overrides to the batch job.
SBATCH_EXPORTS=()
for VAR in SAMPLE_SIZE EXPERIMENT_ID BATCH_SIZE SKIP_DATASET_PREP MODEL_CONFIG MODEL ABLATION RESUME; do
    if [[ -n "${!VAR:-}" ]]; then
        SBATCH_EXPORTS+=(--export="ALL,${VAR}=${!VAR}")
    fi
done

if [[ ${#SBATCH_EXPORTS[@]} -eq 0 ]]; then
    SBATCH_EXPORTS=(--export=ALL)
fi

JOB_ID="$(
    sbatch \
        "${SBATCH_EXPORTS[@]}" \
        --mail-user=luoz23@wfu.edu \
        --mail-type=BEGIN,END,FAIL \
        --output="${LOG_OUT}" \
        --error="${LOG_ERR}" \
        jobs/run_llava_A1.slurm \
        | awk '{print $NF}'
)"

echo "Submitted LLaVA A1 verification experiment"
echo "  Job ID:       ${JOB_ID}"
echo "  Model:        ${MODEL}"
echo "  Ablation:     ${ABLATION}"
echo "  Sample size:  ${SAMPLE_SIZE}"
echo "  Experiment:   ${EXPERIMENT_ID}"
echo "  Skip prep:    ${SKIP_DATASET_PREP}"
echo "  Resume:       ${RESUME}"
echo "  Stdout log:   ${LOG_OUT}"
echo "  Stderr log:   ${LOG_ERR}"
echo "  Results:      outputs/verification/${MODEL}/${EXPERIMENT_ID}/A1/"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_OUT}"
