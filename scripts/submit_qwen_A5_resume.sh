#!/usr/bin/env bash
# Resume an interrupted Qwen A5 ablation run without recomputing completed samples.
#
# Usage:
#   EXPERIMENT_ID=20260708_0020 bash scripts/submit_qwen_A5_resume.sh
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs/slurm

if [[ -z "${EXPERIMENT_ID:-}" ]]; then
    echo "EXPERIMENT_ID must be set to the existing experiment folder." >&2
    echo "Example: EXPERIMENT_ID=20260708_0020 bash scripts/submit_qwen_A5_resume.sh" >&2
    exit 1
fi

LOG_STAMP="$(date +%Y%m%d_%H%M)"
LOG_OUT="logs/slurm/qwen_ablation_${LOG_STAMP}.out"
LOG_ERR="logs/slurm/qwen_ablation_${LOG_STAMP}.err"

export EXPERIMENT_ID
export ABLATION=A5
export RESUME=1
export SKIP_DATASET_PREP=1
export MODEL="${MODEL:-qwen2_5_vl}"

SBATCH_EXPORTS=()
for VAR in SAMPLE_SIZE EXPERIMENT_ID BATCH_SIZE SKIP_DATASET_PREP MODEL_CONFIG ABLATION RESUME; do
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
        jobs/run_qwen_ablation.slurm \
        | awk '{print $NF}'
)"

echo "Submitted Qwen A5 resume job"
echo "  Job ID:       ${JOB_ID}"
echo "  Experiment:   ${EXPERIMENT_ID}"
echo "  Ablation:     A5"
echo "  Resume:       enabled"
echo "  Sample size:  ${SAMPLE_SIZE:-300} (default 300 if unset)"
echo "  Stdout log:   ${LOG_OUT}"
echo "  Stderr log:   ${LOG_ERR}"
echo
echo "--------------------------------------------------"
echo
echo "Email notifications:"
echo
echo "BEGIN -> luoz23@wfu.edu"
echo "END   -> luoz23@wfu.edu"
echo "FAIL  -> luoz23@wfu.edu"
echo
echo "--------------------------------------------------"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_OUT}"
