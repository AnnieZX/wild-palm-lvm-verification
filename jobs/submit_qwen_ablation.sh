#!/usr/bin/env bash
# Submit the overnight Qwen ablation experiment with timestamped Slurm logs.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs/slurm

LOG_STAMP="$(date +%Y%m%d_%H%M)"
EXPERIMENT_ID="${EXPERIMENT_ID:-${LOG_STAMP}}"
LOG_OUT="logs/slurm/qwen_ablation_${LOG_STAMP}.out"
LOG_ERR="logs/slurm/qwen_ablation_${LOG_STAMP}.err"

export EXPERIMENT_ID

# Forward optional environment overrides to the batch job.
SBATCH_EXPORTS=()
for VAR in SAMPLE_SIZE EXPERIMENT_ID BATCH_SIZE SKIP_DATASET_PREP MODEL_CONFIG; do
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

echo "Submitted Qwen ablation experiment"
echo "  Job ID:       ${JOB_ID}"
echo "  Sample size:  ${SAMPLE_SIZE:-300} (default 300 if unset)"
echo "  Experiment:   ${EXPERIMENT_ID:-<auto timestamp at job start>}"
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
