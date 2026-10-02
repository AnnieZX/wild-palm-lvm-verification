#!/usr/bin/env bash
# Submit full-cohort A1–A5 (N=5747) for ONE model that passed paired-probe qualification.
#
# Usage:
#   ./scripts/submit_qualified_model.sh MODEL QUAL_REPORT_DIR [--partition P --gres G --time T] [--submit]
#   Completing a partial model (e.g. A1 exists):
#     ... --experiment-id <existing> --conditions A2,A3,A4,A5 --add-to-existing
#     (refuses if any requested condition directory already exists)
#
# Refuses unless:
#   - QUAL_REPORT_DIR/QUALIFICATION_VERDICT.txt is PASS
#   - tracked src/ scripts/ jobs/ configs/ are clean and HEAD is pushed to its upstream
# Submits five independent jobs through scripts/submit_model_ablation.sh and appends
# one row per job to logs/campaign/submissions.csv. Default is a dry run.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL="${1:?MODEL required}"
REPORT_DIR="${2:?QUAL_REPORT_DIR required}"
shift 2

PARTITION="yangGrp"
GRES="gpu:L40S:1"
TIME_LIMIT="24:00:00"
EXPERIMENT_ID="$(date +%Y%m%d)_${MODEL}_A1A5_5747"
SUBMIT=0
CONDITIONS_CSV="A1,A2,A3,A4,A5"
ADD_TO_EXISTING=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        --gres) GRES="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
        --conditions) CONDITIONS_CSV="$2"; shift 2 ;;
        --add-to-existing) ADD_TO_EXISTING=1; shift ;;
        --submit) SUBMIT=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
IFS=',' read -r -a CONDITIONS <<< "${CONDITIONS_CSV}"

VERDICT="$(tr -d '[:space:]' < "${REPORT_DIR}/QUALIFICATION_VERDICT.txt" 2>/dev/null || true)"
if [[ "${VERDICT}" != "PASS" ]]; then
    echo "ERROR: ${MODEL} qualification verdict is '${VERDICT:-missing}' in ${REPORT_DIR}" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain -- src scripts jobs configs)" ]]; then
    echo "ERROR: tracked code/config changes are uncommitted; commit and push first." >&2
    git status --short -- src scripts jobs configs >&2
    exit 1
fi
git fetch --quiet origin 2>/dev/null || true
if [[ -n "$(git rev-list '@{u}..HEAD' 2>/dev/null)" ]]; then
    echo "ERROR: HEAD has unpushed commits; push the integration checkpoint first." >&2
    exit 1
fi
EXP_DIR="outputs/verification/${MODEL}/${EXPERIMENT_ID}"
if [[ "${ADD_TO_EXISTING}" == "1" ]]; then
    [[ -d "${EXP_DIR}" ]] || { echo "ERROR: --add-to-existing but ${EXP_DIR} does not exist" >&2; exit 1; }
    for c in "${CONDITIONS[@]}"; do
        if [[ -e "${EXP_DIR}/${c}" ]]; then
            echo "ERROR: ${EXP_DIR}/${c} already exists; refusing to touch it." >&2
            exit 1
        fi
    done
elif [[ -d "${EXP_DIR}" ]]; then
    echo "ERROR: ${EXP_DIR} exists; choose a new experiment id." >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
REVISION="$(sed -n 's/^revision: *//p' "configs/models/${MODEL}.yaml" | head -1)"
echo "MODEL=${MODEL} COMMIT=${GIT_COMMIT} REVISION=${REVISION}"
echo "EXPERIMENT_ID=${EXPERIMENT_ID} PARTITION=${PARTITION} GRES=${GRES} TIME=${TIME_LIMIT}"

ARGS=("${MODEL}" --limit 5747 --ablation-size 5747 --experiment-id "${EXPERIMENT_ID}"
      --conditions "${CONDITIONS_CSV}"
      --partition "${PARTITION}" --gres "${GRES}" --time "${TIME_LIMIT}")
if [[ "${SUBMIT}" != "1" ]]; then
    ./scripts/submit_model_ablation.sh "${ARGS[@]}" --dry-run
    exit 0
fi

OUTPUT="$(./scripts/submit_model_ablation.sh "${ARGS[@]}" --submit)"
echo "${OUTPUT}"
mkdir -p logs/campaign
LEDGER="logs/campaign/submissions.csv"
[[ -f "${LEDGER}" ]] || echo "submitted_utc,model,condition,job_id,partition,gres,experiment_id,results_dir,git_commit,checkpoint_revision" > "${LEDGER}"
mapfile -t JOB_IDS < <(echo "${OUTPUT}" | sed -n 's/^Submitted batch job \([0-9]*\)$/\1/p')
if [[ "${#JOB_IDS[@]}" -ne "${#CONDITIONS[@]}" ]]; then
    echo "WARNING: expected ${#CONDITIONS[@]} job ids, got ${#JOB_IDS[@]}" >&2
fi
for i in "${!JOB_IDS[@]}"; do
    echo "$(date -u +%FT%TZ),${MODEL},${CONDITIONS[$i]},${JOB_IDS[$i]},${PARTITION},${GRES},${EXPERIMENT_ID},outputs/verification/${MODEL}/${EXPERIMENT_ID}/${CONDITIONS[$i]},${GIT_COMMIT},${REVISION}" >> "${LEDGER}"
done
echo "Recorded ${#JOB_IDS[@]} jobs in ${LEDGER}"
