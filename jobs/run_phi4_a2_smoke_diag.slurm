#!/bin/bash
#SBATCH --job-name=phi4_a2_smoke
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/phi4_a2_smoke_%j.out
#SBATCH --error=logs/slurm/phi4_a2_smoke_%j.err
#
# Phi-4 A2 diagnostic smoke (does NOT overwrite failed A2-1000 run).
# Samples: previously completed 1–5, hang-in-flight 6, plus later IDs.
#   sbatch --partition=yangGrp --gres=gpu:L40S:1 \
#     --export=ALL,EXPERIMENT_ID=YYYYMMDD_HHMM_phi4_a2_smoke \
#     jobs/run_phi4_a2_smoke_diag.slurm

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

PHI_VENV="${PHI_VENV:-/deac/csc/yangGrp/luoz23/envs/wild-palm-phi4}"
module load apps/python/3.11.8 2>/dev/null || true
# shellcheck disable=SC1091
source "${PHI_VENV}/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export PATH="${PHI_VENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}"

EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M)_phi4_a2_smoke}"
MODEL="${MODEL:-phi4_multimodal}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/Phi-4-multimodal-instruct}"

CANONICAL_DIR="${PROJECT_DIR}/outputs/verification_ablation_1000/A2_overlay_confidence"
CANONICAL_INDEX="${CANONICAL_DIR}/prompt_index.csv"
# Keep CSV beside canonical A2 assets so relative image/prompt paths resolve.
SMOKE_META="${PROJECT_DIR}/outputs/diagnostics/phi4_a2_smoke/${EXPERIMENT_ID}"
PROMPT_INDEX="${CANONICAL_DIR}/prompt_index_phi4_a2_smoke_${EXPERIMENT_ID}.csv"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/${MODEL}/${EXPERIMENT_ID}/A2"
EVAL_DIR="${PROJECT_DIR}/outputs/evaluation/${MODEL}/${EXPERIMENT_ID}/A2"

mkdir -p logs/slurm "${SMOKE_META}" "${RESULTS_DIR}" "${EVAL_DIR}"

echo "===== Phi-4 Multimodal A2 diagnostic smoke ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE:  ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model:         ${MODEL}"
echo "Experiment:    ${EXPERIMENT_ID}"
echo "Prompt index:  ${PROMPT_INDEX}"
echo "Results:       ${RESULTS_DIR}"
echo "Model path:    ${MODEL_PATH}"
echo "Start time:    $(date)"
echo

case "${RESULTS_DIR}" in
    */outputs/verification/qwen/*|*/outputs/verification/qwen2_5_vl/*|*/outputs/evaluation/qwen/*|\
    */outputs/verification/glm_4_6v_flash/*|*/outputs/evaluation/glm_4_6v_flash/*|\
    */20260919_1524_phi4_A1A5_1000/*)
        echo "ERROR: refusing protected path: ${RESULTS_DIR}" >&2
        exit 1
        ;;
esac

case "$(which python)" in
    */envs/wild-palm-phi4/*) ;;
    *)
        echo "ERROR: python is not from wild-palm-phi4: $(which python)" >&2
        exit 1
        ;;
esac

which python
python --version
python -c 'import transformers,torch,peft; print("transformers",transformers.__version__); print("torch",torch.__version__); print("peft",peft.__version__)'
nvidia-smi

# Fail fast on GPUs with active uncorrectable ECC (root cause of A2-1000 pathology).
python - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,ecc.errors.uncorrected.volatile.total",
     "--format=csv,noheader,nounits"],
    text=True,
)
bad = []
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        continue
    idx, ecc = parts[0], parts[1]
    try:
        ecc_n = int(ecc)
    except ValueError:
        ecc_n = -1
    print(f"GPU {idx} volatile_uncorrected_ecc_total={ecc}")
    if ecc_n > 0:
        bad.append((idx, ecc_n))
if bad:
    print("ERROR: refusing GPU(s) with uncorrected ECC:", bad, file=sys.stderr)
    sys.exit(2)
print("ECC_GUARD_PASS")
PY

# Build smoke prompt index: samples 1-6 (failed-run window) + later probes.
python - <<PY
import pandas as pd
from pathlib import Path

canonical = pd.read_csv("${CANONICAL_INDEX}")
want = [
    "sample_000001", "sample_000002", "sample_000003", "sample_000004",
    "sample_000005", "sample_000006",  # completed + hang-in-flight
    "sample_000050", "sample_000100", "sample_000250",
    "sample_000500", "sample_000750", "sample_001000",
]
sub = canonical[canonical.sample_id.isin(want)].copy()
# preserve requested order
order = {s: i for i, s in enumerate(want)}
sub["_ord"] = sub.sample_id.map(order)
sub = sub.sort_values("_ord").drop(columns=["_ord"])
assert len(sub) == len(want), (len(sub), want, sub.sample_id.tolist())
assert set(sub.condition.unique()) == {"A2_overlay_confidence"}
out = Path("${PROMPT_INDEX}")
out.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(out, index=False)
# Mirror path list into diagnostics metadata (not used for IO).
meta = Path("${SMOKE_META}")
meta.mkdir(parents=True, exist_ok=True)
(meta / "smoke_ids.txt").write_text(",".join(sub.sample_id.tolist()) + "\n")
(meta / "prompt_index_path.txt").write_text(str(out) + "\n")
print("SMOKE_IDS:", ",".join(sub.sample_id.tolist()))
print("SMOKE_N:", len(sub))
print("PROMPT_INDEX:", out)
PY

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: missing checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi

VERIFICATION_ARGS=(
    --model phi4_multimodal
    --condition A2
    --limit 12
    --prompt-index "${PROMPT_INDEX}"
    --results-dir "${RESULTS_DIR}"
    --experiment-id "${EXPERIMENT_ID}"
    --batch-size "${BATCH_SIZE}"
    --model-path "${MODEL_PATH}"
    --model-config "${PROJECT_DIR}/configs/models/phi4_multimodal.yaml"
)

echo "===== Inference (A2 diagnostic smoke N=12) ====="
python -u scripts/run_verification.py "${VERIFICATION_ARGS[@]}"

echo
python - <<PY
import json
from pathlib import Path
root = Path("${RESULTS_DIR}")
files = sorted(root.glob("sample_*.json"))
print("N=", len(files))
runtimes = []
for f in files:
    d = json.loads(f.read_text())
    rt = float(d.get("runtime_seconds") or 0)
    runtimes.append(rt)
    pe = d.get("parse_error") or ""
    ie = d.get("inference_error") or ""
    print(
        f"{d.get('sample_id')}: decision={d.get('decision')!r} "
        f"runtime_s={rt:.2f} parse={bool(pe)} infer={bool(ie)} "
        f"raw_len={len(d.get('raw_response') or '')}"
    )
if runtimes:
    print(f"runtime_mean_s={sum(runtimes)/len(runtimes):.2f}")
    print(f"runtime_max_s={max(runtimes):.2f}")
    # Gate: must be comparable to healthy Phi-4 (~10s), not ~4900s pathology.
    if max(runtimes) > 120:
        raise SystemExit(f"SMOKE_FAIL: max runtime {max(runtimes):.1f}s exceeds 120s gate")
print("A2_SMOKE_DIAG_PASS")
PY

echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "A2_SMOKE_DIAG_COMPLETE"
