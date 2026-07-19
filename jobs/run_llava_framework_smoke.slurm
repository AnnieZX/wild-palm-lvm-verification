#!/bin/bash
#SBATCH --job-name=llava_fw_smoke
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodelist=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/llava_framework_smoke_%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# LLaVA-OneVision framework smoke test (verification pipeline end-to-end).
#
# Validates registry → adapter → verifier → parse → JSON → results_index.csv
# on 5 A1 samples. Does NOT run evaluation or full A1–A5.
#
# Submit from the project root:
#   sbatch jobs/run_llava_framework_smoke.slurm
#
# Monitor:
#   tail -f logs/slurm/llava_framework_smoke_<jobid>.out
# ---------------------------------------------------------------------------

echo "===== LLaVA Framework Smoke Test ====="
echo

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

RESULTS_DIR="${PROJECT_DIR}/outputs/verification/llava/smoke_framework/A1"
EXPERIMENT_ID="smoke_framework"
LIMIT=5

mkdir -p logs/slurm
mkdir -p "${RESULTS_DIR}"

echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-unknown}"
echo "Submit dir:  ${PROJECT_DIR}"
echo "Start time:  $(date)"
echo

echo "Loading environment..."
echo

hostname
pwd
python --version
python -c "import transformers; print(transformers.__version__)"
echo
echo "=== nvidia-smi ==="
nvidia-smi
echo

echo "Discovering newest valid A1 prompt_index.csv under outputs/ ..."
PROMPT_INDEX="$(
python - <<'PY'
from pathlib import Path

roots = [Path("outputs")]
candidates = []
for root in roots:
    if not root.is_dir():
        continue
    for path in root.rglob("prompt_index.csv"):
        if path.parent.name != "A1_overlay_only":
            continue
        if not path.is_file():
            continue
        candidates.append(path)

if not candidates:
    raise SystemExit("No A1_overlay_only/prompt_index.csv found under outputs/")

# Newest by modification time; tie-break by path string for stability.
candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
print(candidates[0].resolve())
PY
)"

if [[ ! -f "${PROMPT_INDEX}" ]]; then
    echo "ERROR: Discovered prompt index does not exist: ${PROMPT_INDEX}" >&2
    exit 1
fi

echo "  Prompt index: ${PROMPT_INDEX}"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Limit:        ${LIMIT}"
echo

echo "Running framework smoke inference..."
echo

python -u scripts/run_verification.py \
    --model llava \
    --prompt-index "${PROMPT_INDEX}" \
    --results-dir "${RESULTS_DIR}" \
    --limit "${LIMIT}" \
    --batch-size 1 \
    --experiment-id "${EXPERIMENT_ID}" \
    --condition A1

echo
echo "Validating framework smoke outputs..."
echo

export FRAMEWORK_SMOKE_RESULTS_DIR="${RESULTS_DIR}"
export FRAMEWORK_SMOKE_LIMIT="${LIMIT}"

python -u - <<'PY'
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

results_dir = Path(os.environ["FRAMEWORK_SMOKE_RESULTS_DIR"])
limit = int(os.environ["FRAMEWORK_SMOKE_LIMIT"])
index_path = results_dir / "results_index.csv"
sample_jsons = sorted(results_dir.glob("sample_*.json"))

print("=== Framework smoke validation ===")
print(f"Results dir: {results_dir}")
print(f"results_index.csv exists: {index_path.is_file()}")
print(f"sample_*.json count: {len(sample_jsons)}")
print()

if not index_path.is_file():
    print("FAIL: results_index.csv is missing")
    sys.exit(1)

if len(sample_jsons) == 0:
    print("FAIL: zero sample JSON files produced")
    sys.exit(1)

if len(sample_jsons) != limit:
    print(
        f"NOTE: expected up to {limit} sample JSONs, found {len(sample_jsons)}. "
        "Resume/skip or fewer available jobs may explain a shortfall."
    )
    print()

statuses = []
decisions = []
parse_errors = 0
rows = []

for path in sample_jsons:
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    sample_id = str(record.get("sample_id", path.stem))
    decision = str(record.get("decision") or "").strip()
    parse_error = str(record.get("parse_error") or "").strip()
    inference_error = str(record.get("inference_error") or "").strip()

    if inference_error:
        status = "inference_error"
    elif parse_error or not decision:
        status = "parse_error"
        parse_errors += 1
    else:
        status = "ok"

    statuses.append(status)
    if decision:
        decisions.append(decision)
    rows.append((sample_id, status, decision or "<empty>"))

status_counts = Counter(statuses)
decision_counts = Counter(decisions)

print("status")
for key, count in sorted(status_counts.items()):
    print(f"{key} {count}")
print()

print("decision")
for label in ("Reliable", "Uncertain", "Unreliable"):
    print(f"{label} {decision_counts.get(label, 0)}")
other = sum(
    count
    for key, count in decision_counts.items()
    if key not in {"Reliable", "Uncertain", "Unreliable"}
)
if other:
    print(f"other {other}")
print()

print(f"parse-error count: {parse_errors}")
print()

print("per-sample")
for sample_id, status, decision in rows:
    print(f"{sample_id}\t{status}\t{decision}")
print()

if all(status == "inference_error" for status in statuses):
    print("FAIL: every sample is inference_error")
    sys.exit(1)

print("Framework smoke validation PASSED")
print("(parse_error samples are reported above but do not fail the smoke test)")
PY

echo
echo "Framework smoke completed."
echo
echo "End time: $(date)"
echo "Slurm stdout: ${SLURM_OUTPUT:-logs/slurm/llava_framework_smoke_${SLURM_JOB_ID:-local}.out}"
echo "Results dir:  ${RESULTS_DIR}"
