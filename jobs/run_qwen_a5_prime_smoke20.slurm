#!/bin/bash
#SBATCH --job-name=qwen_a5p_smoke20
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/qwen_a5_prime_smoke20_%j.out
#SBATCH --error=logs/slurm/qwen_a5_prime_smoke20_%j.err
#
# Qwen A5_prime smoke ONLY (N=20). Does NOT run N=1000. Does NOT touch A1–A5.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

EXP_ID="${EXPERIMENT_ID:-20260914_qwen_a5_prime_smoke20}"
ABLATION_DIR="${PROJECT_DIR}/outputs/verification_ablation_1000/A5_prime_a4_crop_panel"
PROMPT_INDEX="${ABLATION_DIR}/prompt_index_smoke20.csv"
RESULTS_DIR="${PROJECT_DIR}/outputs/verification/qwen/${EXP_ID}/A5_prime"
MODEL_PATH="${MODEL_PATH:-/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct}"
PIXEL_REPORT="${ABLATION_DIR}/PIXEL_IDENTITY_REPORT.json"
PARITY_VERDICT="${ABLATION_DIR}/prompt_parity_audit/VERDICT.json"

mkdir -p logs/slurm "${RESULTS_DIR}"

echo "===== Qwen A5_prime smoke20 ====="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-unknown}"
echo "Experiment:   ${EXP_ID}"
echo "Prompt index: ${PROMPT_INDEX}"
echo "Results:      ${RESULTS_DIR}"
echo "Model:        ${MODEL_PATH}"
echo "Start:        $(date)"
echo

python --version
python -c "import transformers, torch; print('transformers', transformers.__version__); print('torch', torch.__version__)"
nvidia-smi
echo

# Hard gates
python - <<PY
import json, sys
from pathlib import Path
rep = json.loads(Path("${PIXEL_REPORT}").read_text())
if not rep.get("all_pass") or rep.get("n_validated") != 1000 or rep.get("n_failed") != 0:
    print("PIXEL IDENTITY GATE FAILED", rep.get("n_validated"), rep.get("n_failed"), file=sys.stderr)
    sys.exit(2)
print("Pixel identity gate OK:", rep["n_validated"], "shape", rep.get("canonical_a5_prime_shape"))
ver = json.loads(Path("${PARITY_VERDICT}").read_text())
if not ver.get("decision_semantics_unchanged") or not ver.get("a5_vs_a5prime_only_condition_name"):
    print("PROMPT PARITY GATE FAILED", ver, file=sys.stderr)
    sys.exit(3)
print("Prompt parity gate OK")
PY

if [[ ! -f "${PROMPT_INDEX}" ]]; then
  echo "ERROR: missing ${PROMPT_INDEX}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: missing model ${MODEL_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "===== Inference (N=20 smoke) ====="
python -u scripts/run_verification.py \
  --model qwen2_5_vl \
  --model-path "${MODEL_PATH}" \
  --prompt-index "${PROMPT_INDEX}" \
  --results-dir "${RESULTS_DIR}" \
  --condition A5_prime \
  --experiment-id "${EXP_ID}" \
  --batch-size 1 \
  --limit 20

echo
echo "===== Smoke summary ====="
python - <<PY
import json, csv
from pathlib import Path
from collections import Counter
results = Path("${RESULTS_DIR}")
files = sorted(results.glob("sample_*.json"))
print("n_json", len(files))
decisions = Counter()
inf_err = parse_err = 0
runtimes = []
model_paths = set()
ids = []
for fp in files:
    o = json.loads(fp.read_text())
    ids.append(o.get("sample_id", fp.stem))
    decisions[str(o.get("decision"))] += 1
    if o.get("inference_error"):
        inf_err += 1
    if o.get("parse_error"):
        parse_err += 1
    if o.get("runtime_seconds") is not None:
        runtimes.append(float(o["runtime_seconds"]))
    # optional fields
    for k in ("model_name", "model_path", "model"):
        if o.get(k):
            model_paths.add(str(o[k]))
print("sample_ids", ids)
print("decisions", dict(decisions))
print("inference_errors", inf_err)
print("parse_errors", parse_err)
print("runtime_sum_s", round(sum(runtimes), 2) if runtimes else None)
print("runtime_mean_s", round(sum(runtimes)/len(runtimes), 2) if runtimes else None)
print("model_fields", sorted(model_paths)[:5])
idx = Path("${RESULTS_DIR}") / "results_index.csv"
if idx.exists():
    rows = list(csv.DictReader(idx.open()))
    print("results_index_rows", len(rows))
    print("results_index_status", Counter(r.get("status") for r in rows))
PY

echo
echo "End: $(date)"
echo "SMOKE_COMPLETE results=${RESULTS_DIR}"
