#!/usr/bin/env bash
# Rebuild every derived analysis artefact under paper/analysis/ from the frozen raw outputs.
# Reads outputs/ only; never writes outside paper/analysis/.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=paper/analysis/scripts
python3 "$S/a00_audit.py"
python3 "$S/a01_master_table.py"
python3 "$S/a02_decision_distributions.py"
python3 "$S/a03_transitions.py"
python3 "$S/a04_effect_sizes.py"
python3 "$S/a05_bootstrap.py"
python3 "$S/a06_agreement.py"
python3 "$S/a07_semantic_review.py"
python3 "$S/a08_iou_sensitivity.py" "$@"
python3 "$S/a09_case_selection.py"
python3 "$S/a10_label_usage_collapse.py"
python3 "$S/a11_detection_difficulty.py"
python3 "$S/a12_size_ladder.py"
