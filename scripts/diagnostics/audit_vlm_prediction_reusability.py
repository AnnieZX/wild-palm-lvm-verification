#!/usr/bin/env python3
"""
Purpose:
    Read-only inventory of stored VLM verification outputs to decide whether they can
    be re-scored against a future, corrected ground-truth table without inference.

    For every <model>/<experiment>/<condition> under outputs/verification it checks
    that per-sample prediction JSONs exist, that sample_id is preserved and belongs to
    the frozen verification index, how many decisions are usable, and whether the
    stored Protocol v2 evaluation CSV carries exactly the same per-sample decisions.
    It also reports how many blind-pilot / positive-QC samples each run covers.

    Nothing is re-scored, no GT is changed and no inference is run.

Input:
    - outputs/verification/<model>/<experiment>/<A*>/sample_*.json, results_index.csv
    - outputs/evaluation_protocol_v2/<model>/<experiment>/<A*>/<A*>_evaluation.csv, _metrics.json
    - outputs/verification_dataset/index.csv
    - outputs/diagnostics/{gt_negative_blind_pilot,gt_positive_blind_qc}/blind_manifest.csv (sample_id only)

Output:
    - outputs/diagnostics/vlm_prediction_reusability/run_inventory.csv
    - outputs/diagnostics/vlm_prediction_reusability/summary.json

Example:
    python scripts/diagnostics/audit_vlm_prediction_reusability.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import EVALUATION_PROTOCOL_V2_ROOT, OUTPUTS_DIR, VERIFICATION_DATASET_INDEX_CSV  # noqa: E402

VERIFICATION_ROOT = OUTPUTS_DIR / "verification"
OUTPUT_DIR = OUTPUTS_DIR / "diagnostics" / "vlm_prediction_reusability"
PILOT_MANIFESTS = {
    "pilot_neg": OUTPUTS_DIR / "diagnostics" / "gt_negative_blind_pilot" / "blind_manifest.csv",
    "qc_pos": OUTPUTS_DIR / "diagnostics" / "gt_positive_blind_qc" / "blind_manifest.csv",
}
DECISIONS = ("Reliable", "Uncertain", "Unreliable")
FULL_N = 5747


def condition_code(name: str) -> str:
    return name.split("_", 1)[0]


def audit_condition(condition_dir: Path, index_ids: set[str], index_image: dict[str, str],
                    pilot_ids: dict[str, set[str]]) -> dict:
    model, experiment = condition_dir.parent.parent.name, condition_dir.parent.name
    code = condition_code(condition_dir.name)
    decisions: dict[str, str] = {}
    id_mismatch = parse_errors = inference_errors = unreadable = 0
    for path in sorted(condition_dir.glob("sample_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            unreadable += 1
            continue
        sample_id = str(record.get("sample_id", ""))
        if sample_id != path.stem:
            id_mismatch += 1
        decisions[path.stem] = str(record.get("decision", "") or "").strip()
        parse_errors += bool(str(record.get("parse_error", "") or "").strip())
        inference_errors += bool(str(record.get("inference_error", "") or "").strip())

    counts = Counter(decisions.values())
    ids = set(decisions)
    row = {
        "model_key": model,
        "experiment_id": experiment,
        "condition": code,
        "prediction_jsons": len(decisions),
        "unreadable_jsons": unreadable,
        "sample_id_field_matches_filename": id_mismatch == 0 and unreadable == 0,
        "all_ids_in_verification_index": ids <= index_ids,
        "covers_full_5747": ids == index_ids,
        "reliable": counts.get("Reliable", 0),
        "uncertain": counts.get("Uncertain", 0),
        "unreliable": counts.get("Unreliable", 0),
        "other_or_empty_decision": sum(v for k, v in counts.items() if k not in DECISIONS),
        "parse_errors": parse_errors,
        "inference_errors": inference_errors,
    }
    results_index = condition_dir / "results_index.csv"
    if results_index.exists():
        status = pd.read_csv(results_index, dtype=str)["status"].fillna("")
        row["results_index_rows"] = len(status)
        row["results_index_ok"] = int((status == "ok").sum())
    else:
        row["results_index_rows"] = row["results_index_ok"] = ""
    for name, sample_ids in pilot_ids.items():
        row[f"{name}_covered"] = len(sample_ids & ids)

    eval_dir = EVALUATION_PROTOCOL_V2_ROOT / model / experiment / code
    eval_csv = eval_dir / f"{code}_evaluation.csv"
    row["v2_evaluation_csv"] = eval_csv.exists()
    row["v2_metrics_json"] = (eval_dir / f"{code}_metrics.json").exists()
    if eval_csv.exists():
        stored = pd.read_csv(eval_csv, dtype=str, keep_default_na=False)
        stored_labels = dict(zip(stored["sample_id"], stored["verification_label"]))
        predicted = {s: l for s, l in stored_labels.items() if l}
        row["v2_eval_rows"] = len(stored)
        row["v2_eval_labels_equal_jsons"] = predicted == {s: d for s, d in decisions.items() if d}
        row["v2_eval_image_ids_match_index"] = all(
            index_image.get(s) == img for s, img in zip(stored["sample_id"], stored["image_name"])
        )
    else:
        row["v2_eval_rows"] = row["v2_eval_labels_equal_jsons"] = row["v2_eval_image_ids_match_index"] = ""
    return row


def main() -> None:
    index = pd.read_csv(VERIFICATION_DATASET_INDEX_CSV, dtype=str)
    index_ids = set(index["sample_id"])
    index_image = dict(zip(index["sample_id"], index["image_name"]))
    pilot_ids = {
        name: set(pd.read_csv(path, dtype=str)["sample_id"]) if path.exists() else set()
        for name, path in PILOT_MANIFESTS.items()
    }

    condition_dirs = sorted(
        path for path in VERIFICATION_ROOT.glob("*/*/*")
        if path.is_dir() and path.name[:1] == "A" and any(path.glob("sample_*.json"))
    )
    rows = [audit_condition(path, index_ids, index_image, pilot_ids) for path in condition_dirs]
    inventory = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT_DIR / "run_inventory.csv", index=False)

    full = inventory[inventory["covers_full_5747"]]
    summary = {
        "verification_index_rows": len(index),
        "condition_runs_with_prediction_jsons": len(inventory),
        "full_5747_condition_runs": len(full),
        "full_runs_by_model": {
            f"{m}/{e}": sorted(g["condition"]) for (m, e), g in full.groupby(["model_key", "experiment_id"])
        },
        "all_runs_sample_id_preserved": bool(inventory["sample_id_field_matches_filename"].all()),
        "all_runs_ids_in_index": bool(inventory["all_ids_in_verification_index"].all()),
        "full_runs_with_v2_eval_labels_equal_jsons": int((full["v2_eval_labels_equal_jsons"] == True).sum()),  # noqa: E712
        "full_runs_parse_or_inference_errors": int((full["parse_errors"] + full["inference_errors"]).sum()),
        "full_runs_covering_all_pilot_and_qc_samples": int(
            ((full["pilot_neg_covered"] == len(pilot_ids["pilot_neg"]))
             & (full["qc_pos_covered"] == len(pilot_ids["qc_pos"]))).sum()
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(inventory[[
        "model_key", "experiment_id", "condition", "prediction_jsons", "covers_full_5747",
        "sample_id_field_matches_filename", "reliable", "uncertain", "unreliable", "other_or_empty_decision",
        "parse_errors", "inference_errors", "pilot_neg_covered", "qc_pos_covered", "v2_evaluation_csv",
        "v2_eval_labels_equal_jsons", "v2_eval_image_ids_match_index",
    ]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
