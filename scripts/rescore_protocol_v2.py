#!/usr/bin/env python3
"""
Purpose:
    Re-score every existing Protocol v1 evaluation run under Evaluation Protocol v2
    from stored VLM predictions (no inference).

Input:
    - outputs/evaluation/<model>/<experiment>/<A*>/<A*>_evaluation.csv   (frozen v1)
    - outputs/evaluation/<model>/<experiment>/<A*>/<A*>_metrics.json
    - outputs/verification/<model>/<experiment>/<A*>/sample_*.json       (predictions)
    - outputs/verification_dataset/index.csv, YOLO predictions, LabelMe Raw_Patches

Output (mirrors the v1 layout; v1 is never written):
    - outputs/evaluation_protocol_v2/<model>/<experiment>/<A*>/{<A*>_evaluation.csv,
      <A*>_metrics.json, summary.csv}
    - outputs/evaluation_protocol_v2/<model>/<experiment>/PROTOCOL.json
    - outputs/evaluation_protocol_v2/PROTOCOL.json
    - outputs/evaluation_protocol_v2/rescore_manifest.csv

All runs are scored and validated in memory first; nothing is written unless every
invariant holds (identical predictions and sample IDs, only GT-derived columns
change, GT- -> GT+ flips only, canonical GT 5,109 / 638).
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compute_verification_metrics as metrics_mod  # noqa: E402
import evaluate_verification_against_groundtruth as eval_mod  # noqa: E402
from src.paths import (  # noqa: E402
    EVALUATION_PROTOCOL_V1_ROOT,
    EVALUATION_PROTOCOL_V2_ROOT,
    OUTPUTS_DIR,
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_INDEX_CSV,
)
from src.preprocessing.gt_palm_bboxes import (  # noqa: E402
    EVALUATION_PROTOCOL_VERSION,
    GT_LABEL_RULE,
)
from src.yolo.predictions_io import group_predictions_by_image, load_predictions  # noqa: E402

FULL_N = 5747
EXPECTED_GT = (5109, 638)
EXPECTED_FLIPS = 424
UNCHANGED_COLUMNS = [
    "image_name", "sample_id", "ablation", "yolo_bbox", "yolo_confidence", "verification_label",
]
GT_COLUMNS = ["gt_bbox", "max_iou", "matched_gt"]
BINARY_KEYS = ["true_positive", "false_positive", "false_negative", "true_negative",
               "precision", "recall", "f1", "accuracy"]
DECISION_KEYS = ["reliable_count", "uncertain_count", "unreliable_count", "evaluated_samples"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--v1-root", type=Path, default=EVALUATION_PROTOCOL_V1_ROOT)
    parser.add_argument("--v2-root", type=Path, default=EVALUATION_PROTOCOL_V2_ROOT)
    parser.add_argument("--verification-root", type=Path, default=OUTPUTS_DIR / "verification")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; write nothing")
    return parser.parse_args()


def git_base_commit() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip()

    return {
        "base_commit": run("rev-parse", "HEAD") or None,
        "working_tree_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        "note": "base_commit is the last commit; Protocol v2 code may be uncommitted "
                "working-tree changes on top of it at generation time.",
    }


def protocol_metadata(generated_at: str, git_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
        "gt_label_rule": GT_LABEL_RULE,
        "protocol_v1_gt_label_rule": 'label == "palm"',
        "gt_box": "axis-aligned envelope of all shape points, any shape_type "
                  "(point shapes give zero-area boxes and are kept)",
        "iou_threshold": eval_mod.IOU_THRESHOLD,
        "matching": "per-image greedy one-to-one matching by descending IoU "
                    "(Pascal VOC / COCO); matched detection = GT+, unmatched = GT-",
        "detection_set": "canonical 5,747-sample YOLO verification index (confidence >= 0.5)",
        "decision_semantics": "Reliable = positive, Unreliable = negative, "
                              "Uncertain excluded from binary metrics",
        "generated_at_utc": generated_at,
        **git_info,
        "definition": "docs/EVALUATION_PROTOCOL.md",
    }


def discover_runs(v1_root: Path) -> list[Path]:
    return sorted(
        path.parent for path in v1_root.glob("*/*/A*/A*_evaluation.csv")
        if path.name == f"{path.parent.name}_evaluation.csv"
    )


def prediction_error_counts(condition_dir: Path) -> tuple[int, int, int]:
    parse_errors = inference_errors = files = 0
    for path in sorted(condition_dir.glob("sample_*.json")):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        files += 1
        parse_errors += bool(record.get("parse_error"))
        inference_errors += bool(record.get("inference_error"))
    return files, parse_errors, inference_errors


def read_str_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.v1_root)
    if not runs:
        sys.exit(f"No v1 evaluation runs under {args.v1_root}")

    index_df = pd.read_csv(VERIFICATION_DATASET_INDEX_CSV)
    predictions = group_predictions_by_image(load_predictions(PREDICTIONS_FULL_JSON))
    greedy = eval_mod.compute_greedy_matches_for_index(
        index_df, predictions, RAW_PATCHES_ROOT, {}, eval_mod.IOU_THRESHOLD,
    )
    gt_pos = sum(match.matched_gt for match in greedy.values())
    if (gt_pos, len(greedy) - gt_pos) != EXPECTED_GT:
        sys.exit(f"ABORT: v2 GT {gt_pos}/{len(greedy) - gt_pos} != expected {EXPECTED_GT}")

    failures: list[str] = []
    staged: list[dict[str, Any]] = []

    for v1_dir in runs:
        rel = v1_dir.relative_to(args.v1_root)
        model, experiment, code = rel.parts
        pred_dir = args.verification_root / rel
        tag = str(rel)
        if not pred_dir.is_dir():
            failures.append(f"{tag}: prediction dir missing {pred_dir}")
            continue

        v2_df = eval_mod.evaluate_ablation_condition(
            condition_dir=pred_dir, index_df=index_df,
            predictions_by_image=predictions, greedy_matches=greedy,
        )
        v2_metrics = metrics_mod.compute_metrics(v2_df, code)

        v1_df = read_str_csv(v1_dir / f"{code}_evaluation.csv")
        with (v1_dir / f"{code}_metrics.json").open(encoding="utf-8") as file:
            v1_metrics = json.load(file)

        # Round-trip v2 through CSV so the comparison uses on-disk representations.
        v2_str = pd.read_csv(io.StringIO(v2_df.to_csv(index=False)),
                             dtype=str, keep_default_na=False)

        if list(v1_df["sample_id"]) != list(v2_str["sample_id"]):
            failures.append(f"{tag}: sample_id order differs")
            continue
        for column in UNCHANGED_COLUMNS:
            if not v1_df[column].equals(v2_str[column]):
                failures.append(f"{tag}: column {column} changed")

        v1_gt = v1_df["matched_gt"].str.lower() == "true"
        v2_gt = v2_str["matched_gt"].str.lower() == "true"
        lost = int((v1_gt & ~v2_gt).sum())
        gained = int((~v1_gt & v2_gt).sum())
        if lost or gained != EXPECTED_FLIPS:
            failures.append(f"{tag}: transitions lost={lost} gained={gained}")
        if int(v2_gt.sum()) != EXPECTED_GT[0]:
            failures.append(f"{tag}: v2 GT+ {int(v2_gt.sum())}")

        changed_rows = (v1_df[GT_COLUMNS] != v2_str[GT_COLUMNS]).any(axis=1)
        changed_images = set(v1_df.loc[changed_rows, "image_name"])
        palm_image_parents = {name.split("_")[2] for name in changed_images}
        if not palm_image_parents <= {f"{p:04d}" for p in range(194, 206)}:
            failures.append(f"{tag}: GT columns changed outside parents 0194-0205")

        for key in DECISION_KEYS:
            if v1_metrics.get(key) != v2_metrics[key]:
                failures.append(f"{tag}: {key} v1={v1_metrics.get(key)} v2={v2_metrics[key]}")

        labeled_flips = int(
            ((~v1_gt & v2_gt) & (v2_str["verification_label"].str.strip() != "")).sum()
        )
        full = v2_metrics["evaluated_samples"] == FULL_N
        if full and labeled_flips != EXPECTED_FLIPS:
            failures.append(f"{tag}: full run but only {labeled_flips} flipped samples labeled")
        if not full:
            if labeled_flips:
                failures.append(f"{tag}: partial run has {labeled_flips} labeled flipped samples")
            for key in BINARY_KEYS:
                if v1_metrics.get(key) != v2_metrics[key]:
                    failures.append(f"{tag}: unaffected run changed {key}")

        files, parse_errors, inference_errors = prediction_error_counts(pred_dir)
        staged.append({
            "model": model, "experiment": experiment, "condition": code,
            "v1_dir": v1_dir, "v2_dir": args.v2_root / rel, "pred_dir": pred_dir,
            "eval_df": v2_df, "metrics": v2_metrics, "v1_metrics": v1_metrics,
            "prediction_files": files, "parse_errors": parse_errors,
            "inference_errors": inference_errors, "full": full,
            "labeled_flips": labeled_flips, "changed_rows": int(changed_rows.sum()),
        })

    n_full = sum(item["full"] for item in staged)
    print(f"Runs discovered: {len(runs)}  staged: {len(staged)}  full N={FULL_N}: {n_full}")
    if failures:
        print("ABORT: invariant failures:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("All invariants hold.")
    if args.dry_run:
        return

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = protocol_metadata(generated_at, git_base_commit())
    args.v2_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    experiments: dict[Path, list[dict[str, Any]]] = {}
    for item in staged:
        out = item["v2_dir"]
        if out.resolve().is_relative_to(args.v1_root.resolve()):
            sys.exit(f"ABORT: refusing to write into v1 tree: {out}")
        out.mkdir(parents=True, exist_ok=True)
        code = item["condition"]
        item["eval_df"].to_csv(out / f"{code}_evaluation.csv", index=False)
        with (out / f"{code}_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(item["metrics"], file, indent=2)
        pd.DataFrame(
            [metrics_mod.metrics_to_summary_row(item["metrics"])],
            columns=metrics_mod.SUMMARY_COLUMNS,
        ).to_csv(out / "summary.csv", index=False)
        experiments.setdefault(out.parent, []).append(item)

        m, v1m = item["metrics"], item["v1_metrics"]
        manifest_rows.append({
            "model": item["model"], "experiment": item["experiment"], "condition": code,
            "full_5747": item["full"], "evaluated_samples": m["evaluated_samples"],
            "reliable": m["reliable_count"], "uncertain": m["uncertain_count"],
            "unreliable": m["unreliable_count"], "prediction_files": item["prediction_files"],
            "parse_errors": item["parse_errors"], "inference_errors": item["inference_errors"],
            "gt_rows_changed": item["changed_rows"], "labeled_flips": item["labeled_flips"],
            **{f"v1_{k}": v1m.get(k) for k in ("accuracy", "precision", "recall", "f1",
                                               "true_positive", "true_negative",
                                               "false_positive", "false_negative")},
            **{f"v2_{k}": m[k] for k in ("accuracy", "precision", "recall", "specificity", "f1",
                                         "balanced_accuracy", "true_positive", "true_negative",
                                         "false_positive", "false_negative")},
            "v1_dir": str(item["v1_dir"].relative_to(PROJECT_ROOT)),
            "v2_dir": str(out.relative_to(PROJECT_ROOT)),
            "predictions_dir": str(item["pred_dir"].relative_to(PROJECT_ROOT)),
        })

    for exp_dir, items in experiments.items():
        exp_meta = {
            **meta,
            "model": items[0]["model"],
            "experiment_id": items[0]["experiment"],
            "conditions": sorted(item["condition"] for item in items),
            "predictions_dir": str(items[0]["pred_dir"].parent.relative_to(PROJECT_ROOT)),
            "protocol_v1_evaluation_dir": str(items[0]["v1_dir"].parent.relative_to(PROJECT_ROOT)),
        }
        with (exp_dir / "PROTOCOL.json").open("w", encoding="utf-8") as file:
            json.dump(exp_meta, file, indent=2)

    with (args.v2_root / "PROTOCOL.json").open("w", encoding="utf-8") as file:
        json.dump({**meta, "rescored_runs": len(staged), "full_5747_runs": n_full,
                   "canonical_gt_positive": EXPECTED_GT[0],
                   "canonical_gt_negative": EXPECTED_GT[1],
                   "v1_to_v2_gt_flips": EXPECTED_FLIPS}, file, indent=2)
    pd.DataFrame(manifest_rows).to_csv(args.v2_root / "rescore_manifest.csv", index=False)
    print(f"Wrote {len(staged)} runs under {args.v2_root}")


if __name__ == "__main__":
    main()
