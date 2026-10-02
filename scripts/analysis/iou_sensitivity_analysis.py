#!/usr/bin/env python3
"""
Purpose:
    IoU-threshold sensitivity of the detection-to-LabelMe matching for the canonical
    5,747 verification detections. Sensitivity analysis only: the canonical Protocol v2
    threshold stays IoU >= 0.5 and no Protocol v2 output is modified. A lower threshold
    is not "better"; it only shows how much of the GT- class depends on box overlap.

    Uses the unchanged Protocol v2 matching (per-image greedy one-to-one by descending
    IoU, same detections, same LabelMe palm rule) with only the threshold varied.
    If the semantic review manifest has labels, also reports how many semantic-palm
    Protocol v2 GT- detections would become matched at each threshold, by review reason.

Input:
    - outputs/verification_dataset/index.csv, outputs/full_inference/predictions_full.json
    - LabelMe JSON under Raw_Patches
    - outputs/semantic_gt_evaluation/reference/protocol_v2_detection_reference.csv
    - outputs/semantic_gt_evaluation/review/semantic_review_manifest.csv (optional labels)

Output (default outputs/semantic_gt_evaluation/iou_sensitivity/):
    - iou_sensitivity_summary.csv
    - iou_sensitivity_per_detection.csv
    - semantic_rematch_by_threshold.csv   (only rows with semantic labels)
    - iou_sensitivity_info.json

Example:
    python scripts/analysis/iou_sensitivity_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_verification_against_groundtruth as eval_mod  # noqa: E402
from src.evaluation.semantic_gt import (  # noqa: E402
    PROTOCOL_V2_REFERENCE_CSV,
    SEMANTIC_GT_ROOT,
    SEMANTIC_REVIEW_MANIFEST_CSV,
    UNREVIEWED,
    V2_NEGATIVE,
    V2_POSITIVE,
    load_reference,
    read_str_csv,
    validate_review_manifest,
)
from src.paths import (  # noqa: E402
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_INDEX_CSV,
)
from src.yolo.predictions_io import group_predictions_by_image, load_predictions  # noqa: E402

THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
CANONICAL_THRESHOLD = eval_mod.IOU_THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    parser.add_argument("--reference", type=Path, default=PROTOCOL_V2_REFERENCE_CSV)
    parser.add_argument("--manifest", type=Path, default=SEMANTIC_REVIEW_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=SEMANTIC_GT_ROOT / "iou_sensitivity")
    return parser.parse_args()


def column_name(threshold: float) -> str:
    return f"matched_iou_{threshold:.2f}"


def compute_matches(thresholds: list[float]) -> pd.DataFrame:
    index_df = pd.read_csv(VERIFICATION_DATASET_INDEX_CSV)
    predictions = group_predictions_by_image(load_predictions(PREDICTIONS_FULL_JSON))
    gt_cache: dict[str, list[tuple[float, float, float, float]]] = {}
    per_detection = pd.DataFrame({"sample_id": index_df["sample_id"].astype(str)})
    for threshold in thresholds:
        matches = eval_mod.compute_greedy_matches_for_index(
            index_df, predictions, RAW_PATCHES_ROOT, gt_cache, threshold,
        )
        per_detection[column_name(threshold)] = per_detection["sample_id"].map(
            lambda s, m=matches: m[s].matched_gt
        )
    return per_detection


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.thresholds) | {CANONICAL_THRESHOLD})
    reference = load_reference(args.reference)
    per_detection = compute_matches(thresholds)

    if list(per_detection["sample_id"]) != list(reference["sample_id"]):
        sys.exit("ABORT: detection order differs from reference")
    v2_positive = reference["original_protocol_v2_gt"].eq(V2_POSITIVE).to_numpy()
    if not (per_detection[column_name(CANONICAL_THRESHOLD)].to_numpy() == v2_positive).all():
        sys.exit("ABORT: recomputation at IoU 0.5 does not reproduce Protocol v2 GT")

    per_detection.insert(1, "image_id", reference["image_id"])
    per_detection.insert(2, "max_iou", reference["max_iou"])
    per_detection.insert(3, "original_protocol_v2_gt", reference["original_protocol_v2_gt"])

    n = len(per_detection)
    summary = []
    for threshold in thresholds:
        matched = per_detection[column_name(threshold)]
        summary.append({
            "iou_threshold": threshold,
            "canonical": threshold == CANONICAL_THRESHOLD,
            "n_detections": n,
            "matched": int(matched.sum()),
            "unmatched": int((~matched).sum()),
            "positive_pct": round(100.0 * matched.sum() / n, 2),
            "negative_pct": round(100.0 * (~matched).sum() / n, 2),
            "v2_negative_now_matched": int((matched & ~v2_positive).sum()),
            "v2_positive_now_unmatched": int((~matched & v2_positive).sum()),
        })

    rematch_rows = []
    labelled = 0
    if args.manifest.is_file():
        manifest = validate_review_manifest(read_str_csv(args.manifest), reference)
        manifest = manifest[manifest["semantic_gt"] != UNREVIEWED]
        labelled = len(manifest)
        pool = per_detection.merge(
            manifest[["sample_id", "semantic_gt", "review_reason"]],
            on="sample_id", how="inner", validate="one_to_one",
        )
        if (pool["original_protocol_v2_gt"] != V2_NEGATIVE).any():
            sys.exit("ABORT: labelled manifest row is not Protocol v2 GT-")
        for (label, reason), group in pool.groupby(["semantic_gt", "review_reason"]):
            for threshold in thresholds:
                rematch_rows.append({
                    "semantic_gt": label, "review_reason": reason,
                    "iou_threshold": threshold, "n_reviewed": len(group),
                    "would_be_matched": int(group[column_name(threshold)].sum()),
                })

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "iou_sensitivity_summary.csv", index=False)
    per_detection.to_csv(out / "iou_sensitivity_per_detection.csv", index=False)
    pd.DataFrame(rematch_rows, columns=["semantic_gt", "review_reason", "iou_threshold",
                                        "n_reviewed", "would_be_matched"]).to_csv(
        out / "semantic_rematch_by_threshold.csv", index=False)
    with (out / "iou_sensitivity_info.json").open("w", encoding="utf-8") as file:
        json.dump({
            "purpose": "sensitivity analysis only; canonical Protocol v2 threshold remains 0.5",
            "matching": "unchanged Protocol v2 greedy one-to-one matching; only the IoU threshold varies",
            "detection_set": "canonical 5,747 verification detections (YOLO conf >= 0.5)",
            "thresholds": thresholds,
            "canonical_threshold_reproduces_protocol_v2": True,
            "semantic_labelled_rows_used": labelled,
        }, file, indent=2)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Semantic-labelled negative-pool rows used: {labelled}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
