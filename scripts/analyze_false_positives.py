#!/usr/bin/env python3
"""
Purpose:
    List false-positive verification samples from an existing evaluation CSV.

Uses the same TP/FP/FN assignment rules as compute_verification_metrics.py
without recomputing IoU or GT matching.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compute_verification_metrics as metrics_module

POSITIVE_LABEL = metrics_module.POSITIVE_LABEL
NEGATIVE_LABEL = metrics_module.NEGATIVE_LABEL
compute_metrics = metrics_module.compute_metrics
normalize_matched_gt = metrics_module.normalize_matched_gt
normalize_verification_label = metrics_module.normalize_verification_label

OUTPUT_COLUMNS = [
    "sample_id",
    "image_name",
    "yolo_confidence",
    "max_iou",
    "verification_label",
    "gt_label",
]

DEFAULT_EVALUATION_DIR = (
    PROJECT_ROOT / "outputs" / "evaluation" / "llava" / "20260719_1734" / "A1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify false-positive samples from evaluation CSV output.",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIR,
        help="Directory containing A*_evaluation.csv",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="A1",
        help="Ablation code for the evaluation CSV (default: A1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <evaluation-dir>/false_positives.csv)",
    )
    return parser.parse_args()


def gt_label_from_matched_gt(value: object) -> str:
    return "positive" if normalize_matched_gt(value) else "negative"


def identify_false_positives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows classified as false positives by compute_verification_metrics.py.

    FP = predicted Reliable on a ground-truth-negative detection (matched_gt=False),
    among rows with definitive verification labels (Reliable or Unreliable).
    """
    labels = df["verification_label"].map(normalize_verification_label)
    gt_positive = df["matched_gt"].map(normalize_matched_gt)

    binary_mask = labels.isin([POSITIVE_LABEL, NEGATIVE_LABEL])
    binary_df = df[binary_mask].copy()
    binary_labels = labels[binary_mask]
    binary_gt = gt_positive[binary_mask]

    predicted_positive = binary_labels == POSITIVE_LABEL
    actual_negative = ~binary_gt
    fp_mask = actual_negative & predicted_positive

    fp_df = binary_df.loc[fp_mask].copy()
    fp_df["gt_label"] = fp_df["matched_gt"].map(gt_label_from_matched_gt)
    return fp_df[OUTPUT_COLUMNS]


def main() -> None:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.resolve()
    evaluation_csv = evaluation_dir / f"{args.ablation}_evaluation.csv"
    output_path = (
        args.output.resolve()
        if args.output is not None
        else evaluation_dir / "false_positives.csv"
    )

    if not evaluation_csv.exists():
        print(f"Evaluation CSV not found: {evaluation_csv}")
        sys.exit(1)

    df = pd.read_csv(evaluation_csv)
    metrics = compute_metrics(df, args.ablation)
    fp_df = identify_false_positives(df)

    expected_fp = metrics["false_positive"]
    if len(fp_df) != expected_fp:
        print(
            f"FP count mismatch: identified {len(fp_df)}, "
            f"compute_metrics reported {expected_fp}",
            file=sys.stderr,
        )
        sys.exit(1)

    metrics_json = evaluation_dir / f"{args.ablation}_metrics.json"
    if metrics_json.exists():
        with metrics_json.open(encoding="utf-8") as file:
            saved_metrics = json.load(file)
        if saved_metrics.get("false_positive") != expected_fp:
            print(
                f"Warning: {metrics_json.name} reports "
                f"false_positive={saved_metrics.get('false_positive')}, "
                f"recomputed={expected_fp}",
                file=sys.stderr,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fp_df.to_csv(output_path, index=False)

    print("False positive analysis")
    print(f"  Evaluation CSV: {evaluation_csv}")
    print(f"  TP={metrics['true_positive']} FP={metrics['false_positive']} "
          f"FN={metrics['false_negative']} TN={metrics['true_negative']}")
    print(f"  Saved {len(fp_df)} false positives: {output_path}")
    print()
    print("First 10 sample_ids:")
    for sample_id in fp_df["sample_id"].head(10):
        print(f"  {sample_id}")


if __name__ == "__main__":
    main()
