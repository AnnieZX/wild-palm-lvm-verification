#!/usr/bin/env python3
"""
Purpose:
    Compute verification metrics from per-ablation GT evaluation CSVs.

Input:
    - outputs/evaluation/A1_evaluation.csv … A5_evaluation.csv

Output:
    - outputs/evaluation/summary.csv
    - outputs/evaluation/A1_metrics.json … (one JSON per ablation)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import EVALUATION_DIR

EVALUATION_CSV_PATTERN = re.compile(r"^(A\d+)_evaluation\.csv$")

POSITIVE_LABEL = "Reliable"
UNCERTAIN_LABEL = "Uncertain"
NEGATIVE_LABEL = "Unreliable"

SUMMARY_COLUMNS = [
    "Ablation",
    "Samples",
    "Precision",
    "Recall",
    "F1",
    "Accuracy",
    "Average_IoU",
    "Average_Confidence",
    "Reliable%",
    "Uncertain%",
    "Unreliable%",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute verification metrics from evaluation CSV files.",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=EVALUATION_DIR,
        help="Directory containing A*_evaluation.csv files",
    )
    return parser.parse_args()


def discover_evaluation_csvs(evaluation_dir: Path) -> list[tuple[str, Path]]:
    """Return sorted (ablation_code, csv_path) pairs."""
    if not evaluation_dir.exists():
        return []

    discovered: list[tuple[str, Path]] = []
    for path in sorted(evaluation_dir.glob("*_evaluation.csv")):
        match = EVALUATION_CSV_PATTERN.match(path.name)
        if match:
            discovered.append((match.group(1), path))
    return discovered


def normalize_verification_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for label in (POSITIVE_LABEL, UNCERTAIN_LABEL, NEGATIVE_LABEL):
        if text.lower() == label.lower():
            return label
    return ""


def normalize_matched_gt(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes"}


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_metrics(df: pd.DataFrame, ablation: str) -> dict[str, Any]:
    """Compute verification metrics for one ablation evaluation CSV."""
    total_samples = len(df)

    labels = df["verification_label"].map(normalize_verification_label)
    gt_positive = df["matched_gt"].map(normalize_matched_gt)

    iou_values = pd.to_numeric(df["max_iou"], errors="coerce")
    confidence_values = pd.to_numeric(df["yolo_confidence"], errors="coerce")

    reliable_count = int((labels == POSITIVE_LABEL).sum())
    uncertain_count = int((labels == UNCERTAIN_LABEL).sum())
    unreliable_count = int((labels == NEGATIVE_LABEL).sum())

    binary_mask = labels.isin([POSITIVE_LABEL, NEGATIVE_LABEL])
    binary_df = df[binary_mask].copy()
    binary_labels = labels[binary_mask]
    binary_gt = gt_positive[binary_mask]

    predicted_positive = binary_labels == POSITIVE_LABEL
    predicted_negative = binary_labels == NEGATIVE_LABEL
    actual_positive = binary_gt
    actual_negative = ~binary_gt

    tp = int((actual_positive & predicted_positive).sum())
    fp = int((actual_negative & predicted_positive).sum())
    fn = int((actual_positive & predicted_negative).sum())
    tn = int((actual_negative & predicted_negative).sum())

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)

    def pct(count: int) -> float:
        if total_samples == 0:
            return 0.0
        return round(100.0 * count / total_samples, 2)

    return {
        "ablation": ablation,
        "samples": total_samples,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "uncertain_predictions": uncertain_count,
        "excluded_from_binary_metrics": int((~binary_mask).sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "average_iou": round(float(iou_values.mean()), 4) if not iou_values.dropna().empty else 0.0,
        "average_confidence": round(float(confidence_values.mean()), 4)
        if not confidence_values.dropna().empty
        else 0.0,
        "reliable_count": reliable_count,
        "uncertain_count": uncertain_count,
        "unreliable_count": unreliable_count,
        "reliable_pct": pct(reliable_count),
        "uncertain_pct": pct(uncertain_count),
        "unreliable_pct": pct(unreliable_count),
        "ground_truth_positive": int(gt_positive.sum()),
        "ground_truth_negative": int((~gt_positive).sum()),
    }


def metrics_to_summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "Ablation": metrics["ablation"],
        "Samples": metrics["samples"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "Accuracy": metrics["accuracy"],
        "Average_IoU": metrics["average_iou"],
        "Average_Confidence": metrics["average_confidence"],
        "Reliable%": metrics["reliable_pct"],
        "Uncertain%": metrics["uncertain_pct"],
        "Unreliable%": metrics["unreliable_pct"],
    }


def print_summary_table(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No metrics to display.")
        return
    print()
    print(summary_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.resolve()

    evaluation_files = discover_evaluation_csvs(evaluation_dir)
    if not evaluation_files:
        print(f"No evaluation CSV files found under: {evaluation_dir}")
        print("Expected files like A1_evaluation.csv")
        sys.exit(1)

    evaluation_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    print("Verification metrics")
    print(f"  Evaluation dir: {evaluation_dir}")

    for ablation, csv_path in evaluation_files:
        df = pd.read_csv(csv_path)
        metrics = compute_metrics(df, ablation)

        json_path = evaluation_dir / f"{ablation}_metrics.json"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        summary_rows.append(metrics_to_summary_row(metrics))
        print(f"  {ablation}: TP={metrics['true_positive']} FP={metrics['false_positive']} "
              f"FN={metrics['false_negative']} F1={metrics['f1']:.4f}")

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_path = evaluation_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print_summary_table(summary_df)
    print()
    print(f"Saved summary: {summary_path}")
    print(f"Saved metrics JSON files: {evaluation_dir}/A*_metrics.json")


if __name__ == "__main__":
    main()
