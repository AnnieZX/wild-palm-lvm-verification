#!/usr/bin/env python3
"""
Purpose:
    Evaluate YOLO detector against LabelMe palm annotations (pre-verification).

Input:
    - outputs/full_inference/predictions_full.json
    - LabelMe JSON under Raw_Patches

Output:
    - outputs/evaluation/detection_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import EVALUATION_DIR, PREDICTIONS_FULL_JSON, RAW_PATCHES_ROOT
from src.preprocessing.gt_palm_bboxes import extract_gt_palm_bboxes
from src.preprocessing.sequential_dataset import find_labelme_json_files
from src.yolo.gt_matching import greedy_match_detections_to_gt
from src.yolo.predictions_io import group_predictions_by_image, load_predictions

IOU_THRESHOLD = 0.5
DETECTION_METRICS_JSON = EVALUATION_DIR / "detection_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO detections against LabelMe palm ground truth.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS_FULL_JSON,
        help="YOLO predictions JSON",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=RAW_PATCHES_ROOT,
        help="LabelMe JSON root (Raw_Patches)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DETECTION_METRICS_JSON,
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_THRESHOLD,
        help="IoU threshold for a valid match (default: 0.5)",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_image(
    detections: list[dict[str, Any]],
    gt_bboxes: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> dict[str, Any]:
    """Return per-image TP/FP/FN counts and matched-pair statistics."""
    matches, matched_detections, matched_gt = greedy_match_detections_to_gt(
        detections,
        gt_bboxes,
        iou_threshold,
    )

    tp = len(matches)
    fp = len(detections) - tp
    fn = len(gt_bboxes) - tp

    matched_ious = [overlap for _, _, overlap in matches]
    tp_confidences = [
        float(detections[det_index]["score"])
        for det_index, _, _ in matches
        if detections[det_index].get("score") is not None
    ]
    fp_confidences = [
        float(detection["score"])
        for det_index, detection in enumerate(detections)
        if det_index not in matched_detections and detection.get("score") is not None
    ]

    return {
        "gt_palms": len(gt_bboxes),
        "yolo_detections": len(detections),
        "matched_detections": tp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "matched_ious": matched_ious,
        "tp_confidences": tp_confidences,
        "fp_confidences": fp_confidences,
    }


def aggregate_metrics(
    per_image_stats: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    """Aggregate per-image counts into dataset-level detection metrics."""
    images = len(per_image_stats)
    gt_palms = sum(stats["gt_palms"] for stats in per_image_stats)
    yolo_detections = sum(stats["yolo_detections"] for stats in per_image_stats)
    matched_detections = sum(stats["matched_detections"] for stats in per_image_stats)

    tp = sum(stats["true_positive"] for stats in per_image_stats)
    fp = sum(stats["false_positive"] for stats in per_image_stats)
    fn = sum(stats["false_negative"] for stats in per_image_stats)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    matched_ious = [
        overlap
        for stats in per_image_stats
        for overlap in stats["matched_ious"]
    ]
    tp_confidences = [
        score
        for stats in per_image_stats
        for score in stats["tp_confidences"]
    ]
    fp_confidences = [
        score
        for stats in per_image_stats
        for score in stats["fp_confidences"]
    ]

    return {
        "images": images,
        "gt_palms": gt_palms,
        "yolo_detections": yolo_detections,
        "matched_detections": matched_detections,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "average_iou": round(safe_mean(matched_ious), 4),
        "average_tp_confidence": round(safe_mean(tp_confidences), 4),
        "average_fp_confidence": round(safe_mean(fp_confidences), 4),
        "iou_threshold": iou_threshold,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    print()
    print(f"Images:                 {metrics['images']}")
    print(f"GT palms:               {metrics['gt_palms']}")
    print(f"YOLO detections:        {metrics['yolo_detections']}")
    print(f"Matched detections:     {metrics['matched_detections']}")
    print(f"TP:                     {metrics['true_positive']}")
    print(f"FP:                     {metrics['false_positive']}")
    print(f"FN:                     {metrics['false_negative']}")
    print(f"Precision:              {metrics['precision']:.4f}")
    print(f"Recall:                 {metrics['recall']:.4f}")
    print(f"F1:                     {metrics['f1']:.4f}")
    print(f"Average IoU:            {metrics['average_iou']:.4f}")
    print(f"Average TP confidence:  {metrics['average_tp_confidence']:.4f}")
    print(f"Average FP confidence:  {metrics['average_fp_confidence']:.4f}")


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        print(f"Predictions file not found: {args.predictions}")
        sys.exit(1)

    if not args.annotations_root.exists():
        print(f"LabelMe annotations root not found: {args.annotations_root}")
        sys.exit(1)

    predictions_by_image = group_predictions_by_image(load_predictions(args.predictions))
    json_files = find_labelme_json_files(args.annotations_root)

    per_image_stats: list[dict[str, Any]] = []
    for json_path in json_files:
        image_name = json_path.stem
        gt_bboxes = extract_gt_palm_bboxes(json_path)
        detections = predictions_by_image.get(image_name, [])
        per_image_stats.append(
            evaluate_image(detections, gt_bboxes, args.iou_threshold),
        )

    metrics = aggregate_metrics(per_image_stats, args.iou_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("YOLO detection vs ground-truth evaluation")
    print(f"  Predictions:   {args.predictions}")
    print(f"  Annotations:   {args.annotations_root}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  Output:        {args.output}")
    print_metrics(metrics)
    print()
    print(f"Saved metrics: {args.output}")


if __name__ == "__main__":
    main()
