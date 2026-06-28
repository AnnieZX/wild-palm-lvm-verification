#!/usr/bin/env python3
"""Analyze a YOLO predictions.json export to infer its processing stage."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

PRED_FILE = Path(
    "/deac/csc/yangGrp/cuij/palm/testing/results/yolo_new/"
    "yolo11x_new_datanew-yolo/val/predictions.json"
)

CONFIDENCE_THRESHOLDS = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.25, 0.10]
SCORE_HISTOGRAM_BINS = [
    (0.00, 0.05),
    (0.05, 0.10),
    (0.10, 0.25),
    (0.25, 0.50),
    (0.50, 0.75),
    (0.75, 1.00),
]
IOU_PAIR_THRESHOLDS = [0.9, 0.8, 0.7]
MAX_DET_CANDIDATE = 100


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")

    return [record for record in data if isinstance(record, dict)]


def extract_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = record.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]

    keys = ("x", "y", "width", "height")
    if all(key in record for key in keys):
        return (
            float(record["x"]),
            float(record["y"]),
            float(record["width"]),
            float(record["height"]),
        )

    return None


def extract_score(record: dict[str, Any]) -> float | None:
    for key in ("score", "confidence", "conf", "probability"):
        if key not in record:
            continue
        value = record[key]
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def iou_xywh(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def group_by_image(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        image_id = prediction.get("image_id")
        if image_id is None:
            continue
        grouped[str(image_id)].append(prediction)
    return dict(grouped)


def score_in_bin(score: float, lower: float, upper: float) -> bool:
    if upper >= 1.0:
        return lower <= score <= upper
    return lower <= score < upper


def print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def analyze_confidence_distribution(scores: list[float]) -> None:
    print_header("1. Confidence distribution")

    if not scores:
        print("No score values found.")
        return

    print(f"Total scored predictions: {len(scores)}")
    print(f"Min score: {min(scores):.6f}")
    print(f"Max score: {max(scores):.6f}")
    print(f"Mean score: {mean(scores):.6f}")
    print()

    for threshold in CONFIDENCE_THRESHOLDS:
        count = sum(score >= threshold for score in scores)
        pct = count / len(scores) * 100
        print(f"score >= {threshold:.2f}: {count:6d} ({pct:5.2f}%)")

    below_10 = sum(score < 0.10 for score in scores)
    below_01 = sum(score < 0.01 for score in scores)
    print()
    print(f"score < 0.10: {below_10:6d} ({below_10 / len(scores) * 100:5.2f}%)")
    print(f"score < 0.01: {below_01:6d} ({below_01 / len(scores) * 100:5.2f}%)")


def analyze_max_det_signal(per_image: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    print_header("2. Images hitting exactly 100 detections")

    counts = {image_id: len(items) for image_id, items in per_image.items()}
    exactly_100 = sorted(image_id for image_id, count in counts.items() if count == MAX_DET_CANDIDATE)
    above_95 = sorted(image_id for image_id, count in counts.items() if count > 95)
    above_100 = sorted(image_id for image_id, count in counts.items() if count > MAX_DET_CANDIDATE)

    total_images = len(counts)
    print(f"Images with exactly {MAX_DET_CANDIDATE} detections: {len(exactly_100)}")
    print(f"Images with >95 detections: {len(above_95)}")
    print(f"Images with >{MAX_DET_CANDIDATE} detections: {len(above_100)}")
    print(f"Max detections on any image: {max(counts.values()) if counts else 0}")

    if exactly_100:
        print()
        print(f"image_ids with exactly {MAX_DET_CANDIDATE} detections:")
        for image_id in exactly_100:
            print(f"  {image_id}")

    if above_95 and len(above_95) != len(exactly_100):
        print()
        print("image_ids with >95 detections:")
        for image_id in above_95:
            print(f"  {image_id} ({counts[image_id]})")

    return {
        "exactly_100_count": len(exactly_100),
        "above_95_count": len(above_95),
        "above_100_count": len(above_100),
        "total_images": total_images,
        "max_count": max(counts.values()) if counts else 0,
        "exactly_100_fraction": len(exactly_100) / total_images if total_images else 0.0,
    }


def analyze_duplicate_boxes(per_image: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    print_header("3. Duplicate-box analysis")

    global_max_iou = 0.0
    pair_counts = {threshold: 0 for threshold in IOU_PAIR_THRESHOLDS}
    total_pairs = 0
    images_with_pairs = 0
    images_missing_bbox = 0
    per_image_max_ious: list[float] = []

    for image_id, predictions in sorted(per_image.items()):
        bboxes = [extract_bbox(prediction) for prediction in predictions]
        bboxes = [bbox for bbox in bboxes if bbox is not None]

        if len(bboxes) != len(predictions):
            images_missing_bbox += 1

        if len(bboxes) < 2:
            continue

        images_with_pairs += 1
        image_max_iou = 0.0

        for box_a, box_b in combinations(bboxes, 2):
            overlap = iou_xywh(box_a, box_b)
            total_pairs += 1
            image_max_iou = max(image_max_iou, overlap)
            global_max_iou = max(global_max_iou, overlap)
            for threshold in IOU_PAIR_THRESHOLDS:
                if overlap > threshold:
                    pair_counts[threshold] += 1

        per_image_max_ious.append(image_max_iou)

    if total_pairs == 0:
        print("Not enough bbox pairs to analyze (missing bbox fields or <2 boxes/image).")
        return {
            "global_max_iou": global_max_iou,
            "total_pairs": total_pairs,
            "pair_counts": pair_counts,
            "images_missing_bbox": images_missing_bbox,
            "images_with_pairs": images_with_pairs,
            "mean_image_max_iou": None,
        }

    print(f"Images analyzed (>=2 boxes with bbox): {images_with_pairs}")
    print(f"Images with missing/invalid bbox entries: {images_missing_bbox}")
    print(f"Total bbox pairs compared: {total_pairs}")
    print(f"Maximum IoU across all pairs: {global_max_iou:.6f}")
    if per_image_max_ious:
        print(f"Mean of per-image maximum IoU: {mean(per_image_max_ious):.6f}")
    print()

    for threshold in IOU_PAIR_THRESHOLDS:
        count = pair_counts[threshold]
        pct = count / total_pairs * 100
        print(f"IoU > {threshold:.1f}: {count:8d} pairs ({pct:6.3f}% of all pairs)")

    return {
        "global_max_iou": global_max_iou,
        "total_pairs": total_pairs,
        "pair_counts": pair_counts,
        "images_missing_bbox": images_missing_bbox,
        "images_with_pairs": images_with_pairs,
        "mean_image_max_iou": mean(per_image_max_ious) if per_image_max_ious else None,
    }


def analyze_score_histogram(scores: list[float]) -> None:
    print_header("4. Score histogram")

    if not scores:
        print("No score values found.")
        return

    for lower, upper in SCORE_HISTOGRAM_BINS:
        count = sum(score_in_bin(score, lower, upper) for score in scores)
        pct = count / len(scores) * 100
        label = f"{lower:.2f}-{upper:.2f}"
        print(f"{label:>10}: {count:6d} ({pct:5.2f}%)")


def print_recommendation(
    scores: list[float],
    max_det_stats: dict[str, Any],
    duplicate_stats: dict[str, Any],
    total_predictions: int,
) -> None:
    print_header("5. Recommendation")

    if total_predictions == 0:
        print("No predictions to analyze.")
        return

    findings: list[str] = []

    # max_det evidence
    exactly_100 = max_det_stats["exactly_100_count"]
    total_images = max_det_stats["total_images"]
    above_100 = max_det_stats["above_100_count"]
    max_count = max_det_stats["max_count"]

    if above_100 > 0:
        findings.append(
            f"AGAINST max_det={MAX_DET_CANDIDATE}: {above_100} image(s) exceed "
            f"{MAX_DET_CANDIDATE} detections (max observed = {max_count})."
        )
    elif exactly_100 > 0:
        fraction = max_det_stats["exactly_100_fraction"] * 100
        findings.append(
            f"FOR possible max_det={MAX_DET_CANDIDATE}: {exactly_100}/{total_images} "
            f"images ({fraction:.1f}%) have exactly {MAX_DET_CANDIDATE} detections."
        )
    else:
        findings.append(
            f"AGAINST max_det={MAX_DET_CANDIDATE}: no image has exactly "
            f"{MAX_DET_CANDIDATE} detections (max observed = {max_count})."
        )

    # NMS evidence
    total_pairs = duplicate_stats["total_pairs"]
    if total_pairs == 0:
        findings.append(
            "INCONCLUSIVE for NMS: bbox pairs could not be evaluated "
            "(missing bbox or fewer than 2 boxes per image)."
        )
    else:
        high_iou_07 = duplicate_stats["pair_counts"][0.7]
        high_iou_09 = duplicate_stats["pair_counts"][0.9]
        pct_07 = high_iou_07 / total_pairs * 100
        pct_09 = high_iou_09 / total_pairs * 100
        global_max = duplicate_stats["global_max_iou"]

        if high_iou_07 == 0:
            findings.append(
                f"FOR NMS already applied: 0/{total_pairs} bbox pairs have IoU > 0.7 "
                f"(global max IoU = {global_max:.4f})."
            )
        elif pct_07 < 0.5:
            findings.append(
                f"FOR NMS likely applied: only {high_iou_07}/{total_pairs} pairs "
                f"({pct_07:.3f}%) have IoU > 0.7 (global max IoU = {global_max:.4f})."
            )
        else:
            findings.append(
                f"FOR raw / pre-NMS output: {high_iou_07}/{total_pairs} pairs "
                f"({pct_07:.3f}%) have IoU > 0.7; {high_iou_09} pairs ({pct_09:.3f}%) "
                f"have IoU > 0.9 (global max IoU = {global_max:.4f})."
            )

    # confidence filtering evidence
    if not scores:
        findings.append("INCONCLUSIVE for confidence filtering: no score field found.")
    else:
        min_score = min(scores)
        below_25 = sum(score < 0.25 for score in scores)
        below_10 = sum(score < 0.10 for score in scores)
        below_01 = sum(score < 0.01 for score in scores)
        pct_below_25 = below_25 / len(scores) * 100

        if below_25 == 0:
            findings.append(
                f"FOR confidence filtering: min score = {min_score:.6f}; "
                "0 predictions have score < 0.25."
            )
        elif below_10 == 0 and min_score >= 0.10:
            findings.append(
                f"FOR possible confidence filtering: min score = {min_score:.6f}; "
                f"0 predictions below 0.10, but {below_25} ({pct_below_25:.2f}%) below 0.25."
            )
        else:
            findings.append(
                f"AGAINST strict confidence filtering at 0.25: min score = {min_score:.6f}; "
                f"{below_25} ({pct_below_25:.2f}%) predictions below 0.25, "
                f"{below_10} below 0.10, {below_01} below 0.01."
            )

    print("Evidence summary:")
    for index, line in enumerate(findings, start=1):
        print(f"  {index}. {line}")

    print()
    print("Conclusion (based only on the evidence above):")

    labels: list[str] = []

    if above_100 == 0 and exactly_100 > 0 and max_det_stats["exactly_100_fraction"] >= 0.05:
        labels.append(f"Likely truncated by max_det={MAX_DET_CANDIDATE}")

    if total_pairs > 0:
        pct_07 = duplicate_stats["pair_counts"][0.7] / total_pairs * 100
        if duplicate_stats["pair_counts"][0.7] == 0:
            labels.append("Likely after NMS")
        elif pct_07 >= 1.0:
            labels.append("Likely raw detector outputs (high duplicate overlap remains)")
        elif pct_07 < 0.5:
            labels.append("Likely after NMS")

    if scores:
        if sum(score < 0.25 for score in scores) == 0:
            labels.append("Likely after confidence filtering (no scores below 0.25)")
        elif sum(score < 0.10 for score in scores) > len(scores) * 0.05:
            labels.append("Likely raw detector outputs (many low-confidence scores present)")

    if not labels:
        print("  INCONCLUSIVE: statistics do not clearly support one processing stage.")
    else:
        for label in labels:
            print(f"  - {label}")

    print()
    print("Interpretation notes:")
    print("  - max_det and NMS are independent checks; both may apply.")
    print("  - Confidence filtering cannot be confirmed without knowing the training/inference threshold.")
    print("  - Absence of sub-0.25 scores suggests filtering, but could also reflect model behavior.")


def main() -> None:
    if not PRED_FILE.exists():
        print(f"Prediction file not found: {PRED_FILE}")
        sys.exit(1)

    predictions = load_predictions(PRED_FILE)
    per_image = group_by_image(predictions)
    scores = [score for prediction in predictions if (score := extract_score(prediction)) is not None]

    print("=" * 60)
    print("Prediction export analysis")
    print("=" * 60)
    print(f"File: {PRED_FILE}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Images with predictions: {len(per_image)}")
    print(f"Predictions with score: {len(scores)}")

    analyze_confidence_distribution(scores)
    max_det_stats = analyze_max_det_signal(per_image)
    duplicate_stats = analyze_duplicate_boxes(per_image)
    analyze_score_histogram(scores)
    print_recommendation(scores, max_det_stats, duplicate_stats, len(predictions))


if __name__ == "__main__":
    main()
