#!/usr/bin/env python3
"""
Purpose:
    Match each LabelMe GT palm to its best YOLO prediction by IoU (threshold 0.5).

Input:
    - outputs/full_inference/predictions_full.json
    - LabelMe JSON under Raw_Patches

Output:
    - outputs/yolo_analysis/gt_matches.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import GT_MATCHES_CSV, PREDICTIONS_FULL_JSON, RAW_PATCHES_ROOT
from src.preprocessing.palm_analyzer import PalmInstance, extract_palm_instances_in_annotation_order
from src.preprocessing.sequential_dataset import find_labelme_json_files
from src.yolo.predictions_io import group_predictions_by_image, iou_xywh, load_predictions

IOU_THRESHOLD = 0.5


def palm_instance_bbox(palm: PalmInstance) -> tuple[float, float, float, float]:
    return palm.bbox_x, palm.bbox_y, palm.bbox_width, palm.bbox_height


def bbox_to_list(bbox: tuple[float, float, float, float]) -> list[float]:
    return [round(value, 2) for value in bbox]


def match_gt_to_best_prediction(
    gt_bbox: tuple[float, float, float, float],
    predictions: list[dict[str, Any]],
) -> tuple[tuple[float, float, float, float] | None, float | None, float]:
    if not predictions:
        return None, None, 0.0

    best_iou = 0.0
    best_pred_bbox = None
    best_score = None

    for prediction in predictions:
        pred_bbox = prediction["bbox"]
        overlap = iou_xywh(gt_bbox, pred_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_pred_bbox = pred_bbox
            best_score = prediction.get("score")

    return best_pred_bbox, best_score, best_iou


def match_image(
    image_name: str,
    json_path: Path,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    palms = extract_palm_instances_in_annotation_order(json_path)
    rows: list[dict[str, Any]] = []

    for index, palm in enumerate(palms, start=1):
        gt_bbox = palm_instance_bbox(palm)
        pred_bbox, score, best_iou = match_gt_to_best_prediction(gt_bbox, predictions)

        rows.append(
            {
                "image_name": image_name,
                "palm_id": f"palm_{index:02d}",
                "gt_bbox": json.dumps(bbox_to_list(gt_bbox)),
                "pred_bbox": json.dumps(bbox_to_list(pred_bbox)) if pred_bbox else "",
                "confidence": score if score is not None else "",
                "IoU": round(best_iou, 4),
                "matched": best_iou >= IOU_THRESHOLD,
            }
        )

    return rows


def main() -> None:
    if not PREDICTIONS_FULL_JSON.exists():
        print(f"Prediction file not found: {PREDICTIONS_FULL_JSON}")
        sys.exit(1)

    if not RAW_PATCHES_ROOT.exists():
        print(f"Raw_Patches directory not found: {RAW_PATCHES_ROOT}")
        sys.exit(1)

    records = load_predictions(PREDICTIONS_FULL_JSON)
    predictions_by_image = group_predictions_by_image(records)
    json_files = find_labelme_json_files(RAW_PATCHES_ROOT)

    rows: list[dict[str, Any]] = []
    for json_path in json_files:
        image_name = json_path.stem
        rows.extend(match_image(image_name, json_path, predictions_by_image.get(image_name, [])))

    df = pd.DataFrame(rows)
    GT_MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GT_MATCHES_CSV, index=False)

    total_gt = len(df)
    matched_count = int(df["matched"].sum()) if total_gt else 0
    match_rate = (matched_count / total_gt * 100) if total_gt else 0.0

    print("YOLO-to-ground-truth matching")
    print(f"  Predictions: {PREDICTIONS_FULL_JSON}")
    print(f"  Raw_Patches: {RAW_PATCHES_ROOT}")
    print(f"  IoU threshold: {IOU_THRESHOLD}")
    print(f"  Output CSV:  {GT_MATCHES_CSV}")
    print()
    print(f"images processed: {len(json_files)}")
    print(f"GT palms: {total_gt}")
    print(f"matched (IoU >= {IOU_THRESHOLD}): {matched_count} ({match_rate:.2f}%)")
    print()
    print(f"Saved {total_gt} rows to {GT_MATCHES_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
