"""Verification dataset index and YOLO prediction lookup helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.yolo.predictions_io import extract_score, iou_xywh


def index_bbox_from_row(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(row["bbox_x"]),
        float(row["bbox_y"]),
        float(row["bbox_width"]),
        float(row["bbox_height"]),
    )


def find_yolo_prediction(
    image_name: str,
    target_bbox: tuple[float, float, float, float],
    predictions_by_image: dict[str, list[dict[str, Any]]],
) -> tuple[tuple[float, float, float, float] | None, float | None]:
    """Locate the YOLO prediction that best matches the verification sample bbox."""
    predictions = predictions_by_image.get(image_name, [])
    if not predictions:
        return None, None

    best_iou = -1.0
    best_bbox: tuple[float, float, float, float] | None = None
    best_score: float | None = None

    for prediction in predictions:
        bbox = prediction.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        pred_bbox = tuple(float(v) for v in bbox)
        overlap = iou_xywh(target_bbox, pred_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_bbox = pred_bbox
            best_score = extract_score(prediction)

    return best_bbox, best_score
