"""Publication-quality OpenCV visualization for verification evaluation samples."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.preprocessing.json_parser import load_json
from src.yolo.predictions_io import iou_xywh

# BGR colors
COLOR_GT = (60, 200, 70)
COLOR_YOLO = (240, 130, 20)
COLOR_GT_CENTER = (40, 220, 80)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_OUTLINE = (20, 20, 20)
COLOR_PANEL = (25, 25, 25)


def parse_bbox_json(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, str):
        data = json.loads(value)
    else:
        data = value
    if not isinstance(data, (list, tuple)) or len(data) != 4:
        return None
    return tuple(float(v) for v in data)


def xywh_to_xyxy(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return int(round(x)), int(round(y)), int(round(x + width)), int(round(y + height))


def extract_gt_palm_bboxes(json_path: Path) -> list[tuple[float, float, float, float]]:
    """Extract rotation palm GT bboxes as xywh."""
    data = load_json(json_path)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for shape in data.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        group_id = shape.get("group_id")
        if group_id is None:
            continue
        grouped.setdefault(int(group_id), []).append(shape)

    bboxes: list[tuple[float, float, float, float]] = []
    for group_id in sorted(grouped):
        palm_shapes = [
            shape
            for shape in grouped[group_id]
            if shape.get("label") == "palm" and shape.get("shape_type") == "rotation"
        ]
        if not palm_shapes:
            continue
        points = palm_shapes[0].get("points", [])
        xs, ys = [], []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
        if not xs:
            continue
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bboxes.append((xmin, ymin, xmax - xmin, ymax - ymin))
    return bboxes


def find_gt_center_for_bbox(
    json_path: Path,
    reference_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float] | None:
    """Return the GT center point for the palm group best matching reference_bbox."""
    if reference_bbox is None:
        return None

    data = load_json(json_path)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for shape in data.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        group_id = shape.get("group_id")
        if group_id is None:
            continue
        grouped.setdefault(int(group_id), []).append(shape)

    best_iou = -1.0
    best_center: tuple[float, float] | None = None

    for group_id in sorted(grouped):
        shapes = grouped[group_id]
        palm_shapes = [
            shape
            for shape in shapes
            if shape.get("label") == "palm" and shape.get("shape_type") == "rotation"
        ]
        if not palm_shapes:
            continue

        points = palm_shapes[0].get("points", [])
        xs, ys = [], []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
        if not xs:
            continue

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        palm_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        overlap = iou_xywh(reference_bbox, palm_bbox)
        if overlap <= best_iou:
            continue

        center_point = None
        for shape in shapes:
            if shape.get("label") != "center":
                continue
            point = shape.get("points", [])
            if point and isinstance(point[0], (list, tuple)) and len(point[0]) >= 2:
                center_point = (float(point[0][0]), float(point[0][1]))
                break

        best_iou = overlap
        best_center = center_point

    return best_center


def _draw_text_with_outline(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        COLOR_TEXT_OUTLINE,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        COLOR_TEXT,
        thickness,
        cv2.LINE_AA,
    )


def _draw_bbox(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
    thickness: int,
    font_scale: float,
) -> None:
    x1, y1, x2, y2 = xywh_to_xyxy(bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    label_y = max(y1 - 8, int(22 * font_scale))
    _draw_text_with_outline(image, label, (x1, label_y), font_scale * 0.9, 1)


def render_verification_example(
    image: np.ndarray,
    *,
    yolo_bbox: tuple[float, float, float, float] | None,
    gt_bbox: tuple[float, float, float, float] | None,
    gt_center: tuple[float, float] | None,
    yolo_confidence: float | None,
    iou: float,
    verification_label: str,
    sample_id: str,
    ablation: str,
) -> np.ndarray:
    """Draw GT/YOLO boxes, center, and annotation panel onto a copy of the image."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    font_scale = max(width, height) / 1800.0
    line_thickness = max(2, int(round(font_scale * 2.5)))

    if gt_bbox is not None:
        _draw_bbox(canvas, gt_bbox, COLOR_GT, "GT", line_thickness, font_scale)
    if yolo_bbox is not None:
        _draw_bbox(canvas, yolo_bbox, COLOR_YOLO, "YOLO", line_thickness, font_scale)
    if gt_center is not None:
        center = (int(round(gt_center[0])), int(round(gt_center[1])))
        radius = max(5, int(round(font_scale * 8)))
        cv2.circle(canvas, center, radius, COLOR_GT_CENTER, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, radius + 1, COLOR_GT, thickness=1, lineType=cv2.LINE_AA)

    panel_height = int(round(150 * font_scale))
    panel = np.full((panel_height, width, 3), COLOR_PANEL, dtype=np.uint8)

    conf_text = f"{yolo_confidence:.3f}" if yolo_confidence is not None and not math.isnan(yolo_confidence) else "n/a"
    label_text = verification_label.strip() or "n/a"
    lines = [
        f"{sample_id}  |  {ablation}",
        f"YOLO confidence: {conf_text}    IoU: {iou:.3f}",
        f"Verification: {label_text}",
    ]

    y_offset = int(round(28 * font_scale))
    for line in lines:
        _draw_text_with_outline(panel, line, (int(round(18 * font_scale)), y_offset), font_scale, 1)
        y_offset += int(round(34 * font_scale))

    legend_x = width - int(round(260 * font_scale))
    legend_y = int(round(24 * font_scale))
    cv2.rectangle(
        panel,
        (legend_x - 10, legend_y - 10),
        (width - 10, legend_y + int(round(70 * font_scale))),
        (45, 45, 45),
        thickness=-1,
    )
    cv2.line(
        panel,
        (legend_x, legend_y + int(round(10 * font_scale))),
        (legend_x + int(round(40 * font_scale)), legend_y + int(round(10 * font_scale))),
        COLOR_GT,
        line_thickness,
        cv2.LINE_AA,
    )
    _draw_text_with_outline(panel, "GT bbox", (legend_x + int(round(50 * font_scale)), legend_y + int(round(14 * font_scale))), font_scale * 0.85, 1)
    cv2.line(
        panel,
        (legend_x, legend_y + int(round(38 * font_scale))),
        (legend_x + int(round(40 * font_scale)), legend_y + int(round(38 * font_scale))),
        COLOR_YOLO,
        line_thickness,
        cv2.LINE_AA,
    )
    _draw_text_with_outline(panel, "YOLO bbox", (legend_x + int(round(50 * font_scale)), legend_y + int(round(42 * font_scale))), font_scale * 0.85, 1)

    return np.vstack([canvas, panel])
