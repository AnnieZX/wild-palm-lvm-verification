"""Extract ground-truth palm bounding boxes from LabelMe annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.preprocessing.json_parser import load_json

# Evaluation Protocol v2 (2026-09-27): palm labels are matched after stripping
# whitespace and lowercasing. Protocol v1 used an exact `label == "palm"` match,
# which silently dropped 486 LabelMe boxes labeled "Palm".
EVALUATION_PROTOCOL_VERSION = "v2"
PALM_LABEL = "palm"
GT_LABEL_RULE = 'isinstance(label, str) and label.strip().lower() == "palm"'


def is_palm_label(label: Any) -> bool:
    """Return True if a LabelMe shape label denotes a palm (Protocol v2)."""
    return isinstance(label, str) and label.strip().lower() == PALM_LABEL


def axis_aligned_bbox_from_points(
    points: list[Any],
) -> tuple[float, float, float, float] | None:
    """
    Compute axis-aligned bbox (x, y, width, height) from shape points.

    Uses xmin/min, ymin/min, xmax/max over every valid point.
    """
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    if not xs or not ys:
        return None

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    return xmin, ymin, xmax - xmin, ymax - ymin


def extract_gt_palm_bboxes(json_path: Path) -> list[tuple[float, float, float, float]]:
    """
    Extract axis-aligned GT palm bboxes from LabelMe JSON.

    Selects every shape whose label satisfies `is_palm_label`, regardless of
    shape_type (rectangle, rotation, polygon, point, etc.), and converts all
    points to an axis-aligned bounding box. Point shapes yield zero-area boxes.
    """
    data = load_json(json_path)
    bboxes: list[tuple[float, float, float, float]] = []

    for shape in data.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        if not is_palm_label(shape.get("label")):
            continue

        bbox = axis_aligned_bbox_from_points(shape.get("points", []))
        if bbox is not None:
            bboxes.append(bbox)

    return bboxes
