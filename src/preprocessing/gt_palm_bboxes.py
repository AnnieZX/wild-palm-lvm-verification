"""Extract ground-truth palm bounding boxes from LabelMe annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.preprocessing.json_parser import load_json


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

    Selects every shape with label == "palm", regardless of shape_type
    (rectangle, rotation, polygon, etc.), and converts all points to an
    axis-aligned bounding box.
    """
    data = load_json(json_path)
    bboxes: list[tuple[float, float, float, float]] = []

    for shape in data.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        if shape.get("label") != "palm":
            continue

        bbox = axis_aligned_bbox_from_points(shape.get("points", []))
        if bbox is not None:
            bboxes.append(bbox)

    return bboxes
