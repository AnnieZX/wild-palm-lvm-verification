"""Ground-truth palm geometry for publication overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.preprocessing.gt_palm_bboxes import axis_aligned_bbox_from_points
from src.preprocessing.json_parser import load_json
from src.yolo.predictions_io import iou_xywh


@dataclass(frozen=True)
class MatchedPalmGeometry:
    """Geometry for the GT palm group best matching a reference bbox."""

    gt_bbox: tuple[float, float, float, float] | None
    center: tuple[float, float] | None
    endpoints: tuple[tuple[float, float], ...]
    group_id: int | None


def extract_matched_palm_geometry(
    json_path: Path,
    reference_bbox: tuple[float, float, float, float] | None,
) -> MatchedPalmGeometry:
    """Return bbox, center, and endpoints for the GT palm group matching reference_bbox."""
    if reference_bbox is None or not json_path.exists():
        return MatchedPalmGeometry(None, None, (), None)

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
    best_group: int | None = None
    best_bbox: tuple[float, float, float, float] | None = None

    for group_id in sorted(grouped):
        palm_shapes = [shape for shape in grouped[group_id] if shape.get("label") == "palm"]
        if not palm_shapes:
            continue
        palm_bbox = axis_aligned_bbox_from_points(palm_shapes[0].get("points", []))
        if palm_bbox is None:
            continue
        overlap = iou_xywh(reference_bbox, palm_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_group = group_id
            best_bbox = palm_bbox

    if best_group is None:
        return MatchedPalmGeometry(None, None, (), None)

    center: tuple[float, float] | None = None
    endpoints: list[tuple[float, float]] = []
    for shape in grouped[best_group]:
        label = shape.get("label")
        points = shape.get("points", [])
        if not points or not isinstance(points[0], (list, tuple)) or len(points[0]) < 2:
            continue
        point = (float(points[0][0]), float(points[0][1]))
        if label == "center":
            center = point
        elif label == "end":
            endpoints.append(point)

    return MatchedPalmGeometry(best_bbox, center, tuple(endpoints), best_group)
