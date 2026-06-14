"""Extract per-palm statistics from LabelMe annotation files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.preprocessing.json_parser import load_json


@dataclass
class PalmInstance:
    """Statistics for one palm annotation group."""

    image_name: str
    group_id: int
    num_endpoints: int
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    center_x: float | None
    center_y: float | None
    endpoint_points: list[tuple[float, float]]
    endpoint_distances: list[float]

    @property
    def dist_mean(self) -> float | None:
        if not self.endpoint_distances:
            return None
        return sum(self.endpoint_distances) / len(self.endpoint_distances)

    @property
    def dist_min(self) -> float | None:
        if not self.endpoint_distances:
            return None
        return min(self.endpoint_distances)

    @property
    def dist_max(self) -> float | None:
        if not self.endpoint_distances:
            return None
        return max(self.endpoint_distances)


def _point_from_shape(shape: dict[str, Any]) -> tuple[float, float] | None:
    points = shape.get("points", [])
    if not points or not isinstance(points[0], (list, tuple)) or len(points[0]) < 2:
        return None
    return float(points[0][0]), float(points[0][1])


def _bbox_from_points(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float, float]:
    """Return x, y, width, height, and area for an axis-aligned bounding box."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox_x = min(xs)
    bbox_y = min(ys)
    width = max(xs) - bbox_x
    height = max(ys) - bbox_y
    return bbox_x, bbox_y, width, height, width * height


def _distance(
    center: tuple[float, float],
    endpoint: tuple[float, float],
) -> float:
    return math.hypot(center[0] - endpoint[0], center[1] - endpoint[1])


def extract_palm_instances(json_path: Path | str) -> list[PalmInstance]:
    """Read one JSON file and return one record per palm instance."""
    json_path = Path(json_path)
    data = load_json(json_path)
    image_name = str(data.get("imagePath") or json_path.stem)

    grouped_shapes: dict[int, list[dict[str, Any]]] = {}
    for shape in data.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        group_id = shape.get("group_id")
        if group_id is None:
            continue
        grouped_shapes.setdefault(int(group_id), []).append(shape)

    palms: list[PalmInstance] = []
    for group_id, shapes in sorted(grouped_shapes.items()):
        palm_shapes = [shape for shape in shapes if shape.get("label") == "palm"]
        if not palm_shapes:
            continue

        palm_shape = palm_shapes[0]
        raw_points = palm_shape.get("points", [])
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
        if not points:
            continue

        bbox_x, bbox_y, width, height, area = _bbox_from_points(points)

        center_point = None
        for shape in shapes:
            if shape.get("label") == "center":
                center_point = _point_from_shape(shape)
                break

        endpoint_points: list[tuple[float, float]] = []
        for shape in shapes:
            if shape.get("label") == "end":
                point = _point_from_shape(shape)
                if point is not None:
                    endpoint_points.append(point)

        distances: list[float] = []
        if center_point is not None:
            distances = [_distance(center_point, endpoint) for endpoint in endpoint_points]

        palms.append(
            PalmInstance(
                image_name=image_name,
                group_id=group_id,
                num_endpoints=len(endpoint_points),
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_width=width,
                bbox_height=height,
                bbox_area=area,
                center_x=center_point[0] if center_point else None,
                center_y=center_point[1] if center_point else None,
                endpoint_points=endpoint_points,
                endpoint_distances=distances,
            )
        )

    return palms


def extract_all_palms(json_dir: Path | str) -> list[PalmInstance]:
    """Extract palm instances from every JSON file in a folder."""
    json_dir = Path(json_dir)
    all_palms: list[PalmInstance] = []
    for json_path in sorted(json_dir.glob("*.json")):
        all_palms.extend(extract_palm_instances(json_path))
    return all_palms
