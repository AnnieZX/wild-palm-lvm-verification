"""Load annotation JSON files and convert them into simple shape objects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Shape:
    """One annotated region from an image."""

    label: str
    points: list[tuple[float, float]]
    shape_type: str  # "bbox", "polygon", or "point"


@dataclass
class AnnotationFile:
    """Parsed contents of one annotation file."""

    format_name: str
    image_path: str | None
    shapes: list[Shape]


def load_json(path: Path | str) -> dict[str, Any]:
    """Read a JSON file and return its contents as a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def detect_format(data: dict[str, Any]) -> str:
    """
    Guess which annotation format the JSON uses.

    Supported formats:
    - labelme: has a top-level "shapes" list
    - coco: has top-level "images" and "annotations" lists
    - unknown: anything else
    """
    if isinstance(data.get("shapes"), list):
        return "labelme"
    if isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list):
        return "coco"
    return "unknown"


def _normalize_points(raw_points: list[Any]) -> list[tuple[float, float]]:
    """Convert point lists like [[x, y], ...] into (x, y) tuples."""
    points: list[tuple[float, float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        points.append((float(point[0]), float(point[1])))
    return points


def _classify_shape(points: list[tuple[float, float]]) -> str:
    """Decide how to draw a shape based on how many points it has."""
    if len(points) == 1:
        return "point"
    if len(points) == 2:
        return "bbox"
    return "polygon"


def parse_labelme_shapes(data: dict[str, Any]) -> list[Shape]:
    """Parse LabelMe-style JSON with 'shapes', 'label', and 'points'."""
    shapes: list[Shape] = []
    for raw_shape in data.get("shapes", []):
        if not isinstance(raw_shape, dict):
            continue

        label = str(raw_shape.get("label", "unknown"))
        points = _normalize_points(raw_shape.get("points", []))
        if not points:
            continue

        shapes.append(
            Shape(
                label=label,
                points=points,
                shape_type=_classify_shape(points),
            )
        )
    return shapes


def parse_coco_shapes(data: dict[str, Any]) -> list[Shape]:
    """Parse a simple COCO-style JSON file."""
    shapes: list[Shape] = []
    for annotation in data.get("annotations", []):
        if not isinstance(annotation, dict):
            continue

        label = str(annotation.get("category_id", "unknown"))

        bbox = annotation.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, width, height = [float(value) for value in bbox]
            points = [(x, y), (x + width, y + height)]
            shapes.append(Shape(label=label, points=points, shape_type="bbox"))
            continue

        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list) and segmentation:
            flat = segmentation[0]
            if isinstance(flat, list) and len(flat) >= 6:
                points = [
                    (float(flat[index]), float(flat[index + 1]))
                    for index in range(0, len(flat) - 1, 2)
                ]
                shapes.append(Shape(label=label, points=points, shape_type="polygon"))

    return shapes


def parse_annotation_file(path: Path | str) -> AnnotationFile:
    """Load one JSON file and return normalized shapes."""
    path = Path(path)
    data = load_json(path)
    format_name = detect_format(data)

    if format_name == "labelme":
        shapes = parse_labelme_shapes(data)
    elif format_name == "coco":
        shapes = parse_coco_shapes(data)
    else:
        shapes = []

    image_path = data.get("imagePath")
    if image_path is not None:
        image_path = str(image_path)

    return AnnotationFile(
        format_name=format_name,
        image_path=image_path,
        shapes=shapes,
    )
