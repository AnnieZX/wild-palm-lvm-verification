"""Load and manipulate YOLO/COCO prediction JSON exports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PredictionDict = dict[str, Any]


def load_predictions(path: Path) -> list[PredictionDict]:
    """Load predictions.json and return a flat list of prediction records."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected predictions root type: {type(data)}")

    if isinstance(data.get("predictions"), list):
        return [record for record in data["predictions"] if isinstance(record, dict)]

    if isinstance(data.get("annotations"), list):
        images = {
            str(image["id"]): image
            for image in data.get("images", [])
            if isinstance(image, dict) and "id" in image
        }
        flat: list[PredictionDict] = []
        for annotation in data["annotations"]:
            if not isinstance(annotation, dict):
                continue
            record = dict(annotation)
            image_ref = record.get("image_id")
            if image_ref is not None and str(image_ref) in images:
                image_info = images[str(image_ref)]
                for key in ("file_name", "filename", "imagePath"):
                    if key in image_info:
                        record.setdefault("file_name", image_info[key])
            flat.append(record)
        return flat

    if all(isinstance(value, dict) for value in data.values()):
        return list(data.values())

    raise ValueError("Could not interpret predictions.json structure.")


def normalize_image_id(value: Any) -> str | None:
    """Return a stem-style image id such as 100_0003_0001_1."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).stem


def extract_image_id(record: PredictionDict) -> str | None:
    """Read image_id from common YOLO / COCO export field names."""
    for key in ("image_id", "imageId", "image", "file_name", "filename", "imagePath"):
        if key in record:
            image_id = normalize_image_id(record[key])
            if image_id:
                return image_id
    return None


def extract_bbox(record: PredictionDict) -> tuple[float, float, float, float] | None:
    """Read bbox as (x, y, width, height)."""
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


def extract_score(record: PredictionDict) -> float | None:
    """Read detector confidence from common field names."""
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


def group_predictions_by_image(
    records: list[PredictionDict],
) -> dict[str, list[PredictionDict]]:
    """
    Group valid predictions by normalized image_id.

    Each grouped entry has keys: bbox (x,y,w,h tuple), score (float|None).
    """
    grouped: dict[str, list[PredictionDict]] = defaultdict(list)
    skipped = 0

    for record in records:
        image_id = extract_image_id(record)
        bbox = extract_bbox(record)
        if image_id is None or bbox is None:
            skipped += 1
            continue
        grouped[image_id].append({"bbox": bbox, "score": extract_score(record)})

    if skipped:
        print(f"Skipped {skipped} prediction record(s) missing image_id or bbox.")

    return dict(grouped)


def count_predictions_by_image(records: list[PredictionDict]) -> dict[str, int]:
    """Count valid predictions grouped by image_id."""
    grouped = group_predictions_by_image(records)
    return {image_id: len(items) for image_id, items in grouped.items()}


def iou_xywh(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU for axis-aligned boxes in (x, y, width, height) format."""
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


def filter_by_score(
    predictions: list[PredictionDict],
    threshold: float,
) -> list[PredictionDict]:
    """Keep predictions with score >= threshold."""
    kept: list[PredictionDict] = []
    for prediction in predictions:
        score = prediction.get("score")
        if score is None:
            continue
        if score >= threshold:
            kept.append(prediction)
    return kept


def apply_nms(
    predictions: list[PredictionDict],
    iou_threshold: float,
) -> list[PredictionDict]:
    """Greedy non-maximum suppression on (x, y, w, h) boxes sorted by score."""
    if not predictions:
        return []

    indexed = list(enumerate(predictions))
    indexed.sort(
        key=lambda item: item[1].get("score") if item[1].get("score") is not None else -1.0,
        reverse=True,
    )

    keep_indices: list[int] = []
    while indexed:
        best_pos, best_item = indexed.pop(0)
        keep_indices.append(best_pos)
        best_bbox = best_item["bbox"]
        indexed = [
            (pos, item)
            for pos, item in indexed
            if iou_xywh(best_bbox, item["bbox"]) < iou_threshold
        ]

    keep_indices.sort()
    return [predictions[index] for index in keep_indices]


def find_png_images(images_root: Path) -> list[Path]:
    """Return all PNG files under images_root, sorted by path."""
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    return sorted(path for path in images_root.rglob("*.png") if path.is_file())
