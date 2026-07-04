"""Build a per-detection verification dataset from YOLO predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from src.preprocessing.verification_overlay import render_single_detection_overlay
from src.yolo.predictions_io import (
    extract_bbox,
    extract_image_id,
    extract_score,
    filter_by_score,
    load_predictions,
)

INDEX_COLUMNS = (
    "sample_id",
    "image_name",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "center_x",
    "center_y",
    "confidence",
    "image_path",
    "metadata_path",
)


def sample_id_label(index: int) -> str:
    """Return a zero-padded sample id such as sample_000001."""
    return f"sample_{index:06d}"


def build_detection_metadata(
    sample_id: str,
    image_name: str,
    bbox: tuple[float, float, float, float],
    confidence: float,
) -> dict[str, Any]:
    """Build metadata JSON for one YOLO detection."""
    x, y, width, height = bbox
    return {
        "sample_id": sample_id,
        "image_name": image_name,
        "bbox": [float(x), float(y), float(width), float(height)],
        "confidence": float(confidence),
        "bbox_width": float(width),
        "bbox_height": float(height),
        "bbox_area": float(width * height),
        "center_x": float(x + width / 2.0),
        "center_y": float(y + height / 2.0),
    }


def metadata_to_index_row(
    metadata: dict[str, Any],
    image_rel_path: str,
    metadata_rel_path: str,
) -> dict[str, Any]:
    """Flatten metadata into one index.csv row."""
    x, y, width, height = metadata["bbox"]
    return {
        "sample_id": metadata["sample_id"],
        "image_name": metadata["image_name"],
        "bbox_x": x,
        "bbox_y": y,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": metadata["bbox_area"],
        "center_x": metadata["center_x"],
        "center_y": metadata["center_y"],
        "confidence": metadata["confidence"],
        "image_path": image_rel_path,
        "metadata_path": metadata_rel_path,
    }


def resolve_patch_image(images_root: Path, image_id: str) -> Path | None:
    """Find the raw patch PNG for a prediction image_id."""
    candidates = [
        images_root / f"{image_id}.png",
        images_root / image_id,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(images_root.rglob(f"{image_id}.png"))
    return matches[0] if matches else None


def iter_verification_samples(
    records: list[dict[str, Any]],
    confidence_threshold: float,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Return (image_id, prediction) pairs above the confidence threshold.

    Predictions are sorted by image_id then by descending confidence for stable ordering.
    """
    samples: list[tuple[str, dict[str, Any]]] = []

    for record in records:
        image_id = extract_image_id(record)
        bbox = extract_bbox(record)
        score = extract_score(record)
        if image_id is None or bbox is None or score is None:
            continue
        if score < confidence_threshold:
            continue
        samples.append(
            (
                image_id,
                {"bbox": bbox, "score": score},
            )
        )

    samples.sort(key=lambda item: (item[0], -item[1]["score"]))
    return samples


def build_verification_dataset(
    images_root: Path,
    predictions_path: Path,
    output_dir: Path,
    confidence_threshold: float = 0.5,
    progress_interval: int = 500,
) -> pd.DataFrame:
    """
    Convert filtered YOLO detections into verification samples.

    Writes:
        output_dir/images/sample_XXXXXX.png
        output_dir/metadata/sample_XXXXXX.json
        output_dir/index.csv
    """
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    records = load_predictions(predictions_path)
    samples = iter_verification_samples(records, confidence_threshold)

    index_rows: list[dict[str, Any]] = []
    missing_images: set[str] = set()
    sample_counter = 0

    for index, (image_id, prediction) in enumerate(samples, start=1):
        image_path = resolve_patch_image(images_root, image_id)
        if image_path is None:
            missing_images.add(image_id)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.add(image_id)
            continue

        sample_counter += 1
        sid = sample_id_label(sample_counter)
        bbox = prediction["bbox"]
        confidence = prediction["score"]

        overlay = render_single_detection_overlay(image, bbox)
        overlay_rel = f"images/{sid}.png"
        metadata_rel = f"metadata/{sid}.json"
        overlay_path = output_dir / overlay_rel
        metadata_path = output_dir / metadata_rel

        cv2.imwrite(str(overlay_path), overlay)

        metadata = build_detection_metadata(sid, image_id, bbox, confidence)
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        index_rows.append(metadata_to_index_row(metadata, overlay_rel, metadata_rel))

        if progress_interval and (index % progress_interval == 0 or index == len(samples)):
            print(f"Processed {index}/{len(samples)} detections ({sample_counter} samples written)")

    index_df = pd.DataFrame(index_rows, columns=list(INDEX_COLUMNS))
    index_df.to_csv(output_dir / "index.csv", index=False)

    if missing_images:
        print(f"Skipped detections for {len(missing_images)} image(s) with missing/unreadable PNG.")

    return index_df
