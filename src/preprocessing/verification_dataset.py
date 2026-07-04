"""Build a per-detection verification dataset from YOLO predictions."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.verification_overlay import (
    DEFAULT_DIM_FACTOR,
    render_single_detection_overlay,
)
from src.yolo.predictions_io import (
    extract_bbox,
    extract_image_id,
    extract_score,
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


@dataclass(frozen=True)
class ImageGenerationJob:
    """Work unit for one Raw Patch and all of its detections."""

    image_id: str
    image_path: str
    output_dir: str
    detections: tuple[tuple[str, tuple[float, float, float, float], float], ...]


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


def build_png_index(images_root: Path) -> dict[str, Path]:
    """
    Build a one-time image_id → PNG path lookup.

    Matches resolve_patch_image priority:
    1. images_root/{image_id}.png
    2. images_root/{image_id}
    3. first sorted rglob match for {image_id}.png
    """
    index: dict[str, Path] = {}
    for path in sorted(images_root.rglob("*.png")):
        if path.is_file():
            index.setdefault(path.stem, path)

    if images_root.is_dir():
        for path in images_root.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() == ".png":
                index[path.stem] = path
            else:
                index[path.name] = path

    return index


def resolve_patch_image(images_root: Path, image_id: str) -> Path | None:
    """Find the raw patch PNG for a prediction image_id."""
    return resolve_patch_image_from_index(images_root, image_id, build_png_index(images_root))


def resolve_patch_image_from_index(
    images_root: Path,
    image_id: str,
    png_index: dict[str, Path],
) -> Path | None:
    """Resolve image path using a prebuilt PNG index."""
    direct_png = images_root / f"{image_id}.png"
    if direct_png.is_file():
        return direct_png

    direct_path = images_root / image_id
    if direct_path.is_file():
        return direct_path

    return png_index.get(image_id)


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


def group_samples_by_image(
    samples: list[tuple[str, dict[str, Any]]],
) -> dict[str, list[tuple[str, tuple[float, float, float, float], float]]]:
    """
    Assign global sample IDs and group detections by source image.

    Preserves the same sample_id ordering as sequential processing.
    """
    grouped: dict[str, list[tuple[str, tuple[float, float, float, float], float]]] = {}
    sample_counter = 0

    for image_id, prediction in samples:
        sample_counter += 1
        sample_id = sample_id_label(sample_counter)
        bbox = prediction["bbox"]
        confidence = prediction["score"]
        grouped.setdefault(image_id, []).append((sample_id, bbox, confidence))

    return grouped


def _write_sample_outputs(
    output_dir: Path,
    image_id: str,
    sample_id: str,
    bbox: tuple[float, float, float, float],
    confidence: float,
    overlay: np.ndarray,
) -> dict[str, Any]:
    """Write overlay image and metadata JSON; return one index row."""
    overlay_rel = f"images/{sample_id}.png"
    metadata_rel = f"metadata/{sample_id}.json"
    overlay_path = output_dir / overlay_rel
    metadata_path = output_dir / metadata_rel

    cv2.imwrite(str(overlay_path), overlay)

    metadata = build_detection_metadata(sample_id, image_id, bbox, confidence)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata_to_index_row(metadata, overlay_rel, metadata_rel)


def _process_image_job(job: ImageGenerationJob) -> dict[str, Any]:
    """
    Load one Raw Patch once and generate all verification samples for that image.

    Runs in a worker process; must remain picklable and self-contained.
    """
    output_dir = Path(job.output_dir)
    image_path = Path(job.image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        return {"index_rows": [], "missing_image_id": job.image_id}

    image_size = (int(image.shape[0]), int(image.shape[1]))
    dimmed_base = (image.astype(np.float32) * DEFAULT_DIM_FACTOR).clip(0, 255).astype(np.uint8)

    index_rows: list[dict[str, Any]] = []
    for sample_id, bbox, confidence in job.detections:
        overlay = render_single_detection_overlay(
            image,
            bbox,
            dimmed_base=dimmed_base,
            image_size=image_size,
        )
        index_rows.append(
            _write_sample_outputs(output_dir, job.image_id, sample_id, bbox, confidence, overlay)
        )

    return {"index_rows": index_rows, "missing_image_id": None}


def build_verification_dataset(
    images_root: Path,
    predictions_path: Path,
    output_dir: Path,
    confidence_threshold: float = 0.5,
    progress_interval: int = 100,
    num_workers: int | None = None,
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
    grouped = group_samples_by_image(samples)

    png_index = build_png_index(images_root)
    output_dir_str = str(output_dir.resolve())

    jobs: list[ImageGenerationJob] = []
    missing_images: set[str] = set()

    for image_id, detections in grouped.items():
        image_path = resolve_patch_image_from_index(images_root, image_id, png_index)
        if image_path is None:
            missing_images.add(image_id)
            continue

        jobs.append(
            ImageGenerationJob(
                image_id=image_id,
                image_path=str(image_path.resolve()),
                output_dir=output_dir_str,
                detections=tuple(detections),
            )
        )

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)

    index_rows: list[dict[str, Any]] = []
    total_images = len(jobs)
    completed_images = 0
    started_at = time.perf_counter()

    if num_workers == 1:
        for job in jobs:
            result = _process_image_job(job)
            if result["missing_image_id"]:
                missing_images.add(result["missing_image_id"])
            else:
                index_rows.extend(result["index_rows"])

            completed_images += 1
            if progress_interval and (
                completed_images % progress_interval == 0 or completed_images == total_images
            ):
                elapsed = time.perf_counter() - started_at
                rate = completed_images / elapsed if elapsed > 0 else 0.0
                print(
                    f"Processed {completed_images}/{total_images} images "
                    f"({len(index_rows)} samples, {rate:.1f} img/s)"
                )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_image_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                if result["missing_image_id"]:
                    missing_images.add(result["missing_image_id"])
                else:
                    index_rows.extend(result["index_rows"])

                completed_images += 1
                if progress_interval and (
                    completed_images % progress_interval == 0 or completed_images == total_images
                ):
                    elapsed = time.perf_counter() - started_at
                    rate = completed_images / elapsed if elapsed > 0 else 0.0
                    print(
                        f"Processed {completed_images}/{total_images} images "
                        f"({len(index_rows)} samples, {rate:.1f} img/s)"
                    )

    index_rows.sort(key=lambda row: row["sample_id"])
    index_df = pd.DataFrame(index_rows, columns=list(INDEX_COLUMNS))
    index_df.to_csv(output_dir / "index.csv", index=False)

    if missing_images:
        print(f"Skipped detections for {len(missing_images)} image(s) with missing/unreadable PNG.")

    return index_df
