#!/usr/bin/env python3
"""Prepare the first 100 palms in deterministic sorted order for LVM verification."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.json_parser import load_json
from src.preprocessing.lvm_input_builder import draw_palm_overlay
from src.preprocessing.palm_analyzer import extract_palm_instances_in_annotation_order

DATASET_ROOT = Path("/deac/csc/yangGrp/cuij/palm/Raw_Patches")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lvm_inputs_100_sequential"
METADATA_CSV = PROJECT_ROOT / "outputs" / "lvm_inputs_metadata_100_sequential.csv"

TARGET_PALM_COUNT = 100
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def find_json_files(dataset_root: Path) -> list[Path]:
    """Return all JSON files under the dataset root, sorted alphabetically."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    return sorted(dataset_root.rglob("*.json"))


def resolve_image_path(json_path: Path, data: dict) -> Path | None:
    """Find the PNG/image file that matches one JSON annotation."""
    image_path_value = data.get("imagePath")
    if image_path_value:
        candidates = [
            json_path.parent / str(image_path_value),
            json_path.parent / Path(str(image_path_value)).name,
            dataset_root_candidate(json_path, str(image_path_value)),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    for extension in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(extension)
        if candidate.exists():
            return candidate

    for extension in IMAGE_EXTENSIONS:
        for candidate in json_path.parent.glob(f"{json_path.stem}{extension}"):
            if candidate.exists():
                return candidate

    return None


def dataset_root_candidate(json_path: Path, image_path_value: str) -> Path:
    """Build a fallback path relative to the JSON directory."""
    return json_path.parent / Path(image_path_value).name


def global_palm_id(index: int) -> str:
    """Return a global sequential palm ID such as palm_001."""
    return f"palm_{index:03d}"


def main() -> None:
    json_files = find_json_files(DATASET_ROOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    json_scanned = 0
    palms_collected = 0

    print("Preparing 100 sequential LVM inputs")
    print(f"  Dataset root: {DATASET_ROOT}")
    print(f"  Target palms: {TARGET_PALM_COUNT}")
    print()

    for json_path in json_files:
        if palms_collected >= TARGET_PALM_COUNT:
            break

        json_scanned += 1
        data = load_json(json_path)
        image_path = resolve_image_path(json_path, data)
        if image_path is None:
            print(f"  SKIP (missing image): {json_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  SKIP (could not read image): {image_path.name}")
            continue

        palms = extract_palm_instances_in_annotation_order(json_path)
        if not palms:
            continue

        print(f"  {json_path.name}: {len(palms)} palm(s)")

        for palm in palms:
            if palms_collected >= TARGET_PALM_COUNT:
                break

            palms_collected += 1
            palm_id = global_palm_id(palms_collected)
            output_name = f"seq_{palms_collected:03d}_{json_path.stem}_{palm_id}.png"
            output_path = OUTPUT_DIR / output_name

            overlay = draw_palm_overlay(image, palm, palm_id)
            cv2.imwrite(str(output_path), overlay)

            row = {
                "image_name": palm.image_name,
                "palm_id": palm_id,
                "source_image_path": str(image_path),
                "source_json_path": str(json_path),
                "lvm_input_path": str(output_path.relative_to(PROJECT_ROOT)),
                "bbox_x": palm.bbox_x,
                "bbox_y": palm.bbox_y,
                "bbox_width": palm.bbox_width,
                "bbox_height": palm.bbox_height,
                "bbox_area": palm.bbox_area,
                "endpoints_count": palm.num_endpoints,
                "yolo_confidence": palm.yolo_confidence,
            }
            rows.append(row)
            print(f"    -> {output_name}")

    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(METADATA_CSV, index=False)

    print()
    print("Sequential LVM input build complete")
    print(f"  Total JSON scanned: {json_scanned}")
    print(f"  Total palm instances collected: {palms_collected}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Metadata CSV: {METADATA_CSV}")

    if palms_collected < TARGET_PALM_COUNT:
        print()
        print(
            f"WARNING: Only collected {palms_collected} palms "
            f"(target was {TARGET_PALM_COUNT})."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
