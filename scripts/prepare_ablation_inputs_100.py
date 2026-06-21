#!/usr/bin/env python3
"""Prepare 100-palm ablation visual inputs for E1-E5 overlay variants."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ablation_overlay import (
    VISUAL_VARIANTS,
    crop_palm_region,
    draw_ablation_variant,
)
from src.preprocessing.palm_analyzer import extract_palm_instances_in_annotation_order
from src.preprocessing.sequential_dataset import (
    TARGET_PALM_COUNT,
    find_json_files,
    global_palm_id,
    load_json_metadata,
    resolve_image_path,
)

DATASET_ROOT = Path("/deac/csc/yangGrp/cuij/palm/Raw_Patches")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ablation_inputs_100"
METADATA_CSV = PROJECT_ROOT / "outputs" / "ablation_metadata_100.csv"


def output_dir_for_variant(visual_variant: str) -> Path:
    return OUTPUT_ROOT / visual_variant


def validate_metadata_files(metadata_df: pd.DataFrame) -> None:
    """Confirm all five visual input paths exist for every row."""
    path_columns = [f"{variant}_path" for variant in VISUAL_VARIANTS]
    missing: list[str] = []

    for _, row in metadata_df.iterrows():
        for column in path_columns:
            path = PROJECT_ROOT / row[column]
            if not path.exists():
                missing.append(str(row[column]))

    if missing:
        sample = missing[:5]
        raise FileNotFoundError(
            "Missing ablation input files:\n  "
            + "\n  ".join(sample)
            + (f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else "")
        )


def main() -> None:
    json_files = find_json_files(DATASET_ROOT)

    for variant in VISUAL_VARIANTS:
        output_dir_for_variant(variant).mkdir(parents=True, exist_ok=True)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    json_scanned = 0
    palms_collected = 0

    print("Preparing 100-palm ablation inputs")
    print(f"  Dataset root: {DATASET_ROOT}")
    print(f"  Target palms: {TARGET_PALM_COUNT}")
    print()

    for json_path in json_files:
        if palms_collected >= TARGET_PALM_COUNT:
            break

        json_scanned += 1
        data = load_json_metadata(json_path)
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
            crop, shifted_palm = crop_palm_region(image, palm)

            row = {
                "sample_index": palms_collected,
                "image_name": palm.image_name,
                "palm_id": palm_id,
                "source_image_path": str(image_path),
                "source_json_path": str(json_path),
                "bbox_x": palm.bbox_x,
                "bbox_y": palm.bbox_y,
                "bbox_width": palm.bbox_width,
                "bbox_height": palm.bbox_height,
                "bbox_area": palm.bbox_area,
                "endpoints_count": palm.num_endpoints,
                "yolo_confidence": palm.yolo_confidence,
            }

            for variant in VISUAL_VARIANTS:
                output_name = f"seq_{palms_collected:03d}_{json_path.stem}_{palm_id}.png"
                output_path = output_dir_for_variant(variant) / output_name
                variant_image = draw_ablation_variant(crop, shifted_palm, palm_id, variant)
                cv2.imwrite(str(output_path), variant_image)
                row[f"{variant}_path"] = str(output_path.relative_to(PROJECT_ROOT))

            rows.append(row)
            print(f"    -> {palm_id}")

    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(METADATA_CSV, index=False)

    print()
    print("Ablation input preparation complete")
    print(f"  JSON files scanned: {json_scanned}")
    print(f"  Palm instances collected: {palms_collected}")
    print(f"  Metadata CSV: {METADATA_CSV}")

    if palms_collected != TARGET_PALM_COUNT:
        print(
            f"WARNING: Expected {TARGET_PALM_COUNT} palms, collected {palms_collected}."
        )
        sys.exit(1)

    validate_metadata_files(metadata_df)
    print("Sanity check passed: all five input paths exist for each row.")


if __name__ == "__main__":
    main()
