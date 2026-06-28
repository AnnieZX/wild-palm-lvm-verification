#!/usr/bin/env python3
"""Render overlay images for sample palm annotations."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from the project root when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.json_parser import parse_annotation_file
from src.preprocessing.overlay_renderer import render_overlay, save_overlay

IMAGE_DIR = PROJECT_ROOT / "data" / "samples" / "images"
JSON_DIR = PROJECT_ROOT / "data" / "samples" / "json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sample_overlays"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_images(image_dir: Path) -> list[Path]:
    """Return all image files in a folder, sorted by name."""
    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def main() -> None:
    if not IMAGE_DIR.exists():
        print(f"Image folder not found: {IMAGE_DIR}")
        sys.exit(1)

    if not JSON_DIR.exists():
        print(f"JSON folder not found: {JSON_DIR}")
        sys.exit(1)

    images = find_images(IMAGE_DIR)
    print(f"Found {len(images)} image(s) in {IMAGE_DIR}")

    processed: list[str] = []
    missing_json: list[str] = []
    skipped: list[str] = []

    for image_path in images:
        json_path = JSON_DIR / f"{image_path.stem}.json"

        if not json_path.exists():
            missing_json.append(image_path.name)
            print(f"  SKIP (missing JSON): {image_path.name}")
            continue

        annotation = parse_annotation_file(json_path)

        if annotation.format_name == "unknown":
            skipped.append(image_path.name)
            print(f"  SKIP (unknown JSON format): {image_path.name}")
            continue

        overlay = render_overlay(image_path, annotation)
        output_path = OUTPUT_DIR / image_path.name
        save_overlay(overlay, output_path)

        processed.append(image_path.name)
        print(
            f"  OK: {image_path.name} "
            f"({annotation.format_name}, {len(annotation.shapes)} shape(s)) "
            f"-> {output_path.relative_to(PROJECT_ROOT)}"
        )

    print()
    print("Summary")
    print(f"  Processed: {len(processed)}")
    print(f"  Missing JSON: {len(missing_json)}")
    print(f"  Skipped: {len(skipped)}")

    if processed:
        print("  Processed files:")
        for name in processed:
            print(f"    - {name}")

    if missing_json:
        print("  Missing JSON for:")
        for name in missing_json:
            print(f"    - {name}")

    if skipped:
        print("  Skipped files:")
        for name in skipped:
            print(f"    - {name}")


if __name__ == "__main__":
    main()
