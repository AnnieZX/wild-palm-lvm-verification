"""Build per-palm overlay images and metadata for LVM verification."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.palm_analyzer import PalmInstance, extract_palm_instances

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# BGR colors for LVM input overlays.
COLOR_BBOX = (0, 200, 0)
COLOR_CENTER = (255, 0, 0)
COLOR_ENDPOINT = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)


def find_images(image_dir: Path) -> list[Path]:
    """Return all image files in a folder, sorted by name."""
    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def palm_id_label(index: int) -> str:
    """Return a readable palm ID such as palm_01."""
    return f"palm_{index:02d}"


def draw_palm_overlay(
    image: np.ndarray,
    palm: PalmInstance,
    palm_id: str,
) -> np.ndarray:
    """Draw one palm's bbox, center, endpoints, and ID onto a copy of the image."""
    overlay = image.copy()

    x = int(palm.bbox_x)
    y = int(palm.bbox_y)
    w = int(palm.bbox_width)
    h = int(palm.bbox_height)

    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BBOX, thickness=2)

    if palm.center_x is not None and palm.center_y is not None:
        center = (int(palm.center_x), int(palm.center_y))
        cv2.circle(overlay, center, radius=7, color=COLOR_CENTER, thickness=-1)
        cv2.putText(
            overlay,
            "center",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_CENTER,
            1,
            cv2.LINE_AA,
        )

    for index, (end_x, end_y) in enumerate(palm.endpoint_points, start=1):
        point = (int(end_x), int(end_y))
        cv2.circle(overlay, point, radius=6, color=COLOR_ENDPOINT, thickness=-1)
        cv2.putText(
            overlay,
            f"end{index}",
            (point[0] + 8, point[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_ENDPOINT,
            1,
            cv2.LINE_AA,
        )

    label_y = max(y - 10, 20)
    cv2.putText(
        overlay,
        palm_id,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        palm_id,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_BBOX,
        1,
        cv2.LINE_AA,
    )

    return overlay


def build_lvm_inputs(
    image_dir: Path,
    json_dir: Path,
    output_dir: Path,
    metadata_csv: Path,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """
    Create per-palm LVM input images and a metadata CSV.

    Returns:
        DataFrame with one row per palm.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    images_processed = 0
    palms_found = 0

    for image_path in find_images(image_dir):
        json_path = json_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            print(f"  SKIP (missing JSON): {image_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  SKIP (could not read image): {image_path.name}")
            continue

        palms = extract_palm_instances(json_path)
        if not palms:
            print(f"  SKIP (no palms): {image_path.name}")
            continue

        images_processed += 1
        print(f"  Processing {image_path.name} ({len(palms)} palm(s))")

        for palm_index, palm in enumerate(palms, start=1):
            palm_id = palm_id_label(palm_index)
            output_name = f"{image_path.stem}_{palm_id}.png"
            output_path = output_dir / output_name

            overlay = draw_palm_overlay(image, palm, palm_id)
            cv2.imwrite(str(output_path), overlay)

            if project_root is not None:
                lvm_input_path = str(output_path.relative_to(project_root))
            else:
                lvm_input_path = str(output_path)

            rows.append(
                {
                    "image_name": palm.image_name,
                    "lvm_input_path": lvm_input_path,
                    "palm_id": palm_id,
                    "bbox_x": palm.bbox_x,
                    "bbox_y": palm.bbox_y,
                    "bbox_width": palm.bbox_width,
                    "bbox_height": palm.bbox_height,
                    "bbox_area": palm.bbox_area,
                    "center_x": palm.center_x,
                    "center_y": palm.center_y,
                    "endpoints_count": palm.num_endpoints,
                }
            )
            palms_found += 1
            print(f"    -> {output_name}")

    df = pd.DataFrame(rows)
    df.to_csv(metadata_csv, index=False)

    print()
    print("LVM input build complete")
    print(f"  Images processed: {images_processed}")
    print(f"  Palms found: {palms_found}")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata CSV: {metadata_csv}")

    return df
