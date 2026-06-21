"""Visual overlay variants for ablation studies."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from src.preprocessing.lvm_input_builder import (
    COLOR_BBOX,
    COLOR_CENTER,
    COLOR_ENDPOINT,
    COLOR_TEXT,
)
from src.preprocessing.palm_analyzer import PalmInstance

# Padding around the palm bbox so crops stay consistent across E1-E5.
CROP_PADDING = 10

VISUAL_VARIANTS = (
    "E1_raw_crop",
    "E2_bbox_only",
    "E3_endpoints_only",
    "E4_bbox_endpoints",
    "E5_full_overlay",
)


def crop_palm_region(
    image: np.ndarray,
    palm: PalmInstance,
    padding: int = CROP_PADDING,
) -> tuple[np.ndarray, PalmInstance]:
    """Crop a fixed region around the palm bbox and shift palm coordinates."""
    height, width = image.shape[:2]
    x1 = max(0, int(palm.bbox_x) - padding)
    y1 = max(0, int(palm.bbox_y) - padding)
    x2 = min(width, int(palm.bbox_x + palm.bbox_width) + padding)
    y2 = min(height, int(palm.bbox_y + palm.bbox_height) + padding)

    crop = image[y1:y2, x1:x2].copy()

    endpoint_points = [(px - x1, py - y1) for px, py in palm.endpoint_points]
    shifted = replace(
        palm,
        bbox_x=float(palm.bbox_x - x1),
        bbox_y=float(palm.bbox_y - y1),
        center_x=(palm.center_x - x1) if palm.center_x is not None else None,
        center_y=(palm.center_y - y1) if palm.center_y is not None else None,
        endpoint_points=endpoint_points,
    )
    return crop, shifted


def draw_ablation_variant(
    crop: np.ndarray,
    palm: PalmInstance,
    palm_id: str,
    visual_variant: str,
) -> np.ndarray:
    """Draw one ablation visual variant on a palm crop."""
    if visual_variant not in VISUAL_VARIANTS:
        raise ValueError(f"Unknown visual variant: {visual_variant}")

    output = crop.copy()
    x = int(palm.bbox_x)
    y = int(palm.bbox_y)
    w = int(palm.bbox_width)
    h = int(palm.bbox_height)

    if visual_variant == "E1_raw_crop":
        return output

    if visual_variant in {"E2_bbox_only", "E4_bbox_endpoints", "E5_full_overlay"}:
        cv2.rectangle(output, (x, y), (x + w, y + h), COLOR_BBOX, thickness=2)

    if visual_variant in {"E3_endpoints_only", "E4_bbox_endpoints", "E5_full_overlay"}:
        for index, (end_x, end_y) in enumerate(palm.endpoint_points, start=1):
            point = (int(end_x), int(end_y))
            cv2.circle(output, point, radius=6, color=COLOR_ENDPOINT, thickness=-1)
            cv2.putText(
                output,
                f"end{index}",
                (point[0] + 8, point[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_ENDPOINT,
                1,
                cv2.LINE_AA,
            )

    if visual_variant == "E5_full_overlay":
        if palm.center_x is not None and palm.center_y is not None:
            center = (int(palm.center_x), int(palm.center_y))
            cv2.circle(output, center, radius=7, color=COLOR_CENTER, thickness=-1)
            cv2.putText(
                output,
                "center",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_CENTER,
                1,
                cv2.LINE_AA,
            )

        label_y = max(y - 10, 20)
        cv2.putText(
            output,
            palm_id,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            palm_id,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COLOR_BBOX,
            1,
            cv2.LINE_AA,
        )

    return output
