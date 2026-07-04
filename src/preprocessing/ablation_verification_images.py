"""Image variants for verification input ablation."""

from __future__ import annotations

import cv2
import numpy as np

A5_CROP_PADDING = 15
A5_OUTPUT_SIZE = (512, 512)


def _clamp_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Return clamped integer bbox as x1, y1, x2, y2."""
    x, y, width, height = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image_width, int(round(x + width)))
    y2 = min(image_height, int(round(y + height)))
    return x1, y1, x2, y2


def _padded_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    """Return padded and clamped crop bounds as x1, y1, x2, y2."""
    x1, y1, x2, y2 = _clamp_bbox(bbox, image_width, image_height)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)
    return x1, y1, x2, y2


def build_a4_combined_image(
    overlay: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Build A4 dual-panel image: smaller full overlay (left) + enlarged crop (right).

    Both panels share the same height as the source overlay.
    """
    image_height, image_width = overlay.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, image_width, image_height)

    if x2 > x1 and y2 > y1:
        crop = overlay[y1:y2, x1:x2]
    else:
        crop = overlay.copy()

    canvas_height = image_height
    canvas_width = image_width * 2
    left_width = canvas_width // 2
    right_width = canvas_width - left_width

    left_panel = cv2.resize(
        overlay,
        (left_width, canvas_height),
        interpolation=cv2.INTER_AREA,
    )
    right_panel = cv2.resize(
        crop,
        (right_width, canvas_height),
        interpolation=cv2.INTER_LINEAR,
    )

    combined = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    combined[:, :left_width] = left_panel
    combined[:, left_width:] = right_panel
    return combined


def build_a5_crop_only_image(
    raw_patch: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding: int = A5_CROP_PADDING,
    output_size: tuple[int, int] = A5_OUTPUT_SIZE,
) -> np.ndarray:
    """
    Build A5 crop-only image from the raw patch (no overlay, no full context).

    Crops the YOLO bbox with padding, then resizes to a fixed output size.
    """
    image_height, image_width = raw_patch.shape[:2]
    x1, y1, x2, y2 = _padded_bbox(bbox, image_width, image_height, padding)

    if x2 > x1 and y2 > y1:
        crop = raw_patch[y1:y2, x1:x2].copy()
    else:
        crop = raw_patch.copy()

    return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
