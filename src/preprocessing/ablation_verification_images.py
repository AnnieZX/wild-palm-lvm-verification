"""Image variants for verification input ablation."""

from __future__ import annotations

import cv2
import numpy as np


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
