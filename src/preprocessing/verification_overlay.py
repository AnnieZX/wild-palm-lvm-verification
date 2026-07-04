"""Render single-detection verification overlays for Qwen2.5-VL input."""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.lvm_input_builder import COLOR_BBOX

DEFAULT_DIM_FACTOR = 0.55


def render_single_detection_overlay(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    dim_factor: float = DEFAULT_DIM_FACTOR,
    box_color: tuple[int, int, int] = COLOR_BBOX,
    box_thickness: int = 2,
) -> np.ndarray:
    """
    Dim the full image, restore the bbox region at full brightness, and draw the target box.

    Args:
        image: BGR source patch.
        bbox: Target detection as (x, y, width, height).
        dim_factor: Brightness multiplier applied outside the bbox (0–1).
    """
    x, y, width, height = bbox
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + width))
    y2 = int(round(y + height))

    img_h, img_w = image.shape[:2]
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    dimmed = (image.astype(np.float32) * dim_factor).clip(0, 255).astype(np.uint8)
    output = dimmed.copy()
    if x2 > x1 and y2 > y1:
        output[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    cv2.rectangle(output, (x1, y1), (x2, y2), box_color, thickness=box_thickness)
    return output
