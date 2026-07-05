"""Debug rendering for verification sample GT/YOLO matching."""

from __future__ import annotations

import cv2
import numpy as np

from src.visualization.verification_visualization import _draw_text_with_outline, xywh_to_xyxy

COLOR_GT_ALL = (60, 200, 70)
COLOR_GT_MATCHED = (0, 230, 255)
COLOR_YOLO = (240, 130, 20)


def render_sample_matching_debug(
    image: np.ndarray,
    *,
    yolo_bbox: tuple[float, float, float, float],
    gt_bboxes: list[tuple[float, float, float, float]],
    matched_gt_index: int | None,
    max_iou: float,
    sample_id: str,
) -> np.ndarray:
    """Draw all GT boxes (green), matched GT (yellow), and YOLO box (blue)."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    font_scale = max(width, height) / 1800.0
    line_thickness = max(2, int(round(font_scale * 2.5)))

    for index, gt_bbox in enumerate(gt_bboxes):
        x1, y1, x2, y2 = xywh_to_xyxy(gt_bbox)
        color = COLOR_GT_MATCHED if matched_gt_index == index else COLOR_GT_ALL
        thickness = line_thickness + 1 if matched_gt_index == index else line_thickness
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label = f"GT{index}" + ("*" if matched_gt_index == index else "")
        _draw_text_with_outline(canvas, label, (x1, max(y1 - 6, int(18 * font_scale))), font_scale * 0.85, 1)

    x1, y1, x2, y2 = xywh_to_xyxy(yolo_bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_YOLO, line_thickness + 1, cv2.LINE_AA)
    _draw_text_with_outline(canvas, "YOLO", (x1, max(y1 - 6, int(18 * font_scale))), font_scale * 0.85, 1)

    title = f"{sample_id}  max IoU={max_iou:.4f}  matched GT={matched_gt_index if matched_gt_index is not None else 'none'}"
    _draw_text_with_outline(canvas, title, (12, int(28 * font_scale)), font_scale, 1)

    return canvas
