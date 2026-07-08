"""Shared matplotlib drawing helpers for publication figures."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.visualization.publication_style import (
    BBOX_LINEWIDTH,
    COLOR_CENTER,
    COLOR_ENDPOINT,
    COLOR_GT,
    COLOR_YOLO,
    ENDPOINT_SIZE,
    POINT_SIZE,
)


def load_image_rgb(path: Path) -> np.ndarray | None:
    """Load an image as RGB uint8 array."""
    if not path.exists():
        return None
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def blank_panel(height: int, width: int, message: str = "Image not found") -> np.ndarray:
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    return panel


def draw_bbox(
    ax: Axes,
    bbox: tuple[float, float, float, float],
    color: tuple[float, float, float],
    label: str | None = None,
) -> None:
    x, y, width, height = bbox
    rect = Rectangle(
        (x, y),
        width,
        height,
        linewidth=BBOX_LINEWIDTH,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x,
            max(y - 6, 4),
            label,
            color=color,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )


def draw_center(ax: Axes, center: tuple[float, float]) -> None:
    ax.scatter(
        [center[0]],
        [center[1]],
        s=POINT_SIZE,
        c=[COLOR_CENTER],
        edgecolors="white",
        linewidths=1.0,
        zorder=5,
    )


def draw_endpoints(ax: Axes, endpoints: tuple[tuple[float, float], ...]) -> None:
    if not endpoints:
        return
    xs = [point[0] for point in endpoints]
    ys = [point[1] for point in endpoints]
    ax.scatter(
        xs,
        ys,
        s=ENDPOINT_SIZE,
        c=[COLOR_ENDPOINT],
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )


def show_image(ax: Axes, image: np.ndarray, title: str = "") -> None:
    ax.imshow(image)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, color="black", pad=8)


def add_bbox_legend(fig: Figure, y: float = 0.02) -> None:
    handles = [
        Rectangle((0, 0), 1, 1, linewidth=BBOX_LINEWIDTH, edgecolor=COLOR_GT, facecolor="none"),
        Rectangle((0, 0), 1, 1, linewidth=BBOX_LINEWIDTH, edgecolor=COLOR_YOLO, facecolor="none"),
    ]
    fig.legend(
        handles,
        ["Ground truth", "YOLO prediction"],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        fontsize=10,
    )


def annotate_points_legend(fig: Figure, y: float = 0.06) -> None:
    fig.text(
        0.5,
        y,
        "Blue = palm center    Yellow = palm endpoints",
        ha="center",
        va="center",
        fontsize=9,
        color="black",
    )


def render_on_image(
    image: np.ndarray,
    *,
    gt_bbox: tuple[float, float, float, float] | None = None,
    yolo_bbox: tuple[float, float, float, float] | None = None,
    center: tuple[float, float] | None = None,
    endpoints: tuple[tuple[float, float], ...] = (),
    draw_gt: bool = True,
    draw_yolo: bool = True,
) -> np.ndarray:
    """Return a copy of image with matplotlib-style overlays rasterized via OpenCV for speed."""
    canvas = image.copy()
    if draw_gt and gt_bbox is not None:
        _draw_cv_bbox(canvas, gt_bbox, COLOR_GT)
    if draw_yolo and yolo_bbox is not None:
        _draw_cv_bbox(canvas, yolo_bbox, COLOR_YOLO)
    if center is not None:
        cv2.circle(
            canvas,
            (int(round(center[0])), int(round(center[1]))),
            6,
            tuple(int(c * 255) for c in COLOR_CENTER[::-1]),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    for point in endpoints:
        cv2.circle(
            canvas,
            (int(round(point[0])), int(round(point[1]))),
            5,
            tuple(int(c * 255) for c in COLOR_ENDPOINT[::-1]),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def _draw_cv_bbox(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    color_rgb: tuple[float, float, float],
) -> None:
    x, y, width, height = bbox
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + width)), int(round(y + height))
    color_bgr = tuple(int(channel * 255) for channel in color_rgb[::-1])
    cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, BBOX_LINEWIDTH, cv2.LINE_AA)
