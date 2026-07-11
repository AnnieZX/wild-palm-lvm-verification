"""Draw annotation overlays on images using OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.json_parser import AnnotationFile, Shape

# BGR colors for common labels in this project.
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "palm": (0, 255, 0),
    "center": (255, 0, 0),
    "end": (0, 165, 255),
    "weed": (0, 255, 255),
}


def color_for_label(label: str) -> tuple[int, int, int]:
    """Pick a stable color for a label name."""
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]

    # Fallback: hash the label into a bright BGR color.
    seed = sum(ord(char) for char in label)
    return (50 + seed % 200, 50 + (seed * 3) % 200, 50 + (seed * 7) % 200)


def load_image(path: Path | str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def draw_shape(image: np.ndarray, shape: Shape) -> None:
    """Draw one shape directly onto the image."""
    color = color_for_label(shape.label)
    points = np.array(shape.points, dtype=np.int32)

    if shape.shape_type == "point":
        x, y = points[0]
        cv2.circle(image, (int(x), int(y)), radius=6, color=color, thickness=-1)
        cv2.putText(
            image,
            shape.label,
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        return

    if shape.shape_type == "bbox":
        x1, y1 = points[0]
        x2, y2 = points[1]
        top_left = (int(min(x1, x2)), int(min(y1, y2)))
        bottom_right = (int(max(x1, x2)), int(max(y1, y2)))
        cv2.rectangle(image, top_left, bottom_right, color, thickness=2)
        cv2.putText(
            image,
            shape.label,
            (top_left[0], max(top_left[1] - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        return

    # Polygon: more than 2 points.
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    cv2.putText(
        image,
        shape.label,
        (int(points[0][0]), max(int(points[0][1]) - 8, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def render_overlay(image_path: Path | str, annotation: AnnotationFile) -> np.ndarray:
    """Load an image, draw all shapes, and return the overlay image."""
    image = load_image(image_path).copy()
    for shape in annotation.shapes:
        draw_shape(image, shape)
    return image


def save_overlay(image: np.ndarray, output_path: Path | str) -> None:
    """Save an overlay image to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
