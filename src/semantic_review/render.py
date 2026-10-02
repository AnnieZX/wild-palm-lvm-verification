"""
Blind review images rendered from the original patch and the target YOLO box only.

Nothing else is drawn: no LabelMe annotation, no other detection, no text.

- context: the full patch with the target rectangle.
- crop: a square window around the target (side = max(CROP_FACTOR x the longer box
  side, CROP_MIN_PX), capped at the patch size, shifted to stay inside the patch),
  upscaled to CROP_OUTPUT_PX, with the target rectangle.
- crop_raw: the same window without the rectangle; short ticks on the crop edges mark
  the target's horizontal and vertical extent.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw

CROP_FACTOR = 3.0
CROP_MIN_PX = 160
CROP_OUTPUT_PX = 640
BOX_COLOR = (255, 235, 0)
OUTLINE_COLOR = (0, 0, 0)
TICK_PX = 14
JPEG_QUALITY = 92
KINDS = ("context", "crop", "crop_raw")


def crop_window(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    side = min(max(CROP_FACTOR * max(w, h), CROP_MIN_PX), width, height)
    side = int(round(side))
    cx, cy = x + w / 2, y + h / 2
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    return left, top, left + side, top + side


def _clip_box(bbox, width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return max(0.0, x), max(0.0, y), min(float(width), x + w), min(float(height), y + h)


def _draw_target(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], width: int) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0 - width, y0 - width, x1 + width, y1 + width], outline=OUTLINE_COLOR, width=1)
    draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=width)


def _encode(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def render(image_path: Path, bbox: tuple[float, float, float, float], kind: str) -> bytes:
    if kind not in KINDS:
        raise ValueError(f"unknown image kind {kind!r}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = _clip_box(bbox, width, height)

    if kind == "context":
        _draw_target(ImageDraw.Draw(image), (x0, y0, x1, y1), width=3)
        return _encode(image)

    left, top, right, bottom = crop_window(bbox, width, height)
    scale = CROP_OUTPUT_PX / (right - left)
    crop = image.crop((left, top, right, bottom)).resize((CROP_OUTPUT_PX, CROP_OUTPUT_PX), Image.LANCZOS)
    box = ((x0 - left) * scale, (y0 - top) * scale, (x1 - left) * scale, (y1 - top) * scale)
    draw = ImageDraw.Draw(crop)
    if kind == "crop":
        _draw_target(draw, box, width=2)
    else:
        bx0, by0, bx1, by1 = box
        edge = CROP_OUTPUT_PX - 1
        for bx in (bx0, bx1):
            draw.line([bx, 0, bx, TICK_PX], fill=BOX_COLOR, width=2)
            draw.line([bx, edge - TICK_PX, bx, edge], fill=BOX_COLOR, width=2)
        for by in (by0, by1):
            draw.line([0, by, TICK_PX, by], fill=BOX_COLOR, width=2)
            draw.line([edge - TICK_PX, by, edge, by], fill=BOX_COLOR, width=2)
    return _encode(crop)
