"""LabelMe annotation path helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_labelme_json(annotations_root: Path, image_name: str) -> Path | None:
    """Find LabelMe JSON for an image stem."""
    candidates = [
        annotations_root / f"{image_name}.json",
        annotations_root / image_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(annotations_root.rglob(f"{image_name}.json"))
    return matches[0] if matches else None
