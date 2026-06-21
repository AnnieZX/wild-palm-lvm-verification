"""Shared helpers for deterministic 100-palm sequential sampling."""

from __future__ import annotations

from pathlib import Path

from src.preprocessing.json_parser import load_json

TARGET_PALM_COUNT = 100
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def find_json_files(dataset_root: Path) -> list[Path]:
    """Return all JSON files under the dataset root, sorted alphabetically."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    return sorted(dataset_root.rglob("*.json"))


def resolve_image_path(json_path: Path, data: dict) -> Path | None:
    """Find the PNG/image file that matches one JSON annotation."""
    image_path_value = data.get("imagePath")
    if image_path_value:
        candidates = [
            json_path.parent / str(image_path_value),
            json_path.parent / Path(str(image_path_value)).name,
            json_path.parent / Path(image_path_value).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    for extension in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(extension)
        if candidate.exists():
            return candidate

    for extension in IMAGE_EXTENSIONS:
        for candidate in json_path.parent.glob(f"{json_path.stem}{extension}"):
            if candidate.exists():
                return candidate

    return None


def global_palm_id(index: int) -> str:
    """Return a global sequential palm ID such as palm_001."""
    return f"palm_{index:03d}"


def load_json_metadata(json_path: Path) -> dict:
    """Load JSON file contents."""
    return load_json(json_path)
