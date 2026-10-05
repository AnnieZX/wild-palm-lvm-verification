"""Serve research images read-only.

Condition inputs are the exact files the VLMs received (resolved through each condition's
prompt_index.csv). They are transcoded to JPEG for bandwidth unless `original=True`, in which
case the stored PNG bytes are returned unchanged.
"""

from __future__ import annotations

import io
import re
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

from PIL import Image

from app.data import loaders as L
from app.data import paths as P

_IMAGE_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")
_SAMPLE_ID = re.compile(r"^sample_\d{6}$")
_MAX_ENTRIES = 256


class _LRU:
    def __init__(self, size: int) -> None:
        self.size = size
        self.data: "OrderedDict[tuple, bytes]" = OrderedDict()
        self.lock = Lock()

    def get(self, key: tuple) -> Optional[bytes]:
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key: tuple, value: bytes) -> None:
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)


_cache = _LRU(_MAX_ENTRIES)


def _prompt_index(condition: str) -> Dict[str, Path]:
    cond_dir = P.ABLATION_ROOT / P.CONDITION_DIRS[condition]
    index = cond_dir / "prompt_index.csv"
    df = L.read_csv(index, dtype={"sample_id": str})
    return {sid: (cond_dir / rel).resolve() for sid, rel in zip(df["sample_id"], df["image_path"])}


def condition_image_path(condition: str, sample_id: str) -> Optional[Path]:
    if condition not in P.CONDITIONS or not _SAMPLE_ID.match(sample_id):
        return None
    key = f"prompt_index_paths:{condition}"
    index_file = P.ABLATION_ROOT / P.CONDITION_DIRS[condition] / "prompt_index.csv"
    mapping = L.CACHE.get(key, [index_file], lambda: _prompt_index(condition))
    path = mapping.get(sample_id)
    return path if path is not None and path.is_file() else None


def raw_patch_path(image_name: str) -> Optional[Path]:
    if not _IMAGE_NAME.match(image_name):
        return None
    path = P.RAW_PATCHES_ROOT / f"{image_name}.png"
    return path if path.is_file() else None


def _jpeg(path: Path, max_side: Optional[int]) -> bytes:
    key = (str(path), path.stat().st_mtime_ns, max_side)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    _cache.put(key, data)
    return data


def load(path: Path, original: bool = False, max_side: Optional[int] = None) -> Tuple[bytes, str]:
    if original:
        return path.read_bytes(), "image/png"
    return _jpeg(path, max_side), "image/jpeg"
