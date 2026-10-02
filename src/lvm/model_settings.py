"""Per-model-key settings and run provenance for family adapters.

Several parameter sizes share one family adapter (e.g. Qwen3-VL 2B/4B/8B/32B).
Each size has its own ``configs/models/<model_key>.yaml``; adapters read
dtype, attention backend, label and pinned revision from it here rather than
hardcoding them per size.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.config.model_config import load_model_config, resolve_model_config_path

GIT_COMMIT_ENV = "WILD_PALM_GIT_COMMIT"


def load_model_settings(model_key: str) -> dict[str, Any]:
    """Return the YAML mapping for ``model_key`` (empty dict if no config exists)."""
    try:
        return load_model_config(resolve_model_config_path(model_key))
    except FileNotFoundError:
        return {}


def read_download_revision(checkpoint: str | Path) -> str:
    """Return ``revision`` from ``<checkpoint>/DOWNLOAD_META.txt`` or ``""``."""
    meta = Path(checkpoint) / "DOWNLOAD_META.txt"
    if not meta.is_file():
        return ""
    for line in meta.read_text(encoding="utf-8").splitlines():
        if line.startswith("revision="):
            return line.split("=", 1)[1].strip()
    return ""


def resolved_attn_implementation(model: Any) -> str:
    """Attention backend Transformers actually selected for a loaded model."""
    config = getattr(model, "config", None)
    value = getattr(config, "_attn_implementation", None)
    return str(value) if value else ""


def run_provenance(*, checkpoint: str | Path, model: Any = None) -> dict[str, str]:
    """Provenance fields recorded in each sample's ``generation`` block."""
    dtype = getattr(model, "dtype", None)
    return {
        "attn_implementation_resolved": resolved_attn_implementation(model),
        "torch_dtype_resolved": str(dtype).replace("torch.", "") if dtype is not None else "",
        "checkpoint_revision": read_download_revision(checkpoint),
        "git_commit": os.environ.get(GIT_COMMIT_ENV, ""),
    }
