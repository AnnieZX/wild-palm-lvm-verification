"""Model-specific raw-response cleanup before shared parsing."""

from __future__ import annotations


def normalize_raw_response(raw_text: str, model_key: str = "") -> str:
    """
    Apply model-specific text cleanup before JSON extraction.

    Identity pass-through for all models until dedicated normalizers are added.
    """
    _ = model_key
    return raw_text
