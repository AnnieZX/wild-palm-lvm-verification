"""Backward-compatible shim; prefer ``src.lvm.parsers.base``."""

from src.lvm.parsers.base import (  # noqa: F401
    DECISION_VALUES,
    normalize_decision,
    parse_verification_response,
)
