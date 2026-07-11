"""Verification adapter registry and factory."""

from __future__ import annotations

from typing import Any, Callable

from src.verification.base_adapter import BaseVerificationAdapter

AdapterFactory = Callable[..., BaseVerificationAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}

# Primary production keys. Aliases map to the same canonical key for lookup.
_REGISTRY_ALIASES: dict[str, str] = {
    "qwen": "qwen2_5_vl",
}


def register_adapter(model_name: str, factory: AdapterFactory) -> None:
    """Register a factory that builds one verification adapter."""
    key = model_name.strip().lower()
    _REGISTRY[key] = factory


def resolve_registry_key(model: str) -> str:
    """Map CLI/registry input to a registered adapter key."""
    key = model.strip().lower()
    return _REGISTRY_ALIASES.get(key, key)


def get_registered_models() -> tuple[str, ...]:
    """Return sorted registered model keys (includes aliases)."""
    return tuple(sorted(_REGISTRY))


def create_adapter(model: str, **kwargs: Any) -> BaseVerificationAdapter:
    """Instantiate a registered verification adapter."""
    key = resolve_registry_key(model)
    if key not in _REGISTRY:
        allowed = ", ".join(get_registered_models()) or "none"
        raise ValueError(f"Unknown verification model {model!r}. Registered: {allowed}")
    return _REGISTRY[key](**kwargs)


def _register_builtin_adapters() -> None:
    from src.lvm.qwen_verification_adapter import build_qwen_adapter

    register_adapter("qwen2_5_vl", build_qwen_adapter)
    register_adapter("qwen", build_qwen_adapter)


_register_builtin_adapters()
