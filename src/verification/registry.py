"""Verification adapter registry and factory."""

from __future__ import annotations

from typing import Any, Callable

from src.verification.base_adapter import BaseVerificationAdapter

AdapterFactory = Callable[..., BaseVerificationAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}

# Primary production keys. Aliases map to the same canonical key for lookup.
_REGISTRY_ALIASES: dict[str, str] = {
    "qwen": "qwen2_5_vl",
    "qwen25_vl_7b": "qwen2_5_vl",
    "qwen2_5_vl_7b": "qwen2_5_vl",
    "internvl3_8b": "internvl3",
    "phi4": "phi4_multimodal",
    "glm46v_flash": "glm_4_6v_flash",
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
    """Return sorted registered model keys (includes CLI aliases)."""
    return tuple(sorted(set(_REGISTRY) | set(_REGISTRY_ALIASES)))


def create_adapter(model: str, **kwargs: Any) -> BaseVerificationAdapter:
    """Instantiate a registered verification adapter."""
    key = resolve_registry_key(model)
    if key not in _REGISTRY:
        allowed = ", ".join(get_registered_models()) or "none"
        raise ValueError(f"Unknown verification model {model!r}. Registered: {allowed}")
    return _REGISTRY[key](**kwargs)


def _register_builtin_adapters() -> None:
    from src.lvm.gemma_verification_adapter import build_gemma_adapter
    from src.lvm.glm_4_6v_flash_verification_adapter import build_glm_4_6v_flash_adapter
    from src.lvm.internvl_verification_adapter import build_internvl_adapter
    from src.lvm.llava_verification_adapter import build_llava_adapter
    from src.lvm.phi4_multimodal_verification_adapter import build_phi4_multimodal_adapter
    from src.lvm.qwen_verification_adapter import build_qwen_adapter

    register_adapter("qwen2_5_vl", build_qwen_adapter)
    register_adapter("qwen", build_qwen_adapter)
    register_adapter("llava", build_llava_adapter)
    register_adapter("gemma", build_gemma_adapter)
    register_adapter("internvl3", build_internvl_adapter)
    register_adapter("internvl", build_internvl_adapter)
    register_adapter("glm_4_6v_flash", build_glm_4_6v_flash_adapter)
    register_adapter("glm46v_flash", build_glm_4_6v_flash_adapter)
    register_adapter("phi4_multimodal", build_phi4_multimodal_adapter)
    register_adapter("phi4", build_phi4_multimodal_adapter)


_register_builtin_adapters()
