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
    "qwen3_vl_8b": "qwen3_vl",
    "qwen3vl": "qwen3_vl",
    "qwen3vl_8b": "qwen3_vl",
    "internvl3_8b": "internvl3",
    "internvl3_5_hf": "internvl3_5_hf",
    "internvl35_hf": "internvl3_5_hf",
    "internvl3.5_hf": "internvl3_5_hf",
    "phi4": "phi4_multimodal",
    "glm46v_flash": "glm_4_6v_flash",
    "molmo2": "molmo2_8b",
    "molmo2-8b": "molmo2_8b",
    "minicpm45": "minicpm_v4_5",
    "minicpm-v-4.5": "minicpm_v4_5",
    "minicpm_v45": "minicpm_v4_5",
    "gemma4_12b": "gemma4",
    "gemma-4": "gemma4",
    "gemma4_12b_it": "gemma4",
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
    from src.lvm.gemma4_verification_adapter import build_gemma4_adapter
    from src.lvm.glm_4_6v_flash_verification_adapter import build_glm_4_6v_flash_adapter
    from src.lvm.internvl_verification_adapter import build_internvl_adapter
    from src.lvm.internvl3_5_hf_verification_adapter import build_internvl3_5_hf_adapter
    from src.lvm.llama3_2_11b_vision_verification_adapter import build_llama3_2_11b_vision_adapter
    from src.lvm.llava_verification_adapter import build_llava_adapter
    from src.lvm.minicpm_v4_5_verification_adapter import build_minicpm_v4_5_adapter
    from src.lvm.ministral3_8b_verification_adapter import build_ministral3_8b_adapter
    from src.lvm.molmo2_verification_adapter import build_molmo2_adapter
    from src.lvm.phi4_multimodal_verification_adapter import build_phi4_multimodal_adapter
    from src.lvm.qwen_verification_adapter import build_qwen_adapter
    from src.lvm.qwen3_vl_verification_adapter import build_qwen3_vl_adapter

    register_adapter("qwen2_5_vl", build_qwen_adapter)
    register_adapter("qwen", build_qwen_adapter)
    register_adapter("qwen3_vl", build_qwen3_vl_adapter)
    register_adapter("qwen3_vl_8b", build_qwen3_vl_adapter)
    register_adapter("qwen3vl", build_qwen3_vl_adapter)
    register_adapter("qwen3vl_8b", build_qwen3_vl_adapter)
    for qwen3_size_key in ("qwen3_vl_2b", "qwen3_vl_4b", "qwen3_vl_32b"):
        register_adapter(qwen3_size_key, build_qwen3_vl_adapter)
    register_adapter("llava", build_llava_adapter)
    register_adapter("gemma", build_gemma_adapter)
    register_adapter("gemma4", build_gemma4_adapter)
    register_adapter("gemma4_12b", build_gemma4_adapter)
    register_adapter("gemma-4", build_gemma4_adapter)
    register_adapter("internvl3", build_internvl_adapter)
    register_adapter("internvl", build_internvl_adapter)
    register_adapter("internvl3_5_hf", build_internvl3_5_hf_adapter)
    register_adapter("internvl35_hf", build_internvl3_5_hf_adapter)
    for internvl35_size_key in ("internvl3_5_hf_2b", "internvl3_5_hf_4b", "internvl3_5_hf_14b"):
        register_adapter(internvl35_size_key, build_internvl3_5_hf_adapter)
    register_adapter("glm_4_6v_flash", build_glm_4_6v_flash_adapter)
    register_adapter("glm46v_flash", build_glm_4_6v_flash_adapter)
    register_adapter("phi4_multimodal", build_phi4_multimodal_adapter)
    register_adapter("phi4", build_phi4_multimodal_adapter)
    register_adapter("molmo2_8b", build_molmo2_adapter)
    register_adapter("molmo2", build_molmo2_adapter)
    register_adapter("molmo2-8b", build_molmo2_adapter)
    register_adapter("minicpm_v4_5", build_minicpm_v4_5_adapter)
    register_adapter("minicpm45", build_minicpm_v4_5_adapter)
    register_adapter("minicpm-v-4.5", build_minicpm_v4_5_adapter)
    register_adapter("ministral3_8b", build_ministral3_8b_adapter)
    register_adapter("llama3_2_11b_vision", build_llama3_2_11b_vision_adapter)


_register_builtin_adapters()
