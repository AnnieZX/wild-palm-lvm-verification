"""Load model configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.paths import PROJECT_ROOT

DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
MODELS_CONFIG_DIR = PROJECT_ROOT / "configs" / "models"

# Registry aliases map to the canonical directory / config key.
CANONICAL_MODEL_KEYS: dict[str, str] = {
    "qwen": "qwen2_5_vl",
    "qwen2_5_vl": "qwen2_5_vl",
    "llava": "llava",
    "gemma": "gemma",
    "gemma4": "gemma4",
    "qwen3_vl": "qwen3_vl",
}

CHECKPOINT_KEYS = ("model_id", "model_path", "active_model")


def normalize_model_key(model_key: str) -> str:
    """Map registry aliases (e.g. qwen) to canonical keys (e.g. qwen2_5_vl)."""
    key = model_key.strip().lower()
    return CANONICAL_MODEL_KEYS.get(key, key)


def load_model_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the model configuration file."""
    path = config_path or DEFAULT_MODEL_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in model config: {path}")

    return config


def default_config_path_for_model(model_key: str) -> Path:
    """Return the default YAML path for a registry model key."""
    canonical = normalize_model_key(model_key)
    model_yaml = MODELS_CONFIG_DIR / f"{canonical}.yaml"
    if model_yaml.exists():
        return model_yaml
    if canonical == "qwen2_5_vl" and DEFAULT_MODEL_CONFIG.exists():
        return DEFAULT_MODEL_CONFIG
    raise FileNotFoundError(
        f"No config found for model {model_key!r}. "
        f"Expected {model_yaml} or pass --model-config explicitly."
    )


def resolve_model_config_path(model_key: str, config_path: Path | None = None) -> Path:
    """Resolve which YAML file to load for a given registry key."""
    if config_path is not None:
        return config_path
    return default_config_path_for_model(model_key)


def get_model_checkpoint(config_path: Path | None = None) -> str:
    """Return the model checkpoint path/id from a config file."""
    config = load_model_config(config_path)
    for key in CHECKPOINT_KEYS:
        value = config.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(
        f"No checkpoint path found in {config_path or DEFAULT_MODEL_CONFIG}. "
        f"Set one of: {', '.join(CHECKPOINT_KEYS)}"
    )


def resolve_model_checkpoint(
    model_key: str,
    *,
    config_path: Path | None = None,
    model_path_override: str | None = None,
) -> str:
    """
    Resolve the checkpoint for inference.

    Priority:
    1. CLI --model-path override
    2. Keys in the resolved model config (model_id, model_path, active_model)
    3. Legacy configs/model.yaml active_model for qwen2_5_vl only
    """
    if model_path_override:
        return model_path_override

    resolved_config = resolve_model_config_path(model_key, config_path)
    try:
        return get_model_checkpoint(resolved_config)
    except ValueError:
        canonical = normalize_model_key(model_key)
        if canonical == "qwen2_5_vl" and resolved_config != DEFAULT_MODEL_CONFIG:
            if DEFAULT_MODEL_CONFIG.exists():
                return get_model_checkpoint(DEFAULT_MODEL_CONFIG)
        raise


def get_active_model_path(config_path: Path | None = None) -> str:
    """Return active_model from config (backward-compatible Qwen2.5 helper)."""
    return get_model_checkpoint(config_path)


def get_model_label(model_key: str, config_path: Path | None = None) -> str:
    """Return human-readable model label from config, or the registry key."""
    config = load_model_config(resolve_model_config_path(model_key, config_path))
    label = config.get("model_label")
    if label:
        return str(label)
    return normalize_model_key(model_key)
