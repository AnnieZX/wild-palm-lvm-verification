"""Load model configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.paths import PROJECT_ROOT

DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"


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


def get_active_model_path(config_path: Path | None = None) -> str:
    """Return active_model from config, raising if unset."""
    config = load_model_config(config_path)
    active_model = config.get("active_model")
    if not active_model:
        raise ValueError(
            f"'active_model' is not set in {config_path or DEFAULT_MODEL_CONFIG}"
        )
    return str(active_model)
