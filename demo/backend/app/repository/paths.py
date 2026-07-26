"""Output path helpers for the demo repository (mirrors src/paths.py without importing src/)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from app.repository.constants import LEGACY_MODEL_DIR_ALIASES

EVALUATION_CSV_PATTERN = re.compile(r"^(A\d+)_evaluation\.csv$")


def canonical_model_key(model_key: str) -> str:
    return LEGACY_MODEL_DIR_ALIASES.get(model_key, model_key)


def verification_model_root(outputs_root: Path, model_key: str) -> Path:
    return outputs_root / "verification" / canonical_model_key(model_key)


def evaluation_model_root(outputs_root: Path, model_key: str) -> Path:
    return outputs_root / "evaluation" / canonical_model_key(model_key)


def _legacy_qwen_verification_experiment(outputs_root: Path, experiment_id: str) -> Path:
    return outputs_root / "verification" / "qwen" / experiment_id


def _legacy_qwen_evaluation_experiment(outputs_root: Path, experiment_id: str) -> Path:
    return outputs_root / "evaluation" / "qwen" / experiment_id


def verification_experiment_dir(
    outputs_root: Path,
    model_key: str,
    experiment_id: str,
) -> Path:
    canonical = canonical_model_key(model_key)
    preferred = verification_model_root(outputs_root, canonical) / experiment_id
    if preferred.exists():
        return preferred
    if canonical == "qwen2_5_vl":
        legacy = _legacy_qwen_verification_experiment(outputs_root, experiment_id)
        if legacy.exists():
            return legacy
    return preferred


def evaluation_experiment_dir(
    outputs_root: Path,
    model_key: str,
    experiment_id: str,
) -> Path:
    canonical = canonical_model_key(model_key)
    preferred = evaluation_model_root(outputs_root, canonical) / experiment_id
    if preferred.exists():
        return preferred
    if canonical == "qwen2_5_vl":
        legacy = _legacy_qwen_evaluation_experiment(outputs_root, experiment_id)
        if legacy.exists():
            return legacy
    return preferred


def verification_condition_dir(
    outputs_root: Path,
    model_key: str,
    experiment_id: str,
    ablation_code: str,
) -> Path:
    return verification_experiment_dir(outputs_root, model_key, experiment_id) / ablation_code


def evaluation_condition_dir(
    outputs_root: Path,
    model_key: str,
    experiment_id: str,
    ablation_code: str,
) -> Path:
    return evaluation_experiment_dir(outputs_root, model_key, experiment_id) / ablation_code


def discover_evaluation_csv(evaluation_dir: Path) -> Optional[Path]:
    if not evaluation_dir.exists():
        return None
    matches: List[Path] = []
    for path in sorted(evaluation_dir.glob("*_evaluation.csv")):
        if EVALUATION_CSV_PATTERN.match(path.name):
            matches.append(path)
    return matches[0] if matches else None


def _has_ablation_inputs(path: Path) -> bool:
    condition_dir = path / "A1_overlay_only"
    if not condition_dir.is_dir():
        return False
    return (condition_dir / "prompt_index.csv").exists() or (condition_dir / "images").is_dir()


def discover_ablation_inputs_dir(outputs_root: Path, sample_size: Optional[int] = None) -> Optional[Path]:
    if sample_size is not None:
        candidate = outputs_root / f"verification_ablation_{sample_size}"
        if _has_ablation_inputs(candidate):
            return candidate

    default_dir = outputs_root / "verification_ablation_100"
    if _has_ablation_inputs(default_dir):
        return default_dir

    matches = sorted(outputs_root.glob("verification_ablation_*"))
    for path in reversed(matches):
        if _has_ablation_inputs(path):
            return path
    return None


def list_experiment_dirs(model_dir: Path) -> List[Path]:
    """Return experiment directories under outputs/verification/<model>/."""
    if not model_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in model_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def list_verification_model_dirs(outputs_root: Path) -> List[Tuple[str, Path]]:
    """Return (directory_name, path) pairs for models with experiment data."""
    verification_root = outputs_root / "verification"
    if not verification_root.exists():
        return []

    discovered: List[Tuple[str, Path]] = []
    for model_dir in sorted(verification_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if any(child.is_dir() for child in model_dir.iterdir()):
            discovered.append((model_dir.name, model_dir))
    return discovered
