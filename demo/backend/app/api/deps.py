"""Shared query parameters for experiment-scoped endpoints."""

from __future__ import annotations

from typing import Dict, Optional, Union

from fastapi import Query

from app.services import get_repository
from shared.models import AblationCode


def experiment_query_params(
    model_key: Optional[str] = Query(
        default=None,
        description="Registry model key (e.g. qwen2_5_vl, llava). Defaults to the latest discovered model.",
    ),
    experiment_id: Optional[str] = Query(
        default=None,
        description="Experiment run identifier from outputs/verification/<model>/<experiment_id>/. Defaults to the latest discovered experiment.",
    ),
    ablation: Optional[AblationCode] = Query(
        default=None,
        description="Ablation code (A1–A5). Defaults to the experiment primary ablation.",
    ),
) -> Dict[str, Union[str, AblationCode]]:
    repository = get_repository()
    resolved_model_key = model_key or repository.default_model_key
    resolved_experiment_id = experiment_id or repository.default_experiment_id
    resolved_ablation = ablation or repository.default_ablation
    return {
        "model_key": resolved_model_key,
        "experiment_id": resolved_experiment_id,
        "ablation": resolved_ablation,
    }
