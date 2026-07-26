"""Shared query parameters for experiment-scoped endpoints."""

from __future__ import annotations

from fastapi import Query

from shared.models import AblationCode


def experiment_query_params(
    model_key: str = Query(
        default="qwen2_5_vl",
        description="Registry model key (e.g. qwen2_5_vl, llava)",
    ),
    experiment_id: str = Query(
        default="20250726_1200",
        description="Experiment run identifier from outputs/verification/<model>/<experiment_id>/",
    ),
    ablation: AblationCode = Query(
        default=AblationCode.A1,
        description="Ablation code (A1–A5)",
    ),
) -> dict[str, str | AblationCode]:
    return {
        "model_key": model_key,
        "experiment_id": experiment_id,
        "ablation": ablation,
    }
