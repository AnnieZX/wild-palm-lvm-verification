"""Aggregate statistics routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services import get_repository
from shared.models import AblationCode, StatisticsResponse

router = APIRouter()


@router.get("/statistics", response_model=StatisticsResponse)
def get_statistics(
    model_key: Optional[str] = Query(default=None),
    experiment_id: Optional[str] = Query(default=None),
    ablation: Optional[AblationCode] = Query(default=None),
) -> StatisticsResponse:
    repository = get_repository()
    resolved_model_key = model_key or repository.default_model_key
    resolved_experiment_id = experiment_id or repository.default_experiment_id
    resolved_ablation = ablation or repository.default_ablation

    payload = repository.get_statistics(resolved_model_key, resolved_experiment_id, resolved_ablation)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No statistics for model={resolved_model_key!r}, "
                f"experiment_id={resolved_experiment_id!r}, ablation={resolved_ablation.value!r}"
            ),
        )
    return payload
