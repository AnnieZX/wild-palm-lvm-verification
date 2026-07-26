"""Aggregate statistics routes."""

from fastapi import APIRouter, HTTPException, Query

from app.services import mock_data
from shared.models import AblationCode, StatisticsResponse

router = APIRouter()


@router.get("/statistics", response_model=StatisticsResponse)
def get_statistics(
    model_key: str = Query(default=mock_data.DEFAULT_MODEL_KEY),
    experiment_id: str = Query(default=mock_data.DEFAULT_EXPERIMENT_ID),
    ablation: AblationCode = Query(default=mock_data.DEFAULT_ABLATION),
) -> StatisticsResponse:
    payload = mock_data.get_statistics(model_key, experiment_id, ablation)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"No statistics for model={model_key!r}, experiment_id={experiment_id!r}, ablation={ablation.value!r}",
        )
    return payload
