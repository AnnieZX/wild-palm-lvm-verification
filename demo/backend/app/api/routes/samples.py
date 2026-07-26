"""Sample listing and detail routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services import mock_data
from shared.models import (
    AblationCode,
    GroundTruthLabel,
    SampleDetailResponse,
    SampleListResponse,
    VerificationDecision,
)

router = APIRouter()


@router.get("/samples", response_model=SampleListResponse)
def list_samples(
    model_key: str = Query(default=mock_data.DEFAULT_MODEL_KEY),
    experiment_id: str = Query(default=mock_data.DEFAULT_EXPERIMENT_ID),
    ablation: AblationCode = Query(default=mock_data.DEFAULT_ABLATION),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    decision: Optional[VerificationDecision] = Query(default=None),
    matched_gt: Optional[bool] = Query(default=None),
    gt_label: Optional[GroundTruthLabel] = Query(default=None),
) -> SampleListResponse:
    model = mock_data.resolve_model(model_key)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Unknown model_key: {model_key!r}")

    experiment = mock_data.resolve_experiment(model, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id!r}")

    if ablation not in experiment.ablations:
        raise HTTPException(
            status_code=404,
            detail=f"Ablation {ablation.value!r} not available for experiment {experiment_id!r}",
        )

    filtered = mock_data.list_sample_summaries(
        decision=decision,
        matched_gt=matched_gt,
        gt_label=gt_label,
    )
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = filtered[start:end]

    return SampleListResponse(
        model_key=model_key,
        experiment_id=experiment_id,
        ablation=ablation,
        total=total,
        page=page,
        page_size=page_size,
        samples=page_items,
    )


@router.get("/sample/{sample_id}", response_model=SampleDetailResponse)
def get_sample(
    sample_id: str,
    model_key: str = Query(default=mock_data.DEFAULT_MODEL_KEY),
    experiment_id: str = Query(default=mock_data.DEFAULT_EXPERIMENT_ID),
    ablation: AblationCode = Query(default=mock_data.DEFAULT_ABLATION),
) -> SampleDetailResponse:
    model = mock_data.resolve_model(model_key)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Unknown model_key: {model_key!r}")

    experiment = mock_data.resolve_experiment(model, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id!r}")

    if ablation not in experiment.ablations:
        raise HTTPException(
            status_code=404,
            detail=f"Ablation {ablation.value!r} not available for experiment {experiment_id!r}",
        )

    record = mock_data.get_sample_record(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id!r}")

    return SampleDetailResponse(
        model_key=model_key,
        experiment_id=experiment_id,
        sample=mock_data.sample_detail(
            record,
            model_key=model_key,
            experiment_id=experiment_id,
        ),
    )
