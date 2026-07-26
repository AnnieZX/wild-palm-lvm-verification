"""Sample listing and detail routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services import get_repository
from shared.models import (
    AblationCode,
    GroundTruthLabel,
    SampleDetailResponse,
    SampleListResponse,
    VerificationDecision,
)

router = APIRouter()


def _resolve_scope(
    model_key: Optional[str],
    experiment_id: Optional[str],
    ablation: Optional[AblationCode],
):
    repository = get_repository()
    resolved_model_key = model_key or repository.default_model_key
    resolved_experiment_id = experiment_id or repository.default_experiment_id
    resolved_ablation = ablation or repository.default_ablation
    return repository, resolved_model_key, resolved_experiment_id, resolved_ablation


def _validate_scope(
    repository,
    model_key: str,
    experiment_id: str,
    ablation: AblationCode,
) -> None:
    model = repository.resolve_model(model_key)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Unknown model_key: {model_key!r}")

    experiment = repository.resolve_experiment(model, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {experiment_id!r}")

    if ablation not in experiment.ablations:
        raise HTTPException(
            status_code=404,
            detail=f"Ablation {ablation.value!r} not available for experiment {experiment_id!r}",
        )


@router.get("/samples", response_model=SampleListResponse)
def list_samples(
    model_key: Optional[str] = Query(default=None),
    experiment_id: Optional[str] = Query(default=None),
    ablation: Optional[AblationCode] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    decision: Optional[VerificationDecision] = Query(default=None),
    matched_gt: Optional[bool] = Query(default=None),
    gt_label: Optional[GroundTruthLabel] = Query(default=None),
) -> SampleListResponse:
    repository, resolved_model_key, resolved_experiment_id, resolved_ablation = _resolve_scope(
        model_key,
        experiment_id,
        ablation,
    )
    _validate_scope(repository, resolved_model_key, resolved_experiment_id, resolved_ablation)

    filtered = repository.list_sample_summaries(
        model_key=resolved_model_key,
        experiment_id=resolved_experiment_id,
        ablation=resolved_ablation,
        decision=decision,
        matched_gt=matched_gt,
        gt_label=gt_label,
    )
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = filtered[start:end]

    return SampleListResponse(
        model_key=resolved_model_key,
        experiment_id=resolved_experiment_id,
        ablation=resolved_ablation,
        total=total,
        page=page,
        page_size=page_size,
        samples=page_items,
    )


@router.get("/sample/{sample_id}", response_model=SampleDetailResponse)
def get_sample(
    sample_id: str,
    model_key: Optional[str] = Query(default=None),
    experiment_id: Optional[str] = Query(default=None),
    ablation: Optional[AblationCode] = Query(default=None),
) -> SampleDetailResponse:
    repository, resolved_model_key, resolved_experiment_id, resolved_ablation = _resolve_scope(
        model_key,
        experiment_id,
        ablation,
    )
    _validate_scope(repository, resolved_model_key, resolved_experiment_id, resolved_ablation)

    sample = repository.get_sample_detail(
        sample_id,
        model_key=resolved_model_key,
        experiment_id=resolved_experiment_id,
        ablation=resolved_ablation,
    )
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id!r}")

    return SampleDetailResponse(
        model_key=resolved_model_key,
        experiment_id=resolved_experiment_id,
        sample=sample,
    )
