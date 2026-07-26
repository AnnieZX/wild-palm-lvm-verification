"""Overlay image routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response

from app.services import get_repository
from shared.models import AblationCode

router = APIRouter()


@router.get(
    "/image/{sample_id}",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Ablation overlay image for the requested sample.",
        }
    },
)
def get_sample_image(
    sample_id: str,
    model_key: Optional[str] = Query(default=None),
    experiment_id: Optional[str] = Query(default=None),
    ablation: Optional[AblationCode] = Query(default=None),
) -> Response:
    repository = get_repository()
    resolved_model_key = model_key or repository.default_model_key
    resolved_experiment_id = experiment_id or repository.default_experiment_id
    resolved_ablation = ablation or repository.default_ablation

    model = repository.resolve_model(resolved_model_key)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Unknown model_key: {resolved_model_key!r}")

    experiment = repository.resolve_experiment(model, resolved_experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id: {resolved_experiment_id!r}")

    if resolved_ablation not in experiment.ablations:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Ablation {resolved_ablation.value!r} not available "
                f"for experiment {resolved_experiment_id!r}"
            ),
        )

    sample = repository.get_sample_detail(
        sample_id,
        model_key=resolved_model_key,
        experiment_id=resolved_experiment_id,
        ablation=resolved_ablation,
    )
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id!r}")

    image_path = repository.get_overlay_image_path(
        sample_id,
        model_key=resolved_model_key,
        experiment_id=resolved_experiment_id,
        ablation=resolved_ablation,
    )
    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Overlay image not found for sample_id: {sample_id!r}",
        )

    headers = {
        "X-Sample-Id": sample_id,
        "X-Model-Key": resolved_model_key,
        "X-Experiment-Id": resolved_experiment_id,
        "X-Ablation": resolved_ablation.value,
    }
    return Response(
        content=image_path.read_bytes(),
        media_type="image/png",
        headers=headers,
    )
