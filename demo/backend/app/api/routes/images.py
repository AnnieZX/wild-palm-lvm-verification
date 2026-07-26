"""Overlay image routes."""

from fastapi import APIRouter, HTTPException, Query, Response

from app.services import mock_data
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
    model_key: str = Query(default=mock_data.DEFAULT_MODEL_KEY),
    experiment_id: str = Query(default=mock_data.DEFAULT_EXPERIMENT_ID),
    ablation: AblationCode = Query(default=mock_data.DEFAULT_ABLATION),
) -> Response:
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

    headers = {
        "X-Sample-Id": sample_id,
        "X-Model-Key": model_key,
        "X-Experiment-Id": experiment_id,
        "X-Ablation": ablation.value,
        "X-Mock-Image": "true",
    }
    return Response(
        content=mock_data.MOCK_OVERLAY_PNG_BYTES,
        media_type="image/png",
        headers=headers,
    )
