"""Model catalog routes."""

from fastapi import APIRouter

from app.services import mock_data
from shared.models import ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(models=mock_data.list_models())
