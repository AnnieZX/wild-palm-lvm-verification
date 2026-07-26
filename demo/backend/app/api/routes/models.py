"""Model catalog routes."""

from fastapi import APIRouter

from app.services import get_repository
from shared.models import ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(models=get_repository().list_models())
