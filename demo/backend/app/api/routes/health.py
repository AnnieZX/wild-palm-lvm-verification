"""Health check route (scaffold only)."""

from fastapi import APIRouter

from shared.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="wild-palm-demo-backend")
