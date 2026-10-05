from typing import Any, Dict

from fastapi import APIRouter

from app.services import analysis_service

router = APIRouter()


@router.get("/summary")
def get_summary() -> Dict[str, Any]:
    return analysis_service.summary()
