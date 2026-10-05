from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.services import model_service

router = APIRouter()


@router.get("/models")
def list_models() -> Dict[str, Any]:
    return {"models": model_service.list_models()}


@router.get("/models/{model_key}")
def get_model(model_key: str) -> Dict[str, Any]:
    model = model_service.get_model(model_key)
    if model is None:
        raise HTTPException(404, f"unknown model {model_key!r}")
    return model
