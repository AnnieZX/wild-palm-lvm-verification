from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from app.services import analysis_service, detection_service

router = APIRouter()


def _filters(q, matched, semantic, conf_min, conf_max, area_q, conf_q, changed_model) -> Dict[str, Any]:
    if semantic and semantic not in detection_service.SEMANTIC_FILTERS:
        raise HTTPException(422, f"semantic must be one of {detection_service.SEMANTIC_FILTERS}")
    return dict(q=q, matched=matched, semantic=semantic, conf_min=conf_min, conf_max=conf_max,
                area_q=area_q, conf_q=conf_q, changed_model=changed_model)


@router.get("/detections")
def list_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    q: Optional[str] = None,
    matched: Optional[bool] = None,
    semantic: Optional[str] = None,
    conf_min: Optional[float] = None,
    conf_max: Optional[float] = None,
    area_q: Optional[str] = Query(None, pattern="^Q[1-5]$"),
    conf_q: Optional[str] = Query(None, pattern="^Q[1-5]$"),
    changed_model: Optional[str] = Query(None, description="keep detections whose A1 and A5 decisions differ"),
) -> Dict[str, Any]:
    f = _filters(q, matched, semantic, conf_min, conf_max, area_q, conf_q, changed_model)
    return detection_service.list_detections(page=page, page_size=page_size, **f)


@router.get("/detections/random")
def random_detection(
    q: Optional[str] = None,
    matched: Optional[bool] = None,
    semantic: Optional[str] = None,
    conf_min: Optional[float] = None,
    conf_max: Optional[float] = None,
    area_q: Optional[str] = Query(None, pattern="^Q[1-5]$"),
    conf_q: Optional[str] = Query(None, pattern="^Q[1-5]$"),
    changed_model: Optional[str] = None,
) -> Dict[str, Any]:
    f = _filters(q, matched, semantic, conf_min, conf_max, area_q, conf_q, changed_model)
    sid = detection_service.random_detection(**f)
    if sid is None:
        raise HTTPException(404, "no detection matches the filters")
    return {"detection_id": sid}


@router.get("/detections/curated")
def curated() -> Dict[str, Any]:
    return analysis_service.curated_cases()


@router.get("/detections/{detection_id}")
def get_detection(detection_id: str) -> Dict[str, Any]:
    det = detection_service.get_detection(detection_id)
    if det is None:
        raise HTTPException(404, f"unknown detection {detection_id!r}")
    return det


@router.get("/detections/{detection_id}/predictions")
def get_predictions(detection_id: str) -> Dict[str, Any]:
    out = detection_service.get_predictions(detection_id)
    if out is None:
        raise HTTPException(404, f"unknown detection {detection_id!r}")
    return out


@router.get("/detections/{detection_id}/reasoning")
def get_reasoning(detection_id: str, model: str, condition: str) -> Dict[str, Any]:
    try:
        return detection_service.get_reasoning(detection_id, model, condition)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@router.get("/detections/{detection_id}/prompt/{condition}")
def get_prompt(detection_id: str, condition: str) -> Dict[str, Any]:
    out = detection_service.get_prompt(detection_id, condition)
    if out is None:
        raise HTTPException(404, "prompt not found")
    return out
