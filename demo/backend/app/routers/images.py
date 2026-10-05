from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.services import image_service

router = APIRouter()

_CACHE_HEADERS = {"Cache-Control": "public, max-age=3600"}


@router.get("/images/raw/{image_name}")
def raw_patch(image_name: str, original: bool = False, max_side: Optional[int] = Query(None, ge=64, le=4096)):
    path = image_service.raw_patch_path(image_name)
    if path is None:
        raise HTTPException(404, "raw patch not found")
    data, media = image_service.load(path, original=original, max_side=max_side)
    return Response(content=data, media_type=media, headers=_CACHE_HEADERS)


@router.get("/images/condition/{condition}/{detection_id}")
def condition_input(condition: str, detection_id: str, original: bool = False,
                    max_side: Optional[int] = Query(None, ge=64, le=4096)):
    path = image_service.condition_image_path(condition, detection_id)
    if path is None:
        raise HTTPException(404, "condition input not found")
    data, media = image_service.load(path, original=original, max_side=max_side)
    return Response(content=data, media_type=media, headers=_CACHE_HEADERS)
