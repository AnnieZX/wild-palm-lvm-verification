from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services import image_service, review_service

router = APIRouter(prefix="/review")


class SessionRequest(BaseModel):
    count: int = Field(10, ge=1, le=review_service.MAX_ITEMS)
    deck: str = "unmatched"


class LabelRequest(BaseModel):
    index: int = Field(..., ge=0)
    label: str


@router.get("/decks")
def decks() -> Dict[str, Any]:
    return review_service.decks()


@router.post("/sessions")
def create_session(body: SessionRequest) -> Dict[str, Any]:
    try:
        return review_service.create_session(body.count, body.deck)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    try:
        return review_service.public_session(session_id)
    except KeyError:
        raise HTTPException(404, "unknown session")


@router.post("/sessions/{session_id}/labels")
def submit_label(session_id: str, body: LabelRequest) -> Dict[str, Any]:
    try:
        return review_service.submit_label(session_id, body.index, body.label)
    except KeyError:
        raise HTTPException(404, "unknown session")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/sessions/{session_id}/results")
def results(session_id: str) -> Dict[str, Any]:
    try:
        return review_service.results(session_id)
    except KeyError:
        raise HTTPException(404, "unknown session")
    except PermissionError as exc:
        raise HTTPException(403, str(exc))


@router.get("/image/{token}")
def review_image(token: str):
    path = review_service.image_for_token(token)
    if path is None:
        raise HTTPException(404, "unknown item")
    data, media = image_service.load(path)
    return Response(content=data, media_type=media, headers={"Cache-Control": "private, max-age=3600"})
