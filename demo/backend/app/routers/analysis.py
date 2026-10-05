from typing import Any, Dict

from fastapi import APIRouter, Query

from app.services import analysis_service

router = APIRouter(prefix="/analysis")

_PAIR = "^A[1-5]->A[1-5]$"


@router.get("/context-shift")
def context_shift(pair: str = Query("A1->A5", pattern=_PAIR)) -> Dict[str, Any]:
    return analysis_service.context_shift(pair)


@router.get("/decision-distribution")
def decision_distribution() -> Dict[str, Any]:
    return analysis_service.decision_distribution()


@router.get("/scaling")
def scaling() -> Dict[str, Any]:
    return analysis_service.scaling()


@router.get("/difficulty")
def difficulty(pair: str = Query("A1->A5", pattern=_PAIR)) -> Dict[str, Any]:
    return analysis_service.difficulty(pair)


@router.get("/semantic-review")
def semantic_review() -> Dict[str, Any]:
    return analysis_service.semantic_review()
