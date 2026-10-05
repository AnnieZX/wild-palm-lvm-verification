"""Research-demo API routes mounted under /api."""

from fastapi import APIRouter

from app.routers import analysis, detections, images, models, review, summary

router = APIRouter(prefix="/api")
router.include_router(summary.router, tags=["summary"])
router.include_router(models.router, tags=["models"])
router.include_router(detections.router, tags=["detections"])
router.include_router(images.router, tags=["images"])
router.include_router(analysis.router, tags=["analysis"])
router.include_router(review.router, tags=["review"])
