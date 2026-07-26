"""Aggregate API routers."""

from fastapi import APIRouter

from app.api.routes import health, images, models, samples, statistics

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(statistics.router, tags=["statistics"])
api_router.include_router(samples.router, tags=["samples"])
api_router.include_router(images.router, tags=["images"])
