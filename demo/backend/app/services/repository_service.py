"""Shared repository singleton for API routes."""

from __future__ import annotations

from typing import Optional

from app.config import settings
from app.repository.experiment_repository import ExperimentRepository

_repository: Optional[ExperimentRepository] = None


def get_repository() -> ExperimentRepository:
    global _repository
    if _repository is None:
        _repository = ExperimentRepository(settings.outputs_root)
    return _repository
