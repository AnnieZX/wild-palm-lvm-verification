"""Abstract verification adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.verification.jobs import VerificationJob


@dataclass(frozen=True)
class VerificationOutcome:
    """Result of verifying one sample."""

    record: dict[str, Any]
    status: str


class BaseVerificationAdapter(ABC):
    """
    Model-agnostic interface for VLM verification inference.

    Subclasses implement provider-specific inference only. The runner owns
    dataset iteration, resume, logging, and output persistence.
    """

    @property
    @abstractmethod
    def model_label(self) -> str:
        """Human-readable model name for logging."""

    @abstractmethod
    def verify(self, job: VerificationJob) -> VerificationOutcome:
        """Run inference for one verification sample."""
