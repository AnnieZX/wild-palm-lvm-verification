"""Model-agnostic verification framework."""

from src.verification.base_adapter import BaseVerificationAdapter, VerificationOutcome
from src.verification.jobs import VerificationJob, load_verification_jobs, validate_jobs
from src.verification.output_manager import VerificationOutputManager
from src.verification.records import build_result_record
from src.verification.registry import create_adapter, get_registered_models, register_adapter
from src.verification.runner import RunnerConfig, VerificationRunner

__all__ = [
    "BaseVerificationAdapter",
    "RunnerConfig",
    "VerificationJob",
    "VerificationOutcome",
    "VerificationOutputManager",
    "VerificationRunner",
    "build_result_record",
    "create_adapter",
    "get_registered_models",
    "load_verification_jobs",
    "register_adapter",
    "validate_jobs",
]
