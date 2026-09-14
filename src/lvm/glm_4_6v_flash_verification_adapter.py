"""GLM-4.6V-Flash verification adapter for the frozen verification framework."""

from __future__ import annotations

import time
import traceback

from src.lvm.glm_4_6v_flash_verifier import (
    Glm46vFlashVerifier,
    extract_final_answer_text,
)
from src.lvm.verification_response_parser import parse_verification_response
from src.verification.base_adapter import BaseVerificationAdapter, VerificationOutcome
from src.verification.jobs import VerificationJob
from src.verification.records import build_result_record


def build_glm_4_6v_flash_adapter(
    *,
    model_name: str,
    batch_size: int = 1,
    device_map: str = "auto",
    max_new_tokens: int = 512,
    model_key: str = "glm_4_6v_flash",
    condition: str = "",
    experiment_id: str = "",
) -> "Glm46vFlashVerificationAdapter":
    """Factory used by the verification adapter registry."""
    return Glm46vFlashVerificationAdapter(
        model_name=model_name,
        batch_size=batch_size,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
        model_key=model_key,
        condition=condition,
        experiment_id=experiment_id,
    )


class Glm46vFlashVerificationAdapter(BaseVerificationAdapter):
    """
    GLM-4.6V-Flash verification adapter.

    Implements model-specific inference only. Orchestration, resume, and
    output persistence are handled by VerificationRunner.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 1,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        model_key: str = "glm_4_6v_flash",
        condition: str = "",
        experiment_id: str = "",
        *,
        enable_thinking: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.model_key = model_key
        self.condition = condition
        self.experiment_id = experiment_id
        self.enable_thinking = enable_thinking
        self._verifier = Glm46vFlashVerifier(
            model_name=model_name,
            device_map=device_map,
            enable_thinking=enable_thinking,
        )

    @property
    def model_label(self) -> str:
        return "GLM-4.6V-Flash"

    def verify(self, job: VerificationJob) -> VerificationOutcome:
        """Run GLM-4.6V-Flash inference for one verification sample."""
        started_at = time.perf_counter()
        metadata = {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "condition": self.condition,
            "experiment_id": self.experiment_id,
        }
        try:
            prompt = job.prompt_path.read_text(encoding="utf-8")
            raw_response = self._verifier.generate_response(
                image_path=job.image_path,
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
            )
            runtime_seconds = time.perf_counter() - started_at
            final_answer_text = extract_final_answer_text(raw_response)

            parsed = None
            parse_error = ""
            try:
                # Parse only the final-answer span; do not repair JSON.
                parsed = parse_verification_response(
                    final_answer_text, model_key=self.model_key
                )
            except ValueError as error:
                parse_error = str(error)
                print(
                    f"PARSE ERROR for {job.sample_id}: {error}\n"
                    f"{traceback.format_exc()}"
                )

            record = build_result_record(
                sample_id=job.sample_id,
                raw_response=raw_response,
                parsed=parsed,
                runtime_seconds=runtime_seconds,
                parse_error=parse_error,
                **metadata,
            )
            record["final_answer_text"] = final_answer_text
            record["enable_thinking"] = self.enable_thinking
            record["generation"] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "enable_thinking": self.enable_thinking,
            }
            status = "ok" if not parse_error else "parse_error"
            return VerificationOutcome(record=record, status=status)

        except Exception as error:
            runtime_seconds = time.perf_counter() - started_at
            tb = traceback.format_exc()
            print(
                f"\nINFERENCE ERROR for {job.sample_id}: "
                f"{type(error).__name__}: {error}\n{tb}",
                flush=True,
            )
            traceback.print_exc()

            record = build_result_record(
                sample_id=job.sample_id,
                raw_response="",
                parsed=None,
                runtime_seconds=runtime_seconds,
                inference_error=tb,
                **metadata,
            )
            record["final_answer_text"] = ""
            record["enable_thinking"] = self.enable_thinking
            return VerificationOutcome(record=record, status="inference_error")
