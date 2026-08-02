#!/usr/bin/env python3
"""
Parity test: production LlavaVerifier vs DiagnosticLlavaEngine inference paths.

Verifies that both code paths produce identical raw text and parsed decisions when
using equivalent settings:
  - same checkpoint
  - same A1 overlay image and prompt file
  - max_new_tokens=512
  - greedy decoding (do_sample=False)
  - P1_original prompt (production A1 prompt text)

Run on a GPU node with the LLaVA checkpoint and A1 ablation inputs available:

    pip install pytest pandas
    pytest tests/test_llava_parity.py -v -s

Or directly:

    python tests/test_llava_parity.py

Optional environment variables:
    LLAVA_PARITY_SAMPLE_COUNT  Number of samples (default 5, minimum 3 when available)
    LLAVA_PARITY_MAX_NEW_TOKENS  Generation limit (default 512)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import pytest
except ImportError:  # pragma: no cover - direct script execution without pytest
    pytest = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.model_config import resolve_model_checkpoint
from src.lvm.llava_verifier import LlavaVerifier
from src.lvm.parsers.base import parse_verification_response
from src.paths import discover_ablation_inputs_dir

DEFAULT_MODEL_KEY = "llava"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_SAMPLE_COUNT = 5
MIN_SAMPLE_COUNT = 3
PROMPT_VARIANT = "P1_original"
DECODING_MODE = "greedy"


@dataclass(frozen=True)
class ParitySample:
    sample_id: str
    image_path: Path
    prompt_path: Path


@dataclass(frozen=True)
class GenerationSettings:
    checkpoint: str
    max_new_tokens: int
    do_sample: bool
    prompt_variant: str
    decoding_mode: str


@dataclass
class ParityResult:
    sample: ParitySample
    production_raw: str
    diagnostic_raw: str
    production_decision: str
    diagnostic_decision: str
    settings: GenerationSettings


def _load_debug_llava_grounding_module():
    module_path = PROJECT_ROOT / "scripts" / "debug_llava_grounding.py"
    spec = importlib.util.spec_from_file_location("debug_llava_grounding", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load diagnostic module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _checkpoint_available() -> bool:
    checkpoint = resolve_model_checkpoint(DEFAULT_MODEL_KEY)
    path = Path(checkpoint)
    if path.is_absolute():
        return path.exists()
    return True


def _discover_parity_samples(limit: int) -> list[ParitySample]:
    ablation_dir = discover_ablation_inputs_dir()
    a1_dir = ablation_dir / "A1_overlay_only"
    prompt_index_path = a1_dir / "prompt_index.csv"
    if not prompt_index_path.is_file():
        return []

    df = pd.read_csv(prompt_index_path)
    base = prompt_index_path.parent.resolve()
    samples: list[ParitySample] = []

    for _, row in df.iterrows():
        sample_id = str(row["sample_id"]).strip()
        image_path = (base / str(row["image_path"])).resolve()
        prompt_path = (base / str(row["prompt_path"])).resolve()
        if not sample_id:
            continue
        if not image_path.is_file() or not prompt_path.is_file():
            continue
        samples.append(
            ParitySample(
                sample_id=sample_id,
                image_path=image_path,
                prompt_path=prompt_path,
            )
        )
        if len(samples) >= limit:
            break

    return samples


def _parse_production_decision(raw_response: str) -> str:
    return parse_verification_response(raw_response, model_key=DEFAULT_MODEL_KEY)["decision"]


def _format_diff(label: str, left: str, right: str) -> str:
    diff_lines = unified_diff(
        left.splitlines(),
        right.splitlines(),
        fromfile=f"production_{label}",
        tofile=f"diagnostic_{label}",
        lineterm="",
    )
    rendered = "\n".join(diff_lines)
    return rendered if rendered else f"(no line diff; {label!r} values differ)"


def _format_failure_report(result: ParityResult) -> str:
    settings = result.settings
    sample = result.sample
    lines = [
        "LLaVA inference parity FAILED",
        "",
        "Generation settings:",
        f"  checkpoint:      {settings.checkpoint}",
        f"  max_new_tokens:  {settings.max_new_tokens}",
        f"  do_sample:       {settings.do_sample}",
        f"  prompt_variant:  {settings.prompt_variant}",
        f"  decoding_mode:   {settings.decoding_mode}",
        "",
        f"sample_id:   {sample.sample_id}",
        f"image_path:  {sample.image_path}",
        f"prompt_path: {sample.prompt_path}",
        "",
    ]

    if result.production_raw != result.diagnostic_raw:
        lines.extend(
            [
                "Raw response diff:",
                _format_diff("raw_response", result.production_raw, result.diagnostic_raw),
                "",
                "Production raw response:",
                result.production_raw,
                "",
                "Diagnostic raw response:",
                result.diagnostic_raw,
                "",
            ]
        )

    if result.production_decision != result.diagnostic_decision:
        lines.extend(
            [
                "Parsed decision diff:",
                f"  production: {result.production_decision!r}",
                f"  diagnostic: {result.diagnostic_decision!r}",
                "",
            ]
        )

    return "\n".join(lines).rstrip()


def _resolve_sample_count(sample_count: Optional[int]) -> int:
    if sample_count is not None:
        return sample_count
    return int(os.environ.get("LLAVA_PARITY_SAMPLE_COUNT", DEFAULT_SAMPLE_COUNT))


def _resolve_max_new_tokens(max_new_tokens: Optional[int]) -> int:
    if max_new_tokens is not None:
        return max_new_tokens
    return int(os.environ.get("LLAVA_PARITY_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS))


def run_parity_checks(
    *,
    sample_count: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> list[ParityResult]:
    requested_count = _resolve_sample_count(sample_count)
    token_limit = _resolve_max_new_tokens(max_new_tokens)

    samples = _discover_parity_samples(requested_count)
    if len(samples) < MIN_SAMPLE_COUNT:
        raise RuntimeError(
            f"Need at least {MIN_SAMPLE_COUNT} A1 overlay samples with image+prompt files; "
            f"found {len(samples)} under verification_ablation_*/A1_overlay_only/."
        )

    checkpoint = resolve_model_checkpoint(DEFAULT_MODEL_KEY)
    settings = GenerationSettings(
        checkpoint=checkpoint,
        max_new_tokens=token_limit,
        do_sample=False,
        prompt_variant=PROMPT_VARIANT,
        decoding_mode=DECODING_MODE,
    )
    debug_module = _load_debug_llava_grounding_module()
    DiagnosticLlavaEngine = debug_module.DiagnosticLlavaEngine
    build_p1_prompt = debug_module.build_p1_prompt

    production_by_id: dict[str, tuple[str, str, str]] = {}
    verifier = LlavaVerifier(model_name=checkpoint, device_map="auto")
    try:
        for sample in samples:
            prompt_text = build_p1_prompt(sample.prompt_path)
            raw_response = verifier.generate_response(
                image_path=sample.image_path,
                prompt=prompt_text,
                max_new_tokens=token_limit,
            )
            production_by_id[sample.sample_id] = (
                raw_response,
                _parse_production_decision(raw_response),
                prompt_text,
            )
    finally:
        del verifier
        if _cuda_available():
            import torch

            torch.cuda.empty_cache()

    engine = DiagnosticLlavaEngine(model_path=checkpoint, device_map="auto")
    results: list[ParityResult] = []
    try:
        for index, sample in enumerate(samples):
            prod_raw, prod_decision, prompt_text = production_by_id[sample.sample_id]
            record = engine.run(
                sample_id=sample.sample_id,
                image_path=sample.image_path,
                prompt=prompt_text,
                prompt_variant=PROMPT_VARIANT,
                decoding_mode=DECODING_MODE,
                image_mode="correct",
                paired_sample_id=sample.sample_id,
                seed=42 + index,
                max_new_tokens=token_limit,
            )
            results.append(
                ParityResult(
                    sample=sample,
                    production_raw=prod_raw,
                    diagnostic_raw=str(record["raw_response"]),
                    production_decision=prod_decision,
                    diagnostic_decision=str(record["parsed_decision"]),
                    settings=settings,
                )
            )
    finally:
        del engine
        if _cuda_available():
            import torch

            torch.cuda.empty_cache()

    return results


def _execute_parity_test() -> None:
    requested_count = _resolve_sample_count(None)
    samples = _discover_parity_samples(requested_count)
    if len(samples) < MIN_SAMPLE_COUNT:
        if pytest is None:
            raise RuntimeError(
                f"Need at least {MIN_SAMPLE_COUNT} A1 overlay samples with image+prompt files; "
                f"found {len(samples)} under verification_ablation_*/A1_overlay_only/."
            )
        pytest.skip(
            f"Need at least {MIN_SAMPLE_COUNT} A1 overlay samples with image+prompt files; "
            f"found {len(samples)} under verification_ablation_*/A1_overlay_only/."
        )

    results = run_parity_checks()
    failures = [
        result
        for result in results
        if result.production_raw != result.diagnostic_raw
        or result.production_decision != result.diagnostic_decision
    ]

    if failures:
        reports = [_format_failure_report(result) for result in failures]
        message = "\n\n".join(reports)
        if pytest is None:
            raise AssertionError(message)
        pytest.fail(message)

    print(
        f"PASS: {len(results)} sample(s) matched for raw text and parsed decision "
        f"({PROMPT_VARIANT}, {DECODING_MODE})."
    )


if pytest is not None:

    @pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required for LLaVA parity test")
    @pytest.mark.skipif(not _checkpoint_available(), reason="LLaVA checkpoint not available locally")
    def test_llava_production_diagnostic_inference_parity() -> None:
        _execute_parity_test()


def main() -> int:
    if not _cuda_available():
        print("SKIP: CUDA GPU not available.")
        return 0
    if not _checkpoint_available():
        print("SKIP: LLaVA checkpoint not found.")
        return 0

    try:
        results = run_parity_checks()
    except RuntimeError as error:
        print(f"SKIP: {error}")
        return 0

    failures = [
        result
        for result in results
        if result.production_raw != result.diagnostic_raw
        or result.production_decision != result.diagnostic_decision
    ]

    if failures:
        for result in failures:
            print(_format_failure_report(result))
            print("-" * 72)
        print(f"FAIL: {len(failures)} of {len(results)} sample(s) differ.")
        return 1

    print(
        f"PASS: {len(results)} sample(s) matched for raw text and parsed decision "
        f"({PROMPT_VARIANT}, {DECODING_MODE})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
