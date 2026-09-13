#!/usr/bin/env python3
"""
InternVL label-semantics probe (ACCEPT / REVIEW / REJECT).

DIAGNOSTIC ONLY — does not use or modify the shared production parser.
Does not overwrite Stage 0 / Stage 1 qualification outputs.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.lvm.internvl_verifier import InternVLVerifier
from src.lvm.response_schema import parse_json_response
from src.paths import PROJECT_ROOT
from src.verification.jobs import load_verification_jobs, validate_jobs

PROBE_DECISIONS = ("ACCEPT", "REVIEW", "REJECT")
PROBE_TO_OFFICIAL = {
    "ACCEPT": "Reliable",
    "REVIEW": "Uncertain",
    "REJECT": "Unreliable",
}


def parse_probe_response(raw_text: str) -> dict[str, str]:
    """
    Probe-local parser: strict JSON, ACCEPT/REVIEW/REJECT only.

    No tolerant JSON repair. Malformed JSON → ValueError (parse failure).
    """
    parsed = parse_json_response(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed response must be a JSON object.")

    decision = str(parsed.get("decision", "")).strip()
    if decision not in PROBE_DECISIONS:
        raise ValueError(
            f"decision must be one of [{', '.join(PROBE_DECISIONS)}], got {decision!r}"
        )

    return {
        "decision": decision,
        "confidence_reasoning": str(parsed.get("confidence_reasoning") or "").strip(),
        "visual_reasoning": str(parsed.get("visual_reasoning") or "").strip(),
        "mapped_official_decision": PROBE_TO_OFFICIAL[decision],
    }


def build_record(
    *,
    sample_id: str,
    raw_response: str,
    parsed: dict[str, str] | None,
    runtime_seconds: float,
    parse_error: str = "",
    inference_error: str = "",
    model_path: str = "",
    experiment_id: str = "",
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "probe": "accept_review_reject",
        "raw_response": raw_response,
        "parsed_response": parsed,
        "decision": parsed.get("decision", "") if parsed else "",
        "mapped_official_decision": (
            parsed.get("mapped_official_decision", "") if parsed else ""
        ),
        "confidence_reasoning": parsed.get("confidence_reasoning", "") if parsed else "",
        "visual_reasoning": parsed.get("visual_reasoning", "") if parsed else "",
        "runtime_seconds": round(runtime_seconds, 4),
        "parse_error": parse_error,
        "inference_error": inference_error,
        "model_key": "internvl3",
        "model_name": model_path,
        "experiment_id": experiment_id,
        "condition": "label_semantics_probe_ARR",
        "generation": {
            "max_new_tokens": 512,
            "do_sample": False,
            "dtype": "bfloat16",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-index",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/diagnostics/model_qualification/internvl_label_semantics_20/prompt_index.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Isolated output directory (must not be Stage0/Stage1 paths).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/deac/csc/yangGrp/luoz23/models/InternVL3-8B-Instruct",
    )
    parser.add_argument("--experiment-id", type=str, default="label_semantics_ARR")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    forbidden = (
        "stage0",
        "balanced100",
    )
    # Soft guard: refuse writing into known Stage0/Stage1 dirs
    path_str = str(results_dir)
    if "/20260909_internvl3_qual/stage0" in path_str or "/20260909_internvl3_qual/balanced100" in path_str:
        raise SystemExit(
            f"Refusing to write into Stage0/Stage1 directory: {results_dir}"
        )

    jobs = load_verification_jobs(prompt_index=args.prompt_index)
    validate_jobs(jobs)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("===== InternVL label-semantics probe (ACCEPT/REVIEW/REJECT) =====")
    print(f"  diagnostic_only: yes")
    print(f"  prompt_index:    {args.prompt_index}")
    print(f"  results_dir:     {results_dir}")
    print(f"  model_path:      {args.model_path}")
    print(f"  samples:         {len(jobs)}")
    print(f"  do_sample:       False")
    print(f"  max_new_tokens:  {args.max_new_tokens}")
    print()

    verifier = InternVLVerifier(model_name=args.model_path, device_map="auto")
    index_rows = []

    for i, job in enumerate(jobs, start=1):
        out_path = results_dir / f"{job.sample_id}.json"
        print(f"[{i}/{len(jobs)}] {job.sample_id}")
        started = time.perf_counter()
        try:
            prompt = job.prompt_path.read_text(encoding="utf-8")
            raw = verifier.generate_response(
                image_path=job.image_path,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )
            runtime = time.perf_counter() - started
            parsed = None
            parse_error = ""
            try:
                parsed = parse_probe_response(raw)
                status = "ok"
            except ValueError as err:
                parse_error = str(err)
                status = "parse_error"
                print(f"  PARSE ERROR: {err}")
            record = build_record(
                sample_id=job.sample_id,
                raw_response=raw,
                parsed=parsed,
                runtime_seconds=runtime,
                parse_error=parse_error,
                model_path=args.model_path,
                experiment_id=args.experiment_id,
            )
        except Exception as err:
            runtime = time.perf_counter() - started
            tb = traceback.format_exc()
            print(f"  INFERENCE ERROR: {err}")
            record = build_record(
                sample_id=job.sample_id,
                raw_response="",
                parsed=None,
                runtime_seconds=runtime,
                inference_error=tb,
                model_path=args.model_path,
                experiment_id=args.experiment_id,
            )
            status = "inference_error"

        out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        index_rows.append(
            {
                "sample_id": job.sample_id,
                "result_path": out_path.name,
                "status": status,
            }
        )

    import pandas as pd

    pd.DataFrame(index_rows).to_csv(results_dir / "results_index.csv", index=False)
    print()
    print("Summary:")
    print(pd.DataFrame(index_rows)["status"].value_counts().to_string())
    print(f"Saved: {results_dir}")


if __name__ == "__main__":
    main()
