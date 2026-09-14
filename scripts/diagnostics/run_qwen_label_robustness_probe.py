#!/usr/bin/env python3
"""
Run Qwen2.5-VL label-robustness probe (S1–S4 × 20 samples = 80 cases).

DIAGNOSTIC ONLY. Probe-local token parser. Does not touch production A1–A5.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lvm.qwen_verifier import QwenVerifier  # noqa: E402
from scripts.diagnostics.qwen_label_robustness_lib import (  # noqa: E402
    CONDITION_SPECS,
    parse_condition_response,
)

DEFAULT_OUT = (
    ROOT / "outputs/diagnostics/model_qualification/qwen_label_robustness_20"
)
DEFAULT_MODEL = "/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct"


def build_record(
    *,
    sample_id: str,
    condition: str,
    image_path: str,
    gt_label: str,
    prompt_path: str,
    runtime_prompt: str,
    raw_response: str,
    parsed: dict[str, str] | None,
    runtime_seconds: float,
    parse_error: str = "",
    inference_error: str = "",
    model_path: str = "",
) -> dict:
    return {
        "sample_id": sample_id,
        "condition": condition,
        "image_path": image_path,
        "gt_label": gt_label,
        "prompt_path": prompt_path,
        "runtime_prompt": runtime_prompt,
        "raw_model_response": raw_response,
        "raw_decision_token": parsed.get("raw_decision_token", "") if parsed else "",
        "mapped_semantic_decision": (
            parsed.get("mapped_semantic_decision", "") if parsed else ""
        ),
        "confidence_reasoning": parsed.get("confidence_reasoning", "") if parsed else "",
        "visual_reasoning": parsed.get("visual_reasoning", "") if parsed else "",
        "parse_success": bool(parsed) and not parse_error and not inference_error,
        "parse_error": parse_error,
        "inference_error": inference_error,
        "runtime_seconds": round(runtime_seconds, 4),
        "checkpoint": model_path,
        "generation": {
            "max_new_tokens": 512,
            "do_sample": False,
            "dtype": "auto",
        },
        "model_key": "qwen2_5_vl",
        "diagnostic_only": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITION_SPECS.keys()),
        help="Subset of conditions to run (default: all).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip samples that already have a result JSON (default: true).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
    )
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    index_path = out_root / "prompt_index.csv"
    if not index_path.is_file():
        raise SystemExit(f"Missing prompt index: {index_path}. Run prepare first.")

    # Safety: only write under diagnostics
    if "/outputs/diagnostics/" not in str(out_root):
        raise SystemExit(f"Refusing non-diagnostics out_root: {out_root}")

    with index_path.open(newline="", encoding="utf-8") as fh:
        jobs = [r for r in csv.DictReader(fh) if r["condition"] in args.conditions]

    print("===== Qwen label-robustness probe =====")
    print(f"  diagnostic_only: yes")
    print(f"  out_root:        {out_root}")
    print(f"  model_path:      {args.model_path}")
    print(f"  conditions:      {args.conditions}")
    print(f"  jobs:            {len(jobs)}")
    print(f"  do_sample:       False")
    print(f"  max_new_tokens:  512")
    print(f"  skip_existing:   {args.skip_existing}")
    print()

    verifier = QwenVerifier(model_name=args.model_path, device_map="auto")
    index_rows: list[dict[str, str]] = []

    for i, job in enumerate(jobs, start=1):
        condition = job["condition"]
        sample_id = job["sample_id"]
        results_dir = out_root / "results" / condition
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{sample_id}.json"

        if args.skip_existing and out_path.exists():
            print(f"[{i}/{len(jobs)}] SKIP existing {condition}/{sample_id}")
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "condition": condition,
                    "result_path": str(out_path.relative_to(out_root)),
                    "status": "skipped_existing",
                }
            )
            continue

        prompt_path = (out_root / job["prompt_path"]).resolve()
        image_path = (out_root / job["image_path"]).resolve()
        prompt = prompt_path.read_text(encoding="utf-8")

        print(f"[{i}/{len(jobs)}] {condition}/{sample_id}")
        started = time.perf_counter()
        try:
            raw = verifier._generate_response(image_path, prompt)
            runtime = time.perf_counter() - started
            parsed = None
            parse_error = ""
            try:
                parsed = parse_condition_response(raw, condition)
                status = "ok"
            except ValueError as err:
                parse_error = str(err)
                status = "parse_error"
                print(f"  PARSE ERROR: {err}")
            record = build_record(
                sample_id=sample_id,
                condition=condition,
                image_path=str(image_path),
                gt_label=job.get("gt_label", ""),
                prompt_path=str(prompt_path),
                runtime_prompt=prompt,
                raw_response=raw,
                parsed=parsed,
                runtime_seconds=runtime,
                parse_error=parse_error,
                model_path=args.model_path,
            )
        except Exception as err:
            runtime = time.perf_counter() - started
            tb = traceback.format_exc()
            print(f"  INFERENCE ERROR: {err}")
            record = build_record(
                sample_id=sample_id,
                condition=condition,
                image_path=str(image_path),
                gt_label=job.get("gt_label", ""),
                prompt_path=str(prompt_path),
                runtime_prompt=prompt,
                raw_response="",
                parsed=None,
                runtime_seconds=runtime,
                inference_error=tb,
                model_path=args.model_path,
            )
            status = "inference_error"

        out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        index_rows.append(
            {
                "sample_id": sample_id,
                "condition": condition,
                "result_path": str(out_path.relative_to(out_root)),
                "status": status,
            }
        )

    import pandas as pd

    idx_path = out_root / "results_index.csv"
    pd.DataFrame(index_rows).to_csv(idx_path, index=False)
    print()
    print("Summary:")
    print(pd.DataFrame(index_rows)["status"].value_counts().to_string())
    print(f"Saved index: {idx_path}")


if __name__ == "__main__":
    main()
