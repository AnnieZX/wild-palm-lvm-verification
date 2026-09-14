#!/usr/bin/env python3
"""
Run GLM-4.6V-Flash label-robustness probe (S1–S4 × 20 = 80 cases).

DIAGNOSTIC ONLY. Reuses probe-local token parser from qwen_label_robustness_lib.
Uses native TF5 Glm46vFlashVerifier (enable_thinking=False).
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

from src.lvm.glm_4_6v_flash_verifier import (  # noqa: E402
    Glm46vFlashVerifier,
    extract_final_answer_text,
)
from scripts.diagnostics.qwen_label_robustness_lib import (  # noqa: E402
    CONDITION_SPECS,
    parse_condition_response,
)

DEFAULT_OUT = (
    ROOT / "outputs/diagnostics/model_qualification/glm46v_label_robustness_20"
)
DEFAULT_MODEL = "/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash"


def build_record(
    *,
    sample_id: str,
    condition: str,
    image_path: str,
    gt_label: str,
    prompt_path: str,
    runtime_prompt: str,
    raw_response: str,
    final_answer_text: str,
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
        "final_answer_text": final_answer_text,
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
            "enable_thinking": False,
        },
        "model_key": "glm_4_6v_flash",
        "diagnostic_only": True,
        "rope_patch_executed": False,
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
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument(
        "--no-skip-existing", action="store_false", dest="skip_existing"
    )
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    index_path = out_root / "prompt_index.csv"
    if not index_path.is_file():
        raise SystemExit(f"Missing prompt index: {index_path}")
    if "/outputs/diagnostics/" not in str(out_root):
        raise SystemExit(f"Refusing non-diagnostics out_root: {out_root}")

    from src.lvm.glm_4_6v_flash_verifier import Glm46vFlashVerifier as _V

    assert not hasattr(_V, "_ensure_rope_scaling"), "RoPE patch must remain disabled"

    with index_path.open(newline="", encoding="utf-8") as fh:
        jobs = [r for r in csv.DictReader(fh) if r["condition"] in args.conditions]

    print("===== GLM-4.6V-Flash label-robustness probe =====")
    print("  diagnostic_only: yes")
    print(f"  out_root:        {out_root}")
    print(f"  model_path:      {args.model_path}")
    print(f"  conditions:      {args.conditions}")
    print(f"  jobs:            {len(jobs)}")
    print("  do_sample:       False")
    print("  max_new_tokens:  512")
    print("  enable_thinking: False")
    print(f"  skip_existing:   {args.skip_existing}")
    print()

    verifier = Glm46vFlashVerifier(
        model_name=args.model_path,
        device_map="auto",
        enable_thinking=False,
    )
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
            raw = verifier.generate_response(
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=512,
            )
            runtime = time.perf_counter() - started
            final_answer = extract_final_answer_text(raw)
            parsed = None
            parse_error = ""
            try:
                parsed = parse_condition_response(final_answer, condition)
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
                final_answer_text=final_answer,
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
                final_answer_text="",
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
