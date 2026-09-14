#!/usr/bin/env python3
"""
Prompt-parity + GLM chat-template audit for the label-robustness probe.

Reuses the exact S1–S4 prompts already prepared for the Qwen probe
(copied under glm46v_label_robustness_20/). Does not regenerate permutations.

DIAGNOSTIC ONLY.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.qwen_label_robustness_lib import (  # noqa: E402
    CONDITION_SPECS,
    audit_prompt_parity,
)

DEFAULT_OUT = (
    ROOT / "outputs/diagnostics/model_qualification/glm46v_label_robustness_20"
)
AUDIT_SAMPLES_DEFAULT = ("sample_000071", "sample_000019", "sample_000163")
MODEL_PATH = "/deac/csc/yangGrp/luoz23/models/GLM-4.6V-Flash"


def dump_glm_chat_template_audit(
    *,
    out_root: Path,
    audit_dir: Path,
    sample_ids: list[str],
    model_path: str,
) -> None:
    """Save processor chat-template wraps for audit samples (no weight load)."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    for sid in sample_ids:
        for key in CONDITION_SPECS:
            user_prompt = (out_root / "prompts" / key / f"{sid}.txt").read_text(
                encoding="utf-8"
            )
            # Image placeholder path for template structure (no decode needed).
            # Use a tiny existing overlay path referenced by the probe.
            img_path = (
                out_root / "prompts" / ".." / ".." / ".." / ".."
            )  # unused; pass string placeholder via messages
            # Resolve real image from prompt_index
            import csv

            with (out_root / "prompt_index.csv").open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            img = None
            for r in rows:
                if r["sample_id"] == sid and r["condition"] == key:
                    img = (out_root / r["image_path"]).resolve()
                    break
            assert img is not None and img.is_file(), (sid, key)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img)},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            # Match verifier: enable_thinking=False when supported.
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            try:
                wrapped = processor.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                wrapped = processor.apply_chat_template(messages, **kwargs)

            payload = {
                "sample_id": sid,
                "condition": key,
                "system_prompt": None,
                "user_prompt_semantic": user_prompt,
                "image_path": str(img),
                "image_token_placement": "chat-template image content block before text",
                "enable_thinking": False,
                "chat_template_text": wrapped,
                "decision_definitions_excerpt": _decision_excerpt(user_prompt),
                "json_instruction_excerpt": _json_excerpt(user_prompt),
            }
            (audit_dir / f"{sid}__{key}__glm_chat_template.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            (audit_dir / f"{sid}__{key}__glm_chat_template.txt").write_text(
                wrapped if isinstance(wrapped, str) else str(wrapped),
                encoding="utf-8",
            )


def _decision_excerpt(text: str) -> str:
    i = text.find("Decision definitions")
    return text[i : i + 600] if i >= 0 else ""


def _json_excerpt(text: str) -> str:
    i = text.find('"decision"')
    return text[max(0, i - 80) : i + 120] if i >= 0 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--audit-sample-ids",
        nargs="+",
        default=list(AUDIT_SAMPLES_DEFAULT),
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    if "/outputs/diagnostics/" not in str(out_root):
        raise SystemExit(f"Refusing non-diagnostics out_root: {out_root}")

    audit_dir = out_root / "prompt_parity_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # README
    (out_root / "README.md").write_text(
        """# GLM-4.6V-Flash Label-Robustness Probe (n=20)

**DIAGNOSTIC ONLY.** Exact methodological parity with
`qwen_label_robustness_20` (same 20 sample_ids, same S1–S4 prompts).

Uses isolated env `wild-palm-glm46v` (Transformers 5.17 native Glm4v).
Does not overwrite Qwen/InternVL robustness outputs.
""",
        encoding="utf-8",
    )

    all_ok = True
    audit_reports = []
    for sid in args.audit_sample_ids:
        prompts = {
            key: (out_root / "prompts" / key / f"{sid}.txt").read_text(encoding="utf-8")
            for key in CONDITION_SPECS
        }
        for key, text in prompts.items():
            (audit_dir / f"{sid}__{key}.txt").write_text(text, encoding="utf-8")
        report = audit_prompt_parity(prompts)
        report["sample_id"] = sid
        all_ok = all_ok and bool(report["ok"])
        audit_reports.append(report)
        for diff_key in ("diff_S1_S2", "diff_S2_S3", "diff_S2_S4"):
            (audit_dir / f"{sid}__{diff_key}.diff").write_text(
                str(report[diff_key]), encoding="utf-8"
            )
        (audit_dir / f"{sid}__parity_report.txt").write_text(
            "\n".join(
                [
                    f"sample_id: {sid}",
                    f"ok: {report['ok']}",
                    f"s1_s2_normalized_identical: {report['s1_s2_normalized_identical']}",
                    f"s2_s3_normalized_identical: {report['s2_s3_normalized_identical']}",
                    f"s2_s4_rest_identical: {report['s2_s4_rest_identical']}",
                    f"s2_s4_blocks_semantically_equal: {report['s2_s4_blocks_semantically_equal']}",
                    f"s2_s4_order_actually_differs: {report['s2_s4_order_actually_differs']}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    print("===== GLM chat-template audit (processor only; no weights) =====")
    dump_glm_chat_template_audit(
        out_root=out_root,
        audit_dir=audit_dir,
        sample_ids=list(args.audit_sample_ids),
        model_path=args.model_path,
    )

    import csv

    with (out_root / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    verdict = {
        "prompt_parity_ok": all_ok,
        "audit_sample_ids": list(args.audit_sample_ids),
        "n_samples": len(rows),
        "n_conditions": len(CONDITION_SPECS),
        "n_inference_cases": len(rows) * len(CONDITION_SPECS),
        "gt_positive": sum(1 for r in rows if r["gt_label"] == "positive"),
        "gt_negative": sum(1 for r in rows if r["gt_label"] == "negative"),
        "sample_ids": [r["sample_id"] for r in rows],
        "checkpoint": args.model_path,
        "prompts_source": str(
            ROOT
            / "outputs/diagnostics/model_qualification/qwen_label_robustness_20/prompts"
        ),
        "generation": {
            "do_sample": False,
            "max_new_tokens": 512,
            "enable_thinking": False,
        },
        "per_sample": [
            {
                "sample_id": r["sample_id"],
                "ok": r["ok"],
                "s1_s2_normalized_identical": r["s1_s2_normalized_identical"],
                "s2_s3_normalized_identical": r["s2_s3_normalized_identical"],
                "s2_s4_rest_identical": r["s2_s4_rest_identical"],
                "s2_s4_blocks_semantically_equal": r["s2_s4_blocks_semantically_equal"],
                "s2_s4_order_actually_differs": r["s2_s4_order_actually_differs"],
            }
            for r in audit_reports
        ],
    }
    (audit_dir / "VERDICT.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )
    (out_root / "PREPARE_SUMMARY.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: verdict[k] for k in verdict if k != "per_sample"}, indent=2))
    for row in verdict["per_sample"]:
        print(row)
    if not all_ok:
        print("PROMPT PARITY FAILED — do not submit GPU job.", file=sys.stderr)
        raise SystemExit(2)
    print("PROMPT PARITY PASSED")


if __name__ == "__main__":
    main()
