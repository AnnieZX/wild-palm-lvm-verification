#!/usr/bin/env python3
"""
Prepare Qwen2.5-VL label-robustness diagnostic assets and run prompt-parity audit.

Writes under:
  outputs/diagnostics/model_qualification/qwen_label_robustness_20/

Does NOT modify InternVL diagnostics, A1–A5 production outputs, or shared parsers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.qwen_label_robustness_lib import (  # noqa: E402
    CONDITION_DIRS,
    CONDITION_SPECS,
    assert_no_unintended_label_residue,
    audit_prompt_parity,
    expected_sample_ids,
    transform_canonical_prompt,
)

INTERN_MANIFEST = (
    ROOT
    / "outputs/diagnostics/model_qualification/internvl_label_semantics_20/manifest.csv"
)
OUT_ROOT = (
    ROOT / "outputs/diagnostics/model_qualification/qwen_label_robustness_20"
)
AUDIT_SAMPLES_DEFAULT = ("sample_000071", "sample_000019", "sample_000163")


def load_intern_manifest() -> list[dict[str, str]]:
    with INTERN_MANIFEST.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["sample_id"] for r in rows]
    expected = expected_sample_ids()
    if ids != expected:
        raise SystemExit(
            "InternVL manifest sample_ids do not match expected fixed set.\n"
            f"  got:      {ids}\n"
            f"  expected: {expected}"
        )
    pos = sum(1 for r in rows if r["gt_label"] == "positive")
    neg = sum(1 for r in rows if r["gt_label"] == "negative")
    if pos != 10 or neg != 10:
        raise SystemExit(f"Expected 10+/10-, got {pos}+/{neg}-")
    return rows


def write_readme(out_root: Path) -> None:
    text = """# Qwen2.5-VL Label-Robustness Probe (n=20)

## Status

**DIAGNOSTIC ONLY** — not an A1–A5 ablation, not prompt optimization.

## Purpose

Test whether Qwen2.5-VL semantic verification decisions are robust to:
1. semantic decision labels (S1)
2. neutral decision tokens A/B/C (S2)
3. token-to-class permutation (S3)
4. definition presentation order (S4)

## Matched sample set

Exact same 20 `sample_id`s as:
`outputs/diagnostics/model_qualification/internvl_label_semantics_20/`

10 GT-positive / 10 GT-negative. A1 overlay images from
`outputs/verification_dataset/images/` with canonical A1 prompts from
`outputs/verification_ablation_1000/A1_overlay_only/prompts/`.

## Conditions

| Key | Verbalizers | Mapping | Definition order |
|-----|-------------|---------|------------------|
| S1_semantic | Reliable/Uncertain/Unreliable | identity | R→U→N |
| S2_neutral | A/B/C | A=R, B=U, C=N | R→U→N |
| S3_token_permutation | B/C/A | B=R, C=U, A=N | R→U→N |
| S4_order_permutation | A/B/C | A=R, B=U, C=N | N→U→R |

## Isolation

- Does not modify production parser
- Does not modify canonical A1–A5 prompts
- Does not overwrite InternVL diagnostic outputs
- All artifacts live under this directory
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def prepare(out_root: Path, audit_sample_ids: list[str]) -> dict:
    if out_root.exists():
        # Refuse clobbering an existing completed diagnostic tree with results.
        existing_results = list((out_root / "results").glob("S*/sample_*.json"))
        if existing_results:
            raise SystemExit(
                f"Refusing to overwrite existing results under {out_root} "
                f"({len(existing_results)} JSON files). Choose a new path or remove manually."
            )
        # Safe to rebuild prompts/manifests if no results yet
        for sub in ("prompts", "prompt_parity_audit"):
            p = out_root / sub
            if p.exists():
                shutil.rmtree(p)

    out_root.mkdir(parents=True, exist_ok=True)
    for key in CONDITION_DIRS:
        (out_root / "prompts" / key).mkdir(parents=True, exist_ok=True)
        (out_root / "results" / key).mkdir(parents=True, exist_ok=True)
    (out_root / "prompt_parity_audit").mkdir(parents=True, exist_ok=True)

    rows = load_intern_manifest()
    write_readme(out_root)

    manifest_out_rows: list[dict[str, str]] = []
    prompt_index_rows: list[dict[str, str]] = []

    for row in rows:
        sid = row["sample_id"]
        canonical_prompt_path = Path(row["canonical_prompt_path"])
        image_path = Path(row["canonical_image_path"])
        if not canonical_prompt_path.is_file():
            raise FileNotFoundError(canonical_prompt_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        canonical_text = canonical_prompt_path.read_text(encoding="utf-8")
        for key in CONDITION_SPECS:
            prompt_text = transform_canonical_prompt(canonical_text, key)
            assert_no_unintended_label_residue(prompt_text, key)
            if key == "S1_semantic" and prompt_text != canonical_text:
                raise SystemExit(f"S1 must be byte-identical to canonical A1 for {sid}")

            rel_prompt = f"prompts/{key}/{sid}.txt"
            abs_prompt = out_root / rel_prompt
            abs_prompt.write_text(prompt_text, encoding="utf-8")

            # Image path relative to out_root
            try:
                rel_image = Path(os_path_rel(out_root, image_path))
            except Exception:
                rel_image = image_path

            prompt_index_rows.append(
                {
                    "sample_id": sid,
                    "condition": key,
                    "prompt_path": rel_prompt,
                    "image_path": str(rel_image),
                    "gt_label": row["gt_label"],
                    "canonical_order": row["canonical_order"],
                    "max_iou": row["max_iou"],
                }
            )

        manifest_out_rows.append(
            {
                "sample_id": sid,
                "gt_label": row["gt_label"],
                "canonical_order": row["canonical_order"],
                "max_iou": row["max_iou"],
                "image_path": str(image_path),
                "canonical_a1_prompt_path": str(canonical_prompt_path),
                "source_internvl_manifest": str(INTERN_MANIFEST),
                "visual_representation": "A1_overlay_only (verification_dataset overlay image)",
            }
        )

    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_out_rows)

    with (out_root / "prompt_index.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(prompt_index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prompt_index_rows)

    # Per-condition indexes for convenience
    for key in CONDITION_SPECS:
        subset = [r for r in prompt_index_rows if r["condition"] == key]
        path = out_root / f"prompt_index_{key}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(subset[0].keys()))
            writer.writeheader()
            writer.writerows(subset)

    # Prompt-parity audit on selected samples
    audit_dir = out_root / "prompt_parity_audit"
    all_ok = True
    audit_reports = []
    for sid in audit_sample_ids:
        prompts = {
            key: (out_root / "prompts" / key / f"{sid}.txt").read_text(encoding="utf-8")
            for key in CONDITION_SPECS
        }
        # Save exact prompts for inspection
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

        summary_lines = [
            f"sample_id: {sid}",
            f"ok: {report['ok']}",
            f"s1_s2_normalized_identical: {report['s1_s2_normalized_identical']}",
            f"s2_s3_normalized_identical: {report['s2_s3_normalized_identical']}",
            f"s2_s4_rest_identical: {report['s2_s4_rest_identical']}",
            f"s2_s4_blocks_semantically_equal: {report['s2_s4_blocks_semantically_equal']}",
            f"s2_s4_order_actually_differs: {report['s2_s4_order_actually_differs']}",
            "",
            "===== RAW DIFF S1 → S2 =====",
            str(report["diff_S1_S2"]),
            "===== RAW DIFF S2 → S3 =====",
            str(report["diff_S2_S3"]),
            "===== RAW DIFF S2 → S4 =====",
            str(report["diff_S2_S4"]),
            "===== NORMALIZED DIFF S1 → S2 (should be empty) =====",
            str(report["normalized_diff_S1_S2"]) or "(empty)",
            "===== NORMALIZED DIFF S2 → S3 (should be empty) =====",
            str(report["normalized_diff_S2_S3"]) or "(empty)",
        ]
        (audit_dir / f"{sid}__parity_report.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

    verdict = {
        "prompt_parity_ok": all_ok,
        "audit_sample_ids": audit_sample_ids,
        "n_samples": len(rows),
        "n_conditions": len(CONDITION_SPECS),
        "n_inference_cases": len(rows) * len(CONDITION_SPECS),
        "gt_positive": sum(1 for r in rows if r["gt_label"] == "positive"),
        "gt_negative": sum(1 for r in rows if r["gt_label"] == "negative"),
        "sample_ids": [r["sample_id"] for r in rows],
        "checkpoint": "/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct",
        "generation": {"do_sample": False, "max_new_tokens": 512},
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
    return verdict


def os_path_rel(base: Path, target: Path) -> str:
    return str(Path(os.path.relpath(target, base)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT,
    )
    parser.add_argument(
        "--audit-sample-ids",
        nargs="+",
        default=list(AUDIT_SAMPLES_DEFAULT),
    )
    args = parser.parse_args()

    print("===== Prepare Qwen label-robustness diagnostic =====")
    print(f"out_root: {args.out_root}")
    verdict = prepare(args.out_root.resolve(), args.audit_sample_ids)
    print(json.dumps({k: verdict[k] for k in verdict if k != "per_sample"}, indent=2))
    print("per_sample:")
    for row in verdict["per_sample"]:
        print(f"  {row}")
    if not verdict["prompt_parity_ok"]:
        print("PROMPT PARITY FAILED — do not submit GPU job.", file=sys.stderr)
        raise SystemExit(2)
    print("PROMPT PARITY PASSED")


if __name__ == "__main__":
    main()
