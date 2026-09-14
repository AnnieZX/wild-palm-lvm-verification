#!/usr/bin/env python3
"""
Evaluate Qwen2.5-VL label-robustness probe results (after 80 inference cases).

Frozen-eval style metrics where meaningful:
  Reliable = positive, Unreliable = negative, Uncertain excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.qwen_label_robustness_lib import CONDITION_SPECS  # noqa: E402

DEFAULT_OUT = (
    ROOT / "outputs/diagnostics/model_qualification/qwen_label_robustness_20"
)


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def binary_metrics(rows: list[dict]) -> dict:
    """Sens/spec/balacc on Reliable vs Unreliable; Uncertain/parse excluded."""
    tp = fp = tn = fn = 0
    excluded = 0
    for r in rows:
        pred = r.get("mapped_semantic_decision") or ""
        gt = r.get("gt_label") or ""
        if pred not in {"Reliable", "Unreliable"}:
            excluded += 1
            continue
        if gt == "positive":
            if pred == "Reliable":
                tp += 1
            else:
                fn += 1
        elif gt == "negative":
            if pred == "Unreliable":
                tn += 1
            else:
                fp += 1
        else:
            excluded += 1
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    bal = (
        (sens + spec) / 2.0
        if sens is not None and spec is not None
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "excluded_uncertain_or_parse": excluded,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": bal,
        "confusion": {
            "Reliable_on_positive": tp,
            "Unreliable_on_positive": fn,
            "Reliable_on_negative": fp,
            "Unreliable_on_negative": tn,
        },
    }


def agreement(a: str, b: str) -> bool | None:
    if not a or not b:
        return None
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_root = args.out_root.resolve()

    with (out_root / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))
    sample_ids = [r["sample_id"] for r in manifest]
    gt_by_id = {r["sample_id"]: r["gt_label"] for r in manifest}

    by_cond: dict[str, dict[str, dict]] = {}
    missing: list[str] = []
    for cond in CONDITION_SPECS:
        by_cond[cond] = {}
        for sid in sample_ids:
            path = out_root / "results" / cond / f"{sid}.json"
            if not path.is_file():
                missing.append(f"{cond}/{sid}")
                continue
            by_cond[cond][sid] = load_result(path)

    if missing:
        raise SystemExit(
            f"Incomplete results ({len(missing)} missing). Examples: {missing[:5]}"
        )

    # Comparison table
    table_rows = []
    changed_s1_s2: list[str] = []
    changed_s2_s3: list[str] = []
    changed_s2_s4: list[str] = []

    for sid in sample_ids:
        s1 = by_cond["S1_semantic"][sid].get("mapped_semantic_decision", "")
        s2 = by_cond["S2_neutral"][sid].get("mapped_semantic_decision", "")
        s3 = by_cond["S3_token_permutation"][sid].get("mapped_semantic_decision", "")
        s4 = by_cond["S4_order_permutation"][sid].get("mapped_semantic_decision", "")
        a12 = agreement(s1, s2)
        a23 = agreement(s2, s3)
        a24 = agreement(s2, s4)
        if a12 is False:
            changed_s1_s2.append(sid)
        if a23 is False:
            changed_s2_s3.append(sid)
        if a24 is False:
            changed_s2_s4.append(sid)
        table_rows.append(
            {
                "sample_id": sid,
                "GT": gt_by_id[sid],
                "S1_semantic": s1,
                "S2_mapped": s2,
                "S3_mapped": s3,
                "S4_mapped": s4,
                "S1_vs_S2": a12,
                "S2_vs_S3": a23,
                "S2_vs_S4": a24,
                "S1_raw": by_cond["S1_semantic"][sid].get("raw_decision_token", ""),
                "S2_raw": by_cond["S2_neutral"][sid].get("raw_decision_token", ""),
                "S3_raw": by_cond["S3_token_permutation"][sid].get(
                    "raw_decision_token", ""
                ),
                "S4_raw": by_cond["S4_order_permutation"][sid].get(
                    "raw_decision_token", ""
                ),
                "S1_parse_ok": by_cond["S1_semantic"][sid].get("parse_success", False),
                "S2_parse_ok": by_cond["S2_neutral"][sid].get("parse_success", False),
                "S3_parse_ok": by_cond["S3_token_permutation"][sid].get(
                    "parse_success", False
                ),
                "S4_parse_ok": by_cond["S4_order_permutation"][sid].get(
                    "parse_success", False
                ),
            }
        )

    table_path = out_root / "comparison_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    def pair_agreement(flag_key: str) -> dict:
        vals = [r[flag_key] for r in table_rows]
        comparable = [v for v in vals if v is not None]
        n_agree = sum(1 for v in comparable if v)
        n_comp = len(comparable)
        return {
            "n_comparable": n_comp,
            "n_agree": n_agree,
            "n_disagree": n_comp - n_agree,
            "agreement_rate": (n_agree / n_comp) if n_comp else None,
        }

    condition_summaries = {}
    for cond in CONDITION_SPECS:
        rows = [by_cond[cond][sid] for sid in sample_ids]
        for r in rows:
            r["gt_label"] = gt_by_id[r["sample_id"]]
        parse_ok = sum(1 for r in rows if r.get("parse_success"))
        dist = Counter(
            (r.get("mapped_semantic_decision") or "PARSE_FAIL") for r in rows
        )
        condition_summaries[cond] = {
            "N": len(rows),
            "parse_success": parse_ok,
            "parse_failure": len(rows) - parse_ok,
            "Reliable": dist.get("Reliable", 0),
            "Uncertain": dist.get("Uncertain", 0),
            "Unreliable": dist.get("Unreliable", 0),
            "PARSE_FAIL": dist.get("PARSE_FAIL", 0),
            "metrics": binary_metrics(rows),
        }

    any_change = sorted(
        set(changed_s1_s2) | set(changed_s2_s3) | set(changed_s2_s4)
    )

    # Descriptive robustness label
    rates = [
        pair_agreement("S1_vs_S2")["agreement_rate"],
        pair_agreement("S2_vs_S3")["agreement_rate"],
        pair_agreement("S2_vs_S4")["agreement_rate"],
    ]
    rates_f = [r for r in rates if r is not None]
    min_agree = min(rates_f) if rates_f else 0.0
    # Collapse heuristic: one class dominates all mapped decisions in a condition
    collapse_flags = []
    for cond, summary in condition_summaries.items():
        mapped_n = summary["Reliable"] + summary["Uncertain"] + summary["Unreliable"]
        if mapped_n > 0:
            max_share = max(
                summary["Reliable"], summary["Uncertain"], summary["Unreliable"]
            ) / mapped_n
            if max_share >= 0.95 and summary["parse_failure"] == 0:
                collapse_flags.append(cond)

    if min_agree >= 0.9 and not collapse_flags:
        interpretation = "ROBUST"
    elif min_agree >= 0.6 and not (
        len(collapse_flags) >= 2
    ):
        interpretation = "MODERATELY_SENSITIVE"
    else:
        interpretation = "SEVERELY_SENSITIVE"

    report = {
        "n_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "condition_summaries": condition_summaries,
        "semantic_agreement": {
            "S1_vs_S2": pair_agreement("S1_vs_S2"),
            "S2_vs_S3": pair_agreement("S2_vs_S3"),
            "S2_vs_S4": pair_agreement("S2_vs_S4"),
        },
        "changed_sample_ids": {
            "S1_vs_S2": changed_s1_s2,
            "S2_vs_S3": changed_s2_s3,
            "S2_vs_S4": changed_s2_s4,
            "any_pair": any_change,
        },
        "n_samples_any_semantic_change": len(any_change),
        "robustness_interpretation": interpretation,
        "internvl_reference": {
            "original_semantic_labels": {
                "Reliable": 10,
                "Uncertain": 8,
                "Unreliable": 0,
                "parse_failures": 2,
            },
            "accept_review_reject": {
                "REJECT": 20,
                "note": "20/20 REJECT under ACCEPT/REVIEW/REJECT verbalizers",
            },
        },
        "comparison_table": str(table_path),
    }

    report_path = out_root / "EVALUATION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Human-readable summary
    lines = [
        "===== Qwen label-robustness evaluation =====",
        f"samples: {len(sample_ids)}",
        f"interpretation: {interpretation}",
        "",
        "--- Per-condition distributions (mapped semantic) ---",
    ]
    for cond, s in condition_summaries.items():
        lines.append(
            f"{cond}: N={s['N']} parse_ok={s['parse_success']}/{s['N']} "
            f"R={s['Reliable']} U={s['Uncertain']} N={s['Unreliable']} "
            f"fail={s['parse_failure']}"
        )
        m = s["metrics"]
        lines.append(
            f"  sens={m['sensitivity']} spec={m['specificity']} "
            f"balacc={m['balanced_accuracy']} excl={m['excluded_uncertain_or_parse']}"
        )
    lines.append("")
    lines.append("--- Semantic agreement ---")
    for k, v in report["semantic_agreement"].items():
        lines.append(
            f"{k}: {v['n_agree']}/{v['n_comparable']} "
            f"({v['agreement_rate']}) disagree={v['n_disagree']}"
        )
    lines.append("")
    lines.append(f"changed S1↔S2 ({len(changed_s1_s2)}): {changed_s1_s2}")
    lines.append(f"changed S2↔S3 ({len(changed_s2_s3)}): {changed_s2_s3}")
    lines.append(f"changed S2↔S4 ({len(changed_s2_s4)}): {changed_s2_s4}")
    lines.append(f"any change ({len(any_change)}): {any_change}")
    lines.append("")
    lines.append("InternVL reference: semantic R10/U8/fail2/N0 → ARR 20/20 REJECT")
    text = "\n".join(lines) + "\n"
    (out_root / "EVALUATION_SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {report_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
