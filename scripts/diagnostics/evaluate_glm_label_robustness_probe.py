#!/usr/bin/env python3
"""
Evaluate GLM-4.6V-Flash label-robustness probe (80 cases).

Uses the same robustness interpretation thresholds as the Qwen probe:
  ROBUST / MODERATELY_SENSITIVE / SEVERELY_SENSITIVE
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
    ROOT / "outputs/diagnostics/model_qualification/glm46v_label_robustness_20"
)


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selective_metrics(rows: list[dict]) -> dict:
    """Canonical selective: Reliable=pos, Unreliable=neg, Uncertain excluded."""
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

    def safe(a: float, b: float) -> float | None:
        return (a / b) if b else None

    prec = safe(tp, tp + fp)
    rec = safe(tp, tp + fn)
    spec = safe(tn, tn + fp)
    f1 = (
        safe(2 * prec * rec, prec + rec)
        if prec is not None and rec is not None
        else None
    )
    acc = safe(tp + tn, tp + tn + fp + fn)
    bacc = (
        (rec + spec) / 2.0
        if rec is not None and spec is not None
        else None
    )
    n = len(rows)
    unc = sum(1 for r in rows if r.get("mapped_semantic_decision") == "Uncertain")
    covered = tp + fp + tn + fn
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1": f1,
        "Accuracy": acc,
        "BalancedAccuracy": bacc,
        "Uncertain": unc,
        "Uncertain_rate": unc / n if n else None,
        "Coverage": covered / n if n else None,
        "excluded_uncertain_or_parse": excluded,
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
        raise SystemExit(f"Incomplete results ({len(missing)}). Examples: {missing[:5]}")

    table_rows = []
    pairs = {
        "S1_vs_S2": ("S1_semantic", "S2_neutral"),
        "S2_vs_S3": ("S2_neutral", "S3_token_permutation"),
        "S2_vs_S4": ("S2_neutral", "S4_order_permutation"),
        "S1_vs_S3": ("S1_semantic", "S3_token_permutation"),
        "S1_vs_S4": ("S1_semantic", "S4_order_permutation"),
    }
    changed: dict[str, list[str]] = {k: [] for k in pairs}

    for sid in sample_ids:
        mapped = {
            c: by_cond[c][sid].get("mapped_semantic_decision", "")
            for c in CONDITION_SPECS
        }
        row = {
            "sample_id": sid,
            "GT": gt_by_id[sid],
            "S1_semantic": mapped["S1_semantic"],
            "S2_mapped": mapped["S2_neutral"],
            "S3_mapped": mapped["S3_token_permutation"],
            "S4_mapped": mapped["S4_order_permutation"],
        }
        for pair_name, (ca, cb) in pairs.items():
            a = mapped[ca]
            b = mapped[cb]
            flag = agreement(a, b)
            row[pair_name] = flag
            if flag is False:
                changed[pair_name].append(sid)
        for c in CONDITION_SPECS:
            row[f"{c}_raw"] = by_cond[c][sid].get("raw_decision_token", "")
            row[f"{c}_parse_ok"] = by_cond[c][sid].get("parse_success", False)
            row[f"{c}_inference_error"] = bool(by_cond[c][sid].get("inference_error"))
        table_rows.append(row)

    table_path = out_root / "comparison_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    def pair_stats(flag_key: str) -> dict:
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
        inf_err = sum(1 for r in rows if r.get("inference_error"))
        parse_ok = sum(1 for r in rows if r.get("parse_success"))
        parse_fail = sum(
            1
            for r in rows
            if (not r.get("parse_success")) and (not r.get("inference_error"))
        )
        dist = Counter(
            (r.get("mapped_semantic_decision") or "PARSE_FAIL") for r in rows
        )
        condition_summaries[cond] = {
            "N": len(rows),
            "inference_errors": inf_err,
            "parse_success": parse_ok,
            "parse_failures": parse_fail,
            "Reliable": dist.get("Reliable", 0),
            "Uncertain": dist.get("Uncertain", 0),
            "Unreliable": dist.get("Unreliable", 0),
            "PARSE_FAIL": dist.get("PARSE_FAIL", 0),
            "metrics": selective_metrics(rows),
        }

    any_change = sorted(
        set().union(*[set(v) for v in changed.values()])
    )

    rates = [
        pair_stats("S1_vs_S2")["agreement_rate"],
        pair_stats("S2_vs_S3")["agreement_rate"],
        pair_stats("S2_vs_S4")["agreement_rate"],
    ]
    rates_f = [r for r in rates if r is not None]
    min_agree = min(rates_f) if rates_f else 0.0
    collapse_flags = []
    for cond, summary in condition_summaries.items():
        mapped_n = summary["Reliable"] + summary["Uncertain"] + summary["Unreliable"]
        if mapped_n > 0:
            max_share = max(
                summary["Reliable"], summary["Uncertain"], summary["Unreliable"]
            ) / mapped_n
            if max_share >= 0.95 and summary["parse_failures"] == 0:
                collapse_flags.append(cond)

    # Same interpretation standard as evaluate_qwen_label_robustness_probe.py
    if min_agree >= 0.9 and not collapse_flags:
        interpretation = "ROBUST"
    elif min_agree >= 0.6 and not (len(collapse_flags) >= 2):
        interpretation = "MODERATELY_SENSITIVE"
    else:
        interpretation = "SEVERELY_SENSITIVE"

    report = {
        "n_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "condition_summaries": condition_summaries,
        "semantic_agreement": {k: pair_stats(k) for k in pairs},
        "changed_sample_ids": {**changed, "any_pair": any_change},
        "n_samples_any_semantic_change": len(any_change),
        "robustness_interpretation": interpretation,
        "interpretation_rule": {
            "primary_pairs_for_min_agree": ["S1_vs_S2", "S2_vs_S3", "S2_vs_S4"],
            "ROBUST": "min_agree>=0.9 and no condition collapse (>=95% one class)",
            "MODERATELY_SENSITIVE": "min_agree>=0.6 and <2 collapsed conditions",
            "SEVERELY_SENSITIVE": "otherwise",
            "same_as": "scripts/diagnostics/evaluate_qwen_label_robustness_probe.py",
        },
        "comparison_table": str(table_path),
    }
    report_path = out_root / "EVALUATION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "===== GLM-4.6V-Flash label-robustness evaluation =====",
        f"samples: {len(sample_ids)}",
        f"interpretation: {interpretation}",
        "",
        "--- Per-condition (mapped semantic) ---",
    ]
    for cond, s in condition_summaries.items():
        m = s["metrics"]
        lines.append(
            f"{cond}: N={s['N']} inf_err={s['inference_errors']} "
            f"parse_fail={s['parse_failures']} "
            f"R={s['Reliable']} U={s['Uncertain']} Ur={s['Unreliable']}"
        )
        lines.append(
            f"  TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']} "
            f"P={m['Precision']} R={m['Recall']} Spec={m['Specificity']} "
            f"F1={m['F1']} Acc={m['Accuracy']} BA={m['BalancedAccuracy']} "
            f"Urate={m['Uncertain_rate']} Cov={m['Coverage']}"
        )
    lines.append("")
    lines.append("--- Semantic agreement ---")
    for k, v in report["semantic_agreement"].items():
        lines.append(
            f"{k}: {v['n_agree']}/{v['n_comparable']} "
            f"({v['agreement_rate']}) disagree={v['n_disagree']}"
        )
    lines.append("")
    lines.append(f"any change ({len(any_change)}): {any_change}")
    text = "\n".join(lines) + "\n"
    (out_root / "EVALUATION_SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {report_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
