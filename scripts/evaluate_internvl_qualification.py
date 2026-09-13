#!/usr/bin/env python3
"""Evaluate InternVL qualification Stage 0 / Stage 1 results (CPU-only)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.paths import PROJECT_ROOT


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("sample_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "sample_id": data.get("sample_id", path.stem),
                "decision": data.get("decision", ""),
                "raw_response": data.get("raw_response", ""),
                "parse_error": data.get("parse_error", ""),
                "inference_error": data.get("inference_error", ""),
                "status": "inference_error"
                if data.get("inference_error")
                else ("parse_error" if data.get("parse_error") else "ok"),
            }
        )
    return pd.DataFrame(rows)


def binary_from_decision(decision: str) -> str | None:
    """Frozen official binary: Reliable=pos, Unreliable=neg, Uncertain excluded."""
    if decision == "Reliable":
        return "positive"
    if decision == "Unreliable":
        return "negative"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", type=str, default="")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    manifest = pd.read_csv(args.manifest)
    merged = manifest.merge(results, on="sample_id", how="left")

    n = len(manifest)
    n_out = results["sample_id"].nunique() if not results.empty else 0
    missing = sorted(set(manifest["sample_id"]) - set(results["sample_id"] if not results.empty else []))
    decisions = Counter(results["decision"].fillna("") if not results.empty else [])
    statuses = Counter(results["status"] if not results.empty else [])
    unique_raw = results["raw_response"].nunique() if not results.empty else 0

    print(f"=== InternVL qualification eval ({args.stage or args.results_dir}) ===")
    print(f"expected={n} outputs={n_out} missing={len(missing)}")
    if missing:
        print("missing_ids:", missing[:20])
    print("status:", dict(statuses))
    print("decisions:", dict(decisions))
    print("unique_raw_texts:", unique_raw)
    parse_fail = int(statuses.get("parse_error", 0))
    infer_fail = int(statuses.get("inference_error", 0))
    parse_success_rate = (n_out - parse_fail - infer_fail) / n if n else 0.0
    print(f"parser_success_rate_over_expected: {parse_success_rate:.4f}")

    # Per-sample GT comparison
    if not results.empty:
        view = merged[["sample_id", "gt_label", "decision", "status"]].copy()
        if "selection_role" in merged.columns:
            view["selection_role"] = merged["selection_role"]
        print("\nper_sample:")
        print(view.to_string(index=False))

    # Binary metrics on Reliable/Unreliable only
    scored = merged[merged["decision"].isin(["Reliable", "Unreliable"])].copy()
    scored["pred"] = scored["decision"].map(binary_from_decision)
    scored["gt"] = scored["gt_label"]
    tp = int(((scored.pred == "positive") & (scored.gt == "positive")).sum())
    fp = int(((scored.pred == "positive") & (scored.gt == "negative")).sum())
    tn = int(((scored.pred == "negative") & (scored.gt == "negative")).sum())
    fn = int(((scored.pred == "negative") & (scored.gt == "positive")).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall)
        else float("nan")
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    bal_acc = (
        ((recall if recall == recall else 0.0) + (specificity if specificity == specificity else 0.0))
        / 2.0
    )

    print("\nconfusion (Reliable=pos, Unreliable=neg; Uncertain excluded):")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn} scored_n={len(scored)}")
    print(
        f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} "
        f"Accuracy={accuracy:.4f} Specificity={specificity:.4f} BalancedAccuracy={bal_acc:.4f}"
    )

    # Diagnostic rejection rate: Uncertain OR Unreliable as flagging
    neg = merged[merged["gt_label"] == "negative"]
    flagged = neg["decision"].isin(["Uncertain", "Unreliable"]).sum()
    print(
        f"\ndiagnostic_negative_detection_rate "
        f"(Uncertain|Unreliable)/GT-: {flagged}/{len(neg)} = "
        f"{(flagged / len(neg) if len(neg) else float('nan')):.4f}"
    )

    # Stage-1 style qualification gates (report only)
    one_class_share = max(decisions.values()) / n if decisions and n else 0.0
    print(f"\nmax_single_class_share_over_expected: {one_class_share:.4f}")


if __name__ == "__main__":
    main()
