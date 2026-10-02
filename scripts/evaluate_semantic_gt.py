#!/usr/bin/env python3
"""
SUPERSEDED / NOT FOR CURRENT RESULTS: predates the official semantic review workflow and does
not consume outputs/semantic_gt_review/human_review.csv. Do not use its outputs for reporting;
see docs/SEMANTIC_VALIDITY_AUDIT.md.

Purpose:
    Re-evaluate stored VLM verification decisions against semantic GT (human-reviewed
    object presence) and compare with Evaluation Protocol v2 (annotation alignment,
    IoU >= 0.5). No inference; Protocol v2 outputs are read, never written.

    Decision policy is identical to Protocol v2 (scripts/compute_verification_metrics.py):
    Reliable = positive prediction, Unreliable = negative prediction, Uncertain excluded
    from TP/TN/FP/FN and reported separately. Semantic GT: palm = positive,
    non_palm = negative, ambiguous and unreviewed rows excluded from binary metrics.
    Unreviewed rows are never treated as negative.

Input:
    - outputs/semantic_gt_evaluation/reference/protocol_v2_detection_reference.csv
    - outputs/semantic_gt_evaluation/review/semantic_review_manifest.csv
    - outputs/verification/<model>/<experiment>/<A*>/sample_*.json         (raw decisions)
    - outputs/evaluation_protocol_v2/<model>/<experiment>/<A*>/A*_evaluation.csv, A*_metrics.json

Output (default outputs/semantic_gt_evaluation/):
    - semantic_gt_table.csv                    one row per canonical detection, with provenance
    - semantic_metrics.csv                     per run, semantic GT
    - protocol_v2_vs_semantic.csv              per run, side-by-side (long format)
    - gt_disagreement_summary.csv / .json      the 638 Protocol v2 GT- detections
    - negative_pool_decisions_by_semantic.csv  per run R/U/Ur within the 638, by semantic category
    - EVALUATION_INFO.json, README.md

Example:
    python scripts/evaluate_semantic_gt.py
    python scripts/evaluate_semantic_gt.py --require-complete
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compute_verification_metrics as metrics_mod  # noqa: E402
import evaluate_verification_against_groundtruth as eval_mod  # noqa: E402
from src.evaluation.semantic_gt import (  # noqa: E402
    AMBIGUOUS,
    DETECTION_KEY,
    DUPLICATE_DETECTION,
    LOCALIZATION_MISMATCH,
    MISSING_ANNOTATION,
    NON_PALM,
    OTHER,
    PALM,
    PROTOCOL_V2_REFERENCE_CSV,
    SEMANTIC_GT_ROOT,
    SEMANTIC_REVIEW_MANIFEST_CSV,
    UNREVIEWED,
    V2_NEGATIVE,
    V2_POSITIVE,
    SemanticGtError,
    build_semantic_gt_table,
    load_reference,
    negative_pool_category,
    read_str_csv,
    semantic_binary_mask,
    semantic_status,
)
from src.paths import (  # noqa: E402
    EVALUATION_PROTOCOL_V1_ROOT,
    EVALUATION_PROTOCOL_V2_ROOT,
    OUTPUTS_DIR,
)

VERIFICATION_ROOT = OUTPUTS_DIR / "verification"
PROTECTED_ROOTS = (EVALUATION_PROTOCOL_V1_ROOT, EVALUATION_PROTOCOL_V2_ROOT, VERIFICATION_ROOT)
FULL_N = 5747

# Keys compared between stored Protocol v2 metrics and the in-memory recomputation.
V2_REPRODUCIBILITY_KEYS = [
    "true_positive", "false_positive", "false_negative", "true_negative",
    "precision", "recall", "f1", "accuracy", "specificity", "balanced_accuracy",
    "reliable_count", "uncertain_count", "unreliable_count", "evaluated_samples",
    "ground_truth_positive", "ground_truth_negative",
]

COMPARISON_COLUMNS = [
    "model", "experiment", "ablation", "full_5747_run",
    "gt_definition", "scope", "semantic_gt_status",
    "n_scope_rows", "n_with_decision", "n_binary_evaluated",
    "n_gt_positive", "n_gt_negative",
    "tp", "tn", "fp", "fn",
    "accuracy", "precision", "sensitivity", "specificity", "f1", "balanced_accuracy",
    "reliable", "uncertain", "unreliable",
    "source",
]
SEMANTIC_METRICS_COLUMNS = [
    "model", "experiment", "ablation", "full_5747_run", "semantic_gt_status", "scope",
    "n_total", "n_semantic_palm", "n_semantic_palm_inherited", "n_semantic_palm_reviewed",
    "n_semantic_non_palm", "n_ambiguous", "n_unreviewed", "n_negative_pool_reviewed",
    "n_with_decision", "n_binary_evaluated",
    "tp", "tn", "fp", "fn",
    "accuracy", "precision", "sensitivity", "specificity", "f1", "balanced_accuracy",
    "reliable", "uncertain", "unreliable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=SEMANTIC_REVIEW_MANIFEST_CSV)
    parser.add_argument("--reference", type=Path, default=PROTOCOL_V2_REFERENCE_CSV)
    parser.add_argument("--v2-root", type=Path, default=EVALUATION_PROTOCOL_V2_ROOT)
    parser.add_argument("--verification-root", type=Path, default=VERIFICATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SEMANTIC_GT_ROOT)
    parser.add_argument("--require-complete", action="store_true",
                        help="Exit with an error unless all 638 negative-pool rows are labelled")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _undefined_to_nan(metrics: dict[str, Any]) -> dict[str, Any]:
    """compute_metrics reports 0.0 for 0/0; semantic outputs report such ratios as undefined."""
    tp, fp, fn, tn = (metrics[k] for k in ("true_positive", "false_positive",
                                            "false_negative", "true_negative"))
    out = dict(metrics)
    if tp + fp == 0:
        out["precision"] = math.nan
    if tp + fn == 0:
        out["recall"] = math.nan
    if tn + fp == 0:
        out["specificity"] = math.nan
    if tp + tn + fp + fn == 0:
        out["accuracy"] = math.nan
    if math.isnan(out["precision"]) or math.isnan(out["recall"]):
        out["f1"] = math.nan
    if math.isnan(out["recall"]) or math.isnan(out["specificity"]):
        out["balanced_accuracy"] = math.nan
    return out


def binary_metrics(df: pd.DataFrame, positive: pd.Series, ablation: str) -> dict[str, Any]:
    """
    Protocol v2 metric arithmetic (compute_verification_metrics.compute_metrics) on rows
    of ``df`` with GT polarity ``positive`` (boolean Series aligned with ``df``).
    """
    scored = df.assign(matched_gt=positive.to_numpy())
    return _undefined_to_nan(metrics_mod.compute_metrics(scored, ablation))


def semantic_binary_metrics(df: pd.DataFrame, ablation: str) -> dict[str, Any]:
    """
    Semantic-GT metrics. ``df`` needs verification_label, semantic_gt, max_iou and
    yolo_confidence. Only palm / non_palm rows are scored; ambiguous and unreviewed
    rows are excluded (never counted as negative).
    """
    mask = semantic_binary_mask(df)
    subset = df[mask]
    if (subset["semantic_gt"] == UNREVIEWED).any():
        raise SemanticGtError("unreviewed rows reached binary scoring")
    return binary_metrics(subset, subset["semantic_gt"] == PALM, ablation)


def metric_row(metrics: dict[str, Any], labels: pd.Series) -> dict[str, Any]:
    normalized = labels.map(metrics_mod.normalize_verification_label)
    return {
        "n_with_decision": int((normalized != "").sum()),
        "n_binary_evaluated": metrics["true_positive"] + metrics["true_negative"]
        + metrics["false_positive"] + metrics["false_negative"],
        "tp": metrics["true_positive"], "tn": metrics["true_negative"],
        "fp": metrics["false_positive"], "fn": metrics["false_negative"],
        "accuracy": metrics["accuracy"], "precision": metrics["precision"],
        "sensitivity": metrics["recall"], "specificity": metrics["specificity"],
        "f1": metrics["f1"], "balanced_accuracy": metrics["balanced_accuracy"],
        "reliable": metrics["reliable_count"], "uncertain": metrics["uncertain_count"],
        "unreliable": metrics["unreliable_count"],
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def display_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(PROJECT_ROOT)) if resolved.is_relative_to(PROJECT_ROOT) else str(path)


def discover_runs(v2_root: Path) -> list[Path]:
    return sorted(
        path.parent for path in v2_root.glob("*/*/A*/A*_evaluation.csv")
        if path.name == f"{path.parent.name}_evaluation.csv"
    )


def load_run(run_dir: Path, v2_root: Path, verification_root: Path,
             reference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one run's Protocol v2 evaluation CSV and verify it against raw predictions."""
    code = run_dir.name
    rel = run_dir.relative_to(v2_root)
    v2_df = pd.read_csv(run_dir / f"{code}_evaluation.csv")
    with (run_dir / f"{code}_metrics.json").open(encoding="utf-8") as file:
        stored_metrics = json.load(file)

    tag = str(rel)
    if list(v2_df["sample_id"]) != list(reference[DETECTION_KEY]):
        raise SemanticGtError(f"{tag}: sample_id order differs from reference")
    expected_gt = reference["original_protocol_v2_gt"].eq(V2_POSITIVE)
    if not v2_df["matched_gt"].map(metrics_mod.normalize_matched_gt).equals(expected_gt):
        raise SemanticGtError(f"{tag}: matched_gt differs from reference")

    pred_dir = verification_root / rel
    if not pred_dir.is_dir():
        raise SemanticGtError(f"{tag}: prediction directory missing: {pred_dir}")
    raw = eval_mod.load_verification_labels(pred_dir)
    raw_labels = v2_df["sample_id"].map(lambda s: raw.get(s, "")).map(
        metrics_mod.normalize_verification_label
    )
    csv_labels = v2_df["verification_label"].map(metrics_mod.normalize_verification_label)
    if not raw_labels.equals(csv_labels):
        raise SemanticGtError(f"{tag}: raw prediction decisions differ from v2 evaluation CSV")

    recomputed = metrics_mod.compute_metrics(v2_df, code)
    diffs = [k for k in V2_REPRODUCIBILITY_KEYS if stored_metrics.get(k) != recomputed[k]]
    if diffs:
        raise SemanticGtError(f"{tag}: stored Protocol v2 metrics not reproducible: {diffs}")
    return v2_df, stored_metrics


def evaluate_run(run_dir: Path, args: argparse.Namespace, reference: pd.DataFrame,
                 table: pd.DataFrame, status: dict[str, Any]) -> dict[str, Any]:
    model, experiment, code = run_dir.relative_to(args.v2_root).parts
    v2_df, stored = load_run(run_dir, args.v2_root, args.verification_root, reference)
    df = v2_df.merge(
        table[[DETECTION_KEY, "semantic_gt", "review_reason", "label_provenance",
               "original_protocol_v2_gt"]],
        on=DETECTION_KEY, how="left", validate="one_to_one",
    )
    if len(df) != len(v2_df) or df["label_provenance"].isna().any():
        raise SemanticGtError(f"{run_dir}: semantic join failed")
    full = stored["evaluated_samples"] == FULL_N
    base = {"model": model, "experiment": experiment, "ablation": code, "full_5747_run": full}
    complete = bool(status["semantic_gt_complete"])
    semantic_scope = "all_detections_excl_ambiguous" if complete else (
        f"reviewed_subset (inherited positives + {status['n_negative_pool_reviewed']}/"
        f"{status['n_negative_pool']} reviewed negative pool, excl. ambiguous)"
    )

    comparison: list[dict[str, Any]] = []
    v2_positive = df["original_protocol_v2_gt"] == V2_POSITIVE
    comparison.append({
        **base, "gt_definition": "protocol_v2_annotation_alignment", "scope": "all_5747",
        "semantic_gt_status": "n/a", "n_scope_rows": len(df),
        "n_gt_positive": int(v2_positive.sum()), "n_gt_negative": int((~v2_positive).sum()),
        **metric_row(stored, df["verification_label"]),
        "source": f"stored {display_path(run_dir / f'{code}_metrics.json')}",
    })
    if not complete:
        comparison.append({
            **base, "gt_definition": "semantic_object_presence", "scope": "all_5747",
            "semantic_gt_status": "semantic GT incomplete", "n_scope_rows": len(df),
            "source": "not computed: semantic review incomplete",
        })

    mask = semantic_binary_mask(df)
    subset = df[mask]
    v2_subset = binary_metrics(subset, subset["original_protocol_v2_gt"] == V2_POSITIVE, code)
    sem_subset = semantic_binary_metrics(df, code)
    subset_counts = {
        "n_scope_rows": int(mask.sum()),
    }
    comparison.append({
        **base, "gt_definition": "protocol_v2_annotation_alignment", "scope": semantic_scope,
        "semantic_gt_status": status["semantic_gt_status"], **subset_counts,
        "n_gt_positive": int((subset["original_protocol_v2_gt"] == V2_POSITIVE).sum()),
        "n_gt_negative": int((subset["original_protocol_v2_gt"] == V2_NEGATIVE).sum()),
        **metric_row(v2_subset, subset["verification_label"]),
        "source": "recomputed on the semantic-evaluable rows (same denominator as semantic row)",
    })
    comparison.append({
        **base, "gt_definition": "semantic_object_presence", "scope": semantic_scope,
        "semantic_gt_status": status["semantic_gt_status"], **subset_counts,
        "n_gt_positive": int((subset["semantic_gt"] == PALM).sum()),
        "n_gt_negative": int((subset["semantic_gt"] == NON_PALM).sum()),
        **metric_row(sem_subset, subset["verification_label"]),
        "source": "semantic review manifest + inherited Protocol v2 positives",
    })

    semantic_row = {
        **base, "semantic_gt_status": status["semantic_gt_status"], "scope": semantic_scope,
        **{k: status[k] for k in ("n_total", "n_semantic_palm", "n_semantic_palm_inherited",
                                  "n_semantic_palm_reviewed", "n_semantic_non_palm",
                                  "n_ambiguous", "n_unreviewed", "n_negative_pool_reviewed")},
        **metric_row(sem_subset, subset["verification_label"]),
    }

    pool = df[df["original_protocol_v2_gt"] == V2_NEGATIVE]
    categories = [negative_pool_category(l, r) for l, r in zip(pool["semantic_gt"], pool["review_reason"])]
    decisions = pool["verification_label"].map(metrics_mod.normalize_verification_label)
    pool_rows = []
    for category, group in pd.Series(decisions.to_numpy(), index=categories).groupby(level=0):
        pool_rows.append({
            **base, "semantic_category": category, "n": len(group),
            "reliable": int((group == metrics_mod.POSITIVE_LABEL).sum()),
            "uncertain": int((group == metrics_mod.UNCERTAIN_LABEL).sum()),
            "unreliable": int((group == metrics_mod.NEGATIVE_LABEL).sum()),
            "no_decision": int((group == "").sum()),
        })
    return {"comparison": comparison, "semantic": semantic_row, "pool": pool_rows}


# ---------------------------------------------------------------------------
# GT disagreement (Protocol v2 GT- pool)
# ---------------------------------------------------------------------------


def disagreement_analysis(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pool = table[table["original_protocol_v2_gt"] == V2_NEGATIVE]
    n_pool = len(pool)
    label, reason = pool["semantic_gt"], pool["review_reason"]
    categories = [
        ("A_true_semantic_negative", "semantic_gt = non_palm", label == NON_PALM),
        ("B_missing_annotation", "semantic_gt = palm, review_reason = missing_annotation",
         (label == PALM) & (reason == MISSING_ANNOTATION)),
        ("C_localization_mismatch", "semantic_gt = palm, review_reason = localization_mismatch",
         (label == PALM) & (reason == LOCALIZATION_MISMATCH)),
        ("palm_duplicate_detection", "semantic_gt = palm, review_reason = duplicate_detection",
         (label == PALM) & (reason == DUPLICATE_DETECTION)),
        ("palm_other", "semantic_gt = palm, review_reason = other",
         (label == PALM) & (reason == OTHER)),
        ("D_ambiguous", "semantic_gt = ambiguous", label == AMBIGUOUS),
        ("unreviewed", "semantic_gt blank", label == UNREVIEWED),
    ]
    rows = [{
        "category": name, "definition": definition, "count": int(mask.sum()),
        "pct_of_protocol_v2_negatives": round(100.0 * int(mask.sum()) / n_pool, 2) if n_pool else math.nan,
    } for name, definition, mask in categories]
    if sum(r["count"] for r in rows) != n_pool:
        raise SemanticGtError("disagreement categories do not partition the negative pool")

    n_palm = int((label == PALM).sum())
    n_unreviewed = int((label == UNREVIEWED).sum())
    n_reviewed = n_pool - n_unreviewed
    summary = {
        "protocol_v2_negatives": n_pool,
        "reviewed": n_reviewed,
        "unreviewed": n_unreviewed,
        "semantic_palm_among_v2_negatives": n_palm,
        "semantic_gt_complete": n_unreviewed == 0,
        "categories": {r["category"]: r["count"] for r in rows},
    }
    if n_unreviewed == 0:
        summary["semantic_palm_among_v2_negatives_fraction"] = round(n_palm / n_pool, 4) if n_pool else None
    else:
        summary["semantic_palm_among_v2_negatives_fraction"] = None
        summary["note"] = ("semantic GT incomplete: the fraction over all negatives is not "
                           "determined; bounds assume every unreviewed row is non_palm (lower) "
                           "or palm (upper)")
        summary["fraction_lower_bound"] = round(n_palm / n_pool, 4) if n_pool else None
        summary["fraction_upper_bound"] = round((n_palm + n_unreviewed) / n_pool, 4) if n_pool else None
        summary["fraction_among_reviewed"] = round(n_palm / n_reviewed, 4) if n_reviewed else None
    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def git_info() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    return {"base_commit": run("rev-parse", "HEAD") or None,
            "working_tree_dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_readme(path: Path, status: dict[str, Any], summary: dict[str, Any], n_runs: int) -> None:
    lines = [
        "# Semantic-GT evaluation outputs",
        "",
        "Generated by `scripts/evaluate_semantic_gt.py`. Definitions and policy: "
        "`docs/SEMANTIC_GT_EVALUATION.md`. Protocol v2 outputs under "
        "`outputs/evaluation_protocol_v2/` are read, never written. No VLM inference was run.",
        "",
        f"- Semantic GT status: **{status['semantic_gt_status']}** "
        f"({status['n_negative_pool_reviewed']}/{status['n_negative_pool']} Protocol v2 GT- rows reviewed)",
        f"- Runs evaluated: {n_runs}",
        f"- Semantic palm among Protocol v2 GT-: {summary['semantic_palm_among_v2_negatives']}"
        f" / {summary['protocol_v2_negatives']}",
        "",
        "| File | Content |",
        "|---|---|",
        "| `semantic_gt_table.csv` | One row per canonical detection: Protocol v2 GT, semantic GT, review reason, `label_provenance` |",
        "| `semantic_metrics.csv` | Semantic-GT metrics per run (palm = positive, non_palm = negative; ambiguous/unreviewed excluded) |",
        "| `protocol_v2_vs_semantic.csv` | Per run: stored Protocol v2 metrics (all 5,747), Protocol v2 recomputed on the semantic-evaluable rows, semantic metrics on the same rows |",
        "| `gt_disagreement_summary.csv/.json` | Composition of the 638 Protocol v2 GT- detections after review |",
        "| `negative_pool_decisions_by_semantic.csv` | Per run Reliable/Uncertain/Unreliable counts within the 638, by semantic category |",
        "| `iou_sensitivity/` | IoU-threshold sensitivity of the matching (`scripts/analysis/iou_sensitivity_analysis.py`) |",
        "| `review/semantic_review_manifest.csv` | Human review manifest (edit `semantic_gt`, `review_reason`, `reviewer_notes` only) |",
        "| `reference/protocol_v2_detection_reference.csv` | Canonical 5,747 detections with Protocol v2 GT |",
        "| `provenance/protected_outputs_baseline.json` | SHA-256 of protected outputs before this path was added |",
        "",
        "While the semantic GT is incomplete, no full-dataset semantic metrics are reported; "
        "semantic rows are computed on the reviewed subset only and labelled as such. "
        "Metrics with a zero denominator are left blank.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    for root in PROTECTED_ROOTS:
        if out == root.resolve() or out.is_relative_to(root.resolve()):
            sys.exit(f"ABORT: refusing to write into protected tree {root}")

    reference = load_reference(args.reference)
    if not args.manifest.is_file():
        sys.exit(f"Manifest not found: {args.manifest} (run scripts/build_semantic_review_manifest.py)")
    table = build_semantic_gt_table(reference, read_str_csv(args.manifest))
    status = semantic_status(table)
    print(json.dumps(status, indent=2))
    if args.require_complete and not status["semantic_gt_complete"]:
        sys.exit(f"ABORT: semantic GT incomplete ({status['n_unreviewed']} unreviewed rows)")

    runs = discover_runs(args.v2_root)
    if not runs:
        sys.exit(f"No Protocol v2 runs under {args.v2_root}")
    comparison, semantic_rows, pool_rows = [], [], []
    for run_dir in runs:
        result = evaluate_run(run_dir, args, reference, table, status)
        comparison.extend(result["comparison"])
        semantic_rows.append(result["semantic"])
        pool_rows.extend(result["pool"])

    disagreement_df, disagreement = disagreement_analysis(table)

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "semantic_gt_table.csv", index=False)
    pd.DataFrame(semantic_rows, columns=SEMANTIC_METRICS_COLUMNS).to_csv(
        out / "semantic_metrics.csv", index=False)
    pd.DataFrame(comparison, columns=COMPARISON_COLUMNS).to_csv(
        out / "protocol_v2_vs_semantic.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(out / "negative_pool_decisions_by_semantic.csv", index=False)
    disagreement_df.to_csv(out / "gt_disagreement_summary.csv", index=False)
    with (out / "gt_disagreement_summary.json").open("w", encoding="utf-8") as file:
        json.dump(disagreement, file, indent=2)
    with (out / "EVALUATION_INFO.json").open("w", encoding="utf-8") as file:
        json.dump({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **git_info(),
            "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
            "reference": str(args.reference), "reference_sha256": sha256_file(args.reference),
            "runs_evaluated": len(runs),
            "full_5747_runs": sum(r["full_5747_run"] for r in semantic_rows),
            **status,
            "decision_policy": "Reliable = positive, Unreliable = negative, Uncertain excluded "
                               "from binary metrics (unchanged from Protocol v2)",
            "semantic_policy": "palm = positive, non_palm = negative, ambiguous excluded, "
                               "unreviewed excluded (never negative); Protocol v2 GT+ inherit palm",
            "definition": "docs/SEMANTIC_GT_EVALUATION.md",
        }, file, indent=2)
    write_readme(out / "README.md", status, disagreement, len(runs))
    print(f"Evaluated {len(runs)} runs; wrote outputs to {out}")


if __name__ == "__main__":
    main()
