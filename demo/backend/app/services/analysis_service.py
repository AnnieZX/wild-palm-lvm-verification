"""Aggregate views built from the thesis analysis layer (paper/analysis/outputs/).

Nothing here recomputes a statistic that the analysis layer already reports; values are
re-shaped for the UI and carry their source file. Missing artifacts yield `available: False`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd

from app.data import loaders as L
from app.data import paths as P
from app.services import model_service


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None:
        return []
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def _src(*parts: str) -> str:
    return str(P.ANALYSIS_OUT.joinpath(*parts).relative_to(P.REPO))


def _display_to_key() -> Dict[str, str]:
    return {m["display"]: m["model_key"] for m in model_service.list_models()}


# ---------------------------------------------------------------------------
def summary() -> Dict[str, Any]:
    det = L.detections()
    models = model_service.list_models()
    audit = {}
    audit_path = P.ANALYSIS_OUT / "audit" / "audit_info.json"
    if audit_path.is_file():
        audit = L.read_json(audit_path)
    sem = L.detections()["semantic_label"].value_counts().to_dict()
    usable_cells = sum(1 for m in models for c in m["conditions"].values() if c["status"] in model_service.USABLE)
    return {
        "n_detections": int(len(det)),
        "n_images": int(det["image_name"].nunique()),
        "conditions": list(P.CONDITIONS),
        "yolo_confidence_min": float(det["confidence"].min()),
        "n_models_registered": len(models),
        "n_models_with_any_usable_condition": sum(1 for m in models if m["n_usable_conditions"] > 0),
        "n_models_all_five_usable": sum(1 for m in models if m["n_usable_conditions"] == 5),
        "n_usable_cells": usable_cells,
        "n_decisions": int(len(L.sample_table())),
        "labelme_matched": int((det["matched_gt"] == True).sum()),  # noqa: E712
        "labelme_unmatched": int((det["matched_gt"] == False).sum()),  # noqa: E712
        "semantic_review": {k: int(v) for k, v in sem.items()},
        "iou_threshold": 0.5,
        "audit_status_counts": audit.get("status_counts"),
        "sources": {
            "detections": str(P.INDEX_CSV.relative_to(P.REPO)),
            "decisions": str(P.SAMPLE_TABLE.relative_to(P.REPO)),
            "semantic_review": str(P.HUMAN_REVIEW_CSV.relative_to(P.REPO)),
            "audit": _src("audit", "audit_info.json"),
        },
    }


# ---------------------------------------------------------------------------
def condition_pairs() -> List[Dict[str, Any]]:
    reg = L.analysis_registry()
    groups = reg.get("PAIR_GROUPS", {})
    notes = reg.get("PAIR_GROUP_NOTES", {})
    out = []
    for group, pairs in groups.items():
        for a, b in pairs:
            out.append({"pair": f"{a}->{b}", "group": group, "note": notes.get(group)})
    return out


def context_shift(pair: str = "A1->A5") -> Dict[str, Any]:
    summary_df = L.analysis_csv("transition_summary.csv")
    matrices = L.analysis_csv("transition_matrices_long.csv")
    dist = L.analysis_csv("decision_distribution_long.csv")
    if summary_df is None:
        return {"available": False, "pair": pair}
    a, b = pair.split("->")
    keys = _display_to_key()
    rows = summary_df[summary_df["pair"] == pair]
    models = []
    for r in records(rows):
        name = r["model"]
        mat = []
        if matrices is not None:
            m = matrices[(matrices["model"] == name) & (matrices["from_condition"] == a)
                         & (matrices["to_condition"] == b)]
            mat = [{"from": x["from_label"], "to": x["to_label"], "count": int(x["count"]),
                    "share_of_n": float(x["share_of_n"])} for x in m.to_dict("records")]
        dists = {}
        if dist is not None:
            for cond in (a, b):
                d = dist[(dist["model"] == name) & (dist["condition"] == cond)]
                dists[cond] = {x["label"]: {"count": int(x["count"]), "rate": float(x["rate"])}
                               for x in d.to_dict("records")}
        r["into_U_rate"] = ((r.get("R->U") or 0) + (r.get("Ur->U") or 0)) / r["n"] if r.get("n") else None
        r["into_Ur_rate"] = ((r.get("R->Ur") or 0) + (r.get("U->Ur") or 0)) / r["n"] if r.get("n") else None
        r["into_R_rate"] = ((r.get("U->R") or 0) + (r.get("Ur->R") or 0)) / r["n"] if r.get("n") else None
        r["model_key"] = keys.get(name)
        r["matrix"] = mat
        r["distributions"] = dists
        models.append(r)
    return {
        "available": True,
        "pair": pair,
        "pairs": condition_pairs(),
        "models": models,
        "sources": [_src("transition_summary.csv"), _src("transition_matrices_long.csv"),
                    _src("decision_distribution_long.csv")],
    }


def decision_distribution() -> Dict[str, Any]:
    dist = L.analysis_csv("decision_distribution_long.csv")
    return {"available": dist is not None, "rows": records(dist),
            "source": _src("decision_distribution_long.csv")}


# ---------------------------------------------------------------------------
def scaling() -> Dict[str, Any]:
    ckpt = L.analysis_csv("size_ladder", "size_ladder_checkpoints.csv")
    order = L.analysis_csv("size_ladder", "size_ladder_ordering_description.csv")
    trans = L.analysis_csv("size_ladder", "size_ladder_transition_matrices.csv")
    usage = L.analysis_csv("label_usage", "label_usage_by_cell.csv")
    if ckpt is None:
        return {"available": False}
    ladders: Dict[str, Dict[str, Any]] = {}
    for r in records(ckpt):
        lad = ladders.setdefault(r["ladder"], {"ladder": r["ladder"], "checkpoints": [], "orderings": []})
        if trans is not None:
            t = trans[(trans["ladder"] == r["ladder"]) & (trans["model"] == r["model"])]
            r["matrix"] = [{"from": x["from_A1"], "to": x["to_A5"], "count": int(x["count"]),
                            "share": float(x["share_of_paired"])} for x in t.to_dict("records")]
        if usage is not None:
            u = usage[usage["model_key"] == r["model_key"]]
            r["label_usage"] = records(u[["condition", "condition_status", "n_valid", "share_R", "share_U",
                                          "share_Ur", "entropy_bits", "majority_label", "majority_share",
                                          "diag_single_label", "diag_majority_ge_0.99"]])
        lad["checkpoints"].append(r)
    for r in records(order):
        if r["ladder"] in ladders:
            ladders[r["ladder"]]["orderings"].append(r)
    return {
        "available": True,
        "ladders": list(ladders.values()),
        "sources": [_src("size_ladder", "size_ladder_checkpoints.csv"),
                    _src("size_ladder", "size_ladder_ordering_description.csv"),
                    _src("label_usage", "label_usage_by_cell.csv")],
    }


# ---------------------------------------------------------------------------
def difficulty(pair: str = "A1->A5") -> Dict[str, Any]:
    rates = L.analysis_csv("detection_difficulty", "event_rates_by_quintile.csv")
    contrasts = L.analysis_csv("detection_difficulty", "q1_vs_q5_contrasts.csv")
    hyp = L.analysis_csv("detection_difficulty", "hypothesis_summary.csv")
    info = L.analysis_csv("detection_difficulty", "property_info.csv")
    cross = L.analysis_csv("detection_difficulty", "confidence_x_area_median_split.csv")
    if rates is None:
        return {"available": False, "pair": pair}
    keys = _display_to_key()
    r = rates[rates["pair"] == pair]
    c = contrasts[contrasts["pair"] == pair] if contrasts is not None else None
    models: Dict[str, Dict[str, Any]] = {}
    for row in records(r):
        m = models.setdefault(row["model"], {"model": row["model"], "model_key": keys.get(row["model"]),
                                             "rates": [], "contrasts": []})
        m["rates"].append({k: row[k] for k in ("property", "quintile", "event", "n_at_risk", "n_event",
                                               "rate", "ci_low", "ci_high")})
    for row in records(c):
        m = models.setdefault(row["model"], {"model": row["model"], "model_key": keys.get(row["model"]),
                                             "rates": [], "contrasts": []})
        m["contrasts"].append({k: row.get(k) for k in ("property", "event", "n_paired", "rate_Q1", "rate_Q5",
                                                       "diff_Q1_minus_Q5", "ci_low", "ci_high", "verdict",
                                                       "single_label_both")})
    cross_rows = records(cross[cross["pair"] == pair]) if cross is not None else []
    return {
        "available": True,
        "pair": pair,
        "pairs": sorted(rates["pair"].unique().tolist()),
        "models": list(models.values()),
        "hypothesis_summary": records(hyp[hyp["pair"] == pair]) if hyp is not None else [],
        "property_info": records(info)[0] if info is not None and len(info) else None,
        "confidence_x_area": cross_rows,
        "sources": [_src("detection_difficulty", "event_rates_by_quintile.csv"),
                    _src("detection_difficulty", "q1_vs_q5_contrasts.csv"),
                    _src("detection_difficulty", "hypothesis_summary.csv")],
    }


# ---------------------------------------------------------------------------
def semantic_review() -> Dict[str, Any]:
    audit_path = P.ANALYSIS_OUT / "semantic" / "semantic_audit.json"
    audit = L.read_json(audit_path) if audit_path.is_file() else {}
    master = L.analysis_csv("master_results.csv")
    alignment = []
    if master is not None:
        cols = ["model_key", "model", "condition", "condition_status", "n_samples", "align_R_unmatched",
                "align_Ur_unmatched", "align_U_unmatched", "n_decided_unmatched", "align_specificity",
                "align_sensitivity", "reliable_rate", "uncertain_rate", "unreliable_rate"]
        alignment = records(master[[c for c in cols if c in master.columns]])
    unmatched = audit.get("unmatched_review", {})
    return {
        "available": bool(audit),
        "unmatched_review": {
            "n": unmatched.get("rows"),
            "label_counts": unmatched.get("label_counts"),
            "reviewers": unmatched.get("reviewers"),
            "review_complete": unmatched.get("review_complete"),
            "labelme_matched_reviewed": unmatched.get("labelme_matched_reviewed"),
            "first_timestamp": unmatched.get("first_timestamp"),
            "last_timestamp": unmatched.get("last_timestamp"),
        },
        "pilot": audit.get("pilot"),
        "provenance": audit.get("provenance"),
        "iou_status_by_semantic": records(L.analysis_csv("semantic", "iou_status_by_semantic.csv")),
        "unmatched_reason_by_semantic": records(L.analysis_csv("semantic", "unmatched_reason_by_semantic.csv")),
        "vlm_verdicts_by_semantic_label": records(L.analysis_csv("semantic", "vlm_verdicts_by_semantic_label.csv")),
        "alignment_cells_semantic_composition": records(
            L.analysis_csv("semantic", "alignment_cells_semantic_composition.csv")),
        "pilot_by_confidence_bin": records(L.analysis_csv("semantic", "pilot_by_confidence_bin.csv")),
        "subsets": records(L.analysis_csv("semantic", "semantic_subsets.csv")),
        "alignment_metrics": alignment,
        "sources": [_src("semantic", "semantic_audit.json"), str(P.HUMAN_REVIEW_CSV.relative_to(P.REPO)),
                    _src("semantic", "vlm_verdicts_by_semantic_label.csv"), _src("master_results.csv")],
    }


def curated_cases() -> Dict[str, Any]:
    df = L.analysis_csv("case_selection", "case_manifest.csv", dtype={"sample_id": str})
    if df is None:
        return {"available": False, "cases": []}
    cols = ["category", "rank_in_category", "sample_id", "matched_gt", "yolo_confidence", "semantic_label"]
    return {"available": True, "cases": records(df[[c for c in cols if c in df.columns]]),
            "source": _src("case_selection", "case_manifest.csv")}
