#!/usr/bin/env python3
"""Analysis 8 — provenance audit and use of the IoU-threshold sensitivity tables.

outputs/semantic_gt_evaluation/ is marked SUPERSEDED for its semantic tables; its
iou_sensitivity/ geometry tables are stated to be semantic-independent. Before using them:
  1. stored 0.50 column == current Protocol v2 matched_gt (from the audited eval CSVs);
  2. stored summary counts == counts recomputed from the stored per-detection table;
  3. no v2-unmatched detection with best IoU < t is matched at t (geometric invariant);
  4. optional independent in-memory recomputation with the frozen Protocol v2 matcher
     (scripts/analysis/iou_sensitivity_analysis.py::compute_matches, which calls
     evaluate_verification_against_groundtruth.compute_greedy_matches_for_index).
     Nothing is written under outputs/.
Then: matched/unmatched counts vs threshold, semantic composition (official review) of
detections whose status flips, and descriptive alignment metrics per threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def col(t: float) -> str:
    return f"matched_iou_{t:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-recompute", action="store_true", help="skip the in-memory recomputation")
    args = ap.parse_args()
    C.ensure_dirs()
    out = C.OUT / "iou_sensitivity"
    out.mkdir(parents=True, exist_ok=True)

    df = C.load_samples()
    ref = df.drop_duplicates("sample_id").set_index("sample_id")[["matched_gt", "max_iou"]].sort_index()
    per = pd.read_csv(C.IOU_SENS_DIR / "iou_sensitivity_per_detection.csv", dtype={"sample_id": str}).set_index("sample_id").sort_index()
    summ = pd.read_csv(C.IOU_SENS_DIR / "iou_sensitivity_summary.csv")
    info = json.loads((C.IOU_SENS_DIR / "iou_sensitivity_info.json").read_text())

    checks = {}
    checks["same_sample_ids_as_cohort"] = bool(per.index.equals(ref.index))
    checks["stored_0.50_equals_v2_matched_gt"] = bool((per[col(0.5)] == ref["matched_gt"]).all())
    checks["stored_v2_label_column_consistent"] = bool(((per["original_protocol_v2_gt"] == "positive") == ref["matched_gt"]).all())
    checks["stored_max_iou_equals_v2"] = bool(np.allclose(per["max_iou"], ref["max_iou"], atol=1e-4))
    recount = all(int(per[col(r.iou_threshold)].sum()) == r.matched for r in summ.itertuples())
    checks["summary_counts_match_per_detection"] = bool(recount)
    um = ~ref["matched_gt"]
    checks["no_unmatched_with_best_iou_below_t_becomes_matched"] = bool(all(
        not (per[col(t)] & um & (ref["max_iou"] < t)).any() for t in THRESHOLDS if t < 0.5))
    checks["matched_counts_monotone_in_threshold"] = bool(np.all(np.diff([per[col(t)].sum() for t in THRESHOLDS]) <= 0))
    checks["info_semantic_labelled_rows_used"] = info.get("semantic_labelled_rows_used")
    checks["stored_files_mtime"] = {p.name: pd.Timestamp(p.stat().st_mtime, unit="s").isoformat()
                                    for p in sorted(C.IOU_SENS_DIR.iterdir())}
    checks["stored_files_sha256"] = {p.name: C.sha256_file(p) for p in sorted(C.IOU_SENS_DIR.iterdir())}

    if not args.no_recompute:
        sys.path.insert(0, str(C.REPO))
        sys.path.insert(0, str(C.REPO / "scripts"))
        sys.path.insert(0, str(C.REPO / "scripts" / "analysis"))
        import iou_sensitivity_analysis as isa  # noqa: E402

        rec = isa.compute_matches(list(THRESHOLDS)).set_index("sample_id").sort_index()
        checks["independent_recompute_equals_stored_all_thresholds"] = bool(
            all((rec[col(t)] == per[col(t)]).all() for t in THRESHOLDS))
    else:
        checks["independent_recompute_equals_stored_all_thresholds"] = "skipped"

    valid = all(v is True for k, v in checks.items()
                if k not in ("info_semantic_labelled_rows_used", "stored_files_mtime", "stored_files_sha256")
                and v != "skipped")
    checks["usable_under_protocol_v2"] = bool(valid)
    (out / "iou_sensitivity_provenance.json").write_text(json.dumps(checks, indent=2))
    if not valid:
        print(json.dumps(checks, indent=2))
        sys.exit("IoU sensitivity tables failed provenance checks; not used.")

    # counts vs threshold + semantic composition of flips
    rev = pd.read_csv(C.HUMAN_REVIEW_CSV, dtype={"sample_id": str}).set_index("sample_id")["semantic_label"]
    rows = []
    for t in THRESHOLDS:
        m = per[col(t)]
        gained = m & ~ref["matched_gt"]
        lost = ~m & ref["matched_gt"]
        g_sem = rev.reindex(gained[gained].index).value_counts()
        rows.append({"iou_threshold": t, "protocol_v2_canonical": t == 0.5, "matched": int(m.sum()),
                     "unmatched": int((~m).sum()), "matched_share": m.mean(),
                     "v2_unmatched_now_matched": int(gained.sum()),
                     "of_which_semantic_palm": int(g_sem.get("palm", 0)),
                     "of_which_semantic_ambiguous": int(g_sem.get("ambiguous", 0)),
                     "of_which_semantic_non_palm": int(g_sem.get("non_palm", 0)),
                     "v2_matched_now_unmatched": int(lost.sum()),
                     "v2_matched_now_unmatched_semantic_status": "not reviewed" if lost.any() else ""})
    counts = pd.DataFrame(rows)
    counts.to_csv(out / "iou_threshold_counts.csv", index=False)

    # descriptive alignment metrics under each threshold (Uncertain still excluded)
    am = []
    d = df.join(per[[col(t) for t in THRESHOLDS]], on="sample_id")
    for (m, c), g in d.groupby(["display", "condition"], sort=False):
        for t in THRESHOLDS:
            mm = g[col(t)].to_numpy()
            dec = g["decision"].to_numpy()
            cnt = {f"{C.LABEL_SHORT[l]}_{s}": int(((dec == l) & (mm if s == "m" else ~mm)).sum())
                   for l in C.LABELS for s in ("m", "u")}
            met = C.metrics_from_counts(cnt)
            am.append({"model": m, "condition": c, "iou_threshold": t,
                       **{k: met[k] for k in ("accuracy", "sensitivity", "specificity", "balanced_accuracy", "f1")}})
    am = pd.DataFrame(am)
    am.to_csv(out / "alignment_metrics_by_iou_threshold.csv", index=False)

    t = counts.copy()
    tex = pd.DataFrame({"IoU thr.": t["iou_threshold"].map(lambda x: f"{x:.2f}" + (" (v2)" if x == 0.5 else "")),
                        "Matched": t["matched"].map(C.fmt), "Unmatched": t["unmatched"].map(C.fmt),
                        "v2-UM $\\to$ M": t["v2_unmatched_now_matched"].map(C.fmt),
                        "\\quad palm / amb.": [f"{a} / {b}" for a, b in zip(t["of_which_semantic_palm"], t["of_which_semantic_ambiguous"])],
                        "v2-M $\\to$ UM": t["v2_matched_now_unmatched"].map(C.fmt)})
    C.write_latex_table(
        C.TABLES / "tab_iou_sensitivity.tex", tex,
        caption=r"IoU-threshold sensitivity of LabelMe matching (same greedy one-to-one matcher; only the "
                r"threshold varies). Protocol v2 uses 0.50. Semantic labels exist only for v2-unmatched detections; "
                r"detections that become unmatched at stricter thresholds were never semantically reviewed.",
        label="tab:iou_sensitivity", colspec="lrrrrr", escape_cells=False)
    print(json.dumps({k: v for k, v in checks.items() if k not in ("stored_files_sha256",)}, indent=2))
    print(counts.to_string(index=False))
    print(am[am["condition"] == "A2"].pivot_table(index="model", columns="iou_threshold", values="specificity").round(3))


if __name__ == "__main__":
    main()
