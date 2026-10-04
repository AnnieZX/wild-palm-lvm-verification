#!/usr/bin/env python3
"""Analysis 7 — human semantic review (RQ4), kept separate from IoU alignment.

Two populations, never pooled:
  (A) Official review of LabelMe-unmatched cohort detections
      outputs/semantic_gt_review/human_review.csv
  (B) Lower-confidence pilot (YOLO conf 0.10-0.50, outside the 5,747 cohort, no VLM
      predictions) outputs/semantic_gt_review/human_confidence_pilot.csv

The reviewer recorded no unmatched *reason*. A geometric reason is derived here from the
Protocol-v2 evaluation's max_iou, which for an unmatched detection is the best IoU with any
LabelMe palm box regardless of assignment (src/evaluation/gt_matching.py::best_gt_overlap):
  no_overlap               max_iou == 0
  partial_overlap          0 < max_iou < 0.5
  iou_ge_0.5_lost_greedy   max_iou >= 0.5 but the LabelMe box was assigned to another detection
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# SHA-256 recorded in docs/SEMANTIC_VALIDITY_AUDIT.md §G
EXPECTED_SHA = {
    "human_review.csv": "b2a0219df9b3310661752e2dcbcd5d9b5bb9501dcdc1d07cfeb5f7284f478485",
    "human_review.log.jsonl": "a1a4d62fe8e1ccfc20edb785eae7efd33e58604913bde0c398675866ad7a0691",
    "human_confidence_pilot.csv": "9a25cd38a1923e135b9ecef83b16397e8bf6254b1a9ba78948b77d01fdb97705",
    "semantic_pilot_reference.csv": "4e02f77ab4ea44b026267833314cba688cea7e204da3eaa0623d2e26321ca062",
}
SEM_LABELS = ("palm", "ambiguous", "non_palm")
BALANCED_100 = C.REPO / "outputs" / "diagnostics" / "model_qualification" / "balanced_A1_100" / "manifest.csv"


def reason(iou: float) -> str:
    if iou == 0:
        return "no_overlap"
    if iou < 0.5:
        return "partial_overlap"
    return "iou_ge_0.5_lost_greedy"


def main() -> None:
    C.ensure_dirs()
    sem_dir = C.OUT / "semantic"
    sem_dir.mkdir(parents=True, exist_ok=True)
    df = C.load_samples()

    provenance = {}
    for path in (C.HUMAN_REVIEW_CSV, C.HUMAN_REVIEW_LOG, C.PILOT_CSV, C.PILOT_REFERENCE_CSV):
        h = C.sha256_file(path)
        provenance[path.name] = {"path": str(path.relative_to(C.REPO)), "sha256": h,
                                 "matches_documented_sha256": h == EXPECTED_SHA[path.name]}

    # ------------------------------------------------------------ (A) unmatched review
    rev = pd.read_csv(C.HUMAN_REVIEW_CSV, dtype={"sample_id": str})
    log = [json.loads(l) for l in C.HUMAN_REVIEW_LOG.read_text().splitlines() if l.strip()]
    ref = df.drop_duplicates("sample_id")[["sample_id", "image_id", "matched_gt", "max_iou", "yolo_confidence"]]
    unmatched_ids = set(ref.loc[~ref["matched_gt"], "sample_id"])
    reviewed_ids = set(rev["sample_id"])
    audit = {
        "rows": len(rev), "unique_sample_ids": int(rev["sample_id"].nunique()),
        "missing_labels": int(rev["semantic_label"].isna().sum() + (rev["semantic_label"] == "").sum()),
        "labels_outside_vocabulary": int((~rev["semantic_label"].isin(SEM_LABELS)).sum()),
        "reviewers": sorted(rev["reviewer"].dropna().unique().tolist()),
        "first_timestamp": rev["review_timestamp"].min(), "last_timestamp": rev["review_timestamp"].max(),
        "log_events": len(log), "log_label_changes": sum(1 for e in log if e.get("previous")),
        "cohort_labelme_unmatched": len(unmatched_ids),
        "reviewed_equals_unmatched_set": reviewed_ids == unmatched_ids,
        "reviewed_not_unmatched": len(reviewed_ids - unmatched_ids),
        "unmatched_not_reviewed": len(unmatched_ids - reviewed_ids),
        "review_complete": bool(reviewed_ids == unmatched_ids and rev["semantic_label"].isin(SEM_LABELS).all()),
        "labelme_matched_reviewed": 0,
        "reason_field_recorded_by_reviewer": False,
    }
    j = rev[["sample_id", "semantic_label"]].merge(ref, on="sample_id", how="left", validate="one_to_one")
    j["unmatched_reason_geometric"] = j["max_iou"].map(reason)
    audit["label_counts"] = j["semantic_label"].value_counts().reindex(SEM_LABELS, fill_value=0).to_dict()
    j.to_csv(sem_dir / "unmatched_review_joined.csv", index=False)

    # IoU status x semantic status (matched detections were not reviewed)
    status = pd.DataFrame([
        {"iou_status": "LabelMe-matched", "n": int(ref["matched_gt"].sum()), "reviewed": 0,
         **{k: "not reviewed" for k in SEM_LABELS}},
        {"iou_status": "LabelMe-unmatched", "n": len(unmatched_ids), "reviewed": len(rev),
         **{k: int((j["semantic_label"] == k).sum()) for k in SEM_LABELS}},
    ])
    status.to_csv(sem_dir / "iou_status_by_semantic.csv", index=False)

    reasons = pd.crosstab(j["unmatched_reason_geometric"], j["semantic_label"]).reindex(
        index=["no_overlap", "partial_overlap", "iou_ge_0.5_lost_greedy"], columns=list(SEM_LABELS), fill_value=0)
    reasons["total"] = reasons.sum(axis=1)
    reasons.to_csv(sem_dir / "unmatched_reason_by_semantic.csv")

    # subsets used historically
    subsets = []
    b100 = pd.read_csv(BALANCED_100, dtype={"sample_id": str})
    b100_neg = set(b100.loc[b100["matched_gt"].astype(str).str.lower() == "false", "sample_id"])
    for name, ids in (("all_unmatched", unmatched_ids),
                      ("balanced_100_negatives", b100_neg),
                      ("A1_1000_slice_unmatched", {s for s in unmatched_ids if int(s.split("_")[1]) <= 1000})):
        sub = j[j["sample_id"].isin(ids)]
        subsets.append({"subset": name, "n": len(sub), **sub["semantic_label"].value_counts()
                        .reindex(SEM_LABELS, fill_value=0).to_dict()})
    pd.DataFrame(subsets).to_csv(sem_dir / "semantic_subsets.csv", index=False)

    # VLM verdicts within semantic categories (unmatched detections only)
    d = df[df["sample_id"].isin(unmatched_ids)].merge(j[["sample_id", "semantic_label", "unmatched_reason_geometric"]],
                                                      on="sample_id")
    rows = []
    for (m, c, s), g in d.groupby(["display", "condition", "semantic_label"], sort=False):
        n = len(g)
        rows.append({"model": m, "condition": c, "semantic_label": s, "n": n,
                     **{f"n_{C.LABEL_SHORT[l]}": int((g["decision"] == l).sum()) for l in C.LABELS},
                     **{f"rate_{C.LABEL_SHORT[l]}": (g["decision"] == l).mean() for l in C.LABELS}})
    vs = pd.DataFrame(rows)
    vs["model"] = pd.Categorical(vs["model"], C.MODEL_ORDER, ordered=True)
    vs = vs.sort_values(["model", "condition", "semantic_label"])
    vs.to_csv(sem_dir / "vlm_verdicts_by_semantic_label.csv", index=False)

    # Composition of alignment "TN" (Unreliable on unmatched) and "FP" (Reliable on unmatched)
    comp = []
    for (m, c), g in d.groupby(["display", "condition"], sort=False):
        for verdict, legacy in (("Unreliable", "TN_alignment"), ("Reliable", "FP_alignment")):
            gg = g[g["decision"] == verdict]
            comp.append({"model": m, "condition": c, "alignment_cell": legacy, "verdict": verdict, "n": len(gg),
                         **{f"n_{k}": int((gg["semantic_label"] == k).sum()) for k in SEM_LABELS}})
    pd.DataFrame(comp).to_csv(sem_dir / "alignment_cells_semantic_composition.csv", index=False)

    # ------------------------------------------------------------ (B) pilot (separate)
    pilot = pd.read_csv(C.PILOT_CSV)
    pref = pd.read_csv(C.PILOT_REFERENCE_CSV)
    pj = pilot[["pilot_id", "semantic_label"]].merge(
        pref[["pilot_id", "confidence_bin", "bin_detections", "yolo_confidence", "geo_max_iou_labelme",
              "geo_annotation_aligned_iou050_any"]], on="pilot_id", how="left", validate="one_to_one")
    pilot_audit = {"rows": len(pilot), "unique_ids": int(pilot["pilot_id"].nunique()),
                   "joined_to_reference": int(pj["confidence_bin"].notna().sum()),
                   "min_conf": float(pj["yolo_confidence"].min()), "max_conf": float(pj["yolo_confidence"].max()),
                   "inside_5747_cohort": False, "has_vlm_predictions": False,
                   "familiarization_snapshot_used": False}
    by_bin = pd.crosstab(pj["confidence_bin"], pj["semantic_label"]).reindex(columns=list(SEM_LABELS), fill_value=0)
    by_bin["n"] = by_bin.sum(axis=1)
    by_bin["bin_population"] = pj.groupby("confidence_bin")["bin_detections"].first()
    by_bin.to_csv(sem_dir / "pilot_by_confidence_bin.csv")
    pilot_align = pd.crosstab(pj["semantic_label"], pj["geo_annotation_aligned_iou050_any"]).reindex(
        index=list(SEM_LABELS), fill_value=0)
    pilot_align.columns = [f"labelme_iou050_any_{c}" for c in pilot_align.columns]
    pilot_align.to_csv(sem_dir / "pilot_semantic_by_labelme_overlap.csv")

    (sem_dir / "semantic_audit.json").write_text(json.dumps(
        {"provenance": provenance, "unmatched_review": audit, "pilot": pilot_audit}, indent=2, default=str))

    # ------------------------------------------------------------ LaTeX
    t = pd.DataFrame([
        {"Population": "LabelMe-matched (cohort)", "n": C.fmt(int(ref["matched_gt"].sum())), "Reviewed": "0",
         "palm": "--", "ambiguous": "--", "non\\_palm": "--"},
        *[{"Population": {"all_unmatched": "LabelMe-unmatched (cohort)",
                          "balanced_100_negatives": r"\quad of which balanced-100 ``negatives''",
                          "A1_1000_slice_unmatched": r"\quad of which A1@1{,}000 slice"}[r["subset"]],
           "n": C.fmt(r["n"]), "Reviewed": C.fmt(r["n"]), "palm": C.fmt(r["palm"]),
           "ambiguous": C.fmt(r["ambiguous"]), "non\\_palm": C.fmt(r["non_palm"])} for r in subsets],
    ])
    C.write_latex_table(
        C.TABLES / "tab_semantic_review.tex", t,
        caption=r"Human semantic review of the 5{,}747-detection cohort. Only LabelMe-unmatched detections were "
                r"reviewed (one reviewer, blind to LabelMe, confidence, IoU and VLM output). LabelMe-unmatched "
                r"is an IoU-matching outcome; it is not a semantic class.",
        label="tab:semantic_review_generated", colspec="lrrrrr", escape_cells=False)

    rt = reasons.reset_index().rename(columns={"unmatched_reason_geometric": "Geometric reason"})
    rt["Geometric reason"] = rt["Geometric reason"].map({
        "no_overlap": "No overlapping LabelMe box (IoU = 0)",
        "partial_overlap": r"Best IoU in $(0, 0.5)$",
        "iou_ge_0.5_lost_greedy": r"IoU $\geq 0.5$, box taken by another detection"})
    rt = rt.rename(columns={"non_palm": "non\\_palm", "total": "Total"})
    C.write_latex_table(
        C.TABLES / "tab_unmatched_reasons.tex", rt,
        caption=r"Why the 638 detections are LabelMe-unmatched, by human semantic label. The reason is derived "
                r"from geometry (best IoU with any LabelMe palm box); the reviewer did not record reasons.",
        label="tab:unmatched_reasons", colspec="lrrrr", escape_cells=False)

    models_c = C.complete_models(df)
    rows = []
    for m in models_c + [x for x in C.MODEL_ORDER if x not in models_c]:
        for c in C.CONDITIONS:
            p = vs[(vs["model"] == m) & (vs["condition"] == c) & (vs["semantic_label"] == "palm")]
            a = vs[(vs["model"] == m) & (vs["condition"] == c) & (vs["semantic_label"] == "ambiguous")]
            if p.empty:
                continue
            p, a = p.iloc[0], a.iloc[0]
            rows.append({"Model": m, "Cond.": c,
                         "palm R": C.fmt(int(p["n_R"])), "palm U": C.fmt(int(p["n_U"])), "palm Ur": C.fmt(int(p["n_Ur"])),
                         "amb. R": C.fmt(int(a["n_R"])), "amb. U": C.fmt(int(a["n_U"])), "amb. Ur": C.fmt(int(a["n_Ur"]))})
    C.write_latex_table(
        C.TABLES / "tab_semantic_by_decision.tex", pd.DataFrame(rows),
        caption=rf"VLM verdicts on the 638 LabelMe-unmatched detections, split by human semantic label "
                rf"({audit['label_counts']['palm']} palm, {audit['label_counts']['ambiguous']} ambiguous, "
                rf"{audit['label_counts']['non_palm']} non-palm). Under Protocol v2 every Unreliable verdict here "
                r"counts as an alignment ``TN''; most such verdicts fall on detections a human judged to be palms.",
        label="tab:semantic_by_decision_generated", colspec="llrrrrrr", group_col="Model", escape_cells=False)

    pb = by_bin.reset_index().rename(columns={"confidence_bin": "YOLO conf. bin", "non_palm": "non\\_palm",
                                              "bin_population": "Detections in bin"})
    C.write_latex_table(
        C.TABLES / "tab_pilot_by_bin.tex", pb[["YOLO conf. bin", "n", "palm", "ambiguous", "non\\_palm",
                                               "Detections in bin"]],
        caption=r"Lower-confidence pilot (outside the 5{,}747 cohort; no VLM predictions). Equal samples per bin, "
                r"so pooled shares are not prevalence estimates; report per bin only.",
        label="tab:pilot_by_bin", colspec="lrrrrr", escape_cells=False)

    # ------------------------------------------------------------ Figure
    colors = {"Reliable": "#4c78a8", "Uncertain": "#bab0ac", "Unreliable": "#e45756"}
    cells = [(m, c) for m in C.MODEL_ORDER for c in C.CONDITIONS
             if not vs[(vs["model"] == m) & (vs["condition"] == c)].empty]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [audit["label_counts"]["palm"] and 3, 3]})
    for ax, lab in zip(axes, ("palm", "ambiguous")):
        x = np.arange(len(cells))
        bottom = np.zeros(len(cells))
        for v in C.LABELS:
            vals = np.array([vs[(vs["model"] == m) & (vs["condition"] == c) & (vs["semantic_label"] == lab)]
                             [f"rate_{C.LABEL_SHORT[v]}"].iloc[0] for m, c in cells])
            ax.bar(x, vals, bottom=bottom, color=colors[v], label=v, width=0.8)
            bottom += vals
        ax.set_xticks(x, [f"{m} {c}" for m, c in cells], rotation=90, fontsize=5.5)
        ax.set_title(f"LabelMe-unmatched, human label = {lab} (n = {audit['label_counts'][lab]})", fontsize=8.5)
    axes[0].set_ylabel("share of verdicts")
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_semantic_verdicts.{ext}", dpi=200)
    plt.close(fig)
    print(json.dumps({"provenance": provenance, "unmatched_review": audit, "pilot": pilot_audit}, indent=2, default=str))
    print(reasons)
    print(by_bin)
    print(pilot_align)


if __name__ == "__main__":
    main()
