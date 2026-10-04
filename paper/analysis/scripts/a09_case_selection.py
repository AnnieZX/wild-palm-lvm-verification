#!/usr/bin/env python3
"""Analysis 9 — deterministic qualitative case-selection manifest (no hand picking).

Pool: the 5,747 cohort detections with decisions from the seven-model panel
(common.CORE_PANEL; all five conditions full_clean; 35 decisions per detection). The panel is
fixed so the manifest does not shift as further checkpoints are added. Every category has a fixed score; cases are taken
in descending score order and ties are broken by sha256(f"{seed}:{category}:{sample_id}")
(seed = CASE_SELECTION_SEED). Categories whose qualifying pool is a plateau (e.g. all 35
decisions Reliable) are therefore a deterministic pseudo-random draw from that plateau.

  consensus_reliable     score = n_R (of 35); qualifying only if n_R == 35
  consensus_unreliable   score = n_Ur (of 35)
  high_disagreement      score = mean over A1..A5 of the Shannon entropy (bits) of the
                         7-model R/U/Ur vote; measures cross-model, not cross-condition, spread
  strong_A2_to_A5_change score = number of models whose A2 and A5 verdicts differ
                         (A2->A5 changes image AND image description; not image-only)
  abstention_heavy       score = n_U (of 35); Phi-4 and Gemma-4 never answer Uncertain, so max 25
  unmatched_semantic_palm  LabelMe-unmatched and human label palm; stratified by geometric
                         unmatched reason (no_overlap / partial_overlap / iou_ge_0.5_lost_greedy),
                         hash-ranked within stratum
  semantic_ambiguous     all LabelMe-unmatched detections labelled ambiguous (19)
  semantic_non_palm      cohort pool is empty (0 non_palm in the official review); the 9 pilot
                         non-palm detections are listed separately — outside the cohort, no VLM output
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

K = 10
REASON_QUOTA = {"no_overlap": 4, "partial_overlap": 3, "iou_ge_0.5_lost_greedy": 3}


def take(pool: pd.DataFrame, category: str, score: str | None, k: int | None) -> pd.DataFrame:
    p = pool.copy()
    p["tiebreak"] = [C.stable_hash_rank(C.CASE_SELECTION_SEED, category, s) for s in p.index]
    by = ([score] if score else []) + ["tiebreak"]
    p = p.sort_values(by, ascending=[False] * bool(score) + [True])
    sel = p if k is None else p.head(k)
    n_top = int((p[score] == p[score].max()).sum()) if score else len(p)
    sel = sel.assign(category=category, pool_size=len(pool), n_at_max_score=n_top,
                     selection_score=sel[score] if score else np.nan)
    sel["rank_in_category"] = np.arange(1, len(sel) + 1)
    return sel


def main() -> None:
    C.ensure_dirs()
    out = C.OUT / "case_selection"
    out.mkdir(parents=True, exist_ok=True)
    df = C.load_samples()
    models = [m for m in C.MODEL_ORDER if m in C.CORE_PANEL]
    assert C.complete_models(df[df["display"].isin(models)]) == models
    d = df[df["display"].isin(models)]
    short = d["decision"].map(C.LABEL_SHORT)

    piv = d.assign(s=short).pivot_table(index="sample_id", columns=["condition", "display"], values="s", aggfunc="first")
    base = df.drop_duplicates("sample_id").set_index("sample_id")[["image_id", "matched_gt", "max_iou", "yolo_confidence"]]
    feats = base.copy()
    allv = piv.to_numpy()
    for lab in ("R", "U", "Ur"):
        feats[f"n_{lab}"] = (allv == lab).sum(axis=1)
    ent = []
    for c in C.CONDITIONS:
        sub = piv[c][models].to_numpy()
        cnt = np.stack([(sub == lab).sum(axis=1) for lab in ("R", "U", "Ur")], axis=1)
        ent.append([C.shannon_entropy(r) for r in cnt])
    feats["mean_vote_entropy"] = np.mean(ent, axis=0)
    feats["n_models_A2_ne_A5"] = (piv["A2"][models].to_numpy() != piv["A5"][models].to_numpy()).sum(axis=1)
    for c in C.CONDITIONS:
        feats[f"votes_{c}"] = ["".join({"R": "R", "U": "U", "Ur": "X"}[v] for v in row) for row in piv[c][models].to_numpy()]

    j = pd.read_csv(C.OUT / "semantic" / "unmatched_review_joined.csv", dtype={"sample_id": str}).set_index("sample_id")
    feats["semantic_label"] = j["semantic_label"].reindex(feats.index).fillna("not_reviewed")
    feats["unmatched_reason_geometric"] = j["unmatched_reason_geometric"].reindex(feats.index).fillna("")

    sels = [
        take(feats[feats["n_R"] == 35], "consensus_reliable", "n_R", K),
        take(feats, "consensus_unreliable", "n_Ur", K),
        take(feats, "high_disagreement", "mean_vote_entropy", K),
        take(feats, "strong_A2_to_A5_change", "n_models_A2_ne_A5", K),
        take(feats, "abstention_heavy", "n_U", K),
    ]
    palm = feats[(~feats["matched_gt"]) & (feats["semantic_label"] == "palm")]
    for reason, q in REASON_QUOTA.items():
        s = take(palm[palm["unmatched_reason_geometric"] == reason], f"unmatched_semantic_palm:{reason}", None, q)
        sels.append(s)
    sels.append(take(feats[feats["semantic_label"] == "ambiguous"], "semantic_ambiguous", None, None))
    man = pd.concat(sels).reset_index().rename(columns={"index": "sample_id"})

    for c in C.CONDITIONS:
        pi = pd.read_csv(C.ABLATION_ROOT / C.ABLATION_DIRS[c] / "prompt_index.csv", dtype={"sample_id": str}).set_index("sample_id")
        root = C.ABLATION_ROOT / C.ABLATION_DIRS[c]
        man[f"image_{c}"] = [str((root / pi.at[s, "image_path"]).resolve().relative_to(C.REPO)) for s in man["sample_id"]]
    counts = man.groupby("sample_id")["category"].transform("count")
    man["n_categories_for_sample"] = counts
    cols = (["category", "rank_in_category", "pool_size", "n_at_max_score", "selection_score", "sample_id", "image_id", "matched_gt",
             "max_iou", "yolo_confidence", "semantic_label", "unmatched_reason_geometric", "n_R", "n_U", "n_Ur",
             "mean_vote_entropy", "n_models_A2_ne_A5"] + [f"votes_{c}" for c in C.CONDITIONS]
            + [f"image_{c}" for c in C.CONDITIONS] + ["n_categories_for_sample"])
    man[cols].to_csv(out / "case_manifest.csv", index=False)

    pilot = pd.read_csv(C.PILOT_CSV).merge(pd.read_csv(C.PILOT_REFERENCE_CSV), on="pilot_id", suffixes=("", "_ref"))
    pn = pilot[pilot["semantic_label"] == "non_palm"].copy()
    pn["tiebreak"] = [C.stable_hash_rank(C.CASE_SELECTION_SEED, "pilot_non_palm", s) for s in pn["pilot_id"]]
    pn = pn.sort_values("tiebreak")
    pn = pn.assign(category="pilot_semantic_non_palm", in_5747_cohort=False, has_vlm_predictions=False)
    pn[["category", "pilot_id", "image_id", "yolo_bbox_xywh", "yolo_confidence", "confidence_bin",
        "geo_max_iou_labelme", "in_5747_cohort", "has_vlm_predictions"]].to_csv(out / "pilot_non_palm_cases.csv", index=False)

    rules = pd.DataFrame([{"category": k, "pool_size": int(g["pool_size"].iloc[0]),
                           "n_at_max_score": int(g["n_at_max_score"].iloc[0]), "selected": len(g),
                           "min_score": g["selection_score"].min(), "max_score": g["selection_score"].max()}
                          for k, g in man.groupby("category", sort=False)])
    rules.to_csv(out / "case_selection_summary.csv", index=False)
    (out / "VOTE_STRING_LEGEND.txt").write_text(
        "votes_<cond> columns: one character per model in this order: " + ", ".join(models)
        + "\nR = Reliable, U = Uncertain, X = Unreliable\n"
        f"Tie-break: sha256('{C.CASE_SELECTION_SEED}:<category>:<sample_id>') ascending.\n")
    print(rules.to_string(index=False))
    print(f"pilot non-palm: {len(pn)}")


if __name__ == "__main__":
    main()
