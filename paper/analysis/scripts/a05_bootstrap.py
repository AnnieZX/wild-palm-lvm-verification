#!/usr/bin/env python3
"""Analysis 5 — image-level (cluster) bootstrap confidence intervals.

Procedure
  * Resampling unit: source image (image_id), because several detections share an
    image. The 870 cohort images are drawn with replacement, B times; a drawn image
    contributes all of its detections (with multiplicity).
  * One set of image weights per replicate is shared by every model and condition,
    so condition contrasts within a model (and cross-model contrasts) are paired.
  * Per replicate, the six cells {R,U,Ur} x {matched, unmatched} are summed per
    model x condition and every metric is recomputed from them.
  * 95% CI = 2.5th / 97.5th percentiles of the replicate distribution (percentile
    method). Replicates where a metric is undefined (zero denominator) are dropped
    for that metric and counted in n_undefined.
  * Seed and B are fixed in common.py (BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES); the weight
    generator is common.image_weights, shared with a11/a12.
  * Partial cells (common.CONDITION_POLICY) enter with their valid rows only. A transition
    rate uses, per model and pair, the detections valid in both conditions as denominator.
  * LaTeX tables cover the seven-model panel; the CSVs cover every usable cell.
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

METRICS = ["reliable_rate", "uncertain_rate", "unreliable_rate", "coverage", "accuracy", "precision",
           "sensitivity", "specificity", "f1", "balanced_accuracy"]
ORDER = {lab: i for i, lab in enumerate(C.LABELS)}


def summarize(point: float, reps: np.ndarray) -> dict:
    ok = reps[~np.isnan(reps)]
    if ok.size == 0:
        return {"estimate": point, "ci_low": np.nan, "ci_high": np.nan, "boot_se": np.nan,
                "n_undefined": int(reps.size)}
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return {"estimate": point, "ci_low": lo, "ci_high": hi, "boot_se": float(ok.std(ddof=1)),
            "n_undefined": int(reps.size - ok.size)}


def main() -> None:
    C.ensure_dirs()
    df = C.load_samples(include_partial=True)
    images = np.array(sorted(df["image_id"].unique()))
    img_pos = {im: i for i, im in enumerate(images)}
    n_img = len(images)
    df["img"] = df["image_id"].map(img_pos)

    cells = [(m, c) for m in C.MODEL_ORDER for c in C.CONDITIONS
             if ((df["display"] == m) & (df["condition"] == c)).any()]
    # per-image counts: columns = cell x 6
    counts = np.zeros((n_img, len(cells) * 6))
    for ci, (m, c) in enumerate(cells):
        sub = df[(df["display"] == m) & (df["condition"] == c)]
        for k, key in enumerate(C.CELL_KEYS):
            lab = {"R": "Reliable", "U": "Uncertain", "Ur": "Unreliable"}[key.split("_")[0]]
            matched = key.endswith("_m")
            sel = sub[(sub["decision"] == lab) & (sub["matched_gt"] == matched)]
            counts[:, ci * 6 + k] = np.bincount(sel["img"], minlength=n_img)

    # per-image transition counts for complete models (changed / toward rejection / toward acceptance)
    models = C.complete_models(df)
    wide = df[df["display"].isin(models)].pivot_table(index=["display", "sample_id"], columns="condition",
                                                      values="decision", aggfunc="first")
    img_of = df.drop_duplicates("sample_id").set_index("sample_id")["img"]
    trans_keys = []
    tcols, dcols = [], []
    for m in models:
        w_all = wide.loc[m]
        for group, pairs in C.PAIR_GROUPS.items():
            for a, b in pairs:
                w = C.paired(w_all, a, b)
                im = img_of.loc[w.index].to_numpy()
                den = np.bincount(im, minlength=n_img)
                fa, fb = w[a].map(ORDER).to_numpy(), w[b].map(ORDER).to_numpy()
                for kind, mask in (("changed", fa != fb), ("toward_rejection", fb > fa),
                                   ("toward_acceptance", fb < fa)):
                    tcols.append(np.bincount(im[mask], minlength=n_img))
                    dcols.append(den)
                    trans_keys.append((m, group, f"{a}->{b}", kind, int(den.sum())))
    tcounts = np.stack(tcols, axis=1).astype(np.float64)
    tden = np.stack(dcols, axis=1).astype(np.float64)
    n_per_img = np.bincount(img_of.to_numpy(), minlength=n_img).astype(np.float64)

    def metric_arrays(tot):
        out = {}
        for ci, cell in enumerate(cells):
            cc = {k: tot[..., ci * 6 + j] for j, k in enumerate(C.CELL_KEYS)}
            out[cell] = C.metrics_from_counts(cc)
        return out

    point = metric_arrays(counts.sum(axis=0))
    point_trans = tcounts.sum(axis=0) / tden.sum(axis=0)

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    boot_tot, boot_trans = [], []
    for wts in C.image_weights(rng, n_img, C.BOOTSTRAP_REPLICATES):
        boot_tot.append(wts @ counts)
        boot_trans.append((wts @ tcounts) / (wts @ tden))
    boot_tot = np.vstack(boot_tot)
    boot_trans = np.vstack(boot_trans)
    boot = metric_arrays(boot_tot)

    # ---- metric CIs
    status = C.cell_status(df)
    n_cell = df.groupby(["display", "condition"]).size().to_dict()
    rows = []
    for cell in cells:
        for met in METRICS:
            rows.append({"model": cell[0], "condition": cell[1], "condition_status": status[cell],
                         "n": int(n_cell[cell]), "metric": met,
                         **summarize(float(point[cell][met]), np.asarray(boot[cell][met]))})
    mci = pd.DataFrame(rows)
    mci.to_csv(C.OUT / "bootstrap_metric_cis.csv", index=False)

    # ---- paired delta CIs (same replicate weights for both conditions)
    rows = []
    for m in models:
        for group, pairs in C.PAIR_GROUPS.items():
            for a, b in pairs:
                for met in METRICS:
                    pd_ = float(point[(m, b)][met] - point[(m, a)][met])
                    reps = np.asarray(boot[(m, b)][met]) - np.asarray(boot[(m, a)][met])
                    s = summarize(pd_, reps)
                    s["ci_excludes_zero"] = bool(not np.isnan(s["ci_low"]) and (s["ci_low"] > 0 or s["ci_high"] < 0))
                    rows.append({"model": m, "group": group, "pair": f"{a}->{b}", "metric": met,
                                 "n_from": int(n_cell[(m, a)]), "n_to": int(n_cell[(m, b)]), **s})
    dci = pd.DataFrame(rows)
    dci.to_csv(C.OUT / "bootstrap_delta_cis.csv", index=False)

    # ---- transition-rate CIs
    rows = []
    for j, (m, group, pair, kind, n_pair) in enumerate(trans_keys):
        rows.append({"model": m, "group": group, "pair": pair, "quantity": f"{kind}_rate", "n_paired": n_pair,
                     **summarize(float(point_trans[j]), boot_trans[:, j])})
    tci = pd.DataFrame(rows)
    tci.to_csv(C.OUT / "bootstrap_transition_cis.csv", index=False)

    info = {"resampling_unit": "image_id (source image)", "n_images": int(n_img), "n_detections": int(n_per_img.sum()),
            "replicates": C.BOOTSTRAP_REPLICATES, "seed": C.BOOTSTRAP_SEED, "rng": "numpy.random.default_rng (PCG64)",
            "ci_method": "percentile, 95% (2.5th/97.5th)",
            "pairing": "one image-weight vector per replicate shared across all models and conditions",
            "undefined_metric_handling": "replicates with zero denominator dropped per metric; counted in n_undefined",
            "partial_cells": "valid rows only; transition denominators = detections valid in both conditions",
            "models_with_transitions": models,
            "detections_per_image": {"min": int(n_per_img.min()), "median": float(np.median(n_per_img)),
                                     "max": int(n_per_img.max())}}
    (C.OUT / "bootstrap_info.json").write_text(json.dumps(info, indent=2))

    # ---- LaTeX: key metrics with CIs
    def ci(m, c, met, d=3):
        r = mci[(mci["model"] == m) & (mci["condition"] == c) & (mci["metric"] == met)].iloc[0]
        if np.isnan(r["estimate"]):
            return "--"
        return f"{r['estimate']:.{d}f} [{r['ci_low']:.{d}f}, {r['ci_high']:.{d}f}]"
    rows = []
    for m, c in [x for x in cells if x[0] in C.CORE_PANEL]:
        rows.append({"Model": m, "Cond.": c, "Coverage": ci(m, c, "coverage"), "Sens": ci(m, c, "sensitivity"),
                     "Spec": ci(m, c, "specificity"), "BalAcc": ci(m, c, "balanced_accuracy")})
    C.write_latex_table(
        C.TABLES / "tab_bootstrap_metrics.tex", pd.DataFrame(rows),
        caption=rf"Point estimates with 95\% image-level bootstrap intervals ({C.BOOTSTRAP_REPLICATES} replicates, "
                rf"seed {C.BOOTSTRAP_SEED}, {n_img} images resampled with replacement). Sens/Spec/BalAcc are "
                r"Protocol-v2 IoU-alignment metrics on decided samples.",
        label="tab:bootstrap_metrics", colspec="llcccc", group_col="Model", escape_cells=False)
    rows = []
    for m, c in [x for x in cells if x[0] not in C.CORE_PANEL]:
        rows.append({"Model": m, "Cond.": c,
                     "$N$": C.fmt(int(n_cell[(m, c)])) + (r"$^\dagger$" if status[(m, c)] == "partial" else ""),
                     "Coverage": ci(m, c, "coverage"), "Sens": ci(m, c, "sensitivity"),
                     "Spec": ci(m, c, "specificity"), "BalAcc": ci(m, c, "balanced_accuracy")})
    C.write_latex_table(
        C.TABLES / "tab_bootstrap_metrics_extended.tex", pd.DataFrame(rows),
        caption=r"As Table~\ref{tab:bootstrap_metrics}, for the further checkpoints. $^\dagger$ partial condition "
                r"(valid rows only). -- = undefined (no decided samples, e.g.\ a checkpoint that answers only "
                r"Uncertain).",
        label="tab:bootstrap_metrics_extended", colspec="llrcccc", group_col="Model", escape_cells=False)

    rows = []
    for m in [x for x in models if x in C.CORE_PANEL]:
        for p in ("A1->A2", "A2->A3", "A2->A4", "A2->A5", "A4->A5"):
            r = {"Model": m, "Pair": p.replace("->", r"$\rightarrow$")}
            for met, lab in (("reliable_rate", r"$\Delta$R"), ("uncertain_rate", r"$\Delta$U"),
                             ("sensitivity", r"$\Delta$Sens"), ("specificity", r"$\Delta$Spec"),
                             ("balanced_accuracy", r"$\Delta$BalAcc")):
                s = dci[(dci["model"] == m) & (dci["pair"] == p) & (dci["metric"] == met)].iloc[0]
                r[lab] = f"{s['estimate']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
            rows.append(r)
    C.write_latex_table(
        C.TABLES / "tab_bootstrap_deltas.tex", pd.DataFrame(rows),
        caption=r"Paired condition contrasts with 95\% image-level bootstrap intervals. A1$\rightarrow$A2 and "
                r"A2$\rightarrow$A3 are text-only (identical image); A2$\rightarrow$A4, A2$\rightarrow$A5 and "
                r"A4$\rightarrow$A5 are image-representation contrasts (image and image-description text change). "
                r"Intervals are descriptive; no multiplicity correction is applied.",
        label="tab:bootstrap_deltas", colspec="llccccc", group_col="Model", escape_cells=False, size=r"\tiny")

    # ---- Figure: forest plot of paired deltas
    pairs = ["A1->A2", "A2->A3", "A2->A4", "A2->A5", "A4->A5"]
    mets = [("reliable_rate", "ΔR rate"), ("uncertain_rate", "ΔU rate"), ("sensitivity", "ΔSens (align.)"),
            ("specificity", "ΔSpec (align.)"), ("balanced_accuracy", "ΔBalAcc (align.)")]
    colors = {"A1->A2": "#9ecae1", "A2->A3": "#3182bd", "A2->A4": "#fdae6b", "A2->A5": "#e6550d", "A4->A5": "#a63603"}
    fig, axes = plt.subplots(1, len(mets), figsize=(14, 4.2), sharey=True)
    for ax, (met, title) in zip(axes, mets):
        for j, p in enumerate(pairs):
            sub = dci[(dci["pair"] == p) & (dci["metric"] == met)].set_index("model").reindex(models)
            y = np.arange(len(models)) + (j - 2) * 0.15
            ax.errorbar(sub["estimate"], y, xerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
                        fmt="o", ms=3, lw=0.8, color=colors[p], label=p.replace("->", "→"))
        ax.axvline(0, color="0.5", lw=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(range(len(models)), models, fontsize=7)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.text(0.01, 0.01, f"Paired image-level bootstrap, {C.BOOTSTRAP_REPLICATES} replicates, seed {C.BOOTSTRAP_SEED}. "
             "Blue: text-only contrasts (same image). Orange: image-representation contrasts.", fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_bootstrap_deltas.{ext}", dpi=200)
    plt.close(fig)
    print(json.dumps(info, indent=2))
    print(dci[dci["pair"].isin(["A2->A4", "A2->A5"]) & dci["metric"].isin(["balanced_accuracy", "specificity"])]
          .round(3).to_string())


if __name__ == "__main__":
    main()
