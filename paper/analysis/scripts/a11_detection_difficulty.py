#!/usr/bin/env python3
"""Analysis 11 — do decisions change more for small / low-confidence detections?

For every model with the needed conditions usable, and for the pairs A1->A5 (primary) and
A2->A5 (secondary; identical confidence text), the detections valid in BOTH conditions are
split by cohort quintiles (common.detection_properties; Q1 = lowest) of
  * YOLO confidence, and
  * normalised box area (bbox area / source-image area).
Events per detection: changed, toward_rejection (R<U<Ur index increases), into_U (A5 = U and
A1 != U), into_Ur (A5 = Ur and A1 != Ur); rates are over the paired detections in a quintile.

Contrast: D = rate(Q1) - rate(Q5), with a 95% image-cluster bootstrap percentile interval
(common.image_weights, BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES; same replicates as a05).
Verdict for "small / low-confidence detections change more" (event = changed), derived from
the interval, never typed in:
  supports    CI entirely > 0
  reverses    CI entirely < 0
  null        CI contains 0
  not evaluable  the model uses a single label in both conditions (no change is possible)
The collapse diagnostics of a10 are attached so concentrated outputs can be read with care.
No multiplicity correction; intervals are descriptive.
Confounding check: Spearman correlation of confidence and area in the cohort, and changed
rates in the 2x2 table of median splits (low/high confidence x small/large box).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ORDER = {lab: i for i, lab in enumerate(C.LABELS)}
PAIRS = [("A1", "A5"), ("A2", "A5")]
PROPS = {"confidence": "confidence_q", "box_area": "area_q"}
QUINTILES = [f"Q{i}" for i in range(1, 6)]
EVENTS = ("changed", "toward_rejection", "into_U", "into_Ur")


def events(fa: np.ndarray, fb: np.ndarray) -> dict:
    return {"changed": fa != fb, "toward_rejection": fb > fa,
            "into_U": (fb == 1) & (fa != 1), "into_Ur": (fb == 2) & (fa != 2)}


def verdict(lo: float, hi: float, evaluable: bool) -> str:
    if not evaluable:
        return "not evaluable (single label in both conditions)"
    if np.isnan(lo):
        return "not evaluable (undefined)"
    if lo > 0:
        return "supports"
    if hi < 0:
        return "reverses"
    return "null (CI includes 0)"


def main() -> None:
    C.ensure_dirs()
    out = C.OUT / "detection_difficulty"
    out.mkdir(parents=True, exist_ok=True)
    df = C.load_samples(include_partial=True)
    props = C.detection_properties()
    usage = pd.read_csv(C.OUT / "label_usage" / "label_usage_by_cell.csv")
    usage = usage.set_index(["model", "condition"])

    images = np.array(sorted(df["image_id"].unique()))
    img_pos = {im: i for i, im in enumerate(images)}
    n_img = len(images)
    assert set(props["image_name"]) == set(images)
    props["img"] = props["image_name"].map(img_pos)

    wide = df.pivot_table(index=["display", "sample_id"], columns="condition", values="decision", aggfunc="first")
    have = df.groupby("display")["condition"].unique().map(set)

    keys, num_cols, den_cols = [], [], []
    meta = {}
    for m in C.MODEL_ORDER:
        if m not in have.index:
            continue
        w_all = wide.loc[m]
        for a, b in PAIRS:
            if not {a, b} <= have[m]:
                continue
            w = C.paired(w_all, a, b)
            fa, fb = w[a].map(ORDER).to_numpy(), w[b].map(ORDER).to_numpy()
            ev = events(fa, fb)
            p = props.loc[w.index]
            single = bool(usage.loc[(m, a), "diag_single_label"] and usage.loc[(m, b), "diag_single_label"])
            meta[(m, f"{a}->{b}")] = {"n_paired": len(w), "single_label_both": single,
                                      "majority_ge_0.99_from": bool(usage.loc[(m, a), "diag_majority_ge_0.99"]),
                                      "majority_ge_0.99_to": bool(usage.loc[(m, b), "diag_majority_ge_0.99"]),
                                      "status_from": usage.loc[(m, a), "condition_status"],
                                      "status_to": usage.loc[(m, b), "condition_status"]}
            img = p["img"].to_numpy()
            for prop, qcol in PROPS.items():
                q = p[qcol].to_numpy()
                for qq in QUINTILES:
                    inq = q == qq
                    den = np.bincount(img[inq], minlength=n_img)
                    for e in EVENTS:
                        num_cols.append(np.bincount(img[inq & ev[e]], minlength=n_img))
                        den_cols.append(den)
                        keys.append((m, f"{a}->{b}", prop, qq, e))
    num = np.stack(num_cols, axis=1).astype(np.float64)
    den = np.stack(den_cols, axis=1).astype(np.float64)
    point = num.sum(axis=0) / np.where(den.sum(axis=0) > 0, den.sum(axis=0), np.nan)
    reps = C.bootstrap_ratio(num, den, n_img)
    kidx = {k: i for i, k in enumerate(keys)}

    rows = []
    for k, i in kidx.items():
        lo, hi, nu = C.percentile_ci(reps[:, i])
        rows.append({"model": k[0], "pair": k[1], "property": k[2], "quintile": k[3], "event": k[4],
                     "n_at_risk": int(den[:, i].sum()), "n_event": int(num[:, i].sum()),
                     "rate": point[i], "ci_low": lo, "ci_high": hi})
    rates = pd.DataFrame(rows)
    rates.to_csv(out / "event_rates_by_quintile.csv", index=False)

    rows = []
    for (m, pair), md in meta.items():
        for prop in PROPS:
            for e in EVENTS:
                i1, i5 = kidx[(m, pair, prop, "Q1", e)], kidx[(m, pair, prop, "Q5", e)]
                d = point[i1] - point[i5]
                lo, hi, nu = C.percentile_ci(reps[:, i1] - reps[:, i5])
                r = {"model": m, "model_key": C.DISPLAY_TO_KEY[m], "pair": pair, "property": prop, "event": e,
                     **md, "rate_Q1": point[i1], "rate_Q5": point[i5],
                     **{f"rate_{q}": point[kidx[(m, pair, prop, q, e)]] for q in QUINTILES},
                     "diff_Q1_minus_Q5": d, "ci_low": lo, "ci_high": hi, "n_undefined": nu,
                     "verdict": verdict(lo, hi, not md["single_label_both"])}
                rows.append(r)
    con = pd.DataFrame(rows)
    con.to_csv(out / "q1_vs_q5_contrasts.csv", index=False)

    # data-derived summary of the hypothesis (event = changed)
    summ = []
    for pair in ("A1->A5", "A2->A5"):
        for prop in PROPS:
            g = con[(con["pair"] == pair) & (con["property"] == prop) & (con["event"] == "changed")]
            vc = g["verdict"].value_counts()
            ev_ = g[~g["verdict"].str.startswith("not evaluable")]
            summ.append({"pair": pair, "property": prop, "n_models": len(g), "n_evaluable": len(ev_),
                         "n_supports": int(vc.get("supports", 0)), "n_reverses": int(vc.get("reverses", 0)),
                         "n_null": int(vc.get("null (CI includes 0)", 0)),
                         "supports": ";".join(g.loc[g["verdict"] == "supports", "model"]),
                         "reverses": ";".join(g.loc[g["verdict"] == "reverses", "model"]),
                         "null": ";".join(g.loc[g["verdict"].str.startswith("null"), "model"]),
                         "not_evaluable": ";".join(g.loc[g["verdict"].str.startswith("not evaluable"), "model"]),
                         "supports_among_core_panel": int(((g["verdict"] == "supports")
                                                           & g["model"].isin(C.CORE_PANEL)).sum()),
                         "core_panel_evaluable": int((g["model"].isin(C.CORE_PANEL)
                                                      & ~g["verdict"].str.startswith("not evaluable")).sum())})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "hypothesis_summary.csv", index=False)

    # confounding check
    rho = float(props["confidence"].rank().corr(props["norm_area"].rank()))
    lowc = props["confidence"] <= props["confidence"].median()
    small = props["norm_area"] <= props["norm_area"].median()
    cross = []
    for m in C.MODEL_ORDER:
        if m not in have.index or not {"A1", "A5"} <= have[m]:
            continue
        w = C.paired(wide.loc[m], "A1", "A5")
        ch = pd.Series(w["A1"].to_numpy() != w["A5"].to_numpy(), index=w.index)
        for cl, cm in (("low_conf", lowc), ("high_conf", ~lowc)):
            for al, am in (("small_box", small), ("large_box", ~small)):
                sel = (cm & am).reindex(w.index)
                cross.append({"model": m, "pair": "A1->A5", "confidence_half": cl, "area_half": al,
                              "n": int(sel.sum()), "changed_rate": float(ch[sel].mean())})
    cross = pd.DataFrame(cross)
    cross.to_csv(out / "confidence_x_area_median_split.csv", index=False)
    pd.DataFrame([{"spearman_confidence_vs_norm_area": rho, "n": len(props),
                   "confidence_quintile_edges": ";".join(f"{x:.4f}" for x in
                                                        props["confidence"].quantile([0, .2, .4, .6, .8, 1])),
                   "norm_area_quintile_edges": ";".join(f"{x:.6f}" for x in
                                                       props["norm_area"].quantile([0, .2, .4, .6, .8, 1]))}]
                 ).to_csv(out / "property_info.csv", index=False)

    # ---------------- LaTeX: A1->A5 changed, Q1-Q5 for both properties
    g = con[(con["pair"] == "A1->A5") & (con["event"] == "changed")]
    tab = []
    for m in [x for x in C.MODEL_ORDER if x in set(g["model"])]:
        r = {"Model": m}
        for prop, lab in (("confidence", "conf."), ("box_area", "area")):
            x = g[(g["model"] == m) & (g["property"] == prop)].iloc[0]
            r[f"Q1 {lab}"] = f"{100*x['rate_Q1']:.1f}"
            r[f"Q5 {lab}"] = f"{100*x['rate_Q5']:.1f}"
            if x["verdict"].startswith("not evaluable"):
                r[f"Q1$-$Q5 {lab}"] = "n/e"
            else:
                r[f"Q1$-$Q5 {lab}"] = (f"{100*x['diff_Q1_minus_Q5']:+.1f} [{100*x['ci_low']:+.1f}, "
                                       f"{100*x['ci_high']:+.1f}]")
            r[f"{lab} verdict"] = {"supports": "+", "reverses": "$-$"}.get(
                x["verdict"], "0" if x["verdict"].startswith("null") else "n/e")
        if x["majority_ge_0.99_from"] or x["majority_ge_0.99_to"]:
            r["Model"] = m + r"$^{c}$"
        if "partial" in (x["status_from"], x["status_to"]):
            r["Model"] += r"$^\dagger$"
        tab.append(r)
    s_c = summ[(summ["pair"] == "A1->A5") & (summ["property"] == "confidence")].iloc[0]
    s_a = summ[(summ["pair"] == "A1->A5") & (summ["property"] == "box_area")].iloc[0]
    C.write_latex_table(
        C.TABLES / "tab_detection_difficulty.tex", pd.DataFrame(tab),
        caption=rf"A1$\rightarrow$A5 changed rate (\%) in the lowest (Q1) and highest (Q5) cohort quintile of YOLO "
                rf"confidence and of normalised box area, and Q1$-$Q5 with 95\% image-cluster bootstrap intervals "
                rf"({C.BOOTSTRAP_REPLICATES} replicates, seed {C.BOOTSTRAP_SEED}). Verdict: + interval above 0 "
                rf"(low-confidence / small boxes change more), $-$ below 0, 0 includes 0, n/e not evaluable "
                rf"(single label in both conditions). Confidence: {s_c['n_supports']} of {s_c['n_evaluable']} "
                rf"evaluable checkpoints +, {s_c['n_reverses']} $-$, {s_c['n_null']} 0; area: {s_a['n_supports']} "
                rf"+, {s_a['n_reverses']} $-$, {s_a['n_null']} 0. Confidence and area are correlated in the cohort "
                rf"(Spearman $\rho={rho:.2f}$), so the two contrasts are not independent. $^{{c}}$ majority share "
                rf"$\geq 0.99$ in A1 or A5 (descriptive); $^\dagger$ partial condition, valid pairs only.",
        label="tab:detection_difficulty", colspec="lrrcc" + "rrcc", escape_cells=False, size=r"\tiny")

    # ---------------- Figure: changed rate by quintile, A1->A5
    models = [x for x in C.MODEL_ORDER if x in set(g["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (prop, title) in zip(axes, (("confidence", "YOLO confidence quintile"),
                                        ("box_area", "normalised box-area quintile"))):
        for i, m in enumerate(models):
            sub = rates[(rates["model"] == m) & (rates["pair"] == "A1->A5") & (rates["property"] == prop)
                        & (rates["event"] == "changed")].set_index("quintile").reindex(QUINTILES)
            ax.plot(range(5), sub["rate"], marker="osD^v<>ph*xP"[i % 12], lw=0.8, ms=3.5, label=m,
                    color=plt.cm.tab20(i % 20))
        ax.set_xticks(range(5), QUINTILES)
        ax.set_xlabel(title + " (Q1 = lowest)", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("A1→A5 changed rate")
    axes[1].legend(fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGURES / f"fig_detection_difficulty.{ext}", dpi=200)
    plt.close(fig)
    print(f"Spearman(conf, area) = {rho:.3f}")
    print(summ.drop(columns=["supports", "reverses", "null"]).to_string())
    print(g[["model", "property", "n_paired", "rate_Q1", "rate_Q5", "diff_Q1_minus_Q5", "ci_low", "ci_high",
             "verdict"]].round(3).to_string())


if __name__ == "__main__":
    main()
