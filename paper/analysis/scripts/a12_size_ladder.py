#!/usr/bin/env python3
"""Analysis 12 — within-family size ladders (common.LADDERS), A1 vs A5.

Per checkpoint on a ladder (registry order = nominal size):
  status (all five conditions usable / pending / not run), counted parameters (a00 inventory);
  A1 and A5 decision distributions (each over its own valid rows) and entropy;
  over detections valid in BOTH A1 and A5 (n_paired): changed, toward-rejection,
  toward-acceptance, into-Uncertain (A5 = U, A1 != U) and into-Unreliable (A5 = Ur, A1 != Ur)
  fractions with 95% image-cluster bootstrap intervals (same replicates as a05/a11), and the
  3x3 A1->A5 transition matrix;
  changed rate in the lowest / highest confidence and box-area quintile (from a11).
Ordering description (NOT a test of monotonic scaling): for ladders with >= 3 evaluable
checkpoints, the observed pattern of a quantity along nominal size is labelled
"increasing", "decreasing" or "non-monotone", and the share of bootstrap replicates showing
the same strict pattern is reported. With 3-4 points an ordered sequence can easily arise
by chance and says nothing about checkpoints in between or beyond; with 2 points only the
difference is reported. Checkpoints that use a single label in both A1 and A5 are listed but
excluded from the ordering description (their change rate is 0 by construction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ORDER = {lab: i for i, lab in enumerate(C.LABELS)}
QUANTITIES = ("changed", "toward_rejection", "toward_acceptance", "into_U", "into_Ur")


def pattern(v: np.ndarray) -> str:
    d = np.diff(v)
    if np.all(d > 0):
        return "increasing"
    if np.all(d < 0):
        return "decreasing"
    return "non-monotone"


def main() -> None:
    C.ensure_dirs()
    out = C.OUT / "size_ladder"
    out.mkdir(parents=True, exist_ok=True)
    df = C.load_samples(include_partial=True)
    inv = pd.read_csv(C.OUT / "audit" / "model_inventory.csv").set_index("model_key")
    usage = pd.read_csv(C.OUT / "label_usage" / "label_usage_by_cell.csv").set_index(["model_key", "condition"])
    con = pd.read_csv(C.OUT / "detection_difficulty" / "q1_vs_q5_contrasts.csv")
    con = con[(con["pair"] == "A1->A5") & (con["event"] == "changed")].set_index(["model_key", "property"])

    images = np.array(sorted(df["image_id"].unique()))
    img_pos = {im: i for i, im in enumerate(images)}
    n_img = len(images)
    df["img"] = df["image_id"].map(img_pos)
    have = df.groupby("model_key")["condition"].unique().map(set)
    wide = df.pivot_table(index=["model_key", "sample_id"], columns="condition", values="decision", aggfunc="first")
    img_of = df.drop_duplicates("sample_id").set_index("sample_id")["img"]

    rows, mats, num_cols, den_cols, keys = [], [], [], [], []
    for family, ladder in C.LADDERS.items():
        for rank, key in enumerate(ladder):
            base = {"ladder": family, "rank_in_ladder": rank + 1, "model_key": key}
            if key in C.NOT_RUN:
                disp, nb, why = C.NOT_RUN[key]
                rows.append({**base, "model": disp, "nominal_b": nb, "status": "not run", "note": why})
                continue
            run = C.RUN_BY_KEY[key]
            base.update(model=run["display"], nominal_b=run["nominal_b"],
                        params_billions_counted=inv.loc[key, "parameters_billions"])
            conds = have.get(key, set())
            if len(conds) < 5:
                rows.append({**base, "status": f"pending ({len(conds)} of 5 conditions usable)",
                             "note": run["expected"]})
                continue
            partial = inv.loc[key, "conditions_partial"]
            partial = partial if isinstance(partial, str) else ""
            w = C.paired(wide.loc[key], "A1", "A5")
            fa, fb = w["A1"].map(ORDER).to_numpy(), w["A5"].map(ORDER).to_numpy()
            ev = {"changed": fa != fb, "toward_rejection": fb > fa, "toward_acceptance": fb < fa,
                  "into_U": (fb == 1) & (fa != 1), "into_Ur": (fb == 2) & (fa != 2)}
            im = img_of.loc[w.index].to_numpy()
            den = np.bincount(im, minlength=n_img)
            for q in QUANTITIES:
                num_cols.append(np.bincount(im[ev[q]], minlength=n_img))
                den_cols.append(den)
                keys.append((key, q))
            single = bool(usage.loc[(key, "A1"), "diag_single_label"] and usage.loc[(key, "A5"), "diag_single_label"])
            r = {**base, "status": "five conditions usable" + (f" (partial: {partial})" if partial else ""),
                 "single_label_A1_and_A5": single, "n_paired_A1_A5": len(w)}
            for c in ("A1", "A5"):
                u = usage.loc[(key, c)]
                r.update({f"{c}_n": int(u["n_valid"]), f"{c}_R": u["share_R"], f"{c}_U": u["share_U"],
                          f"{c}_Ur": u["share_Ur"], f"{c}_entropy": u["entropy_bits"],
                          f"{c}_majority_share": u["majority_share"]})
            for q in QUANTITIES:
                r[q] = float(ev[q].mean())
            for prop in ("confidence", "box_area"):
                x = con.loc[(key, prop)]
                r[f"changed_Q1_{prop}"] = x["rate_Q1"]
                r[f"changed_Q5_{prop}"] = x["rate_Q5"]
                r[f"Q1_minus_Q5_{prop}"] = x["diff_Q1_minus_Q5"]
                r[f"Q1_minus_Q5_{prop}_ci"] = f"[{x['ci_low']:.3f}, {x['ci_high']:.3f}]"
                r[f"verdict_{prop}"] = x["verdict"]
            rows.append(r)
            ct = pd.crosstab(w["A1"], w["A5"]).reindex(index=list(C.LABELS), columns=list(C.LABELS), fill_value=0)
            for fr in C.LABELS:
                for to in C.LABELS:
                    mats.append({"ladder": family, "model": run["display"], "from_A1": fr, "to_A5": to,
                                 "count": int(ct.loc[fr, to]), "share_of_paired": ct.loc[fr, to] / len(w)})

    num = np.stack(num_cols, axis=1).astype(np.float64)
    den = np.stack(den_cols, axis=1).astype(np.float64)
    reps = C.bootstrap_ratio(num, den, n_img)
    kidx = {k: i for i, k in enumerate(keys)}
    lad = pd.DataFrame(rows)
    for q in QUANTITIES:
        lo, hi = [], []
        for key in lad["model_key"]:
            if (key, q) in kidx:
                a, b, _ = C.percentile_ci(reps[:, kidx[(key, q)]])
            else:
                a = b = np.nan
            lo.append(a)
            hi.append(b)
        lad[f"{q}_ci_low"] = lo
        lad[f"{q}_ci_high"] = hi
    lad.to_csv(out / "size_ladder_checkpoints.csv", index=False)
    pd.DataFrame(mats).to_csv(out / "size_ladder_transition_matrices.csv", index=False)

    # ordering description
    orows = []
    for family, ladder in C.LADDERS.items():
        g = lad[(lad["ladder"] == family) & lad["model_key"].isin([k for k, _ in keys])]
        excluded = g[g["single_label_A1_and_A5"] == True]  # noqa: E712
        g = g[g["single_label_A1_and_A5"] != True]  # noqa: E712
        pts = list(g["model"])
        for q in QUANTITIES:
            base = {"ladder": family, "quantity": f"{q}_rate_A1_to_A5", "checkpoints_in_size_order": ";".join(pts),
                    "n_points": len(pts), "values": ";".join(f"{v:.3f}" for v in g[q]),
                    "excluded_single_label": ";".join(excluded["model"]),
                    "not_available": ";".join(lad.loc[(lad["ladder"] == family) & ~lad["model_key"].isin(
                        [k for k, _ in keys]), "model"])}
            if len(pts) >= 3:
                obs = pattern(g[q].to_numpy())
                rmat = np.stack([reps[:, kidx[(k, q)]] for k in g["model_key"]], axis=1)
                same = np.mean([pattern(r) == obs for r in rmat])
                orows.append({**base, "observed_pattern": obs, "bootstrap_share_same_pattern": float(same),
                              "note": "descriptive ordering of few points; not a test of monotonic scaling"})
            elif len(pts) == 2:
                i0, i1 = kidx[(g["model_key"].iloc[0], q)], kidx[(g["model_key"].iloc[1], q)]
                lo, hi, _ = C.percentile_ci(reps[:, i1] - reps[:, i0])
                orows.append({**base, "observed_pattern": "two points only",
                              "larger_minus_smaller": float(g[q].iloc[1] - g[q].iloc[0]),
                              "larger_minus_smaller_ci": f"[{lo:.3f}, {hi:.3f}]",
                              "note": "two checkpoints: difference only, no ordering claim"})
            else:
                orows.append({**base, "observed_pattern": "fewer than two evaluable checkpoints"})
    order = pd.DataFrame(orows)
    order.to_csv(out / "size_ladder_ordering_description.csv", index=False)

    # ---------------- LaTeX
    tab = []
    for r in lad.itertuples():
        if not isinstance(getattr(r, "A1_n", None), (int, float)) or pd.isna(getattr(r, "A1_n", np.nan)):
            tab.append({"Ladder": r.ladder, "Checkpoint": r.model, "Params": C.fmt(getattr(r, "params_billions_counted",
                                                                                          np.nan), 2),
                        "A1 R/U/Ur": r.status, "A5 R/U/Ur": "", "Changed": "", "into U": "", "into Ur": "",
                        r"$\Delta$conf": "", r"$\Delta$area": ""})
            continue

        def ci(q):
            return (f"{100*getattr(r, q):.1f} [{100*getattr(r, q + '_ci_low'):.1f}, "
                    f"{100*getattr(r, q + '_ci_high'):.1f}]")

        def dq(prop):
            v = getattr(r, f"verdict_{prop}")
            if v.startswith("not evaluable"):
                return "n/e"
            return f"{100*getattr(r, f'Q1_minus_Q5_{prop}'):+.1f}" + \
                {"supports": "", "reverses": "", }.get(v, r"$^{0}$")
        name = r.model + (r"$^{s}$" if r.single_label_A1_and_A5 else "") + (r"$^\dagger$" if "partial" in r.status else "")
        tab.append({"Ladder": r.ladder, "Checkpoint": name, "Params": C.fmt(r.params_billions_counted, 2),
                    "A1 R/U/Ur": f"{100*r.A1_R:.0f}/{100*r.A1_U:.0f}/{100*r.A1_Ur:.0f}",
                    "A5 R/U/Ur": f"{100*r.A5_R:.0f}/{100*r.A5_U:.0f}/{100*r.A5_Ur:.0f}",
                    "Changed": ci("changed"), "into U": ci("into_U"), "into Ur": ci("into_Ur"),
                    r"$\Delta$conf": dq("confidence"), r"$\Delta$area": dq("box_area")})
    C.write_latex_table(
        C.TABLES / "tab_size_ladder.tex", pd.DataFrame(tab),
        caption=r"Within-family size ladders, A1 (overlay only) vs A5 (crop only). Params: counted from checkpoint "
                r"headers (billions). R/U/Ur in \% of valid decisions. Changed / into U / into Ur: \% of detections "
                r"valid in both conditions, with 95\% image-cluster bootstrap intervals. $\Delta$conf, $\Delta$area: "
                r"A1$\rightarrow$A5 changed rate in the lowest minus the highest quintile (pp; $^{0}$ interval "
                r"includes 0; n/e not evaluable). $^{s}$ single label in both A1 and A5; $^\dagger$ partial condition "
                r"(valid rows only). Points are few per family; an ordered sequence is not evidence of monotonic "
                r"scaling.",
        label="tab:size_ladder", colspec="llrcccccrr", group_col="Ladder", escape_cells=False, size=r"\tiny")
    print(lad[["ladder", "model", "status", "n_paired_A1_A5", "changed", "into_U", "into_Ur"]].round(3).to_string())
    print(order[["ladder", "quantity", "checkpoints_in_size_order", "values", "observed_pattern",
                 "bootstrap_share_same_pattern", "excluded_single_label", "not_available"]].to_string())


if __name__ == "__main__":
    main()
