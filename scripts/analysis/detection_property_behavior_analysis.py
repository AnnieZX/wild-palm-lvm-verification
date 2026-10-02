#!/usr/bin/env python3
"""
Purpose:
    Ground-truth-independent analysis of which detection-level properties are
    associated with VLM decision changes between input conditions (primary A1->A5,
    secondary A2->A5). Uses only stored YOLO detection properties and stored VLM
    decisions; no LabelMe, matching, GT, or semantic label enters the analysis, and
    no association reported here measures whether a decision is correct.
    See docs/DETECTION_PROPERTY_BEHAVIOR_ANALYSIS.md.

    Uncertainty is quantified with an image-clustered bootstrap (resampling whole
    source images, never individual detections); a coarser naming-derived parent-frame
    bootstrap is reported as a sensitivity check for contrasts.

Input (read-only):
    - outputs/verification_dataset/index.csv               (detection properties)
    - outputs/verification_dataset/images/*.png            (PNG header only: source size)
    - outputs/a1_a5_behavior_analysis/per_detection_decisions.csv
    - outputs/a1_a5_behavior_analysis/pairwise_transitions.csv   (count cross-check)

Output (default outputs/detection_property_behavior_analysis/):
    - property_table.csv, property_summary.csv, property_correlations.csv
    - clustering_summary.csv
    - detection_outcomes.csv
    - category_property_medians.csv, median_differences.csv
    - event_rates_by_bin.csv, event_contrasts.csv
    - stratified_change_rates.csv
    - summary.md, analysis_info.json

Example:
    python scripts/analysis/detection_property_behavior_analysis.py
    python scripts/visualization/plot_detection_property_behavior.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

import a1_a5_behavior_analysis as behavior  # noqa: E402
from src.paths import OUTPUTS_DIR, VERIFICATION_DATASET_DIR, VERIFICATION_DATASET_INDEX_CSV  # noqa: E402
from src.preprocessing.ablation_verification_images import (  # noqa: E402
    A5_CROP_PADDING,
    A5_OUTPUT_SIZE,
    _padded_bbox,
)

BEHAVIOR_DIR = behavior.DEFAULT_OUT_DIR
PER_DETECTION_CSV = BEHAVIOR_DIR / "per_detection_decisions.csv"
PAIRWISE_CSV = BEHAVIOR_DIR / "pairwise_transitions.csv"
DEFAULT_OUT_DIR = OUTPUTS_DIR / "detection_property_behavior_analysis"
INPUT_FILES = (VERIFICATION_DATASET_INDEX_CSV, PER_DETECTION_CSV, PAIRWISE_CSV)

EXPECTED_N = behavior.EXPECTED_N
MODELS = behavior.MODEL_RUNS
COMPARISONS = (("A1", "A5"), ("A2", "A5"))
PRIMARY = "A1->A5"

# Only these index.csv columns are read; everything else is derived from them.
INDEX_COLUMNS = ("sample_id", "image_name", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
                 "bbox_area", "center_x", "center_y", "confidence")
FORBIDDEN_TOKENS = {"gt", "iou", "tp", "fp", "tn", "fn", "accuracy", "sensitivity", "specificity",
                    "precision", "recall", "f1", "matched", "labelme", "semantic", "palm", "correct"}

ABBREV = {"Reliable": "R", "Uncertain": "U", "Unreliable": "Ur"}
LABEL_FROM_ABBREV = {v: k for k, v in ABBREV.items()}
RANK = behavior.REJECTION_RANK

N_BOOT = 2000
SEED = 20261001

CONFIDENCE_EDGES = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 1.00)
DETECTIONS_PER_IMAGE_EDGES = (0, 3, 6, 9, 12, 20)
# (property, binning) used for rate-by-bin tables. "quintile" bins are cohort quintiles.
BINNED_PROPERTIES = {
    "confidence": "fixed_confidence",
    "norm_area": "quintile",
    "elongation": "quintile",
    "center_edge_dist_norm": "quintile",
    "center_x_norm": "quintile",
    "center_y_norm": "quintile",
    "touches_edge": "binary",
    "detections_in_image": "fixed_detections",
    "a5_upscale": "quintile",
    "a5_aspect_distortion": "quintile",
}
# Lowest-vs-highest contrasts use cohort quintiles for every continuous property.
CONTRAST_PROPERTIES = ("confidence", "norm_area", "elongation", "center_edge_dist_norm",
                       "a5_upscale", "a5_aspect_distortion", "touches_edge", "detections_in_image")
MEDIAN_PROPERTIES = ("confidence", "norm_area", "elongation", "center_edge_dist_norm",
                     "detections_in_image", "a5_upscale")
CONTINUOUS_PROPERTIES = ("confidence", "bbox_width", "bbox_height", "bbox_area", "norm_area",
                         "aspect_ratio", "elongation", "center_x_norm", "center_y_norm",
                         "edge_gap_norm", "center_edge_dist_norm", "detections_in_image",
                         "a5_crop_width", "a5_crop_height", "a5_upscale", "a5_aspect_distortion")

# Events: (name, source label required or None). "into_X" = changed into X among
# detections not already X; "a->b" = conditional on the source decision a.
GENERIC_EVENTS = ("changed", "toward_rejection", "toward_acceptance")
MODEL_EVENTS = {
    "qwen3_vl": ("R->U", "Ur->U", "R->Ur", "into_U"),
    "internvl3_5_hf": ("into_U", "R->U"),
    "qwen2_5_vl": ("into_Ur", "R->Ur", "U->Ur"),
    "glm_4_6v_flash": ("into_U", "into_Ur", "R->U", "R->Ur"),
    "gemma4": ("R->Ur", "Ur->R"),
    "phi4_multimodal": ("R->Ur", "Ur->R"),
}
SPARSE_EVENT_COUNT = 30
SPARSE_CONTRAST_AT_RISK = 50


# --------------------------------------------------------------------------- properties

def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def parent_frame(image_name: str) -> str:
    """Naming-derived parent prefix (drop the trailing _<k> patch suffix)."""
    return image_name.rsplit("_", 1)[0]


def build_property_table(index: pd.DataFrame, image_sizes: pd.DataFrame,
                         expected_n: int | None = EXPECTED_N) -> pd.DataFrame:
    """Derive detection properties from stored index fields and source-image sizes."""
    df = index[list(INDEX_COLUMNS)].merge(image_sizes, on="sample_id", how="left", validate="one_to_one")
    W, H = df["image_width"].astype(float), df["image_height"].astype(float)
    x, y, w, h = df["bbox_x"], df["bbox_y"], df["bbox_width"], df["bbox_height"]
    short_side = np.minimum(W, H)
    df["parent_frame"] = df["image_name"].map(parent_frame)
    df["norm_area"] = df["bbox_area"] / (W * H)
    df["aspect_ratio"] = w / h
    df["elongation"] = np.abs(np.log(df["aspect_ratio"]))
    df["center_x_norm"] = df["center_x"] / W
    df["center_y_norm"] = df["center_y"] / H
    edge_gap_px = np.minimum.reduce([x, y, W - (x + w), H - (y + h)])
    df["edge_gap_norm"] = edge_gap_px / short_side
    df["touches_edge"] = (edge_gap_px < 1.0).astype(int)
    df["center_edge_dist_norm"] = np.minimum.reduce(
        [df["center_x"], df["center_y"], W - df["center_x"], H - df["center_y"]]) / short_side
    df["detections_in_image"] = df.groupby("image_name")["sample_id"].transform("size")
    crops = [
        _padded_bbox((bx, by, bw, bh), int(iw), int(ih), A5_CROP_PADDING)
        for bx, by, bw, bh, iw, ih in zip(x, y, w, h, df["image_width"], df["image_height"])
    ]
    df["a5_crop_width"] = [x2 - x1 for x1, _, x2, _ in crops]
    df["a5_crop_height"] = [y2 - y1 for _, y1, _, y2 in crops]
    out_w, out_h = A5_OUTPUT_SIZE
    df["a5_upscale"] = np.sqrt(out_w * out_h / (df["a5_crop_width"] * df["a5_crop_height"]))
    df["a5_aspect_distortion"] = np.abs(np.log(
        (df["a5_crop_width"] / df["a5_crop_height"]) / (out_w / out_h)))
    validate_property_table(df, expected_n)
    return df.sort_values("sample_id").reset_index(drop=True)


def validate_property_table(df: pd.DataFrame, expected_n: int | None = EXPECTED_N) -> None:
    if expected_n is not None and len(df) != expected_n:
        raise AssertionError(f"Property table has {len(df)} rows, expected {expected_n}")
    if df["sample_id"].duplicated().any():
        raise AssertionError("Duplicate sample_id in property table")
    if df.isna().any().any():
        missing = df.columns[df.isna().any()].tolist()
        raise AssertionError(f"Missing property values in {missing}")
    W, H = df["image_width"], df["image_height"]
    checks = {
        "confidence in [0.5, 1]": df["confidence"].between(0.5, 1.0).all(),
        "positive box size": ((df["bbox_width"] > 0) & (df["bbox_height"] > 0)).all(),
        "area = width * height": np.allclose(df["bbox_area"], df["bbox_width"] * df["bbox_height"], rtol=1e-6),
        "center = x + w/2": np.allclose(df["center_x"], df["bbox_x"] + df["bbox_width"] / 2)
        and np.allclose(df["center_y"], df["bbox_y"] + df["bbox_height"] / 2),
        "box inside image": ((df["bbox_x"] >= 0) & (df["bbox_y"] >= 0)
                             & (df["bbox_x"] + df["bbox_width"] <= W + 1e-6)
                             & (df["bbox_y"] + df["bbox_height"] <= H + 1e-6)).all(),
        "norm_area in (0, 1]": ((df["norm_area"] > 0) & (df["norm_area"] <= 1)).all(),
        "center_x_norm in (0, 1)": ((df["center_x_norm"] > 0) & (df["center_x_norm"] < 1)).all(),
        "center_y_norm in (0, 1)": ((df["center_y_norm"] > 0) & (df["center_y_norm"] < 1)).all(),
        "edge_gap_norm >= 0": (df["edge_gap_norm"] >= -1e-9).all(),
        "center_edge_dist_norm in (0, 0.5]": ((df["center_edge_dist_norm"] > 0)
                                              & (df["center_edge_dist_norm"] <= 0.5)).all(),
        "positive A5 crop": ((df["a5_crop_width"] > 0) & (df["a5_crop_height"] > 0)).all(),
        "elongation >= 0": (df["elongation"] >= 0).all(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"Property range checks failed: {failed}")
    assert_no_forbidden_columns(df.columns)


def assert_no_forbidden_columns(columns) -> None:
    for column in columns:
        tokens = set(str(column).lower().replace("__", "_").replace("-", "_").split("_"))
        bad = tokens & FORBIDDEN_TOKENS
        if bad:
            raise AssertionError(f"Ground-truth-like column {column!r} ({sorted(bad)})")


def add_bins(props: pd.DataFrame) -> pd.DataFrame:
    """Add <prop>__bin (display bins) and <prop>__q (cohort quintiles, 1..5) columns."""
    df = props.copy()
    for prop, scheme in BINNED_PROPERTIES.items():
        if scheme == "quintile":
            df[f"{prop}__bin"] = pd.qcut(df[prop], 5, labels=[f"Q{i}" for i in range(1, 6)])
        elif scheme == "fixed_confidence":
            df[f"{prop}__bin"] = pd.cut(df[prop], CONFIDENCE_EDGES, right=False,
                                        labels=[f"[{a:.2f},{b:.2f})" for a, b in
                                                zip(CONFIDENCE_EDGES, CONFIDENCE_EDGES[1:])])
        elif scheme == "fixed_detections":
            df[f"{prop}__bin"] = pd.cut(df[prop], DETECTIONS_PER_IMAGE_EDGES,
                                        labels=[f"{a + 1}-{b}" for a, b in
                                                zip(DETECTIONS_PER_IMAGE_EDGES, DETECTIONS_PER_IMAGE_EDGES[1:])])
        elif scheme == "binary":
            df[f"{prop}__bin"] = pd.Categorical(np.where(df[prop] == 1, "yes", "no"), categories=["no", "yes"])
        if df[f"{prop}__bin"].isna().any():
            raise AssertionError(f"Unbinned values for {prop}")
    for prop in CONTRAST_PROPERTIES:
        if prop == "touches_edge":
            df[f"{prop}__q"] = df[prop].map({0: 1, 1: 5})
        elif prop == "detections_in_image":
            codes = df[f"{prop}__bin"].cat.codes
            df[f"{prop}__q"] = np.select([codes == 0, codes == codes.max()], [1, 5], 3)
        else:
            df[f"{prop}__q"] = pd.qcut(df[prop], 5, labels=False) + 1
    return df


CONTRAST_LABELS = {
    "touches_edge": ("not touching edge", "touching edge"),
    "detections_in_image": ("1-3 per image", "13-20 per image"),
}


# --------------------------------------------------------------------------- outcomes

def transition_code(src: str, dst: str) -> str:
    return f"{ABBREV[src]}->{ABBREV[dst]}"


def build_outcomes(decisions: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per detection x model x comparison."""
    frames = []
    for key in MODELS:
        for a, b in COMPARISONS:
            src, dst = decisions[f"{key}__{a}"], decisions[f"{key}__{b}"]
            rank_src, rank_dst = src.map(RANK), dst.map(RANK)
            frames.append(pd.DataFrame({
                "sample_id": decisions["sample_id"],
                "model_key": key,
                "comparison": f"{a}->{b}",
                "from_label": src.map(ABBREV),
                "to_label": dst.map(ABBREV),
                "transition": [transition_code(s, d) for s, d in zip(src, dst)],
                "direction": np.select([rank_dst > rank_src, rank_dst < rank_src],
                                       ["toward_rejection", "toward_acceptance"], "unchanged"),
            }))
    out = pd.concat(frames, ignore_index=True)
    out["changed"] = (out["direction"] != "unchanged").astype(int)
    return out


def validate_against_behavior(outcomes: pd.DataFrame, pairwise: pd.DataFrame,
                              expected_n: int = EXPECTED_N) -> None:
    """Transition counts must equal the stored A1-A5 behavior analysis."""
    for (key, comparison), group in outcomes.groupby(["model_key", "comparison"]):
        if len(group) != expected_n:
            raise AssertionError(f"{key} {comparison}: {len(group)} rows")
        ref = pairwise[(pairwise["model_key"] == key) & (pairwise["comparison"] == comparison)]
        if len(ref) != 1:
            raise AssertionError(f"No reference transition row for {key} {comparison}")
        ref = ref.iloc[0]
        counts = group["transition"].value_counts()
        for a in behavior.LABELS:
            for b in behavior.LABELS:
                if int(counts.get(transition_code(a, b), 0)) != int(ref[behavior.cell_name(a, b)]):
                    raise AssertionError(f"{key} {comparison} {a}->{b} differs from behavior analysis")
        for direction in ("toward_rejection", "toward_acceptance"):
            if int((group["direction"] == direction).sum()) != int(ref[direction]):
                raise AssertionError(f"{key} {comparison} {direction} differs from behavior analysis")


def event_masks(group: pd.DataFrame, event: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (denominator mask, event mask) for one model x comparison group."""
    if event in GENERIC_EVENTS:
        den = np.ones(len(group), dtype=bool)
        num = (group["direction"] != "unchanged").to_numpy() if event == "changed" else \
            (group["direction"] == event).to_numpy()
        return den, num
    if event.startswith("into_"):
        label = event.split("_", 1)[1]
        den = (group["from_label"] != label).to_numpy()
        return den, den & (group["to_label"] == label).to_numpy()
    src, dst = event.split("->")
    den = (group["from_label"] == src).to_numpy()
    return den, den & (group["to_label"] == dst).to_numpy()


def event_description(event: str) -> str:
    if event == "changed":
        return "decision changed"
    if event == "toward_rejection":
        return "shift toward rejection (R->U, R->Ur, U->Ur)"
    if event == "toward_acceptance":
        return "shift toward acceptance (Ur->U, Ur->R, U->R)"
    if event.startswith("into_"):
        label = event.split("_", 1)[1]
        return f"changed into {LABEL_FROM_ABBREV[label]} among detections not already {LABEL_FROM_ABBREV[label]}"
    src, dst = event.split("->")
    return f"{LABEL_FROM_ABBREV[src]} -> {LABEL_FROM_ABBREV[dst]} among source-{LABEL_FROM_ABBREV[src]} detections"


def events_for(key: str) -> tuple[str, ...]:
    return GENERIC_EVENTS + MODEL_EVENTS.get(key, ())


# --------------------------------------------------------------------------- bootstrap

def cluster_bootstrap_weights(n_clusters: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot x n_clusters) multiplicities: each resample draws n_clusters whole clusters with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n_boot)


def cluster_counts(codes: np.ndarray, mask: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.bincount(codes, weights=mask.astype(float), minlength=n_clusters)


def boot_ratio(weights: np.ndarray, num_c: np.ndarray, den_c: np.ndarray) -> np.ndarray:
    num, den = weights @ num_c, weights @ den_c
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size < 0.95 * samples.size:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def boot_weighted_median(values: np.ndarray, codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-resample median of `values`, each detection weighted by its cluster's multiplicity."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    w = weights[:, codes[order]].astype(float)
    cum = np.cumsum(w, axis=1)
    total = cum[:, -1]
    out = np.full(weights.shape[0], np.nan)
    ok = total > 0
    idx = (cum[ok] >= (total[ok] / 2.0)[:, None]).argmax(axis=1)
    out[ok] = sorted_values[idx]
    return out


# --------------------------------------------------------------------------- analysis

def property_summary(props: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prop in CONTINUOUS_PROPERTIES + ("touches_edge",):
        s = props[prop].astype(float)
        q = s.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).to_numpy()
        rows.append({"property": prop, "n": int(s.notna().sum()), "missing": int(s.isna().sum()),
                     "mean": s.mean(), "min": q[0], "p05": q[1], "p25": q[2], "median": q[3],
                     "p75": q[4], "p95": q[5], "max": q[6]})
    return pd.DataFrame(rows)


def clustering_summary(props: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, column in (("source_image", "image_name"), ("parent_frame", "parent_frame")):
        sizes = props.groupby(column).size()
        rows.append({"level": level, "column": column, "clusters": int(sizes.size),
                     "detections": int(sizes.sum()), "min": int(sizes.min()),
                     "median": float(sizes.median()), "mean": float(sizes.mean()),
                     "p90": float(sizes.quantile(0.9)), "max": int(sizes.max()),
                     "singleton_clusters": int((sizes == 1).sum())})
    return pd.DataFrame(rows)


def category_medians(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (key, comparison), group in data.groupby(["model_key", "comparison"], sort=False):
        for grouping in ("direction", "transition", "to_label"):
            for category, sub in group.groupby(grouping):
                for prop in MEDIAN_PROPERTIES:
                    s = sub[prop].astype(float)
                    rows.append({"model_key": key, "model": MODELS[key][0], "comparison": comparison,
                                 "grouping": grouping, "category": category, "n": len(sub),
                                 "property": prop, "median": s.median(),
                                 "q25": s.quantile(0.25), "q75": s.quantile(0.75)})
    return pd.DataFrame(rows)


def median_differences(data: pd.DataFrame, codes: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    """Median(property | shift category) - median(property | unchanged), image-clustered CI."""
    rows = []
    for (key, comparison), group in data.groupby(["model_key", "comparison"], sort=False):
        idx = group.index.to_numpy()
        direction = group["direction"].to_numpy()
        for prop in ("confidence", "norm_area"):
            values = group[prop].to_numpy(dtype=float)
            base_mask = direction == "unchanged"
            base_boot = boot_weighted_median(values[base_mask], codes[idx][base_mask], weights)
            for category in ("toward_rejection", "toward_acceptance"):
                mask = direction == category
                if mask.sum() < SPARSE_EVENT_COUNT:
                    continue
                cat_boot = boot_weighted_median(values[mask], codes[idx][mask], weights)
                lo, hi = percentile_ci(cat_boot - base_boot)
                rows.append({"model_key": key, "model": MODELS[key][0], "comparison": comparison,
                             "property": prop, "category": category, "n_category": int(mask.sum()),
                             "n_unchanged": int(base_mask.sum()),
                             "median_category": float(np.median(values[mask])),
                             "median_unchanged": float(np.median(values[base_mask])),
                             "difference": float(np.median(values[mask]) - np.median(values[base_mask])),
                             "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def event_tables(data: pd.DataFrame, codes: np.ndarray, weights: np.ndarray,
                 parent_codes: np.ndarray, parent_weights: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_img, n_par = weights.shape[1], parent_weights.shape[1]
    bin_rows, contrast_rows = [], []
    for (key, comparison), group in data.groupby(["model_key", "comparison"], sort=False):
        idx = group.index.to_numpy()
        g_codes, g_parent = codes[idx], parent_codes[idx]
        for event in events_for(key):
            den, num = event_masks(group, event)
            n_den, n_num = int(den.sum()), int(num.sum())
            base = {"model_key": key, "model": MODELS[key][0], "comparison": comparison, "event": event,
                    "event_description": event_description(event), "sparse": n_num < SPARSE_EVENT_COUNT}
            overall = boot_ratio(weights, cluster_counts(g_codes, num, n_img), cluster_counts(g_codes, den, n_img))
            lo, hi = percentile_ci(overall)
            contrast_rows.append({**base, "property": "(overall)", "n_denominator": n_den, "n_event": n_num,
                                  "rate": n_num / n_den if n_den else np.nan, "ci_low": lo, "ci_high": hi})
            for prop in BINNED_PROPERTIES:
                bins = group[f"{prop}__bin"]
                for order, label in enumerate(bins.cat.categories):
                    in_bin = (bins == label).to_numpy()
                    d, n = den & in_bin, num & in_bin
                    boot = boot_ratio(weights, cluster_counts(g_codes, n, n_img), cluster_counts(g_codes, d, n_img))
                    lo, hi = percentile_ci(boot)
                    bin_rows.append({**base, "property": prop, "bin_order": order, "bin": str(label),
                                     "n_denominator": int(d.sum()), "n_event": int(n.sum()),
                                     "rate": n.sum() / d.sum() if d.sum() else np.nan,
                                     "ci_low": lo, "ci_high": hi})
            for prop in CONTRAST_PROPERTIES:
                q = group[f"{prop}__q"].to_numpy()
                low, high = q == 1, q == 5
                d_lo, n_lo, d_hi, n_hi = den & low, num & low, den & high, num & high
                diff = (boot_ratio(weights, cluster_counts(g_codes, n_hi, n_img), cluster_counts(g_codes, d_hi, n_img))
                        - boot_ratio(weights, cluster_counts(g_codes, n_lo, n_img), cluster_counts(g_codes, d_lo, n_img)))
                diff_p = (boot_ratio(parent_weights, cluster_counts(g_parent, n_hi, n_par),
                                     cluster_counts(g_parent, d_hi, n_par))
                          - boot_ratio(parent_weights, cluster_counts(g_parent, n_lo, n_par),
                                       cluster_counts(g_parent, d_lo, n_par)))
                lo, hi = percentile_ci(diff)
                plo, phi = percentile_ci(diff_p)
                rate_lo = n_lo.sum() / d_lo.sum() if d_lo.sum() else np.nan
                rate_hi = n_hi.sum() / d_hi.sum() if d_hi.sum() else np.nan
                low_label, high_label = CONTRAST_LABELS.get(prop, ("Q1 (lowest)", "Q5 (highest)"))
                contrast_rows.append({**base, "property": prop, "low_group": low_label, "high_group": high_label,
                                      "n_denominator_low": int(d_lo.sum()), "n_event_low": int(n_lo.sum()),
                                      "n_denominator_high": int(d_hi.sum()), "n_event_high": int(n_hi.sum()),
                                      "rate_low": rate_lo, "rate_high": rate_hi, "difference": rate_hi - rate_lo,
                                      "ci_low": lo, "ci_high": hi,
                                      "ci_low_parent_frame": plo, "ci_high_parent_frame": phi,
                                      "sparse_contrast": bool(min(d_lo.sum(), d_hi.sum()) < SPARSE_CONTRAST_AT_RISK)})
    return pd.DataFrame(bin_rows), pd.DataFrame(contrast_rows)


TERTILE_LABELS = ["T1 (low)", "T2", "T3 (high)"]
# (property binned in quintiles, stratifier) pairs used to separate correlated properties.
STRATIFICATIONS = (("norm_area", "confidence"), ("confidence", "norm_area"), ("elongation", "touches_edge"))


def stratified_change(data: pd.DataFrame) -> pd.DataFrame:
    """A1->A5 changed rate by property quintile within strata of a correlated property."""
    sub = data[data["comparison"] == PRIMARY].copy()
    frames = []
    for prop, stratifier in STRATIFICATIONS:
        if stratifier == "touches_edge":
            stratum = sub["touches_edge__bin"].astype(str).map({"no": "not touching edge", "yes": "touching edge"})
        else:
            stratum = pd.qcut(sub[stratifier], 3, labels=TERTILE_LABELS).astype(str).map(
                lambda t, s=stratifier: f"{s} {t}")
        bins = pd.qcut(sub[prop], 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
        out = (sub.assign(stratum=stratum, property_bin=bins)
               .groupby(["model_key", "stratum", "property_bin"])["changed"]
               .agg(n="size", changed="sum").reset_index())
        out.insert(1, "property", prop)
        out.insert(2, "stratifier", stratifier)
        frames.append(out)
    out = pd.concat(frames, ignore_index=True)
    out["changed_rate"] = out["changed"] / out["n"]
    out.insert(1, "model", out["model_key"].map(lambda k: MODELS[k][0]))
    return out


# --------------------------------------------------------------------------- reporting

def _md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([header, sep, *("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))])


def _pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{100 * x:.1f}"


def render_summary(contrasts: pd.DataFrame, medians: pd.DataFrame, clusters: pd.DataFrame) -> str:
    parts = [
        "# Detection-property behavior analysis (ground-truth independent)",
        "Generated by `scripts/analysis/detection_property_behavior_analysis.py`. Associations between "
        "detection properties and VLM decision changes; they do not measure whether any decision is correct. "
        "Rates in %; 95% CIs from an image-clustered bootstrap "
        f"({N_BOOT} resamples of whole source images, seed {SEED}).",
        "## Source-image clustering", _md(clusters),
    ]
    for comparison in (PRIMARY, "A2->A5"):
        c = contrasts[(contrasts["comparison"] == comparison)]
        overall = c[c["property"] == "(overall)"]
        parts.append(f"## {comparison}: overall event rates")
        parts.append(_md(pd.DataFrame({
            "Model": overall["model"], "Event": overall["event"], "n at risk": overall["n_denominator"],
            "n event": overall["n_event"], "Rate %": overall["rate"].map(_pct),
            "95% CI": [f"{_pct(a)}–{_pct(b)}" for a, b in zip(overall["ci_low"], overall["ci_high"])],
        })))
        for prop in ("confidence", "norm_area", "elongation", "center_edge_dist_norm", "touches_edge"):
            sub = c[c["property"] == prop]
            parts.append(f"### {comparison}: highest vs lowest group, `{prop}`")
            parts.append(_md(pd.DataFrame({
                "Model": sub["model"], "Event": sub["event"],
                "Low group": sub["low_group"], "Rate low %": sub["rate_low"].map(_pct),
                "High group": sub["high_group"], "Rate high %": sub["rate_high"].map(_pct),
                "Diff (pp)": (100 * sub["difference"]).map("{:+.1f}".format),
                "95% CI image": [f"{100 * a:+.1f} to {100 * b:+.1f}" for a, b in zip(sub["ci_low"], sub["ci_high"])],
                "95% CI parent": [f"{100 * a:+.1f} to {100 * b:+.1f}"
                                  for a, b in zip(sub["ci_low_parent_frame"], sub["ci_high_parent_frame"])],
                "Sparse": sub["sparse_contrast"].map({True: "yes", False: ""}),
            })))
    m = medians[(medians["grouping"] == "direction") & medians["property"].isin(["confidence", "norm_area"])]
    wide = m.pivot_table(index=["model", "comparison", "category", "n"], columns="property", values="median",
                         sort=False).reset_index()
    wide["confidence"] = wide["confidence"].map("{:.3f}".format)
    wide["norm_area"] = (100 * wide["norm_area"]).map("{:.2f}%".format)
    parts.append("## Median property by direction category")
    parts.append(_md(wide))
    return "\n\n".join(parts) + "\n"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_safe_output_dir(out_dir: Path) -> None:
    behavior.assert_safe_output_dir(out_dir)
    resolved, protected = out_dir.resolve(), BEHAVIOR_DIR.resolve()
    if resolved == protected or protected in resolved.parents or resolved in protected.parents:
        raise SystemExit(f"Refusing to write into or above the behavior-analysis inputs: {BEHAVIOR_DIR}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.read_csv(VERIFICATION_DATASET_INDEX_CSV, usecols=list(INDEX_COLUMNS) + ["image_path"])
    sizes = [png_size(VERIFICATION_DATASET_DIR / p) for p in index["image_path"]]
    image_sizes = pd.DataFrame({"sample_id": index["sample_id"],
                                "image_width": [s[0] for s in sizes], "image_height": [s[1] for s in sizes]})
    decisions = pd.read_csv(PER_DETECTION_CSV, dtype=str)
    pairwise = pd.read_csv(PAIRWISE_CSV)
    return index.drop(columns="image_path"), image_sizes, decisions, pairwise


def validate_decisions(decisions: pd.DataFrame, cohort: list[str]) -> None:
    if len(decisions) != EXPECTED_N or decisions["sample_id"].duplicated().any():
        raise AssertionError("per_detection_decisions.csv is not a unique 5,747-row cohort")
    if sorted(decisions["sample_id"]) != cohort:
        raise AssertionError("per_detection_decisions.csv cohort differs from verification index")
    for key in MODELS:
        for code in behavior.CONDITIONS:
            if not decisions[f"{key}__{code}"].isin(behavior.LABELS).all():
                raise AssertionError(f"Invalid decision label in {key}__{code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    assert_safe_output_dir(args.out_dir)
    input_hashes_before = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in INPUT_FILES}

    index, image_sizes, decisions, pairwise = load_inputs()
    cohort = behavior.load_cohort(VERIFICATION_DATASET_INDEX_CSV)
    validate_decisions(decisions, cohort)
    props = build_property_table(index, image_sizes)
    if props["sample_id"].tolist() != cohort:
        raise AssertionError("Property table cohort differs from verification index")
    binned = add_bins(props)

    outcomes = build_outcomes(decisions)
    validate_against_behavior(outcomes, pairwise)
    data = outcomes.merge(binned, on="sample_id", how="left", validate="many_to_one")
    if len(data) != len(outcomes) or data[list(BINNED_PROPERTIES)].isna().any().any():
        raise AssertionError("Property join is incomplete")
    assert_no_forbidden_columns(data.columns)

    image_codes_all = pd.Categorical(props["image_name"])
    parent_codes_all = pd.Categorical(props["parent_frame"])
    code_of = dict(zip(props["sample_id"], image_codes_all.codes))
    parent_of = dict(zip(props["sample_id"], parent_codes_all.codes))
    codes = data["sample_id"].map(code_of).to_numpy()
    parent_codes = data["sample_id"].map(parent_of).to_numpy()
    weights = cluster_bootstrap_weights(len(image_codes_all.categories), args.n_boot, SEED)
    parent_weights = cluster_bootstrap_weights(len(parent_codes_all.categories), args.n_boot, SEED + 1)

    clusters = clustering_summary(props)
    bins_table, contrasts = event_tables(data, codes, weights, parent_codes, parent_weights)
    medians = category_medians(data)
    med_diff = median_differences(data, codes, weights)
    correlations = props[list(CONTINUOUS_PROPERTIES)].corr(method="spearman")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    props.to_csv(out / "property_table.csv", index=False)
    property_summary(props).to_csv(out / "property_summary.csv", index=False)
    correlations.to_csv(out / "property_correlations.csv")
    clusters.to_csv(out / "clustering_summary.csv", index=False)
    props.groupby("image_name").size().value_counts().sort_index().rename_axis("detections_per_image") \
        .rename("images").to_csv(out / "detections_per_image_distribution.csv")
    outcomes.to_csv(out / "detection_outcomes.csv", index=False)
    medians.to_csv(out / "category_property_medians.csv", index=False)
    med_diff.to_csv(out / "median_differences.csv", index=False)
    bins_table.to_csv(out / "event_rates_by_bin.csv", index=False)
    contrasts.to_csv(out / "event_contrasts.csv", index=False)
    stratified_change(data).to_csv(out / "stratified_change_rates.csv", index=False)
    (out / "summary.md").write_text(render_summary(contrasts, medians, clusters), encoding="utf-8")

    input_hashes_after = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in INPUT_FILES}
    if input_hashes_after != input_hashes_before:
        raise AssertionError("An input file changed during the analysis")
    info = {
        "description": "GT-independent detection-property behavior analysis; "
                       "see docs/DETECTION_PROPERTY_BEHAVIOR_ANALYSIS.md",
        "cohort_n": len(cohort),
        "comparisons": [f"{a}->{b}" for a, b in COMPARISONS],
        "primary_comparison": PRIMARY,
        "index_columns_read": list(INDEX_COLUMNS),
        "source_image_size": "PNG header of outputs/verification_dataset/images/<sample_id>.png",
        "a5_crop": {"padding_px": A5_CROP_PADDING, "output_size": list(A5_OUTPUT_SIZE)},
        "bootstrap": {"unit": "source image (image_name)", "n_resamples": args.n_boot, "seed": SEED,
                      "sensitivity_unit": "parent frame (image_name without trailing _<k>)",
                      "sensitivity_seed": SEED + 1, "ci": "percentile 2.5/97.5",
                      "n_source_images": int(len(image_codes_all.categories)),
                      "n_parent_frames": int(len(parent_codes_all.categories))},
        "confidence_bins": list(CONFIDENCE_EDGES),
        "quintile_edges": {p: [float(v) for v in props[p].quantile([0, .2, .4, .6, .8, 1]).round(6)]
                           for p in BINNED_PROPERTIES if BINNED_PROPERTIES[p] == "quintile"},
        "events": {k: list(events_for(k)) for k in MODELS},
        "input_sha256": input_hashes_before,
    }
    (out / "analysis_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
