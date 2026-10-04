"""Shared constants and helpers for the thesis analysis layer.

Read-only with respect to the repository: every function here reads from
outputs/ or the checkpoint directories and writes only under paper/analysis/.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ANALYSIS = REPO / "paper" / "analysis"
OUT = ANALYSIS / "outputs"
TABLES = ANALYSIS / "tables"
FIGURES = ANALYSIS / "figures"
DERIVED = ANALYSIS / "derived"

VERIFICATION_ROOT = REPO / "outputs" / "verification"
EVAL_V2_ROOT = REPO / "outputs" / "evaluation_protocol_v2"
INDEX_CSV = REPO / "outputs" / "verification_dataset" / "index.csv"
ABLATION_ROOT = REPO / "outputs" / "verification_ablation_5747"
HUMAN_REVIEW_CSV = REPO / "outputs" / "semantic_gt_review" / "human_review.csv"
HUMAN_REVIEW_LOG = REPO / "outputs" / "semantic_gt_review" / "human_review.log.jsonl"
PILOT_CSV = REPO / "outputs" / "semantic_gt_review" / "human_confidence_pilot.csv"
PILOT_REFERENCE_CSV = REPO / "outputs" / "semantic_candidate_audit" / "semantic_pilot_reference.csv"
IOU_SENS_DIR = REPO / "outputs" / "semantic_gt_evaluation" / "iou_sensitivity"
MODELS_ROOT = Path("/deac/csc/yangGrp/luoz23/models")

N_COHORT = 5747
CONDITIONS = ("A1", "A2", "A3", "A4", "A5")
ABLATION_DIRS = {
    "A1": "A1_overlay_only",
    "A2": "A2_overlay_confidence",
    "A3": "A3_overlay_confidence_geometry",
    "A4": "A4_overlay_crop_confidence",
    "A5": "A5_crop_only",
}
LABELS = ("Reliable", "Uncertain", "Unreliable")
LABEL_SHORT = {"Reliable": "R", "Uncertain": "U", "Unreliable": "Ur"}

BOOTSTRAP_SEED = 20261002
BOOTSTRAP_REPLICATES = 2000
CASE_SELECTION_SEED = 20261002

# Condition pairs, grouped by what the executed prompts/images actually change
# (verified by diffing outputs/verification_ablation_5747/*/prompts/*.txt and
# the image_path column of each prompt_index.csv).
PAIR_GROUPS = {
    "text_only": [("A1", "A2"), ("A2", "A3"), ("A1", "A3")],
    "image_representation": [("A2", "A4"), ("A2", "A5"), ("A4", "A5")],
    "mixed_image_and_metadata": [("A1", "A4"), ("A1", "A5"), ("A3", "A4"), ("A3", "A5")],
}
PAIR_GROUP_NOTES = {
    "text_only": (
        "Identical image file; prompt differs in the metadata block, the "
        "metadata instruction lines and the condition label."
    ),
    "image_representation": (
        "Identical YOLO-confidence text; image differs. The 'Input image' "
        "description paragraph and the condition label also differ, and "
        "A2->A4/A5 change one metadata-instruction line. Not image-only."
    ),
    "mixed_image_and_metadata": (
        "Image and metadata text both change; effects cannot be attributed "
        "to either factor."
    ),
}


def pair_group(a: str, b: str) -> str:
    for group, pairs in PAIR_GROUPS.items():
        if (a, b) in pairs:
            return group
    raise KeyError((a, b))


# ---------------------------------------------------------------------------
# Run registry. Paths verified on disk 2026-10-04.
#   expected: free-text run state at registry time (informational only; the audit in
#             a00 derives the actual per-condition status from the files).
#   nominal_b: size in the checkpoint name (billions), None if the name states none.
#   checkpoint: local checkpoint directory used for parameter counting.
# ---------------------------------------------------------------------------
def _run(model_key, display, family, nominal_size, nominal_b, rel, checkpoint, expected, eval_key=None):
    return dict(model_key=model_key, eval_key=eval_key or model_key, display=display, family=family,
                nominal_size=nominal_size, nominal_b=nominal_b, pred_rel=rel, eval_rel=rel,
                checkpoint=checkpoint, expected=expected)


RUNS = [
    _run("qwen2_5_vl", "Qwen2.5-VL-7B", "Qwen2.5-VL", "7B", 7, "qwen/20260708_0020",
         "Qwen2.5-VL-7B-Instruct", "complete", eval_key="qwen"),
    _run("qwen3_vl", "Qwen3-VL-8B", "Qwen3-VL", "8B", 8, "qwen3_vl/qwen3vl_A1A5_5747",
         "Qwen3-VL-8B-Instruct", "complete"),
    _run("qwen3_vl_4b", "Qwen3-VL-4B", "Qwen3-VL", "4B", 4, "qwen3_vl_4b/20261001_qwen3_vl_4b_A1A5_5747",
         "Qwen3-VL-4B-Instruct", "complete"),
    _run("internvl3_5_hf", "InternVL3.5-8B", "InternVL3.5", "8B", 8,
         "internvl3_5_hf/20260924_internvl3_5_hf_A1A5_5747", "InternVL3_5-8B-HF", "complete"),
    _run("glm_4_6v_flash", "GLM-4.6V-Flash", "GLM-4.6V", "not stated", None,
         "glm_4_6v_flash/20260919_glm46v_flash_A1A5_5747", "GLM-4.6V-Flash", "complete"),
    _run("phi4_multimodal", "Phi-4-multimodal", "Phi-4", "not stated", None,
         "phi4_multimodal/20260921_phi4_A1A5_5747", "Phi-4-multimodal-instruct", "complete"),
    _run("gemma4", "Gemma-4-12B", "Gemma 4", "12B", 12, "gemma4/20260925_gemma4_A1A5_5747",
         "gemma-4-12B-it", "complete"),
    _run("minicpm_v4_5", "MiniCPM-V-4.5", "MiniCPM-V", "not stated", None,
         "minicpm_v4_5/20260923_minicpm_A1A5_5747", "MiniCPM-V-4_5",
         "A1-A3 complete; A4/A5 queued (Slurm 8365886-87)"),
    _run("molmo2_8b", "Molmo2-8B", "Molmo2", "8B", 8, "molmo2_8b/20260923_molmo2_A1A5_5747", "Molmo2-8B",
         "A1 complete; A2-A5 queued (Slurm 8365888-91)"),
    _run("qwen3_vl_2b", "Qwen3-VL-2B", "Qwen3-VL", "2B", 2, "qwen3_vl_2b/20261001_qwen3_vl_2b_A1A5_5747",
         "Qwen3-VL-2B-Instruct", "complete; A3 and A5 each contain one parse error (see CONDITION_POLICY)"),
    _run("qwen2_5_vl_3b", "Qwen2.5-VL-3B", "Qwen2.5-VL", "3B", 3,
         "qwen2_5_vl_3b/20261002_qwen2_5_vl_3b_A1A5_5747", "Qwen2.5-VL-3B-Instruct", "complete"),
    _run("internvl3_5_hf_2b", "InternVL3.5-2B", "InternVL3.5", "2B", 2,
         "internvl3_5_hf_2b/20261001_internvl3_5_hf_2b_A1A5_5747", "InternVL3_5-2B-HF",
         "complete; parse errors in every condition (see CONDITION_POLICY)"),
    _run("internvl3_5_hf_4b", "InternVL3.5-4B", "InternVL3.5", "4B", 4,
         "internvl3_5_hf_4b/20261001_internvl3_5_hf_4b_A1A5_5747", "InternVL3_5-4B-HF", "complete"),
    _run("internvl3_5_hf_14b", "InternVL3.5-14B", "InternVL3.5", "14B", 14,
         "internvl3_5_hf_14b/20261001_internvl3_5_hf_14b_A1A5_5747", "InternVL3_5-14B-HF", "complete"),
    _run("qwen3_vl_32b", "Qwen3-VL-32B", "Qwen3-VL", "32B", 32,
         "qwen3_vl_32b/20261004_qwen3_vl_32b_A1A5_5747", "Qwen3-VL-32B-Instruct",
         "submitted 2026-10-04 (Slurm 8370171-75), pending"),
]
RUN_BY_KEY = {r["model_key"]: r for r in RUNS}
MODEL_ORDER = [r["display"] for r in RUNS]
DISPLAY_TO_KEY = {r["display"]: r["model_key"] for r in RUNS}

# The seven models that formed the original complete A1-A5 panel. Thesis tables that were
# built on this panel (agreement summary, case-selection votes) stay on it so their values
# remain comparable; extended checkpoints are reported in separate tables/CSVs.
CORE_PANEL = ["Qwen2.5-VL-7B", "Qwen3-VL-8B", "Qwen3-VL-4B", "InternVL3.5-8B", "GLM-4.6V-Flash",
              "Phi-4-multimodal", "Gemma-4-12B"]

# Within-family size ladders (model_key order = nominal size). Entries not in RUNS are
# listed so that tables can state "not run" explicitly.
LADDERS = {
    "Qwen3-VL": ["qwen3_vl_2b", "qwen3_vl_4b", "qwen3_vl", "qwen3_vl_32b"],
    "InternVL3.5": ["internvl3_5_hf_2b", "internvl3_5_hf_4b", "internvl3_5_hf", "internvl3_5_hf_14b"],
    "Qwen2.5-VL": ["qwen2_5_vl_3b", "qwen2_5_vl", "qwen2_5_vl_32b"],
}
NOT_RUN = {"qwen2_5_vl_32b": ("Qwen2.5-VL-32B", 32, "qualification probe only; full cohort not run")}

# ---------------------------------------------------------------------------
# Condition-status policy (applied by a00_audit.py; recorded in the audit outputs).
#   full_clean  all 5,747 prediction files present and every integrity check passes.
#   partial     all 5,747 files present; the ONLY defects are parse errors: each parse-error
#               record has an empty decision, results_index marks exactly those IDs
#               "parse_error", the Protocol-v2 CSV leaves exactly those labels empty, and the
#               stored metrics equal a recomputation over the remaining valid rows.
#               Analyses use the valid rows only and report N; nothing is repaired,
#               re-parsed or imputed, and the frozen protocol is unchanged.
#   failed      any other defect. incomplete / not_run: fewer files / no directory.
# Cells that are "usable" = full_clean or partial. Analyses that require identical N across
# models (cross-model agreement, case selection, semantic tables, IoU sensitivity) use
# full_clean cells only. Paired analyses (transitions, bootstrap contrasts, detection
# difficulty, size ladders) use detections valid in BOTH conditions and report that N.
# Degeneracy (single-label output) is a property derived from the data in
# a10_label_usage_collapse.py, not a registry flag.
# ---------------------------------------------------------------------------
CONDITION_POLICY = {
    "qwen3_vl_2b": "A3 and A5 each have one parse error (512-token repetition loop; "
                   "sample_000061 in A3, sample_005495 in A5). Those conditions are partial at "
                   "N=5,746; the other three are full_clean. The model is not re-run.",
    "internvl3_5_hf_2b": "Every condition has parse errors (malformed JSON). All five conditions "
                         "are partial; only parseable rows are analysed. Every parseable decision is "
                         "Uncertain, so the checkpoint is degenerate; no repair, no re-parsing, no "
                         "re-run, no protocol change.",
}

# Runs on disk that are deliberately NOT analysed, with the reason.
KNOWN_EXCLUSIONS = {
    "qwen2_5_vl/20260708_0020/A5": "cancelled wrong-path resume job 8318954 (partial A5, no index); "
                                   "documented in docs/A1_A5_BEHAVIOR_ANALYSIS.md §2",
}


def ensure_dirs() -> None:
    for d in (OUT, TABLES, FIGURES, DERIVED):
        d.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_safetensors_parameters(checkpoint_dir: Path) -> int | None:
    """Exact parameter count from safetensors headers (sum of tensor numel)."""
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        return None
    total = 0
    for path in files:
        with path.open("rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            total += int(np.prod(meta["shape"])) if meta["shape"] else 1
    return total


# ---------------------------------------------------------------------------
# Derived sample-level table
# ---------------------------------------------------------------------------
SAMPLE_TABLE = DERIVED / "sample_decisions.csv.gz"


def load_samples(analysed_only: bool = True, include_partial: bool = False) -> pd.DataFrame:
    """Load the derived sample-level table written by a00_audit.py.

    The table holds valid-decision rows of usable cells only (parse-error rows are absent).
    analysed_only=True  -> full_clean cells (every cell has N = 5,747).
    include_partial=True -> full_clean + partial cells (N varies; see condition_status).
    """
    if not SAMPLE_TABLE.is_file():
        raise SystemExit("Run a00_audit.py first (derived sample table missing).")
    df = pd.read_csv(SAMPLE_TABLE, dtype={"sample_id": str, "image_id": str})
    if include_partial:
        return df[df["usable"]].copy()
    if analysed_only:
        df = df[df["analysed"]].copy()
    return df


def complete_models(df: pd.DataFrame) -> list[str]:
    """Display names of models with all five conditions present in df (registry order)."""
    have = df.groupby("display")["condition"].nunique()
    return [m for m in MODEL_ORDER if have.get(m, 0) == 5]


def cell_status(df: pd.DataFrame) -> dict:
    """(display, condition) -> condition_status for the cells in df."""
    s = df.drop_duplicates(["display", "condition"])
    return {(r.display, r.condition): r.condition_status for r in s.itertuples()}


def paired(w: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Rows of a (sample_id x condition) decision pivot valid in both a and b."""
    return w[[a, b]].dropna()


def image_weights(rng, n_img: int, b: int, chunk: int = 250):
    """Image-cluster bootstrap weights: B multinomial draws of n_img images, in chunks.

    Every script that bootstraps uses this generator with BOOTSTRAP_SEED and the same
    sorted image list, so replicate r draws the same images in every analysis.
    """
    for start in range(0, b, chunk):
        k = min(chunk, b - start)
        idx = rng.integers(0, n_img, size=(k, n_img))
        flat = (np.arange(k)[:, None] * n_img + idx).ravel()
        yield np.bincount(flat, minlength=k * n_img).reshape(k, n_img).astype(np.float64)


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, n_img: int) -> np.ndarray:
    """Replicates of sum(num)/sum(den) for per-image count columns (n_img x k).

    Returns a (B x k) array; NaN where the replicate denominator is zero.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = []
    for wts in image_weights(rng, n_img, BOOTSTRAP_REPLICATES):
        n_ = wts @ num
        d_ = wts @ den
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(np.where(d_ > 0, n_ / d_, np.nan))
    return np.vstack(out)


def percentile_ci(reps: np.ndarray) -> tuple[float, float, int]:
    ok = reps[~np.isnan(reps)]
    if ok.size == 0:
        return float("nan"), float("nan"), int(reps.size)
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return float(lo), float(hi), int(reps.size - ok.size)


# ---------------------------------------------------------------------------
# Detection properties (YOLO confidence, normalised box area) with cohort quintiles
# ---------------------------------------------------------------------------
def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    w, h = struct.unpack(">II", header[16:24])
    return int(w), int(h)


def detection_properties() -> pd.DataFrame:
    """sample_id -> confidence, norm_area and their cohort quintiles (Q1 = lowest).

    Area is normalised by the source-image size read from each PNG header. Quintile edges
    are computed once on the 5,747-detection cohort and are identical for every model.
    """
    from concurrent.futures import ThreadPoolExecutor
    idx = pd.read_csv(INDEX_CSV, dtype={"sample_id": str, "image_name": str})
    root = INDEX_CSV.parent
    with ThreadPoolExecutor(max_workers=16) as pool:
        sizes = list(pool.map(lambda p: _png_size(root / p), idx["image_path"]))
    idx["image_width"] = [s[0] for s in sizes]
    idx["image_height"] = [s[1] for s in sizes]
    idx["norm_area"] = idx["bbox_area"] / (idx["image_width"] * idx["image_height"])
    labels = [f"Q{i}" for i in range(1, 6)]
    idx["confidence_q"] = pd.qcut(idx["confidence"], 5, labels=labels).astype(str)
    idx["area_q"] = pd.qcut(idx["norm_area"], 5, labels=labels).astype(str)
    return idx.set_index("sample_id")[["image_name", "confidence", "norm_area", "image_width", "image_height",
                                       "confidence_q", "area_q"]]


# ---------------------------------------------------------------------------
# Metrics (Protocol v2 alignment; Uncertain excluded from binary metrics)
# ---------------------------------------------------------------------------
def _safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def metrics_from_counts(c: dict) -> dict:
    """c has keys R_m, R_u, U_m, U_u, Ur_m, Ur_u (m = LabelMe-matched, u = unmatched).

    Works elementwise on numpy arrays (used by the bootstrap).
    """
    tp, fp, fn, tn = c["R_m"], c["R_u"], c["Ur_m"], c["Ur_u"]
    n_r = c["R_m"] + c["R_u"]
    n_u = c["U_m"] + c["U_u"]
    n_ur = c["Ur_m"] + c["Ur_u"]
    n = n_r + n_u + n_ur
    decided = tp + fp + fn + tn
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    return {
        "n": n,
        "reliable_rate": _safe_div(n_r, n),
        "uncertain_rate": _safe_div(n_u, n),
        "unreliable_rate": _safe_div(n_ur, n),
        "coverage": _safe_div(decided, n),
        "abstention_rate": _safe_div(n_u, n),
        "accuracy": _safe_div(tp + tn, decided),
        "precision": _safe_div(tp, tp + fp),
        "sensitivity": sens,
        "specificity": spec,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sens + spec) / 2.0,
    }


CELL_KEYS = ("R_m", "R_u", "U_m", "U_u", "Ur_m", "Ur_u")


def cell_counts(sub: pd.DataFrame) -> dict:
    out = {}
    for lab in LABELS:
        s = LABEL_SHORT[lab]
        out[f"{s}_m"] = int(((sub["decision"] == lab) & sub["matched_gt"]).sum())
        out[f"{s}_u"] = int(((sub["decision"] == lab) & ~sub["matched_gt"]).sum())
    return out


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------
def cohen_kappa(a: np.ndarray, b: np.ndarray, labels=LABELS) -> tuple[float, float, float]:
    """Return (observed agreement, expected agreement, kappa) for two raters."""
    n = len(a)
    po = float(np.mean(a == b))
    pe = 0.0
    for lab in labels:
        pe += float(np.mean(a == lab)) * float(np.mean(b == lab))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, pe, kappa


def fleiss_kappa(matrix: np.ndarray) -> float:
    """matrix: items x categories counts, each row summing to the number of raters."""
    n_items, _ = matrix.shape
    n_raters = matrix[0].sum()
    p_j = matrix.sum(axis=0) / (n_items * n_raters)
    p_i = (np.sum(matrix * matrix, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e = np.sum(p_j * p_j)
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def shannon_entropy(counts) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def stable_hash_rank(seed: int, category: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{category}:{sample_id}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# LaTeX table writer (booktabs, no external template dependency)
# ---------------------------------------------------------------------------
def latex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s.replace("->", r"$\rightarrow$")


def fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", "{,}")
    return f"{float(x):.{digits}f}"


def write_latex_table(path: Path, df: pd.DataFrame, caption: str, label: str,
                      colspec: str | None = None, notes: str | None = None,
                      size: str = r"\scriptsize", group_col: str | None = None,
                      escape_cells: bool = True) -> None:
    cols = list(df.columns)
    colspec = colspec or ("l" * 1 + "r" * (len(cols) - 1))
    lines = [
        "% AUTO-GENERATED by paper/analysis/scripts — do not edit by hand.",
        r"\begin{table}[t]", r"\centering", size,
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", r"\toprule",
        " & ".join(rf"\textbf{{{latex_escape(c)}}}" for c in cols) + r" \\", r"\midrule",
    ]
    prev = None
    for _, row in df.iterrows():
        if group_col is not None and prev is not None and row[group_col] != prev:
            lines.append(r"\midrule")
        prev = row[group_col] if group_col is not None else None
        cells = [latex_escape(v) if escape_cells else str(v) for v in row.tolist()]
        # a row starting with "[" would be read as the optional argument of the preceding \\
        cells = ["{}" + c if c.startswith("[") else c for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize {notes}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
