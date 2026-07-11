"""Central path constants for data roots and output directories."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Cluster data roots
# ---------------------------------------------------------------------------
RAW_PATCHES_ROOT = Path("/deac/csc/yangGrp/cuij/palm/Raw_Patches")
YOLO_MODEL_PATH = Path(
    "/deac/csc/yangGrp/cuij/palm/training/yolonew/results/yolo11x_palm_new/weights/best.pt"
)
QWEN_MODEL_PATH = Path("/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct")

# ---------------------------------------------------------------------------
# outputs/ layout
# ---------------------------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# YOLO inference
FULL_INFERENCE_DIR = OUTPUTS_DIR / "full_inference"
PREDICTIONS_FULL_JSON = FULL_INFERENCE_DIR / "predictions_full.json"
FULL_INFERENCE_OVERLAYS_DIR = FULL_INFERENCE_DIR / "overlays"

# YOLO vs LabelMe analysis
YOLO_GT_OVERLAP_DIR = OUTPUTS_DIR / "yolo_gt_overlap_full"
YOLO_ANALYSIS_DIR = OUTPUTS_DIR / "yolo_analysis"
PREDICTION_STATISTICS_CSV = YOLO_ANALYSIS_DIR / "prediction_statistics.csv"
GT_MATCHES_CSV = YOLO_ANALYSIS_DIR / "gt_matches.csv"

# Verification ablation (100 YOLO detections × input/prompt conditions)
VERIFICATION_ABLATION_DIR = OUTPUTS_DIR / "verification_ablation_100"
VERIFICATION_ABLATION_SUMMARY_CSV = VERIFICATION_ABLATION_DIR / "ablation_prompt_summary.csv"

# YOLO → LVM verification dataset (one sample per detection)
VERIFICATION_DATASET_DIR = OUTPUTS_DIR / "verification_dataset"
VERIFICATION_DATASET_IMAGES_DIR = VERIFICATION_DATASET_DIR / "images"
VERIFICATION_DATASET_METADATA_DIR = VERIFICATION_DATASET_DIR / "metadata"
VERIFICATION_DATASET_INDEX_CSV = VERIFICATION_DATASET_DIR / "index.csv"
VERIFICATION_PROMPTS_DIR = VERIFICATION_DATASET_DIR / "prompts"
VERIFICATION_PROMPT_INDEX_CSV = VERIFICATION_DATASET_DIR / "prompt_index.csv"

# Legacy flat verification results (non-experiment runs)
VERIFICATION_RESULTS_DIR = OUTPUTS_DIR / "verification_results"
VERIFICATION_ABLATION_RESULTS_DIR = OUTPUTS_DIR / "verification_ablation_results"
VERIFICATION_ABLATION_ANALYSIS_CSV = VERIFICATION_ABLATION_RESULTS_DIR / "ablation_summary.csv"
VERIFICATION_ABLATION_ANALYSIS_MD = VERIFICATION_ABLATION_RESULTS_DIR / "ablation_summary.md"

# Verification vs LabelMe GT evaluation (legacy flat layout)
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
EVALUATION_SUMMARY_CSV = EVALUATION_DIR / "summary.csv"

# Publication visualization
VISUALIZATION_DIR = OUTPUTS_DIR / "visualization"

# Run logs (cluster and local)
LOGS_DIR = PROJECT_ROOT / "logs"
SLURM_LOG_DIR = LOGS_DIR / "slurm"

# Ablation condition names (full folder names under ablation inputs root)
ABLATION_CONDITION_NAMES = (
    "A1_overlay_only",
    "A2_overlay_confidence",
    "A3_overlay_confidence_geometry",
    "A4_overlay_crop_confidence",
    "A5_crop_only",
)

# Short codes used in production output folders (A1 … A5)
ABLATION_CODES = tuple(name.split("_", 1)[0] for name in ABLATION_CONDITION_NAMES)

assert len(ABLATION_CODES) == len(ABLATION_CONDITION_NAMES), \
    "zip inputs have different lengths"
ABLATION_CODE_TO_CONDITION = {
    code: name for code, name in zip(ABLATION_CODES, ABLATION_CONDITION_NAMES)
}

# Legacy Qwen output roots (pre-framework-freeze experiments under verification/qwen/)
LEGACY_QWEN_MODEL_KEY = "qwen"
QWEN_VERIFICATION_ROOT = OUTPUTS_DIR / "verification" / LEGACY_QWEN_MODEL_KEY
QWEN_EVALUATION_ROOT = OUTPUTS_DIR / "evaluation" / LEGACY_QWEN_MODEL_KEY


def _canonical_model_key(model_key: str) -> str:
    from src.config.model_config import normalize_model_key

    return normalize_model_key(model_key)


def verification_root(model_key: str) -> Path:
    """Return outputs/verification/<registry_key>/ for a model."""
    return OUTPUTS_DIR / "verification" / _canonical_model_key(model_key)


def evaluation_root(model_key: str) -> Path:
    """Return outputs/evaluation/<registry_key>/ for a model."""
    return OUTPUTS_DIR / "evaluation" / _canonical_model_key(model_key)


def _legacy_verification_experiment_dir(experiment_id: str) -> Path:
    return QWEN_VERIFICATION_ROOT / experiment_id


def _legacy_evaluation_experiment_dir(experiment_id: str) -> Path:
    return QWEN_EVALUATION_ROOT / experiment_id


def verification_experiment_dir(model_key: str, experiment_id: str) -> Path:
    """
    Resolve the experiment directory for verification outputs.

    Prefers outputs/verification/<canonical_key>/<experiment_id>.
    Falls back to legacy outputs/verification/qwen/<experiment_id> when the
    canonical key is qwen2_5_vl and only the legacy path exists.
    """
    canonical = _canonical_model_key(model_key)
    preferred = verification_root(model_key) / experiment_id
    if preferred.exists():
        return preferred
    if canonical == "qwen2_5_vl":
        legacy = _legacy_verification_experiment_dir(experiment_id)
        if legacy.exists():
            return legacy
    return preferred


def evaluation_experiment_dir(model_key: str, experiment_id: str) -> Path:
    """
    Resolve the experiment directory for evaluation outputs.

    Same legacy fallback rules as verification_experiment_dir().
    """
    canonical = _canonical_model_key(model_key)
    preferred = evaluation_root(model_key) / experiment_id
    if preferred.exists():
        return preferred
    if canonical == "qwen2_5_vl":
        legacy = _legacy_evaluation_experiment_dir(experiment_id)
        if legacy.exists():
            return legacy
    return preferred


def verification_condition_dir(
    model_key: str,
    condition_code: str,
    experiment_id: str | None = None,
) -> Path:
    """Return inference results directory for one ablation code (e.g. A1)."""
    if experiment_id:
        return verification_experiment_dir(model_key, experiment_id) / condition_code
    return verification_root(model_key) / condition_code


def evaluation_condition_dir(
    model_key: str,
    condition_code: str,
    experiment_id: str | None = None,
) -> Path:
    """Return evaluation output directory for one ablation code (e.g. A1)."""
    if experiment_id:
        return evaluation_experiment_dir(model_key, experiment_id) / condition_code
    return evaluation_root(model_key) / condition_code


def experiment_visualization_dir(model_key: str, experiment_id: str) -> Path:
    """Return visualization output root for one model experiment run."""
    canonical = _canonical_model_key(model_key)
    preferred = VISUALIZATION_DIR / canonical / experiment_id
    legacy_flat = VISUALIZATION_DIR / experiment_id
    if legacy_flat.exists() and not preferred.exists():
        return legacy_flat
    return preferred


def discover_ablation_inputs_dir(sample_size: int | None = None) -> Path:
    """
    Locate ablation prompt/image inputs.

    Prefers outputs/verification_ablation_{sample_size} when sample_size is set,
    otherwise falls back to VERIFICATION_ABLATION_DIR.
    """
    if sample_size is not None:
        candidate = OUTPUTS_DIR / f"verification_ablation_{sample_size}"
        if (candidate / "A1_overlay_only" / "prompt_index.csv").exists():
            return candidate
    if (VERIFICATION_ABLATION_DIR / "A1_overlay_only" / "prompt_index.csv").exists():
        return VERIFICATION_ABLATION_DIR
    matches = sorted(OUTPUTS_DIR.glob("verification_ablation_*"))
    for path in reversed(matches):
        if (path / "A1_overlay_only" / "prompt_index.csv").exists():
            return path
    return VERIFICATION_ABLATION_DIR


def resolve_ablation_condition_name(condition: str) -> str:
    """Map A1 or A1_overlay_only to the full ablation condition folder name."""
    if condition in ABLATION_CONDITION_NAMES:
        return condition
    if condition in ABLATION_CODE_TO_CONDITION:
        return ABLATION_CODE_TO_CONDITION[condition]
    allowed = ", ".join([*ABLATION_CODES, *ABLATION_CONDITION_NAMES])
    raise ValueError(f"Unknown ablation condition {condition!r}. Expected one of: {allowed}")


def ablation_code_from_condition(condition: str) -> str:
    """Return short code (A1) for a full or short condition name."""
    return resolve_ablation_condition_name(condition).split("_", 1)[0]


# ---------------------------------------------------------------------------
# Deprecated aliases — prefer verification_root() / verification_condition_dir()
# ---------------------------------------------------------------------------


def qwen_experiment_visualization_dir(experiment_id: str) -> Path:
    """Deprecated: use experiment_visualization_dir('qwen2_5_vl', experiment_id)."""
    legacy = VISUALIZATION_DIR / experiment_id
    if legacy.exists():
        return legacy
    return experiment_visualization_dir("qwen2_5_vl", experiment_id)


def qwen_verification_condition_dir(
    condition_code: str,
    experiment_id: str | None = None,
) -> Path:
    """Deprecated: use verification_condition_dir('qwen2_5_vl', ...)."""
    return verification_condition_dir("qwen2_5_vl", condition_code, experiment_id)


def qwen_evaluation_condition_dir(
    condition_code: str,
    experiment_id: str | None = None,
) -> Path:
    """Deprecated: use evaluation_condition_dir('qwen2_5_vl', ...)."""
    return evaluation_condition_dir("qwen2_5_vl", condition_code, experiment_id)
