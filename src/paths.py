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

# Qwen verification inference results
VERIFICATION_RESULTS_DIR = OUTPUTS_DIR / "verification_results"
VERIFICATION_ABLATION_RESULTS_DIR = OUTPUTS_DIR / "verification_ablation_results"
VERIFICATION_ABLATION_ANALYSIS_CSV = VERIFICATION_ABLATION_RESULTS_DIR / "ablation_summary.csv"
VERIFICATION_ABLATION_ANALYSIS_MD = VERIFICATION_ABLATION_RESULTS_DIR / "ablation_summary.md"

# Verification vs LabelMe GT evaluation
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
EVALUATION_SUMMARY_CSV = EVALUATION_DIR / "summary.csv"

# Publication visualization examples
VISUALIZATION_DIR = OUTPUTS_DIR / "visualization"

# Run logs (cluster and local)
LOGS_DIR = PROJECT_ROOT / "logs"
SLURM_LOG_DIR = LOGS_DIR / "slurm"

# Production Qwen ablation experiment layout
QWEN_VERIFICATION_ROOT = OUTPUTS_DIR / "verification" / "qwen"
QWEN_EVALUATION_ROOT = OUTPUTS_DIR / "evaluation" / "qwen"

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

ABLATION_CODE_TO_CONDITION = {
    code: name for code, name in zip(ABLATION_CODES, ABLATION_CONDITION_NAMES, strict=True)
}


def qwen_verification_condition_dir(
    condition_code: str,
    experiment_id: str | None = None,
) -> Path:
    """Return inference results directory for one ablation code (e.g. A1)."""
    root = QWEN_VERIFICATION_ROOT if not experiment_id else QWEN_VERIFICATION_ROOT / experiment_id
    return root / condition_code


def qwen_evaluation_condition_dir(
    condition_code: str,
    experiment_id: str | None = None,
) -> Path:
    """Return evaluation output directory for one ablation code (e.g. A1)."""
    root = QWEN_EVALUATION_ROOT if not experiment_id else QWEN_EVALUATION_ROOT / experiment_id
    return root / condition_code


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
