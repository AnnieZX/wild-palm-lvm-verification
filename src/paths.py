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
