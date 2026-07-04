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

# LVM ablation (100 palms)
ABLATION_INPUTS_DIR = OUTPUTS_DIR / "ablation_inputs_100"
ABLATION_METADATA_CSV = OUTPUTS_DIR / "ablation_metadata_100.csv"
ABLATION_RESULTS_DIR = OUTPUTS_DIR / "ablation_results_100"
ABLATION_RAW_RESPONSES_DIR = OUTPUTS_DIR / "ablation_raw_responses_100"
ABLATION_COMBINED_CSV = OUTPUTS_DIR / "ablation_results_100_combined.csv"
ABLATION_SUMMARY_CSV = OUTPUTS_DIR / "ablation_summary_100.csv"
ABLATION_ENDPOINT_SUMMARY_CSV = OUTPUTS_DIR / "ablation_endpoint_summary_100.csv"
ABLATION_ANCHORING_SUMMARY_CSV = OUTPUTS_DIR / "ablation_anchoring_summary_100.csv"

# Ablation smoke test
SMOKE_TEST_RESULTS_DIR = OUTPUTS_DIR / "smoke_test_results"
SMOKE_TEST_RAW_RESPONSES_DIR = OUTPUTS_DIR / "smoke_test_raw_responses"
SMOKE_TEST_COMBINED_CSV = OUTPUTS_DIR / "smoke_test_combined.csv"

# YOLO → LVM verification dataset (one sample per detection)
VERIFICATION_DATASET_DIR = OUTPUTS_DIR / "verification_dataset"
VERIFICATION_DATASET_IMAGES_DIR = VERIFICATION_DATASET_DIR / "images"
VERIFICATION_DATASET_METADATA_DIR = VERIFICATION_DATASET_DIR / "metadata"
VERIFICATION_DATASET_INDEX_CSV = VERIFICATION_DATASET_DIR / "index.csv"
VERIFICATION_PROMPTS_DIR = VERIFICATION_DATASET_DIR / "prompts"
VERIFICATION_PROMPT_INDEX_CSV = VERIFICATION_DATASET_DIR / "prompt_index.csv"

# Qwen verification inference results
VERIFICATION_RESULTS_DIR = OUTPUTS_DIR / "verification_results"
