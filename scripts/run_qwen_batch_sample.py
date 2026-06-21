#!/usr/bin/env python3
"""Run Qwen2.5-VL verification on all sample palm inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvm.qwen_verifier import QwenVerifier

METADATA_CSV = PROJECT_ROOT / "outputs" / "lvm_inputs_metadata.csv"
RESULTS_CSV = PROJECT_ROOT / "outputs" / "qwen_sample_results.csv"
RAW_RESPONSE_DIR = PROJECT_ROOT / "outputs" / "raw_responses"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"

DEFAULT_MODEL_PATH = "/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct"

RESULT_COLUMNS = [
    "image_name",
    "palm_id",
    "classification",
    "confidence",
    "bbox_alignment",
    "palm_structure",
    "occlusion_level",
    "reasoning",
    "model_name",
]


def load_model_config(config_path: Path) -> dict:
    """Load model settings from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_model_path(config: dict) -> str:
    """Return the local cluster model path, with config override if present."""
    return config.get("active_model", DEFAULT_MODEL_PATH)


def raw_response_path(lvm_input_path: Path) -> Path:
    """Build a raw response file path from one LVM input image path."""
    return RAW_RESPONSE_DIR / f"{lvm_input_path.stem}.txt"


def result_row_from_response(result: dict) -> dict:
    """Extract CSV columns from one verifier result."""
    row = {column: result.get(column) for column in RESULT_COLUMNS}
    return row


def empty_result_row(metadata: dict, model_name: str, reasoning: str = "") -> dict:
    """Build a placeholder row when inference or parsing fails."""
    return {
        "image_name": metadata.get("image_name"),
        "palm_id": metadata.get("palm_id"),
        "classification": None,
        "confidence": None,
        "bbox_alignment": None,
        "palm_structure": None,
        "occlusion_level": None,
        "reasoning": reasoning,
        "model_name": model_name,
    }


def main() -> None:
    if not METADATA_CSV.exists():
        print(f"Metadata file not found: {METADATA_CSV}")
        print("Run prepare_lvm_inputs.py first.")
        sys.exit(1)

    config = load_model_config(MODEL_CONFIG) if MODEL_CONFIG.exists() else {}
    model_path = resolve_model_path(config)

    metadata_df = pd.read_csv(METADATA_CSV)
    if metadata_df.empty:
        print("Metadata CSV is empty.")
        sys.exit(1)

    print("Batch Qwen2.5-VL verification (sample scale)")
    print(f"  Model path: {model_path}")
    print(f"  Palms to process: {len(metadata_df)}")
    print()

    try:
        verifier = QwenVerifier(model_name=model_path)
    except (RuntimeError, FileNotFoundError) as error:
        print(error)
        sys.exit(1)

    RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    parse_errors = 0
    inference_errors = 0

    for _, row in tqdm(
        metadata_df.iterrows(),
        total=len(metadata_df),
        desc="Verifying palms",
    ):
        metadata = row.to_dict()
        image_path = PROJECT_ROOT / row["lvm_input_path"]
        palm_id = metadata.get("palm_id", "unknown")

        try:
            result = verifier.verify_image(str(image_path), metadata)
        except Exception as error:
            inference_errors += 1
            tqdm.write(f"  ERROR {palm_id}: inference failed ({error})")
            results.append(
                empty_result_row(
                    metadata,
                    model_name=model_path,
                    reasoning=f"Inference error: {error}",
                )
            )
            continue

        raw_response = result.get("raw_response", "")
        raw_path = raw_response_path(image_path)
        raw_path.write_text(raw_response, encoding="utf-8")

        if "parse_error" in result:
            parse_errors += 1
            tqdm.write(f"  WARN {palm_id}: {result['parse_error']}")

        results.append(result_row_from_response(result))

    results_df = pd.DataFrame(results)[RESULT_COLUMNS]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    print()
    print("Batch verification complete")
    print(f"  Processed: {len(metadata_df)}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Inference errors: {inference_errors}")
    print(f"  Results CSV: {RESULTS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  Raw responses: {RAW_RESPONSE_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
