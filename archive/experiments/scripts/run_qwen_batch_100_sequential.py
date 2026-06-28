#!/usr/bin/env python3
"""Run Qwen2.5-VL verification on the first 100 sequential palm inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvm.qwen_verifier import QwenVerifier

METADATA_CSV = PROJECT_ROOT / "outputs" / "lvm_inputs_metadata_100_sequential.csv"
RESULTS_CSV = PROJECT_ROOT / "outputs" / "qwen_100_sequential_results.csv"
RAW_RESPONSE_DIR = PROJECT_ROOT / "outputs" / "raw_responses_100_sequential"

MODEL_PATH = "/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct"
EXPECTED_ROW_COUNT = 100

RESULT_COLUMNS = [
    "image_name",
    "palm_id",
    "is_palm",
    "palm_confidence",
    "detection_quality",
    "bbox_alignment",
    "palm_structure",
    "occlusion_level",
    "radial_crown_visible",
    "fronds_visible",
    "trunk_visible",
    "reasoning",
    "model_name",
]


def raw_response_path(lvm_input_path: Path) -> Path:
    """Build a raw response file path from one LVM input image path."""
    return RAW_RESPONSE_DIR / f"{lvm_input_path.stem}.txt"


def validate_metadata(metadata_df: pd.DataFrame) -> None:
    """Confirm metadata and overlay files are ready before inference."""
    if len(metadata_df) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_ROW_COUNT} rows in metadata CSV, "
            f"found {len(metadata_df)}."
        )

    missing_files: list[str] = []
    for _, row in metadata_df.iterrows():
        image_path = PROJECT_ROOT / row["lvm_input_path"]
        if not image_path.exists():
            missing_files.append(str(row["lvm_input_path"]))

    if missing_files:
        sample = missing_files[:5]
        raise FileNotFoundError(
            "Missing LVM input files:\n  "
            + "\n  ".join(sample)
            + (f"\n  ... and {len(missing_files) - 5} more" if len(missing_files) > 5 else "")
        )


def result_row_from_response(result: dict, metadata: dict, model_name: str) -> dict:
    """Extract CSV columns, supporting both new and legacy response schemas."""
    row = {column: "" for column in RESULT_COLUMNS}

    row["image_name"] = result.get("image_name") or metadata.get("image_name") or ""
    row["palm_id"] = result.get("palm_id") or metadata.get("palm_id") or ""
    row["model_name"] = result.get("model_name") or model_name
    row["reasoning"] = result.get("reasoning") or ""

    field_sources = [
        "is_palm",
        "palm_confidence",
        "detection_quality",
        "bbox_alignment",
        "palm_structure",
        "occlusion_level",
        "radial_crown_visible",
        "fronds_visible",
        "trunk_visible",
    ]
    for field in field_sources:
        value = result.get(field)
        if value is not None and value != "":
            row[field] = value

    # Legacy schema fallbacks from current QwenVerifier output.
    if not row["detection_quality"] and result.get("classification"):
        row["detection_quality"] = result["classification"]
    if not row["palm_confidence"] and result.get("confidence") is not None:
        row["palm_confidence"] = result["confidence"]

    return row


def empty_result_row(metadata: dict, model_name: str, reasoning: str = "") -> dict:
    """Build a placeholder row when inference fails."""
    row = {column: "" for column in RESULT_COLUMNS}
    row["image_name"] = metadata.get("image_name", "")
    row["palm_id"] = metadata.get("palm_id", "")
    row["model_name"] = model_name
    row["reasoning"] = reasoning
    return row


def main() -> None:
    if not METADATA_CSV.exists():
        print(f"Metadata file not found: {METADATA_CSV}")
        print("Run prepare_lvm_inputs_100_sequential.py first.")
        sys.exit(1)

    metadata_df = pd.read_csv(METADATA_CSV)
    try:
        validate_metadata(metadata_df)
    except (ValueError, FileNotFoundError) as error:
        print(f"Metadata validation failed: {error}")
        sys.exit(1)

    print("Batch Qwen2.5-VL verification (100 sequential palms)")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Palms to process: {len(metadata_df)}")
    print()

    try:
        verifier = QwenVerifier(model_name=MODEL_PATH)
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
                    model_name=MODEL_PATH,
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

        results.append(result_row_from_response(result, metadata, MODEL_PATH))

    results_df = pd.DataFrame(results)
    for column in RESULT_COLUMNS:
        if column not in results_df.columns:
            results_df[column] = ""
    results_df = results_df[RESULT_COLUMNS]

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
