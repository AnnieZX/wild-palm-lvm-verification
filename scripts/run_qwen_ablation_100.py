#!/usr/bin/env python3
"""Run the 100-palm Qwen2.5-VL ablation matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lvm.ablation_response_parser import parse_ablation_response
from src.lvm.qwen_verifier import QwenVerifier
from src.prompts.ablation_prompts import build_ablation_prompt

METADATA_CSV = PROJECT_ROOT / "outputs" / "ablation_metadata_100.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "ablation_results_100"
RAW_RESPONSE_ROOT = PROJECT_ROOT / "outputs" / "ablation_raw_responses_100"
COMBINED_CSV = PROJECT_ROOT / "outputs" / "ablation_results_100_combined.csv"

MODEL_PATH = "/deac/csc/yangGrp/luoz23/models/Qwen2.5-VL-7B-Instruct"
EXPECTED_PALMS = 100
EXPECTED_CONDITIONS = 10
EXPECTED_COMBINED_ROWS = EXPECTED_PALMS * EXPECTED_CONDITIONS

# 10 unique ablation conditions.
ABLATION_CONDITIONS: list[tuple[str, str, str]] = [
    ("E1_raw_crop__P2_two_step", "E1_raw_crop_path", "P2_two_step"),
    ("E2_bbox_only__P2_two_step", "E2_bbox_only_path", "P2_two_step"),
    ("E3_endpoints_only__P2_two_step", "E3_endpoints_only_path", "P2_two_step"),
    ("E4_bbox_endpoints__P2_two_step", "E4_bbox_endpoints_path", "P2_two_step"),
    ("E5_full_overlay__P2_two_step", "E5_full_overlay_path", "P2_two_step"),
    ("E5_full_overlay__P1_direct_reliability", "E5_full_overlay_path", "P1_direct_reliability"),
    ("E5_full_overlay__P3_reasoning", "E5_full_overlay_path", "P3_reasoning"),
    ("E5_full_overlay__P4_yolo_confidence", "E5_full_overlay_path", "P4_yolo_confidence"),
    ("E5_full_overlay__P5_geometric_metadata", "E5_full_overlay_path", "P5_geometric_metadata"),
    ("E5_full_overlay__P6_full_metadata", "E5_full_overlay_path", "P6_full_metadata"),
]

COMBINED_COLUMNS = [
    "condition_name",
    "visual_variant",
    "prompt_variant",
    "sample_index",
    "image_name",
    "palm_id",
    "is_palm",
    "palm_confidence",
    "detection_quality",
    "confidence",
    "bbox_alignment",
    "palm_structure",
    "occlusion_level",
    "radial_crown_visible",
    "fronds_visible",
    "trunk_visible",
    "reasoning",
    "model_name",
    "parse_error",
    "inference_error",
]


def parse_condition_parts(condition_name: str) -> tuple[str, str]:
    visual_variant, prompt_variant = condition_name.split("__", maxsplit=1)
    return visual_variant, prompt_variant


def validate_metadata(metadata_df: pd.DataFrame) -> None:
    if len(metadata_df) != EXPECTED_PALMS:
        raise ValueError(
            f"Expected exactly {EXPECTED_PALMS} metadata rows, found {len(metadata_df)}."
        )

    path_columns = [
        "E1_raw_crop_path",
        "E2_bbox_only_path",
        "E3_endpoints_only_path",
        "E4_bbox_endpoints_path",
        "E5_full_overlay_path",
    ]
    missing: list[str] = []
    for _, row in metadata_df.iterrows():
        for column in path_columns:
            path = PROJECT_ROOT / row[column]
            if not path.exists():
                missing.append(str(row[column]))

    if missing:
        sample = missing[:5]
        raise FileNotFoundError(
            "Missing ablation input files:\n  "
            + "\n  ".join(sample)
            + (f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else "")
        )


def empty_row(
    condition_name: str,
    metadata: dict,
    model_name: str,
    parse_error: str = "",
    inference_error: str = "",
) -> dict:
    visual_variant, prompt_variant = parse_condition_parts(condition_name)
    row = {column: "" for column in COMBINED_COLUMNS}
    row.update(
        {
            "condition_name": condition_name,
            "visual_variant": visual_variant,
            "prompt_variant": prompt_variant,
            "sample_index": metadata.get("sample_index", ""),
            "image_name": metadata.get("image_name", ""),
            "palm_id": metadata.get("palm_id", ""),
            "model_name": model_name,
            "parse_error": parse_error,
            "inference_error": inference_error,
        }
    )
    return row


def result_row(
    condition_name: str,
    metadata: dict,
    parsed: dict,
    model_name: str,
) -> dict:
    visual_variant, prompt_variant = parse_condition_parts(condition_name)
    row = empty_row(condition_name, metadata, model_name)
    row.update(
        {
            "sample_index": metadata.get("sample_index", ""),
            "image_name": metadata.get("image_name", ""),
            "palm_id": parsed.get("palm_id") or metadata.get("palm_id", ""),
            "is_palm": parsed.get("is_palm", ""),
            "palm_confidence": parsed.get("palm_confidence", ""),
            "detection_quality": parsed.get("detection_quality", ""),
            "confidence": parsed.get("confidence", ""),
            "bbox_alignment": parsed.get("bbox_alignment", ""),
            "palm_structure": parsed.get("palm_structure", ""),
            "occlusion_level": parsed.get("occlusion_level", ""),
            "radial_crown_visible": parsed.get("radial_crown_visible", ""),
            "fronds_visible": parsed.get("fronds_visible", ""),
            "trunk_visible": parsed.get("trunk_visible", ""),
            "reasoning": parsed.get("reasoning", ""),
        }
    )
    row["visual_variant"] = visual_variant
    row["prompt_variant"] = prompt_variant
    return row


def main() -> None:
    if not METADATA_CSV.exists():
        print(f"Metadata file not found: {METADATA_CSV}")
        print("Run prepare_ablation_inputs_100.py first.")
        sys.exit(1)

    metadata_df = pd.read_csv(METADATA_CSV)
    try:
        validate_metadata(metadata_df)
    except (ValueError, FileNotFoundError) as error:
        print(f"Metadata validation failed: {error}")
        sys.exit(1)

    print("100-palm Qwen ablation study")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Palms: {len(metadata_df)}")
    print(f"  Unique conditions: {len(ABLATION_CONDITIONS)}")
    print()

    try:
        verifier = QwenVerifier(model_name=MODEL_PATH)
    except (RuntimeError, FileNotFoundError) as error:
        print(error)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESPONSE_ROOT.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict] = []
    total_parse_errors = 0
    total_inference_errors = 0

    for condition_name, path_column, prompt_variant in ABLATION_CONDITIONS:
        print(f"Running condition: {condition_name}")
        condition_rows: list[dict] = []
        raw_dir = RAW_RESPONSE_ROOT / condition_name
        raw_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(
            metadata_df.iterrows(),
            total=len(metadata_df),
            desc=condition_name,
        ):
            metadata = row.to_dict()
            image_path = PROJECT_ROOT / row[path_column]
            palm_id = metadata.get("palm_id", "unknown")
            prompt = build_ablation_prompt(metadata, prompt_variant)

            try:
                result = verifier.verify_image(
                    str(image_path),
                    metadata,
                    prompt=prompt,
                    use_legacy_validation=False,
                )
            except Exception as error:
                total_inference_errors += 1
                tqdm.write(f"  ERROR {palm_id}: inference failed ({error})")
                row_out = empty_row(
                    condition_name,
                    metadata,
                    MODEL_PATH,
                    inference_error=str(error),
                )
                condition_rows.append(row_out)
                combined_rows.append(row_out)
                continue

            raw_response = result.get("raw_response", "")
            raw_path = raw_dir / f"{Path(row[path_column]).stem}.txt"
            raw_path.write_text(raw_response, encoding="utf-8")

            try:
                parsed = parse_ablation_response(raw_response, prompt_variant)
                row_out = result_row(condition_name, metadata, parsed, MODEL_PATH)
            except ValueError as error:
                total_parse_errors += 1
                tqdm.write(f"  WARN {palm_id}: parse failed ({error})")
                row_out = empty_row(
                    condition_name,
                    metadata,
                    MODEL_PATH,
                    parse_error=str(error),
                )

            condition_rows.append(row_out)
            combined_rows.append(row_out)

        condition_df = pd.DataFrame(condition_rows)
        for column in COMBINED_COLUMNS:
            if column not in condition_df.columns:
                condition_df[column] = ""
        condition_df = condition_df[COMBINED_COLUMNS]
        condition_csv = RESULTS_DIR / f"{condition_name}.csv"
        condition_df.to_csv(condition_csv, index=False)
        print(f"  Saved: {condition_csv.relative_to(PROJECT_ROOT)}")

    combined_df = pd.DataFrame(combined_rows)
    for column in COMBINED_COLUMNS:
        if column not in combined_df.columns:
            combined_df[column] = ""
    combined_df = combined_df[COMBINED_COLUMNS]
    combined_df.to_csv(COMBINED_CSV, index=False)

    print()
    print("Ablation run complete")
    print(f"  Combined rows: {len(combined_df)} (expected {EXPECTED_COMBINED_ROWS})")
    if len(combined_df) < EXPECTED_COMBINED_ROWS:
        print("  WARNING: combined results has fewer rows than expected.")
    print(f"  Parse errors: {total_parse_errors}")
    print(f"  Inference errors: {total_inference_errors}")
    print(f"  Combined CSV: {COMBINED_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
