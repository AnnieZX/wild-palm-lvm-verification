#!/usr/bin/env python3
"""
Controlled LLaVA-OneVision grounding diagnostic.

Standalone experiment — does NOT use the production verification adapter.
Tests prompt variants and decoding modes on 12 stratified A1 samples.

Example:
    python scripts/debug_llava_grounding.py
    python scripts/debug_llava_grounding.py --select-only
    python scripts/debug_llava_grounding.py --skip-selection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

_eval_spec = importlib.util.spec_from_file_location(
    "evaluate_verification_against_groundtruth",
    PROJECT_ROOT / "scripts" / "evaluate_verification_against_groundtruth.py",
)
_eval_module = importlib.util.module_from_spec(_eval_spec)
assert _eval_spec.loader is not None
_eval_spec.loader.exec_module(_eval_module)
compute_greedy_matches_for_index = _eval_module.compute_greedy_matches_for_index
from src.config.model_config import resolve_model_checkpoint
from src.lvm.response_schema import parse_json_response
from src.paths import (
    OUTPUTS_DIR,
    PREDICTIONS_FULL_JSON,
    RAW_PATCHES_ROOT,
    VERIFICATION_DATASET_INDEX_CSV,
    discover_ablation_inputs_dir,
)
from src.prompts.ablation_verification_prompts import (
    DECISION_DEFINITIONS,
    JSON_RESPONSE_TEMPLATE,
)
from src.utils.verification_index import find_yolo_prediction, index_bbox_from_row
from src.yolo.predictions_io import group_predictions_by_image, load_predictions

DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "debug" / "llava_grounding"
DEFAULT_MODEL_KEY = "llava"
PROMPT_VARIANTS = ("P1_original", "P2_reordered_labels", "P3_neutral_labels", "P4_short_visual")
DECODING_MODES = ("greedy", "sampled")
P4_SHORT_PROMPT = (
    "Inspect only the object inside the green bounding box.\n"
    "Decide whether it is:\n"
    "A: clearly a wild palm\n"
    "B: visually ambiguous\n"
    "C: clearly not a wild palm\n"
    "\n"
    "Base the answer only on visible image evidence.\n"
    "Return JSON:\n"
    "{\n"
    '  "decision": "A | B | C",\n'
    '  "visible_evidence": "specific features actually visible inside the box"\n'
    "}"
)

DECISION_DEFINITIONS_REORDERED = """Decision definitions

Unreliable
The highlighted object is not a palm or the detection is clearly incorrect.

Uncertain
Evidence is ambiguous due to occlusion, overlap, low image quality, partial visibility, or confusing vegetation.

Reliable
The highlighted bounding box clearly contains a valid wild palm."""

JSON_RESPONSE_TEMPLATE_REORDERED = """{
  "decision": "Unreliable | Uncertain | Reliable",
  "confidence_reasoning": "",
  "visual_reasoning": ""
}"""

P3_JSON_TEMPLATE = """{
  "decision": "A | B | C",
  "confidence_reasoning": "",
  "visual_reasoning": ""
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLaVA-OneVision visual grounding diagnostic experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diagnostic output directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="LLaVA checkpoint (default: configs/models/llava.yaml)",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=None,
        help="A1 ablation inputs root (default: discover verification_ablation_<N>)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size suffix for ablation dir discovery (default: 1000)",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=VERIFICATION_DATASET_INDEX_CSV,
        help="Verification dataset index.csv",
    )
    parser.add_argument(
        "--evaluation-csv",
        type=Path,
        default=None,
        help="Optional precomputed A1_evaluation.csv (uses GT recompute if missing)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS_FULL_JSON,
        help="YOLO predictions JSON",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=RAW_PATCHES_ROOT,
        help="LabelMe JSON root",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection and P5 image swaps",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per inference",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Only build selected_samples.csv and exit",
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Reuse existing selected_samples.csv",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference; recompute summary from existing results only",
    )
    return parser.parse_args()


def resolve_ablation_a1_dir(ablation_dir: Path | None, sample_size: int) -> Path:
    root = ablation_dir or discover_ablation_inputs_dir(sample_size)
    a1_dir = root / "A1_overlay_only"
    if not (a1_dir / "prompt_index.csv").is_file():
        raise FileNotFoundError(f"A1 prompt index not found under {a1_dir}")
    return a1_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_values_fingerprint(pixel_values: Any) -> str:
    if pixel_values is None:
        return ""
    tensor = pixel_values.detach().float().cpu()
    payload = f"{tuple(tensor.shape)}:{float(tensor.mean()):.8f}:{float(tensor.std()):.8f}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def serialize_image_sizes(image_sizes: Any) -> str:
    if image_sizes is None:
        return ""
    if hasattr(image_sizes, "tolist"):
        return json.dumps(image_sizes.tolist())
    return str(image_sizes)


def load_prompt_index(a1_dir: Path) -> pd.DataFrame:
    prompt_index = a1_dir / "prompt_index.csv"
    df = pd.read_csv(prompt_index)
    base = prompt_index.parent.resolve()
    df["image_path_resolved"] = df["image_path"].map(lambda p: (base / p).resolve())
    df["prompt_path_resolved"] = df["prompt_path"].map(lambda p: (base / p).resolve())
    return df


def build_evaluation_frame(
    index_df: pd.DataFrame,
    predictions_path: Path,
    annotations_root: Path,
    evaluation_csv: Path | None,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    if evaluation_csv is not None and evaluation_csv.is_file():
        eval_df = pd.read_csv(evaluation_csv)
        required = {"sample_id", "matched_gt", "max_iou"}
        missing = required - set(eval_df.columns)
        if missing:
            raise ValueError(f"Evaluation CSV missing columns: {sorted(missing)}")
        return eval_df

    predictions_by_image = group_predictions_by_image(load_predictions(predictions_path))
    gt_cache: dict[str, list[tuple[float, float, float, float]]] = {}
    greedy_matches = compute_greedy_matches_for_index(
        index_df,
        predictions_by_image,
        annotations_root,
        gt_cache,
        iou_threshold,
    )

    rows: list[dict[str, Any]] = []
    for _, sample in index_df.iterrows():
        sample_id = str(sample["sample_id"])
        image_name = str(sample["image_name"])
        index_bbox = index_bbox_from_row(sample)
        yolo_bbox, yolo_confidence = find_yolo_prediction(
            image_name,
            index_bbox,
            predictions_by_image,
        )
        if yolo_bbox is None:
            yolo_bbox = index_bbox
            yolo_confidence = float(sample["confidence"]) if pd.notna(sample.get("confidence")) else 0.0

        match = greedy_matches.get(sample_id)
        matched_gt = bool(match.matched_gt) if match else False
        max_iou = float(match.max_iou) if match else 0.0
        if yolo_confidence is None:
            yolo_confidence = float(sample["confidence"]) if pd.notna(sample.get("confidence")) else 0.0

        rows.append(
            {
                "sample_id": sample_id,
                "image_name": image_name,
                "matched_gt": matched_gt,
                "max_iou": round(max_iou, 4),
                "yolo_confidence": float(yolo_confidence),
            }
        )
    return pd.DataFrame(rows)


def categorize_row(row: pd.Series) -> str:
    matched = bool(row["matched_gt"])
    iou = float(row["max_iou"])
    conf = float(row["yolo_confidence"]) if pd.notna(row["yolo_confidence"]) else 0.0

    if matched and iou >= 0.7 and conf >= 0.6:
        return "gt_positive_easy"
    if (not matched) and iou < 0.3 and conf >= 0.5:
        return "gt_negative_fp"
    if (not matched) and 0.3 <= iou < 0.5:
        return "visually_ambiguous"
    if matched and 0.5 <= iou < 0.7:
        return "visually_ambiguous"
    if (not matched) and iou >= 0.5:
        return "visually_ambiguous"
    return "other"


def select_samples(
    eval_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    merged = eval_df.merge(
        prompt_df[["sample_id", "image_path_resolved", "prompt_path_resolved"]],
        on="sample_id",
        how="inner",
    )
    merged["category"] = merged.apply(categorize_row, axis=1)

    rng = random.Random(seed)
    selected_rows: list[pd.Series] = []

    for category, count in (
        ("gt_positive_easy", 4),
        ("gt_negative_fp", 4),
        ("visually_ambiguous", 4),
    ):
        pool = merged[merged["category"] == category].copy()
        if pool.empty:
            pool = merged[merged["category"] == "other"].copy()
        if len(pool) < count:
            raise ValueError(
                f"Not enough samples for category {category}: found {len(pool)}, need {count}"
            )
        if category == "gt_positive_easy":
            pool = pool.sort_values(["max_iou", "yolo_confidence"], ascending=False)
        elif category == "gt_negative_fp":
            pool = pool.sort_values(["yolo_confidence", "max_iou"], ascending=[False, True])
        else:
            pool = pool.sort_values(["max_iou"], ascending=False)
        picks = pool.head(max(count * 3, count))
        indices = list(picks.index)
        rng.shuffle(indices)
        chosen = picks.loc[indices[:count]]
        selected_rows.extend(chosen.to_dict("records"))

    out = pd.DataFrame(selected_rows)
    out = out.rename(
        columns={
            "image_path_resolved": "image_path",
            "prompt_path_resolved": "prompt_path",
            "matched_gt": "gt_match",
        }
    )
    out["image_path"] = out["image_path"].astype(str)
    out["gt_match"] = out["gt_match"].astype(bool)
    return out[
        ["sample_id", "image_path", "gt_match", "max_iou", "yolo_confidence", "category"]
    ]


def build_p1_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def build_p2_prompt(original: str) -> str:
    text = original.replace(DECISION_DEFINITIONS, DECISION_DEFINITIONS_REORDERED)
    text = text.replace(JSON_RESPONSE_TEMPLATE, JSON_RESPONSE_TEMPLATE_REORDERED)
    text = text.replace(
        '"decision": "Reliable | Uncertain | Unreliable"',
        '"decision": "Unreliable | Uncertain | Reliable"',
    )
    return text


def build_p3_prompt(original: str, sample_id: str) -> str:
    # Reuse the narrative body but swap decision language to neutral A/B/C labels.
    text = original
    text = re.sub(
        r"Decision definitions[\s\S]*?(?=--------------------------------------------------\n\nReasoning requirements)",
        (
            "Decision definitions\n\n"
            "A\n"
            "The highlighted bounding box clearly contains a valid wild palm.\n\n"
            "B\n"
            "Evidence is ambiguous due to occlusion, overlap, low image quality, partial visibility, "
            "or confusing vegetation.\n\n"
            "C\n"
            "The highlighted object is not a palm or the detection is clearly incorrect.\n\n"
            "--------------------------------------------------\n\n"
        ),
        text,
        count=1,
    )
    text = text.replace(JSON_RESPONSE_TEMPLATE, P3_JSON_TEMPLATE)
    text = re.sub(
        r"\bReliable\b",
        "a valid palm (option A)",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bUncertain\b",
        "ambiguous (option B)",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bUnreliable\b",
        "invalid detection (option C)",
        text,
        flags=re.IGNORECASE,
    )
    forbidden = ("Reliable", "Uncertain", "Unreliable")
    if any(word in text for word in forbidden):
        raise ValueError(f"P3 prompt for {sample_id} still contains forbidden labels")
    return text


def build_prompt(variant: str, sample_id: str, prompt_path: Path) -> str:
    original = build_p1_prompt(prompt_path)
    if variant == "P1_original":
        return original
    if variant == "P2_reordered_labels":
        return build_p2_prompt(original)
    if variant == "P3_neutral_labels":
        return build_p3_prompt(original, sample_id)
    if variant == "P4_short_visual":
        return P4_SHORT_PROMPT
    raise ValueError(f"Unknown prompt variant: {variant}")


def canonical_decision(raw_decision: str, variant: str) -> str:
    text = (raw_decision or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if variant in {"P3_neutral_labels", "P4_short_visual"} or variant.startswith("P5"):
        if upper.startswith("A"):
            return "A"
        if upper.startswith("B"):
            return "B"
        if upper.startswith("C"):
            return "C"
        return text
    lowered = text.lower()
    if "unreliable" in lowered:
        return "Unreliable"
    if "uncertain" in lowered:
        return "Uncertain"
    if "reliable" in lowered:
        return "Reliable"
    return text


def gt_expected_label(category: str, variant: str) -> str | None:
    if category == "gt_positive_easy":
        expected = "positive"
    elif category == "gt_negative_fp":
        expected = "negative"
    else:
        return None

    if variant in {"P3_neutral_labels", "P4_short_visual"} or variant.startswith("P5"):
        return "A" if expected == "positive" else "C"
    return "Reliable" if expected == "positive" else "Unreliable"


def parse_decision_from_response(raw_response: str, variant: str) -> str:
    try:
        parsed = parse_json_response(raw_response)
    except ValueError:
        return ""
    decision = parsed.get("decision", "")
    return canonical_decision(str(decision), variant)


class DiagnosticLlavaEngine:
    """Standalone LLaVA-OneVision inference with runtime diagnostics."""

    def __init__(self, model_path: str, device_map: str = "auto") -> None:
        import torch
        from transformers import (
            LlavaOnevisionForConditionalGeneration,
            LlavaOnevisionProcessor,
        )

        self.torch = torch
        self.processor = LlavaOnevisionProcessor.from_pretrained(model_path)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    @staticmethod
    def _shape(value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "shape"):
            return str(tuple(value.shape))
        return str(type(value))

    def _move_inputs(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            try:
                return inputs.to(self.model.device, dtype=self.torch.bfloat16)
            except TypeError:
                return inputs.to(self.model.device)
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if value.is_floating_point():
                    moved[key] = value.to(self.model.device, dtype=self.torch.bfloat16)
                else:
                    moved[key] = value.to(self.model.device)
            else:
                moved[key] = value
        return moved

    def run(
        self,
        *,
        sample_id: str,
        image_path: Path,
        prompt: str,
        prompt_variant: str,
        decoding_mode: str,
        image_mode: str,
        paired_sample_id: str,
        seed: int,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        from PIL import Image

        image_path = image_path.resolve()
        started = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_template = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            images=image,
            text=chat_template,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)

        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if decoding_mode == "greedy":
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            self.torch.manual_seed(seed)

        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0, input_len:]
        raw_response = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        runtime_seconds = time.perf_counter() - started

        pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else None
        image_sizes = inputs.get("image_sizes") if hasattr(inputs, "get") else None

        variant_key = prompt_variant.replace("_swap", "")
        parsed_decision = parse_decision_from_response(raw_response, variant_key)

        return {
            "sample_id": sample_id,
            "prompt_variant": prompt_variant,
            "decoding_mode": decoding_mode,
            "image_mode": image_mode,
            "paired_sample_id": paired_sample_id,
            "resolved_image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "image_width": width,
            "image_height": height,
            "chat_template": chat_template,
            "input_ids_shape": self._shape(inputs.get("input_ids")),
            "pixel_values_present": pixel_values is not None,
            "pixel_values_shape": self._shape(pixel_values),
            "pixel_values_fingerprint": pixel_values_fingerprint(pixel_values),
            "image_sizes": serialize_image_sizes(image_sizes),
            "raw_response": raw_response,
            "parsed_decision": parsed_decision,
            "runtime_seconds": round(runtime_seconds, 4),
        }


def build_swap_map(sample_ids: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = sample_ids[:]
    rng.shuffle(shuffled)
    mapping: dict[str, str] = {}
    for index, sample_id in enumerate(sample_ids):
        if shuffled[index] == sample_id:
            shuffled[index], shuffled[(index + 1) % len(shuffled)] = (
                shuffled[(index + 1) % len(shuffled)],
                shuffled[index],
            )
        mapping[sample_id] = shuffled[index]
    for sample_id in sample_ids:
        if mapping[sample_id] == sample_id:
            others = [sid for sid in sample_ids if sid != sample_id]
            mapping[sample_id] = rng.choice(others)
    return mapping


def run_inference(
    selected_df: pd.DataFrame,
    engine: DiagnosticLlavaEngine,
    output_dir: Path,
    seed: int,
    max_new_tokens: int,
) -> pd.DataFrame:
    sample_ids = selected_df["sample_id"].tolist()
    image_by_id = {
        str(row["sample_id"]): Path(str(row["image_path"]))
        for _, row in selected_df.iterrows()
    }
    prompt_path_by_id = {
        str(row["sample_id"]): Path(str(row.get("prompt_path", "")))
        for _, row in selected_df.iterrows()
        if row.get("prompt_path")
    }
    swap_map = build_swap_map(sample_ids, seed + 17)

    rows: list[dict[str, Any]] = []
    total = len(sample_ids) * (
        len(PROMPT_VARIANTS) * len(DECODING_MODES) + len(DECODING_MODES) * 2
    )
    done = 0

    for _, sample in selected_df.iterrows():
        sample_id = str(sample["sample_id"])
        prompt_path = prompt_path_by_id.get(sample_id)
        if prompt_path is None or not prompt_path.is_file():
            metadata_path = None
            # Fallback: rebuild A1 prompt from metadata if only selected_samples.csv exists.
            raise FileNotFoundError(
                f"Missing prompt path for {sample_id}. Re-run selection without --skip-selection "
                "or include prompt_path in selected_samples.csv."
            )

        for variant in PROMPT_VARIANTS:
            prompt = build_prompt(variant, sample_id, prompt_path)
            for decoding_mode in DECODING_MODES:
                done += 1
                print(
                    f"[{done}/{total}] {sample_id} {variant} {decoding_mode} correct_image",
                    flush=True,
                )
                record = engine.run(
                    sample_id=sample_id,
                    image_path=image_by_id[sample_id],
                    prompt=prompt,
                    prompt_variant=variant,
                    decoding_mode=decoding_mode,
                    image_mode="correct",
                    paired_sample_id=sample_id,
                    seed=seed + done,
                    max_new_tokens=max_new_tokens,
                )
                rows.append(record)

        for decoding_mode in DECODING_MODES:
            for image_mode, image_sample_id in (
                ("correct", sample_id),
                ("swapped", swap_map[sample_id]),
            ):
                done += 1
                print(
                    f"[{done}/{total}] {sample_id} P5_counterfactual {decoding_mode} {image_mode}",
                    flush=True,
                )
                record = engine.run(
                    sample_id=sample_id,
                    image_path=image_by_id[image_sample_id],
                    prompt=P4_SHORT_PROMPT,
                    prompt_variant="P5_counterfactual",
                    decoding_mode=decoding_mode,
                    image_mode=image_mode,
                    paired_sample_id=image_sample_id,
                    seed=seed + done,
                    max_new_tokens=max_new_tokens,
                )
                rows.append(record)

    results_df = pd.DataFrame(rows)
    csv_path = output_dir / "results.csv"
    jsonl_path = output_dir / "results.jsonl"
    results_df.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {csv_path}")
    print(f"Saved {jsonl_path}")
    return results_df


def _duplicate_rate(series: pd.Series) -> float:
    values = series.astype(str).tolist()
    if not values:
        return 0.0
    counts = Counter(values)
    duplicates = sum(count for count in counts.values() if count > 1)
    return round(100.0 * duplicates / len(values), 2)


def _response_similarity(series: pd.Series) -> float:
    values = series.astype(str).tolist()
    if len(values) <= 1:
        return 100.0
    counts = Counter(values)
    most_common = counts.most_common(1)[0][1]
    return round(100.0 * most_common / len(values), 2)


def compute_summary(results_df: pd.DataFrame, selected_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    summary["decision_by_prompt"] = (
        results_df.groupby(["prompt_variant", "parsed_decision"]).size().unstack(fill_value=0).to_dict()
    )
    summary["decision_by_decoding"] = (
        results_df.groupby(["decoding_mode", "parsed_decision"]).size().unstack(fill_value=0).to_dict()
    )
    summary["unique_raw_responses"] = int(results_df["raw_response"].nunique())
    summary["duplicate_response_rate_pct"] = _duplicate_rate(results_df["raw_response"])
    summary["response_similarity_pct"] = _response_similarity(results_df["raw_response"])

    # Label-order sensitivity (greedy only).
    p1 = results_df[
        (results_df["prompt_variant"] == "P1_original") & (results_df["decoding_mode"] == "greedy")
    ].set_index("sample_id")["parsed_decision"]
    p2 = results_df[
        (results_df["prompt_variant"] == "P2_reordered_labels")
        & (results_df["decoding_mode"] == "greedy")
    ].set_index("sample_id")["parsed_decision"]
    common = p1.index.intersection(p2.index)
    if len(common):
        changed = sum(p1[sid] != p2[sid] for sid in common)
        summary["label_order_change_pct"] = round(100.0 * changed / len(common), 2)
        summary["p1_first_label_rate_pct"] = round(
            100.0 * sum(val == "Reliable" for val in p1.loc[common]) / len(common),
            2,
        )
        summary["p2_first_label_rate_pct"] = round(
            100.0 * sum(val == "Unreliable" for val in p2.loc[common]) / len(common),
            2,
        )
    else:
        summary["label_order_change_pct"] = 0.0
        summary["p1_first_label_rate_pct"] = 0.0
        summary["p2_first_label_rate_pct"] = 0.0

    # Image swap sensitivity (P5).
    swap_rows = results_df[results_df["prompt_variant"] == "P5_counterfactual"]
    swap_changed = 0
    swap_total = 0
    for (sample_id, decoding_mode), group in swap_rows.groupby(["sample_id", "decoding_mode"]):
        correct = group[group["image_mode"] == "correct"]
        swapped = group[group["image_mode"] == "swapped"]
        if correct.empty or swapped.empty:
            continue
        swap_total += 1
        if correct.iloc[0]["parsed_decision"] != swapped.iloc[0]["parsed_decision"]:
            swap_changed += 1
    summary["image_swap_change_pct"] = (
        round(100.0 * swap_changed / swap_total, 2) if swap_total else 0.0
    )

    # Accuracy on clear positive/negative categories.
    category_by_id = selected_df.set_index("sample_id")["category"].to_dict()
    accuracy_rows = []
    for _, row in results_df.iterrows():
        category = category_by_id.get(row["sample_id"])
        expected = gt_expected_label(str(category), str(row["prompt_variant"]))
        if expected is None:
            continue
        accuracy_rows.append(
            {
                "correct": row["parsed_decision"] == expected,
                "prompt_variant": row["prompt_variant"],
                "decoding_mode": row["decoding_mode"],
            }
        )
    if accuracy_rows:
        acc_df = pd.DataFrame(accuracy_rows)
        summary["accuracy_by_prompt"] = (
            acc_df.groupby("prompt_variant")["correct"].mean().round(4).to_dict()
        )
        summary["accuracy_by_decoding"] = (
            acc_df.groupby("decoding_mode")["correct"].mean().round(4).to_dict()
        )
    else:
        summary["accuracy_by_prompt"] = {}
        summary["accuracy_by_decoding"] = {}

    # Pixel value diagnostics.
    summary["pixel_values_present_rate_pct"] = round(
        100.0 * results_df["pixel_values_present"].mean(),
        2,
    )
    summary["unique_image_hashes"] = int(results_df["image_sha256"].nunique())
    summary["unique_pixel_shapes"] = int(results_df["pixel_values_shape"].nunique())
    summary["unique_pixel_fingerprints"] = int(
        results_df["pixel_values_fingerprint"].astype(str).nunique()
    )

    # P4 vs P1 diversity.
    p1_g = results_df[
        (results_df["prompt_variant"] == "P1_original") & (results_df["decoding_mode"] == "greedy")
    ]["parsed_decision"]
    p4_g = results_df[
        (results_df["prompt_variant"] == "P4_short_visual") & (results_df["decoding_mode"] == "greedy")
    ]["parsed_decision"]
    summary["p1_greedy_unique_decisions"] = int(p1_g.nunique())
    summary["p4_greedy_unique_decisions"] = int(p4_g.nunique())

    sampled = results_df[results_df["decoding_mode"] == "sampled"]
    greedy = results_df[results_df["decoding_mode"] == "greedy"]
    summary["sampled_unique_decisions"] = int(sampled["parsed_decision"].nunique())
    summary["greedy_unique_decisions"] = int(greedy["parsed_decision"].nunique())

    return summary


def write_diagnosis_md(output_dir: Path, summary: dict[str, Any], selected_df: pd.DataFrame) -> None:
    def yn(condition: bool) -> str:
        return "Yes" if condition else "No / inconclusive"

    p1_first = summary.get("p1_first_label_rate_pct", 0.0)
    p2_first = summary.get("p2_first_label_rate_pct", 0.0)
    swap_change = summary.get("image_swap_change_pct", 0.0)
    label_change = summary.get("label_order_change_pct", 0.0)
    p4_unique = summary.get("p4_greedy_unique_decisions", 0)
    p1_unique = summary.get("p1_greedy_unique_decisions", 0)
    dup_rate = summary.get("duplicate_response_rate_pct", 0.0)
    pixel_rate = summary.get("pixel_values_present_rate_pct", 0.0)
    unique_hashes = summary.get("unique_image_hashes", 0)

    lines = [
        "# LLaVA-OneVision Grounding Diagnostic",
        "",
        "Automated report from `scripts/debug_llava_grounding.py`.",
        "",
        "## Selected samples",
        "",
        f"- Count: {len(selected_df)}",
        f"- Categories: {selected_df['category'].value_counts().to_dict()}",
        "",
        "## Summary statistics",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Diagnostic questions",
        "",
        "### A. Does LLaVA react to different images?",
        "",
        f"- Unique image SHA256 hashes in results: **{unique_hashes}**",
        f"- Unique pixel_values fingerprints: **{summary.get('unique_pixel_fingerprints', 0)}**",
        f"- Assessment: {'Images differ across runs.' if unique_hashes > 1 else 'Insufficient evidence.'}",
        "",
        "### B. Does swapping images change the output?",
        "",
        f"- P5 image-swap decision change rate: **{swap_change}%**",
        f"- Assessment: {'Likely visually grounded if >25%.' if swap_change > 25 else 'Weak image sensitivity if low.'}",
        "",
        "### C. Does changing label order change the output?",
        "",
        f"- P1 vs P2 greedy decision change rate: **{label_change}%**",
        f"- Assessment: {'Label order affects outputs.' if label_change > 20 else 'Limited label-order sensitivity.'}",
        "",
        "### D. Is the model anchored to the first label?",
        "",
        f"- P1 first-label (Reliable) rate: **{p1_first}%**",
        f"- P2 first-label (Unreliable) rate: **{p2_first}%**",
        f"- Assessment: {'Possible first-label anchoring.' if p1_first > 70 or p2_first > 70 else 'No strong anchoring signal.'}",
        "",
        "### E. Does the shorter prompt improve visual differentiation?",
        "",
        f"- P1 greedy unique decisions: **{p1_unique}**",
        f"- P4 greedy unique decisions: **{p4_unique}**",
        f"- Assessment: {'Short prompt increases diversity.' if p4_unique > p1_unique else 'No clear improvement from short prompt.'}",
        "",
        "### F. Does sampling improve decision diversity without destroying consistency?",
        "",
        f"- Greedy unique decisions: **{summary.get('greedy_unique_decisions', 0)}**",
        f"- Sampled unique decisions: **{summary.get('sampled_unique_decisions', 0)}**",
        f"- Duplicate raw-response rate: **{dup_rate}%**",
        "",
        "### G. Is pixel_values present and different across samples?",
        "",
        f"- pixel_values present rate: **{pixel_rate}%**",
        f"- Assessment: {yn(pixel_rate >= 99.0)}",
        "",
        "### H. Is LLaVA suitable for the main ablation study under the current prompt?",
        "",
        "See accuracy_by_prompt in summary JSON. If P1 greedy accuracy is low on clear "
        "positive/negative samples, or duplicate-response rate is very high, the current "
        "full A1 prompt is likely unsuitable without revision.",
        "",
    ]
    path = output_dir / "DIAGNOSIS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_samples.csv"

    a1_dir = resolve_ablation_a1_dir(args.ablation_dir, args.sample_size)
    prompt_df = load_prompt_index(a1_dir)

    if args.skip_selection and selected_path.is_file():
        selected_df = pd.read_csv(selected_path)
        if "prompt_path" not in selected_df.columns:
            merged = selected_df.merge(
                prompt_df[["sample_id", "prompt_path_resolved"]],
                on="sample_id",
                how="left",
            )
            selected_df["prompt_path"] = merged["prompt_path_resolved"].astype(str)
    else:
        if not args.index_csv.is_file():
            print(f"Verification index not found: {args.index_csv}")
            sys.exit(1)
        index_df = pd.read_csv(args.index_csv)
        eval_df = build_evaluation_frame(
            index_df=index_df.head(args.sample_size),
            predictions_path=args.predictions,
            annotations_root=args.annotations_root,
            evaluation_csv=args.evaluation_csv,
        )
        selected_df = select_samples(eval_df, prompt_df.head(args.sample_size), args.seed)
        if "prompt_path" not in selected_df.columns:
            selected_df = selected_df.merge(
                prompt_df[["sample_id", "prompt_path_resolved"]],
                on="sample_id",
                how="left",
            )
            selected_df["prompt_path"] = selected_df["prompt_path_resolved"].astype(str)
            selected_df = selected_df.drop(columns=["prompt_path_resolved"])
        selected_df[
            ["sample_id", "image_path", "gt_match", "max_iou", "yolo_confidence", "category"]
        ].to_csv(selected_path, index=False)
        print(f"Saved {selected_path}")

    if args.select_only:
        print(selected_df.to_string(index=False))
        return

    if args.skip_inference:
        results_path = args.output_dir / "results.csv"
        if not results_path.is_file():
            print(f"Results not found: {results_path}")
            sys.exit(1)
        results_df = pd.read_csv(results_path)
    else:
        model_path = args.model_path or resolve_model_checkpoint(DEFAULT_MODEL_KEY)
        print(f"Loading LLaVA-OneVision from: {model_path}")
        engine = DiagnosticLlavaEngine(model_path=model_path)
        results_df = run_inference(
            selected_df=selected_df,
            engine=engine,
            output_dir=args.output_dir,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )

    summary = compute_summary(results_df, selected_df)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")
    write_diagnosis_md(args.output_dir, summary, selected_df)


if __name__ == "__main__":
    main()
