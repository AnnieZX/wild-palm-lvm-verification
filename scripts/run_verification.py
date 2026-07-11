#!/usr/bin/env python3
"""
Purpose:
    Run batched VLM verification inference on a verification dataset.

Input:
    - Verification dataset (images/, prompts/, index.csv or prompt_index.csv)
    - Or an explicit --prompt-index CSV (e.g. ablation condition prompt_index.csv)
    - Model path from configs/model.yaml (override with --model-path)

Output:
    - Per-sample JSON under --results-dir (default: outputs/verification_results/)
    - results_index.csv under --results-dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.model_config import DEFAULT_MODEL_CONFIG, get_active_model_path
from src.paths import VERIFICATION_DATASET_DIR, VERIFICATION_RESULTS_DIR
from src.utils.verification_resume import RESULTS_INDEX_FILENAME
from src.verification.jobs import load_verification_jobs
from src.verification.output_manager import VerificationOutputManager
from src.verification.registry import create_adapter, get_registered_models
from src.verification.runner import RunnerConfig, VerificationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM verification inference on a verification dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=get_registered_models(),
        help="Verification model adapter (default: qwen)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=VERIFICATION_DATASET_DIR,
        help="Verification dataset root (default when --prompt-index is not set)",
    )
    parser.add_argument(
        "--prompt-index",
        type=Path,
        default=None,
        help="Path to prompt_index.csv; image/prompt paths resolve relative to its parent",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=VERIFICATION_RESULTS_DIR,
        help="Directory for per-sample result JSON files",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="YAML config file containing active_model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override active_model from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples per inference batch (default: 4)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Transformers device_map setting (default: auto)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per sample",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose result JSON already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N samples (for debugging)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples already present in results_index.csv or as result JSON files",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment id for resume logging (optional)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Ablation condition code for resume logging (optional)",
    )
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    return get_active_model_path(args.model_config)


def main() -> None:
    args = parse_args()

    if args.prompt_index is not None:
        if not args.prompt_index.exists():
            print(f"Prompt index not found: {args.prompt_index}")
            sys.exit(1)
        dataset_dir = args.prompt_index.parent
        prompt_index = args.prompt_index
    else:
        if not args.dataset_dir.exists():
            print(f"Verification dataset not found: {args.dataset_dir}")
            print(
                "Run scripts/pipeline/generate_verification_dataset.py and "
                "scripts/pipeline/build_verification_prompts.py first."
            )
            sys.exit(1)
        dataset_dir = args.dataset_dir
        prompt_index = None

    try:
        model_path = resolve_model_path(args)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        sys.exit(1)

    try:
        jobs = load_verification_jobs(
            dataset_dir=dataset_dir if prompt_index is None else None,
            prompt_index=prompt_index,
        )
    except FileNotFoundError as error:
        print(error)
        sys.exit(1)

    if args.limit is not None:
        jobs = jobs[: args.limit]

    results_dir = args.results_dir.resolve()
    output_manager = VerificationOutputManager(results_dir)
    results_index_path = results_dir / RESULTS_INDEX_FILENAME

    try:
        adapter = create_adapter(
            args.model,
            model_name=model_path,
            batch_size=args.batch_size,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(error)
        sys.exit(1)

    runner = VerificationRunner(adapter, output_manager)

    print(f"{adapter.model_label} verification inference")
    print(f"  Model key:    {args.model}")
    print(f"  Dataset:      {dataset_dir}")
    if prompt_index is not None:
        print(f"  Prompt index: {prompt_index}")
    print(f"  Results:      {results_dir}")
    print(f"  Model path:   {model_path}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Samples:      {len(jobs)}")
    print()

    summary_df = runner.run(
        jobs,
        RunnerConfig(
            resume=args.resume,
            skip_existing=args.skip_existing,
            experiment_id=args.experiment_id,
            condition=args.condition,
            batch_size=args.batch_size,
        ),
    )

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if summary_df.empty:
        print("No samples processed.")
    else:
        print(summary_df["status"].value_counts().to_string())
    print()
    print(f"Saved results: {results_dir}/")
    print(f"Saved index:     {results_index_path}")


if __name__ == "__main__":
    main()
