#!/usr/bin/env python3
"""
Purpose:
    Run verification inference on an ablation condition (A1–A5).

Delegates to ``run_verification.py`` with ablation-specific paths.
No separate inference logic is implemented here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import (
    ABLATION_CODES,
    VERIFICATION_ABLATION_DIR,
    VERIFICATION_ABLATION_RESULTS_DIR,
    resolve_ablation_condition_name,
)
from src.prompts.ablation_verification_prompts import ABLATION_CONDITIONS

RUN_VERIFICATION = PROJECT_ROOT / "scripts" / "run_verification.py"

CONDITION_CHOICES = sorted({*ABLATION_CONDITIONS, *ABLATION_CODES})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM verification on one ablation condition.",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=CONDITION_CHOICES,
        help="Ablation condition (A1 … A5 or full folder name)",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=VERIFICATION_ABLATION_DIR,
        help="Root directory containing ablation condition folders",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=VERIFICATION_ABLATION_RESULTS_DIR,
        help="Root directory for ablation inference results",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Explicit results directory (overrides --results-root / condition)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N samples (forwarded to run_verification.py)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose result JSON already exists",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size (forwarded to run_verification.py)",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Model YAML config (forwarded to run_verification.py)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model path (forwarded to run_verification.py)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    condition_name = resolve_ablation_condition_name(args.condition)
    condition_dir = args.ablation_dir / condition_name
    prompt_index = condition_dir / "prompt_index.csv"
    if not prompt_index.exists():
        print(f"Prompt index not found: {prompt_index}")
        print("Run scripts/pipeline/build_ablation_verification_prompts.py first.")
        sys.exit(1)

    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        results_dir = args.results_root / condition_name

    command = [
        sys.executable,
        str(RUN_VERIFICATION),
        "--prompt-index",
        str(prompt_index),
        "--results-dir",
        str(results_dir),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.skip_existing:
        command.append("--skip-existing")
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.model_config is not None:
        command.extend(["--model-config", str(args.model_config)])
    if args.model_path is not None:
        command.extend(["--model-path", args.model_path])

    print("Ablation verification inference")
    print(f"  Condition:    {condition_name}")
    print(f"  Prompt index: {prompt_index}")
    print(f"  Results:      {results_dir}")
    print()

    completed = subprocess.run(command, check=False)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
