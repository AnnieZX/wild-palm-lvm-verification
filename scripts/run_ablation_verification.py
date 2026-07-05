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

from src.paths import VERIFICATION_ABLATION_DIR, VERIFICATION_ABLATION_RESULTS_DIR
from src.prompts.ablation_verification_prompts import ABLATION_CONDITIONS

RUN_VERIFICATION = PROJECT_ROOT / "scripts" / "run_verification.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM verification on one ablation condition.",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(ABLATION_CONDITIONS),
        help="Ablation condition folder name",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    condition_dir = args.ablation_dir / args.condition
    prompt_index = condition_dir / "prompt_index.csv"
    if not prompt_index.exists():
        print(f"Prompt index not found: {prompt_index}")
        print("Run scripts/pipeline/build_ablation_verification_prompts.py first.")
        sys.exit(1)

    results_dir = args.results_root / args.condition
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

    print("Ablation verification inference")
    print(f"  Condition:    {args.condition}")
    print(f"  Prompt index: {prompt_index}")
    print(f"  Results:      {results_dir}")
    print()

    completed = subprocess.run(command, check=False)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
