#!/usr/bin/env python3
"""Deprecated entry point. Use scripts/run_full_inference.py instead."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "run_full_inference.py"), run_name="__main__")
