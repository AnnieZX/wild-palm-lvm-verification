#!/usr/bin/env python3
"""Check Python, PyTorch, and CUDA availability for cluster deployment."""

from __future__ import annotations

import sys


def main() -> None:
    print(f"Python version: {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        print("Torch version: not installed")
        print("CUDA available: unknown (torch not installed)")
        print("CUDA device count: unknown (torch not installed)")
        print("GPU names: unavailable (torch not installed)")
        print()
        print("Cluster environment check complete.")
        return

    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for index in range(device_count):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    else:
        print("CUDA device count: 0")
        print("GPU names: unavailable (CUDA not available)")

    print()
    print("Cluster environment check complete.")


if __name__ == "__main__":
    main()
