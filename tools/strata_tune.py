#!/usr/bin/env python3
"""Equivalent `strata tune` entry point; opt-in and never launches an engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backends/qwen35_llama/tools"))
from strata_tune import main

if __name__ == "__main__":
    raise SystemExit(main())
