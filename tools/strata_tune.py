#!/usr/bin/env python3
"""Opt-in tuner entry; only experiment --execute launches a leased engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backends/qwen35_llama/tools"))
from strata_tune import main

if __name__ == "__main__":
    raise SystemExit(main())
