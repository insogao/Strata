"""Opt-in unified single-executor API. See docs/UNIFIED_SINGLE_GPU_DELIVERY.md."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backends/qwen35_llama/tools'))
from strata_unified import main
if __name__ == '__main__':
    raise SystemExit(main())
