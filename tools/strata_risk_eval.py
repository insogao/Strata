"""Opt-in P4 risk evaluation entry point."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backends/qwen35_llama/tools'))
from strata_risk_eval import main
if __name__=='__main__':raise SystemExit(main())
