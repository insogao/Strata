"""Finite P4 synthetic-reference comparison entry point."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backends/qwen35_llama/tools'))
from strata_risk_campaign import main
if __name__=='__main__':raise SystemExit(main())
