"""Smoke test for INDIC (Hindi Pivot) [ben -> eng]."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from smoke_helper import run_smoke_test

if __name__ == '__main__':
    run_smoke_test(
        model_type="indic",
        pivot_mode=True,
        src_lang="ben",
        tgt_lang="eng",
    )
