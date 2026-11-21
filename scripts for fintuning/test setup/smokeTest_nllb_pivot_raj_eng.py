"""Smoke test for NLLB (Hindi Pivot) [raj -> eng]."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from smoke_helper import run_smoke_test

if __name__ == '__main__':
    run_smoke_test(
        model_type="nllb",
        pivot_mode=True,
        src_lang="raj",
        tgt_lang="eng",
    )
