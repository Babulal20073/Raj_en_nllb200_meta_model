"""Smoke test for NLLB (Direct) [har -> eng]."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from smoke_helper import run_smoke_test

if __name__ == '__main__':
    run_smoke_test(
        model_type="nllb",
        pivot_mode=False,
        src_lang="har",
        tgt_lang="eng",
    )
