#!/usr/bin/env python3
"""Unified pipeline runner for all 4 fine-tuning approaches.

Runs: Data Setup -> Bayesian Search CV + K-Fold CV -> Loss Tracking -> Test Set Evaluation.
"""
import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

PIPELINES = {
    "nllb_pivot": {
        "script": BASE_DIR / "finetune_nllb_lora.py",
        "output_dir": ROOT_DIR / "nllb_lora_rajasthani_clean",
        "name": "NLLB-200 LoRA (Hindi Pivot)",
    },
    "nllb_direct": {
        "script": BASE_DIR / "finetune_nllb_lora_direct.py",
        "output_dir": ROOT_DIR / "nllb_lora_rajasthani_direct",
        "name": "NLLB-200 LoRA (Direct)",
    },
    "indic_pivot": {
        "script": BASE_DIR / "finetune_indic_lora.py",
        "output_dir": ROOT_DIR / "indictrans2_lora_rajasthani_pivot",
        "name": "IndicTrans2 LoRA (Hindi Pivot)",
    },
    "indic_direct": {
        "script": BASE_DIR / "finetune_indic_lora_direct.py",
        "output_dir": ROOT_DIR / "indictrans2_lora_rajasthani_direct",
        "name": "IndicTrans2 LoRA (Direct)",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run complete fine-tuning + eval pipeline.")
    parser.add_argument(
        "--approach",
        choices=list(PIPELINES.keys()) + ["all"],
        default="nllb_pivot",
        help="Fine-tuning approach to run (or 'all').",
    )
    parser.add_argument("--k_folds", type=int, default=3, help="Number of cross-validation folds.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Bayesian search trials.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training; run evaluation only.")
    args = parser.parse_args()

    targets = list(PIPELINES.keys()) if args.approach == "all" else [args.approach]

    for target in targets:
        info = PIPELINES[target]
        print("\n" + "=" * 70)
        print(f" PIPELINE: {info['name']} [{target}]")
        print(f" Script: {info['script'].name}")
        print(f" K-Folds: {args.k_folds} | Trials: {args.n_trials}")
        print("=" * 70)

        if not args.eval_only:
            env = {"K_FOLDS": str(args.k_folds), "N_TRIALS": str(args.n_trials)}
            cmd = [sys.executable, str(info["script"])]
            print(f"\n[1/2] Running Training & Bayesian Search CV ({info['script'].name})...")
            ret = subprocess.run(cmd, cwd=str(ROOT_DIR), env={**sys.path_importer_cache, **env, **dict(subprocess.os.environ)})
            if ret.returncode != 0:
                print(f"Pipeline {target} failed during training (code {ret.returncode}).")
                continue

        print(f"\n[2/2] Running Evaluation Suite...")
        eval_script = BASE_DIR / "evaluate_clean_comparison.py"
        subprocess.run([sys.executable, str(eval_script)], cwd=str(ROOT_DIR))

    print("\nPipeline execution finished.")


if __name__ == "__main__":
    main()

