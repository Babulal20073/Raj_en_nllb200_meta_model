#!/usr/bin/env python3
"""Unified pipeline runner for all 4 fine-tuning approaches.

Runs: Data Setup -> Bayesian Search CV + K-Fold CV -> Loss Tracking -> Test Set Evaluation.
Usage:
    python3 "scripts for fintuning/run_pipeline.py" --data_dir data_split --output_dir my_output --approach nllb_pivot
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

PIPELINES = {
    "nllb_pivot": {
        "script": BASE_DIR / "finetune_nllb_lora.py",
        "default_out": "nllb_lora_rajasthani_clean",
        "name": "NLLB-200 LoRA (Hindi Pivot)",
    },
    "nllb_direct": {
        "script": BASE_DIR / "finetune_nllb_lora_direct.py",
        "default_out": "nllb_lora_rajasthani_direct",
        "name": "NLLB-200 LoRA (Direct)",
    },
    "indic_pivot": {
        "script": BASE_DIR / "finetune_indic_lora.py",
        "default_out": "indictrans2_lora_rajasthani_pivot",
        "name": "IndicTrans2 LoRA (Hindi Pivot)",
    },
    "indic_direct": {
        "script": BASE_DIR / "finetune_indic_lora_direct.py",
        "default_out": "indictrans2_lora_rajasthani_direct",
        "name": "IndicTrans2 LoRA (Direct)",
    },
}


def detect_lang(data_dir):
    d = Path(data_dir)
    if not d.exists():
        return "raj"
    for f in d.glob("train.*"):
        ext = f.suffix.lstrip(".")
        if ext and ext not in ["eng", "en", "txt", "csv", "json"]:
            return ext
    return "raj"


def main():
    parser = argparse.ArgumentParser(description="Run complete fine-tuning + eval pipeline.")
    parser.add_argument("--data_dir", type=str, default="data_split", help="Directory containing train/val/test splits.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for model, plots, and metrics.")
    parser.add_argument("--src_lang", type=str, default=None, help="Source language code (e.g. raj, har, odia, ben).")
    parser.add_argument(
        "--approach",
        choices=list(PIPELINES.keys()) + ["auto", "all"],
        default="auto",
        help="Fine-tuning approach to run (auto-selected by language if 'auto').",
    )
    parser.add_argument("--k_folds", type=int, default=3, help="Number of cross-validation folds.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Bayesian search trials.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training; run evaluation only.")
    args = parser.parse_args()

    src_lang = args.src_lang or detect_lang(args.data_dir)

    # Auto-select approach based on language if requested
    if args.approach == "auto":
        approach = "nllb_pivot" if src_lang in ["raj", "har"] else "nllb_direct"
        targets = [approach]
    elif args.approach == "all":
        targets = list(PIPELINES.keys())
    else:
        targets = [args.approach]

    for target in targets:
        info = PIPELINES[target]
        out_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / info["default_out"]
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f" PIPELINE: {info['name']} [{target}]")
        print(f" Source Lang: {src_lang} | Data Dir: {args.data_dir}")
        print(f" Output Dir: {out_dir}")
        print(f" K-Folds: {args.k_folds} | Trials: {args.n_trials}")
        print("=" * 70)

        env = {
            **os.environ,
            "DATA_DIR": str(Path(args.data_dir).resolve()),
            "OUTPUT_DIR": str(out_dir.resolve()),
            "SRC_LANG": src_lang,
            "K_FOLDS": str(args.k_folds),
            "N_TRIALS": str(args.n_trials),
        }

        if not args.eval_only:
            cmd = [sys.executable, str(info["script"])]
            print(f"\n[1/2] Running Training & Bayesian Search CV ({info['script'].name})...")
            ret = subprocess.run(cmd, cwd=str(ROOT_DIR), env=env)
            if ret.returncode != 0:
                print(f"Error: Pipeline {target} training failed with code {ret.returncode}.")
                continue

        print(f"\n[2/2] Running Evaluation Suite...")
        eval_script = BASE_DIR / "evaluate_clean_comparison.py"
        subprocess.run([sys.executable, str(eval_script)], cwd=str(ROOT_DIR), env=env)

        # Print report summary
        loss_png = out_dir / "loss_curve.png"
        metrics_csv = ROOT_DIR / "comparison_outputs_clean/metrics.csv"
        print("\n" + "=" * 70)
        print(" PIPELINE RESULT SUMMARY:")
        print(f" - Model Checkpoint: {out_dir}")
        if loss_png.exists():
            print(f" - Loss Curve Plot:  {loss_png}")
        if metrics_csv.exists():
            print(f" - Metrics Table:    {metrics_csv}")
            try:
                import pandas as pd
                print("\n" + pd.read_csv(metrics_csv).to_string(index=False))
            except Exception:
                pass
        print("=" * 70)


if __name__ == "__main__":
    main()
