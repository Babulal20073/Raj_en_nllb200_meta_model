---
name: indic-finetune
description: >
  Autonomous pipeline to fine-tune NLLB-200 or IndicTrans2 on regional Indic languages using Bayesian Search CV and K-Fold Cross-Validation, plot publication loss curves, and evaluate BLEU, chrF, chrF++, Exact Acc, and Semantic Acc. Trigger on "train model", "finetune", "run pipeline on data".
---

# indic-finetune

Autonomous fine-tuning and evaluation pipeline for regional Indic-to-English translation.

## When to Use
Use when user says "train model on <data_location> and output to <output_dir>", "run training", or "/finetune".

## Workflow
1. Identify `data_dir` and `output_dir` from user request.
2. Run pipeline script:
```bash
python3 "scripts for fintuning/run_pipeline.py" --data_dir "<data_dir>" --output_dir "<output_dir>"
```
3. Report results to user:
   - Path to publication loss plot (`<output_dir>/loss_curve.png`).
   - Evaluation metrics table from `comparison_outputs_clean/metrics.csv`.
   - Best hyperparameters found by Bayesian Search CV.

