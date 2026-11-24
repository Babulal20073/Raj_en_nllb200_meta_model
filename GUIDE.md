# User Guide

Train Indic-to-English translation models (**Rajasthani**, **Haryanvi**, **Odia**, **Bengali**).

---

### 1. Setup (One-time)
```bash
./install_deps.sh
```

---

### 2. Method A: Train via Agent Skill (Prompt Only)

Tell the agent in chat:
> *"Train model on `data_split` and output to `my_output`"*

The agent automatically:
- Detects language (`raj`, `har`, `odia`, `ben`).
- Runs Bayesian Search CV + 3-Fold Cross-Validation (10 trials).
- Saves winning adapter in `<output_dir>`.
- Displays loss curve plot and 5 evaluation scores (BLEU, chrF, chrF++, Exact Acc, Semantic Acc).

---

### 3. Method B: Train via Terminal CLI

```bash
python3 "scripts for fintuning/run_pipeline.py" --data_dir data_split --output_dir my_output
```

*Options:*
- `--approach`: `nllb_pivot` (dialects), `nllb_direct` (native script), `indic_pivot`, `indic_direct`.
- `--k_folds`: Cross-validation folds (default: `3`).
- `--n_trials`: Bayesian trials (default: `10`).

---

### 4. Method C: Interactive Jupyter Notebooks

Launch Jupyter and open any pipeline from `notebooks/`:
```bash
jupyter notebook notebooks/
```

---

### 5. Deliverables for Paper

Located in `<output_dir>`:
1. `loss_curve.png` / `loss_curve.pdf` (Loss curve plot)
2. `best_params.json` (Winning hyperparameters)
3. `comparison_outputs_clean/metrics.csv` (BLEU, chrF, chrF++, Exact Acc, Semantic Acc)
