# GUIDE

Fine-tune your assigned regional language to English: **Rajasthani (`raj`)**, **Haryanvi (`har`)**, **Odia (`odia`)**, or **Bengali (`ben`)**.

---

### 1. Setup (One-time)
```bash
./install_deps.sh
```
*(Verify anytime with `./install_deps.sh --check`)*

---

### 2. Add Data to `data_split/`
Place line-matched, parallel UTF-8 text files in `data_split/`:
- `train.<lang>` & `train.eng`
- `val.<lang>` & `val.eng`
- `test.<lang>` & `test.eng`

---

### 3. Test & Train (Pick your language)

| Language | 1. Smoke Test (30 sec dry-run) | 2. Full Train Script (`scripts for fintuning/`) |
| :--- | :--- | :--- |
| **Rajasthani (`raj`)** | `python3 "scripts for fintuning/test setup/smokeTest_nllb_pivot_raj_eng.py"` | `python3 "scripts for fintuning/finetune_nllb_lora.py"` |
| **Haryanvi (`har`)** | `python3 "scripts for fintuning/test setup/smokeTest_nllb_pivot_har_eng.py"` | `python3 "scripts for fintuning/finetune_nllb_lora.py"` |
| **Odia (`odia`)** | `python3 "scripts for fintuning/test setup/smokeTest_nllb_not_odia_eng.py"` | `python3 "scripts for fintuning/finetune_nllb_lora_direct.py"` |
| **Bengali (`ben`)** | `python3 "scripts for fintuning/test setup/smokeTest_nllb_not_ben_eng.py"` | `python3 "scripts for fintuning/finetune_nllb_lora_direct.py"` |

> **20 GB GPU tip**: In your script, set `TRAIN_BATCH_SIZE = 16`, `GRAD_ACCUM = 2`, and `fp16 = True`.  
> Bayesian Search automatically optimizes hyperparameters across 10 trials and saves the best model.

---

### 4. Evaluate & Submit

Run evaluation on your test set:
```bash
python3 "scripts for fintuning/evaluate_clean_comparison.py"
```

**Send these 3 files for the paper:**
1. `loss_curve.png` *(from your model output folder)*
2. `best_params.json` *(winning hyperparameters)*
3. `comparison_outputs_clean/metrics.csv` *(BLEU, chrF, chrF++, Exact Acc, Semantic Acc)*

---
*For in-depth architecture diagrams, token mappings, or MoE adapter merging, see [`agent.md`](agent.md).*
