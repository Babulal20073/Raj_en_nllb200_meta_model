# Machine Translation (Low-Resource)

```text
prepare_dataset.py -> split and prepare dataset for training

if you want to generate more data :
	use the scripts not named here
else :

	for finetuning: (*change the path)
		├── finetune_indic_lora_direct.py <---- finetune indic trans2 on dataset without hindi pivot    -
		├── finetune_indic_lora.py        <---- finetune indic trans2 on dataset with hindi pivot       |---- hyperparameter tuning (LoRA rank up)
		├── finetune_nllb_lora_direct.py  <---- finetune nllb on dataset without hindi pivot            |
		└── finetune_nllb_lora.py         <---- finetune nllb on dataset with hindi pivot                -

evaluate :
	└── evaluate_lora.py <-- evaluate model (*change the path)
```

---

## File Structure

```text
Raj_en_nllb200_meta_model/
├── scripts for fintuning/            # Primary active pipelines (UNTOUCHED)
│   ├── finetune_nllb_lora.py         # NLLB LoRA (Hindi pivot) + Bayesian Search + K-Fold
│   ├── finetune_nllb_lora_direct.py  # NLLB LoRA (Direct) + Bayesian Search + K-Fold
│   ├── finetune_indic_lora.py        # IndicTrans2 LoRA (Hindi pivot) + Bayesian Search + K-Fold
│   ├── finetune_indic_lora_direct.py # IndicTrans2 LoRA (Direct) + Bayesian Search + K-Fold
│   ├── bayesian_search.py            # Bayesian Optimization CV & Loss Tracker
│   ├── evaluate_clean_comparison.py  # Evaluation (BLEU, chrF, chrF++, Acc, SemAcc)
│   ├── evaluate_indic_lora.py        # IndicTrans2 evaluation
│   ├── evaluate_lora.py              # LoRA checkpoint evaluation
│   ├── visualize_results.py          # HTML & chart visualizer
│   ├── merge_moe_NLLB.py             # Model merging / MoE router script
│   ├── run_pipeline.py               # Unified CLI pipeline runner
│   ├── run_it                        # Pipeline guide
│   └── test setup/                   # 16-test smoke suite (4 approaches x 4 languages)
│
├── notebooks/                        # Interactive Jupyter pipeline notebooks
│   ├── pipeline_1_nllb_pivot.ipynb
│   ├── pipeline_2_nllb_direct.ipynb
│   ├── pipeline_3_indic_pivot.ipynb
│   ├── pipeline_4_indic_direct.ipynb
│   └── all_4_pipelines_workbench.ipynb
│
├── extra_scripts/                    # Neat archive of non-active root scripts
│   ├── data_pipeline/                # Historical corpus & cleaning scripts
│   │   ├── back_translate.py
│   │   ├── build_corpus.py
│   │   ├── labse_filter.py
│   │   ├── prepare_dataset.py
│   │   └── split_dataset.py
│   │
│   ├── early_experiments/            # Early/baseline experiment scripts
│   │   ├── app.py
│   │   ├── evaluate.py
│   │   ├── evaluate_lora.py
│   │   ├── finetune_nllb.py
│   │   ├── finetune_nllb_lora.py
│   │   ├── main.py
│   │   ├── predictions.txt
│   │   └── zero_shot_translate.py
│   │
│   └── stray_artifacts/              # Duplicate/stray files from root
│       ├── =4.57.3
│       ├── re
│       ├── README (2).md
│       └── .git (2)/
│
├── data_clean/                       # Processed datasets
├── data_final/                       # Final training datasets
├── data_split/                       # Train / Val / Test parallel splits
├── comparison_outputs_clean/         # Evaluation benchmarks & metrics
├── agent.md                          # Full technical handbook for autonomous agents
├── GUIDE.md                          # 1-minute quickstart for teammates
├── requirements.txt                  # Complete project dependencies
├── install_deps.sh                   # Environment installer & verifier
└── pyproject.toml                    # UV / Pip manifest
```

---

## About
Built low-resource Indic (Rajasthani, Haryanvi, Odia, Bengali) to English machine translation system using NLLB-200 and IndicTrans2. Applied data augmentation (zero-shot, back-translation, LaBSE filtering). Used LoRA for parameter-efficient fine-tuning with Bayesian Search CV and K-Fold Cross-Validation. Evaluated on BLEU, chrF, chrF++, Exact Accuracy, and Semantic Accuracy.
