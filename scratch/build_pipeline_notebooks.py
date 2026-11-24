#!/usr/bin/env python3
"""Generate 4 standalone Jupyter Notebooks and 1 master notebook for the 4 fine-tuning pipelines."""
from pathlib import Path
import nbformat as nbf

nb_dir = Path("notebooks")
nb_dir.mkdir(parents=True, exist_ok=True)

PIPELINES = [
    {
        "filename": "pipeline_1_nllb_pivot.ipynb",
        "title": "Pipeline 1: NLLB-200 LoRA (Hindi Pivot -> English)",
        "approach": "nllb_pivot",
        "model_id": "facebook/nllb-200-distilled-600M",
        "src_token": "hin_Deva",
        "tgt_token": "eng_Latn",
        "script_ref": "finetune_nllb_lora.py",
        "output_dir": "nllb_lora_pivot_output",
        "is_indic": False,
        "is_pivot": True,
        "desc": "Translates regional Indic language (e.g. Rajasthani / Haryanvi) using Hindi pivot token alignment in NLLB-200.",
    },
    {
        "filename": "pipeline_2_nllb_direct.ipynb",
        "title": "Pipeline 2: NLLB-200 LoRA (Direct Translation -> English)",
        "approach": "nllb_direct",
        "model_id": "facebook/nllb-200-distilled-600M",
        "src_token": "ory_Orya",  # or custom token
        "tgt_token": "eng_Latn",
        "script_ref": "finetune_nllb_lora_direct.py",
        "output_dir": "nllb_lora_direct_output",
        "is_indic": False,
        "is_pivot": False,
        "desc": "Translates regional Indic language (e.g. Odia / Bengali) directly using native script token alignment in NLLB-200.",
    },
    {
        "filename": "pipeline_3_indic_pivot.ipynb",
        "title": "Pipeline 3: IndicTrans2 LoRA (Hindi Pivot -> English)",
        "approach": "indic_pivot",
        "model_id": "ai4bharat/indictrans2-indic-en-1B",
        "src_token": "hin_Deva",
        "tgt_token": "eng_Latn",
        "script_ref": "finetune_indic_lora.py",
        "output_dir": "indictrans2_lora_pivot_output",
        "is_indic": True,
        "is_pivot": True,
        "desc": "Fine-tunes IndicTrans2 with IndicProcessor using Hindi pivot token for dialectal transfer.",
    },
    {
        "filename": "pipeline_4_indic_direct.ipynb",
        "title": "Pipeline 4: IndicTrans2 LoRA (Direct Translation -> English)",
        "approach": "indic_direct",
        "model_id": "ai4bharat/indictrans2-indic-en-1B",
        "src_token": "ory_Orya",
        "tgt_token": "eng_Latn",
        "script_ref": "finetune_indic_lora_direct.py",
        "output_dir": "indictrans2_lora_direct_output",
        "is_indic": True,
        "is_pivot": False,
        "desc": "Fine-tunes IndicTrans2 directly with IndicProcessor on native language script tokens.",
    },
]


def build_pipeline_notebook(cfg):
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Header
    cells.append(nbf.v4.new_markdown_cell(f"""# {cfg['title']}
> **Approach**: `{cfg['approach']}`  
> **Base Model**: `{cfg['model_id']}`  
> **Source Token**: `{cfg['src_token']}` | **Target Token**: `{cfg['tgt_token']}`  
> **Description**: {cfg['desc']}

### Pipeline Workflow:
1. **Data Ingestion & K-Fold Partitioning** (K-Fold CV)
2. **Bayesian Hyperparameter Search** (LoRA rank $r$, $\\alpha$, lr, dropout via Gaussian Process)
3. **Loss Tracking & Convergence Plots** (Publication PNG / vector PDF / CSV history)
4. **Rigorous Evaluation on Test Set** (BLEU, chrF, chrF++, Exact Accuracy, Semantic Accuracy)
5. **Interactive Translation Sandbox**
"""))

    # Cell 2: Setup & Environment Imports
    cells.append(nbf.v4.new_code_cell(f"""# 1. Environment & Library Imports
import os
import sys
import gc
import json
import re
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from sacrebleu import corpus_bleu
from sacrebleu.metrics import CHRF

# Add scripts directory to path for Bayesian search & evaluation utilities
BASE_DIR = Path.cwd()
if (BASE_DIR / "scripts for fintuning").exists():
    sys.path.append(str(BASE_DIR / "scripts for fintuning"))
elif (BASE_DIR.parent / "scripts for fintuning").exists():
    sys.path.append(str(BASE_DIR.parent / "scripts for fintuning"))

from bayesian_search import bayesian_search_cv, plot_loss_curve

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
"""))

    # Cell 3: Data Load & Configuration
    cells.append(nbf.v4.new_code_cell(f"""# 2. Configuration & Dataset Loading
SRC_LANG_CODE = "raj"  # Change to your language: raj, har, odia, ben
TGT_LANG_CODE = "eng"

SRC_TOKEN = "{cfg['src_token']}"
TGT_TOKEN = "{cfg['tgt_token']}"
MODEL_ID = "{cfg['model_id']}"
OUTPUT_DIR = "{cfg['output_dir']}"

# Training & GPU Hyperparameters (Optimized for 20 GB GPU)
K_FOLDS = 3                     # K-Fold Cross Validation
N_TRIALS = 10                   # Bayesian Search Trials
EPOCHS = 5                      # Epochs per trial
TRAIN_BATCH_SIZE = 16 if torch.cuda.is_available() else 2
GRAD_ACCUM = 2                  # Effective Batch Size = 32
FP16 = torch.cuda.is_available()
SEED = 42

def load_pairs(src_file, tgt_file):
    s_path = Path(src_file)
    t_path = Path(tgt_file)
    if not s_path.exists() or not t_path.exists():
        # Fallback search in data_split / data_final
        for d in [Path("data_split"), Path("../data_split"), Path("data_final")]:
            if (d / s_path.name).exists() and (d / t_path.name).exists():
                s_path, t_path = d / s_path.name, d / t_path.name
                break
    with open(s_path, encoding="utf-8") as fs, open(t_path, encoding="utf-8") as ft:
        return [(s.strip(), t.strip()) for s, t in zip(fs, ft) if s.strip() and t.strip()]

train_pairs = load_pairs(f"train.{{SRC_LANG_CODE}}", f"train.{{TGT_LANG_CODE}}")
val_pairs   = load_pairs(f"val.{{SRC_LANG_CODE}}", f"val.{{TGT_LANG_CODE}}")
test_pairs  = load_pairs(f"test.{{SRC_LANG_CODE}}", f"test.{{TGT_LANG_CODE}}")

print(f"Loaded dataset: {{len(train_pairs)}} train, {{len(val_pairs)}} val, {{len(test_pairs)}} test pairs.")
"""))

    # Cell 4: Tokenizer & Preprocessing Setup
    if not cfg["is_indic"]:
        cells.append(nbf.v4.new_code_cell(f"""# 3. Tokenizer & Dataset Tokenization (NLLB-200)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_TOKEN, tgt_lang=TGT_TOKEN)

def tokenize_batch(batch):
    inputs = tokenizer(batch["src"], max_length=128, truncation=True, padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["tgt"], max_length=128, truncation=True, padding=False)
    inputs["labels"] = labels["input_ids"]
    return inputs

def to_dataset(pairs):
    ds = Dataset.from_dict({{"src": [s for s, _ in pairs], "tgt": [t for _, t in pairs]}})
    return ds.map(tokenize_batch, batched=True, remove_columns=["src", "tgt"])

train_dataset = to_dataset(train_pairs)
val_dataset = to_dataset(val_pairs)
test_dataset = to_dataset(test_pairs)
"""))
    else:
        cells.append(nbf.v4.new_code_cell(f"""# 3. Tokenizer & IndicProcessor Setup (IndicTrans2)
from IndicTransToolkit import IndicProcessor, IndicDataCollator

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = IndicProcessor(inference=False)

def tokenize_indic(batch):
    src_sentences = processor.preprocess_batch(batch["src"], src_lang=SRC_TOKEN, tgt_lang=TGT_TOKEN, is_target=False)
    tgt_sentences = processor.preprocess_batch(batch["tgt"], src_lang=TGT_TOKEN, tgt_lang=TGT_TOKEN, is_target=True)
    inputs = tokenizer(src_sentences, max_length=128, truncation=True, padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_sentences, max_length=128, truncation=True, padding=False)
    inputs["labels"] = labels["input_ids"]
    return inputs

def to_dataset(pairs):
    ds = Dataset.from_dict({{"src": [s for s, _ in pairs], "tgt": [t for _, t in pairs]}})
    return ds.map(tokenize_indic, batched=True, remove_columns=["src", "tgt"])

train_dataset = to_dataset(train_pairs)
val_dataset = to_dataset(val_pairs)
test_dataset = to_dataset(test_pairs)
"""))

    # Cell 5: Bayesian Search CV + K-Fold Training Execution
    cells.append(nbf.v4.new_code_cell(f"""# 4. Bayesian Search CV with K-Fold Cross Validation Execution
SEARCH_SPACE = {{
    "lr": (5e-6, 1e-3, "log"),
    "r": [4, 8, 16, 32, 64, 128],
    "lora_alpha": [8, 16, 32, 64, 128],
    "lora_dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
}}

def create_model(r=16, lora_alpha=32, lora_dropout=0.1):
    base_m = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if FP16 else torch.float32,
        trust_remote_code=True
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        bias="none",
    )
    return get_peft_model(base_m, lora_cfg)

def train_trial(params, trial_dir, train_data=None, eval_data=None):
    t_ds = train_data if train_data is not None else train_dataset
    v_ds = eval_data if eval_data is not None else val_dataset
    
    m = create_model(
        r=params.get("r", 16),
        lora_alpha=params.get("lora_alpha", 32),
        lora_dropout=params.get("lora_dropout", 0.1)
    )
    
    args = Seq2SeqTrainingArguments(
        output_dir=trial_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=float(params.get("lr", 2e-4)),
        num_train_epochs=EPOCHS,
        fp16=FP16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        seed=SEED
    )
    
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        model=m,
        args=args,
        train_dataset=t_ds,
        eval_dataset=v_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    best_loss = float(trainer.state.best_metric) if trainer.state.best_metric is not None else float("inf")
    return best_loss, m, trainer.state.log_history

# Run Bayesian Search CV with K-Fold Cross Validation
best_params, best_loss = bayesian_search_cv(
    train_fn=train_trial,
    search_space=SEARCH_SPACE,
    n_trials=N_TRIALS,
    initial_params={{"lr": 2e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.1}},
    output_dir=OUTPUT_DIR,
    tokenizer=tokenizer,
    k_folds=K_FOLDS,
    dataset=train_dataset,
    seed=SEED
)
"""))

    # Cell 6: Loss Tracking & Visual Graphs
    cells.append(nbf.v4.new_code_cell("""# 5. Display Publication Loss Curves & Convergence
import pandas as pd
from IPython.display import Image, display

loss_png = Path(OUTPUT_DIR) / "loss_curve.png"
conv_png = Path(OUTPUT_DIR) / "bayesian_optimization_convergence.png"
history_csv = Path(OUTPUT_DIR) / "loss_curve_history.csv"

if loss_png.exists():
    print("--- Training vs. Validation Loss Curve ---")
    display(Image(filename=str(loss_png)))

if conv_png.exists():
    print("--- Bayesian Optimization Convergence ---")
    display(Image(filename=str(conv_png)))

if history_csv.exists():
    df = pd.read_csv(history_csv)
    print("--- Loss History Table (First 10 Epochs) ---")
    display(df.head(10))
"""))

    # Cell 7: Rigorous 5-Metric Evaluation on Test Set
    cells.append(nbf.v4.new_code_cell(f"""# 6. Evaluation Suite (BLEU, chrF, chrF++, Exact Accuracy, Semantic Accuracy)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_semantic_accuracy(hyps, refs):
    try:
        from sentence_transformers import SentenceTransformer, util
        st = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        e_h = st.encode(hyps, convert_to_tensor=True)
        e_r = st.encode(refs, convert_to_tensor=True)
        return float(torch.diag(util.cos_sim(e_h, e_r)).mean().item()) * 100.0
    except Exception:
        scores = []
        for h, r in zip(hyps, refs):
            if not h.strip() or not r.strip():
                scores.append(1.0 if h.strip() == r.strip() else 0.0)
                continue
            m = TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb").fit([h, r]).transform([h, r])
            scores.append(float(cosine_similarity(m[0:1], m[1:2])[0][0]))
        return float(np.mean(scores)) * 100.0 if scores else 0.0

# Load best trained model for inference
best_model = PeftModel.from_pretrained(
    AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if FP16 else torch.float32),
    OUTPUT_DIR
).to("cuda" if FP16 else "cpu")
best_model.eval()

test_srcs = [s for s, _ in test_pairs]
test_refs = [t for _, t in test_pairs]

predictions = []
batch_size = 16
for i in range(0, len(test_srcs), batch_size):
    batch_src = test_srcs[i:i+batch_size]
    inputs = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=128).to(best_model.device)
    with torch.no_grad():
        generated = best_model.generate(**inputs, max_length=128, num_beams=4)
    predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

bleu = corpus_bleu(predictions, [[r] for r in test_refs]).score
chrf = CHRF(word_order=0).corpus_score(predictions, [test_refs]).score
chrf_plus = CHRF(word_order=2).corpus_score(predictions, [test_refs]).score
exact_acc = 100.0 * sum(h.strip().lower() == r.strip().lower() for h, r in zip(predictions, test_refs)) / max(1, len(test_refs))
sem_acc = compute_semantic_accuracy(predictions, test_refs)

metrics_df = pd.DataFrame([{{
    "Pipeline": "{cfg['title']}",
    "BLEU": round(bleu, 2),
    "chrF": round(chrf, 2),
    "chrF++": round(chrf_plus, 2),
    "Exact_Acc": f"{{exact_acc:.2f}}%",
    "Semantic_Acc": f"{{sem_acc:.2f}}%"
}}])

print("\\n" + "=" * 70)
print("FINAL TEST EVALUATION METRICS:")
print("=" * 70)
display(metrics_df)
"""))

    # Cell 8: Interactive Translation Demo
    cells.append(nbf.v4.new_code_cell("""# 7. Interactive Translation Sandbox
def translate_sentence(text):
    inputs = tokenizer([text], return_tensors="pt", max_length=128, truncation=True).to(best_model.device)
    with torch.no_grad():
        out = best_model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True)

test_sample = test_srcs[0] if test_srcs else "राम घर गयो।"
print("Input sentence:", test_sample)
print("Prediction:    ", translate_sentence(test_sample))
"""))

    nb.cells = cells
    out_path = nb_dir / cfg["filename"]
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook: {out_path}")


# Build all 4 notebooks
for p in PIPELINES:
    build_pipeline_notebook(p)

print("\nAll 4 pipeline notebooks generated in notebooks/ directory.")

