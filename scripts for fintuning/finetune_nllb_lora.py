"""LoRA fine-tuning of NLLB-600M, one dataset per run.
Switch the dataset by changing DATASET in the CONFIG block below.
"""
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sacrebleu.metrics import CHRF
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ================= CONFIG =================
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# ---------- SWITCH DATASET HERE: "rajasthani" | "haryanvi" | "odia" | "bengali" ----------
DATASET = "rajasthani"
REVERSE = False          # False: dialect -> English   True: English -> dialect
# ------------------------------------------------------------------------------------------

DATASETS = {
    "rajasthani": dict(
        dialect_file="data_final/train.raj",
        eng_file="data_final/train.eng",
        dialect_lang="hin_Deva",     # proxy for Rajasthani (try "mwr_Deva" if your tokenizer has it)
    ),
    "haryanvi": dict(
        dialect_file="PATH/TO/train.hry",   # <-- placeholder, fill in
        eng_file="PATH/TO/train.eng",       # <-- placeholder, fill in
        dialect_lang="hin_Deva",     # proxy for Haryanvi
    ),
    "odia": dict(
        dialect_file="PATH/TO/train.ory",   # <-- placeholder, fill in
        eng_file="PATH/TO/train.eng",       # <-- placeholder, fill in
        dialect_lang="ory_Orya",
    ),
    "bengali": dict(
        dialect_file="PATH/TO/train.ben",   # <-- placeholder, fill in
        eng_file="PATH/TO/train.eng",       # <-- placeholder, fill in
        dialect_lang="ben_Beng",
    ),
}

ENG_LANG = "eng_Latn"
cfg = DATASETS[DATASET]
if REVERSE:
    SRC_FILE, TGT_FILE = cfg["eng_file"], cfg["dialect_file"]
    SRC_LANG, TGT_LANG = ENG_LANG, cfg["dialect_lang"]
else:
    SRC_FILE, TGT_FILE = cfg["dialect_file"], cfg["eng_file"]
    SRC_LANG, TGT_LANG = cfg["dialect_lang"], ENG_LANG

OUTPUT_DIR = f"nllb_lora_{DATASET}" + ("_en2x" if REVERSE else "")

MAX_LEN = 160
BATCH_SIZE = 16          # lower to 8 (and raise GRAD_ACC to 4) if you run out of GPU memory
GRAD_ACC = 2             # effective batch = BATCH_SIZE * GRAD_ACC
EPOCHS = 10              # early stopping picks the best epoch
LR = 2e-4
LORA_R = 32
LORA_ALPHA = 64
DEV_SIZE = 500           # held-out sentences used for chrF++
FILTER_DATA = True       # drop duplicates and badly misaligned pairs
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
# =========================================

print(f"Dataset: {DATASET}  |  {SRC_LANG} -> {TGT_LANG}  |  output: {OUTPUT_DIR}")


# ---------- Load data ----------
print("Loading training data...")
with open(SRC_FILE, encoding="utf-8") as f:
    src = [l.strip() for l in f]
with open(TGT_FILE, encoding="utf-8") as f:
    tgt = [l.strip() for l in f]
assert len(src) == len(tgt), f"Line count mismatch: {len(src)} vs {len(tgt)}"

pairs = [(s, t) for s, t in zip(src, tgt) if s and t]
if FILTER_DATA:
    seen, kept = set(), []
    for s, t in pairs:
        ratio = (len(s.split()) + 1) / (len(t.split()) + 1)
        if (s, t) in seen or ratio > 3 or ratio < 1 / 3:
            continue
        seen.add((s, t))
        kept.append((s, t))
    print(f"Filtered {len(pairs) - len(kept)} duplicate / misaligned pairs")
    pairs = kept
print(f"Total sentence pairs: {len(pairs)}")

dataset = Dataset.from_dict({"src": [p[0] for p in pairs], "tgt": [p[1] for p in pairs]})
dataset = dataset.train_test_split(test_size=min(DEV_SIZE, max(1, len(pairs) // 10)), seed=SEED)


# ---------- Tokenizer ----------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG


def preprocess(batch):
    # No padding here: the collator pads each batch dynamically and
    # pads labels with -100 so padding is ignored by the loss.
    return tokenizer(
        batch["src"],
        text_target=batch["tgt"],
        max_length=MAX_LEN,
        truncation=True,
    )


print("Tokenizing dataset...")
dataset = dataset.map(preprocess, batched=True, remove_columns=["src", "tgt"])


# ---------- Model ----------
print("Loading base NLLB model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)   # fp32 weights, bf16 mixed precision below
# Needed so evaluation-time generation produces the right target language
model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)


# ---------- LoRA ----------
print("Attaching LoRA adapters...")
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ---------- Metric (chrF++) ----------
chrf = CHRF(word_order=2)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    hyps = tokenizer.batch_decode(preds, skip_special_tokens=True)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return {"chrf": chrf.corpus_score(hyps, [refs]).score}


# ---------- Training ----------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    group_by_length=True,
    bf16=USE_BF16,
    predict_with_generate=True,
    generation_max_length=MAX_LEN,
    generation_num_beams=1,              
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="chrf",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# ---------- Train ----------
print("Starting LoRA fine-tuning...")
trainer.train()

print("Saving best LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Done. Best dev chrF++: {trainer.state.best_metric:.2f}  ->  {OUTPUT_DIR}")