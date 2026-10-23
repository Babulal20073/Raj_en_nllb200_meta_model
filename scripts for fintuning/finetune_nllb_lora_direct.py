"""Train an NLLB LoRA adapter directly on Rajasthani→English pairs.

Unlike finetune_nllb_lora.py, this script does NOT use Hindi (hin_Deva) as a
proxy language code for the source side.  Instead the source text is tokenised
with *no* forced source-language token so the model must learn the mapping
purely from the paired data without any intermediate-language detour.
The target side still uses eng_Latn so that NLLB's forced-BOS decoding works.
"""
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


MODEL_REPO = "facebook/nllb-200-distilled-600M"
_snapshots = Path.home() / ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots"
_cached = [p for p in _snapshots.glob("*") if (p / "pytorch_model.bin").is_file()]
MODEL_NAME = str(_cached[0]) if _cached else MODEL_REPO
TRAIN_SRC = "data_split/train.raj"
TRAIN_TGT = "data_split/train.eng"
DEV_SRC   = "data_split/val.raj"
DEV_TGT   = "data_split/val.eng"
# No intermediate language: we do NOT set src_lang to hin_Deva.
# The tokenizer will encode source text without prepending any language token,
# letting the model learn Rajasthani patterns from scratch via LoRA.
TGT_LANG  = "eng_Latn"
OUTPUT_DIR = "nllb_lora_rajasthani_direct"
MAX_LEN   = 128
TRAIN_BATCH_SIZE = 1   # CPU-safe; increase only when GPU memory is available
GRAD_ACCUM = 8
EPOCHS    = 5
LR        = 2e-4
SEED      = 42


def load_pairs(src_path, tgt_path):
    with open(src_path, encoding="utf-8") as f:
        src = [line.strip() for line in f]
    with open(tgt_path, encoding="utf-8") as f:
        tgt = [line.strip() for line in f]
    if len(src) != len(tgt):
        raise ValueError(f"Line mismatch: {src_path}={len(src)}, {tgt_path}={len(tgt)}")
    seen, pairs = set(), []
    for source, target in zip(src, tgt):
        if not source or not target or (source, target) in seen:
            continue
        ratio = (len(source.split()) + 1) / (len(target.split()) + 1)
        if ratio < 1 / 3 or ratio > 3:
            continue
        seen.add((source, target))
        pairs.append((source, target))
    return pairs


train_pairs = load_pairs(TRAIN_SRC, TRAIN_TGT)
dev_pairs   = load_pairs(DEV_SRC, DEV_TGT)
if not train_pairs or not dev_pairs:
    raise ValueError("Train or validation split is empty")
print(f"NLLB (direct) | genuine train pairs={len(train_pairs)} | validation pairs={len(dev_pairs)}")
print("Final test set is not used for training or checkpoint selection.")

# Load tokenizer WITHOUT setting src_lang so no Hindi/proxy BOS is prepended.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# src_lang is intentionally left unset (no intermediate language).
tokenizer.tgt_lang = TGT_LANG

use_cuda  = torch.cuda.is_available()
use_bf16  = use_cuda and torch.cuda.is_bf16_supported()
dtype     = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
# Force English BOS during decoding so beam search stays in English.
model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)


def tokenize(batch):
    """Tokenise source with no language prefix and target as English."""
    # Encode source text directly — no src_lang token prepended.
    model_inputs = tokenizer(
        batch["src"],
        max_length=MAX_LEN,
        truncation=True,
    )
    # Encode target with the English language token.
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["tgt"],
            max_length=MAX_LEN,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def to_dataset(pairs):
    ds = Dataset.from_dict({"src": [s for s, _ in pairs], "tgt": [t for _, t in pairs]})
    return ds.map(tokenize, batched=True, remove_columns=["src", "tgt"])


train_dataset, eval_dataset = to_dataset(train_pairs), to_dataset(dev_pairs)

model = get_peft_model(
    model,
    LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj"],
    ),
)
model.print_trainable_parameters()

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=1,
    label_names=["labels"],
    remove_unused_columns=False,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    bf16=use_bf16,
    fp16=use_cuda and not use_bf16,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    predict_with_generate=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=25,
    report_to="none",
    dataloader_num_workers=0,
    seed=SEED,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Best validation loss: {trainer.state.best_metric:.4f} | saved {OUTPUT_DIR}")
