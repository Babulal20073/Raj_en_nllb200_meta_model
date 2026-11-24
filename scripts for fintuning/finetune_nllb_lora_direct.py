#Train an NLLB LoRA adapter directly on Rajasthani→English pairs.

import sys
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

sys.path.append(str(Path(__file__).resolve().parent))
from bayesian_search import bayesian_search_cv


def resolve_path(p):
    path = Path(p)
    if path.exists():
        return str(path)
    parent_data = Path(__file__).resolve().parent.parent / "data_split" / path.name
    if parent_data.exists():
        return str(parent_data)
    return str(path)


MODEL_REPO = "facebook/nllb-200-distilled-600M"
_snapshots = Path.home() / ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots"
_cached = [p for p in _snapshots.glob("*") if (p / "pytorch_model.bin").is_file()]
MODEL_NAME = str(_cached[0]) if _cached else MODEL_REPO
TRAIN_SRC = resolve_path("../data_split/train.raj")
TRAIN_TGT = resolve_path("../data_split/train.eng")
DEV_SRC   = resolve_path("../data_split/val.raj")
DEV_TGT   = resolve_path("../data_split/val.eng")
TGT_LANG  = "eng_Latn"
OUTPUT_DIR = "nllb_lora_rajasthani_direct"
MAX_LEN   = 128
TRAIN_BATCH_SIZE = 4   
GRAD_ACCUM = 8
EPOCHS    = 10
LR        = 2e-2
SEED      = 80085

N_TRIALS = 10
K_FOLDS = int(os.environ.get("K_FOLDS", 3))
SEARCH_SPACE = {
    "lr": (1e-5, 1e-2, "log"),
    "r": [4, 8, 16, 20, 32, 64, 128],
    "lora_alpha": [8, 16, 32, 48, 64, 128],
    "lora_dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
}


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
def create_model(r=20, lora_alpha=48, lora_dropout=0.25):
    m = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    m.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    m = get_peft_model(
        m,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj"],
        ),
    )
    return m


def tokenize(batch):
    """Tokenise source with no language prefix and target as English."""
    model_inputs = tokenizer(
        batch["src"],
        max_length=MAX_LEN,
        truncation=True,
    )
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


def train_trial(params, trial_dir, train_data=None, eval_data=None):
    train_ds = train_data if train_data is not None else train_dataset
    eval_ds = eval_data if eval_data is not None else eval_dataset
    m = create_model(
        r=int(params.get("r", 20)),
        lora_alpha=int(params.get("lora_alpha", 48)),
        lora_dropout=float(params.get("lora_dropout", 0.25)),
    )
    args = Seq2SeqTrainingArguments(
        output_dir=trial_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=1,
        label_names=["labels"],
        remove_unused_columns=False,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=float(params.get("lr", LR)),
        num_train_epochs=EPOCHS,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.1,
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
        model=m,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    best_loss = float(trainer.state.best_metric) if trainer.state.best_metric is not None else float("inf")
    return best_loss, m, trainer.state.log_history


bayesian_search_cv(
    train_fn=train_trial,
    search_space=SEARCH_SPACE,
    n_trials=N_TRIALS,
    initial_params={"lr": LR, "r": 20, "lora_alpha": 48, "lora_dropout": 0.25},
    output_dir=OUTPUT_DIR,
    tokenizer=tokenizer,
    seed=SEED,
    k_folds=K_FOLDS,
    dataset=train_dataset,
)
