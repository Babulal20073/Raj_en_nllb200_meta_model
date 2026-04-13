import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

# ================= CONFIG =================

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"

TRAIN_RAJ = "data_final/train.raj"
TRAIN_ENG = "data_final/train.eng"

MAX_LEN = 128          # CRITICAL
TRAIN_BATCH = 1
GRAD_ACCUM = 8
EPOCHS = 5
LR = 2e-5

# =========================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

DEVICE = "cuda"

# ---------------- DATA ----------------
def load_parallel(src, tgt):
    with open(src, encoding="utf-8") as fs, open(tgt, encoding="utf-8") as ft:
        s = [l.strip() for l in fs if l.strip()]
        t = [l.strip() for l in ft if l.strip()]
    return {"src": s, "tgt": t}

print("Loading training data...")
train_ds = Dataset.from_dict(load_parallel(TRAIN_RAJ, TRAIN_ENG))

# ---------------- MODEL ----------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16
).to(DEVICE)

# 🔑 MEMORY SAVERS
model.gradient_checkpointing_enable()
model.config.use_cache = False

# 🔑 FREEZE ENCODER (THIS IS THE KEY)
for param in model.model.encoder.parameters():
    param.requires_grad = False

# ---------------- PREPROCESS ----------------
def preprocess(batch):
    return tokenizer(
        batch["src"],
        text_target=batch["tgt"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

print("Tokenizing...")
train_ds = train_ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])

collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ---------------- TRAINING ----------------
training_args = TrainingArguments(
    output_dir="nllb_finetuned",
    per_device_train_batch_size=TRAIN_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    bf16=True,
    max_grad_norm=0.0,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=1,
    report_to="none",
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=collator
)

print("Starting fine-tuning (encoder frozen, RTX 4050 safe)...")
trainer.train()

trainer.save_model("nllb_finetuned")
tokenizer.save_pretrained("nllb_finetuned")

print("Fine-tuning complete.")
