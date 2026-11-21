import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# ================= CONFIG =================
MODEL_NAME = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = "nllb_lora_finetuned"

TRAIN_RAJ = "data_final/train.raj"
TRAIN_ENG = "data_final/train.eng"

SRC_LANG = "hin_Deva"   # proxy for Rajasthani
TGT_LANG = "eng_Latn"

MAX_LEN = 160
BATCH_SIZE = 4
GRAD_ACC = 4
EPOCHS = 5
LR = 2e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================


# ---------- Load data ----------
print("Loading training data...")
with open(TRAIN_RAJ, encoding="utf-8") as f:
    src = [l.strip() for l in f if l.strip()]
with open(TRAIN_ENG, encoding="utf-8") as f:
    tgt = [l.strip() for l in f if l.strip()]

assert len(src) == len(tgt)
print(f"Total sentence pairs: {len(src)}")

dataset = Dataset.from_dict({"src": src, "tgt": tgt})


# ---------- Tokenizer ----------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG


def preprocess(batch):
    model_inputs = tokenizer(
        batch["src"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        text_target=batch["tgt"],
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing dataset...")
dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=["src", "tgt"]
)


# ---------- Model ----------
print("Loading base NLLB model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
)

# IMPORTANT: DO NOT enable gradient checkpointing for LoRA
model.config.use_cache = False


# ---------- LoRA ----------
print("Attaching LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(DEVICE)


# ---------- Training ----------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    bf16=True if DEVICE == "cuda" else False,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)


# ---------- Train ----------
print("Starting LoRA fine-tuning...")
trainer.train()

print("Saving LoRA model...")
trainer.save_model(OUTPUT_DIR)

print("LoRA fine-tuning complete.")
