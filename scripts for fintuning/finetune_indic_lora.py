"""Train an IndicTrans2 LoRA adapter on the same genuine split as NLLB."""
import torch
from datasets import Dataset
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import IndicTransToolkit.collator as indic_collator
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# IndicTransToolkit currently defines this helper inside the collator class,
# but its __call__ method resolves it as a module global. Expose Transformers'
# implementation in that module so the collator can find it.
indic_collator.pad_without_fast_tokenizer_warning = pad_without_fast_tokenizer_warning

from IndicTransToolkit import IndicDataCollator, IndicProcessor


MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
TRAIN_SRC = "data_split/train.raj"
TRAIN_TGT = "data_split/train.eng"
DEV_SRC = "data_split/val.raj"
DEV_TGT = "data_split/val.eng"
SRC_LANG = "hin_Deva"  # Hindi proxy: IndicTrans2 has no Rajasthani language code
TGT_LANG = "eng_Latn"
OUTPUT_DIR = "indictrans2_lora_rajasthani_clean"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM = 2
EPOCHS = 5
LR = 2e-4
SEED = 42


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
dev_pairs = load_pairs(DEV_SRC, DEV_TGT)
if not train_pairs or not dev_pairs:
    raise ValueError("Train or validation split is empty")
print(f"IndicTrans2 | genuine train pairs={len(train_pairs)} | validation pairs={len(dev_pairs)}")
print("Final test set is not used for training or checkpoint selection.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
processor = IndicProcessor(inference=False)
use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=dtype,
    attn_implementation="eager",
)


def tokenize_batch(batch):
    sources = processor.preprocess_batch(
        batch["src"], src_lang=SRC_LANG, tgt_lang=TGT_LANG, is_target=False
    )
    targets = processor.preprocess_batch(
        batch["tgt"], src_lang=TGT_LANG, tgt_lang=SRC_LANG, is_target=True
    )
    model_inputs = tokenizer(sources, max_length=MAX_LEN, truncation=True, padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_LEN, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_split(pairs):
    ds = Dataset.from_dict({"src": [s for s, _ in pairs], "tgt": [t for _, t in pairs]})
    return ds.map(tokenize_batch, batched=True, remove_columns=["src", "tgt"])


train_dataset, eval_dataset = tokenize_split(train_pairs), tokenize_split(dev_pairs)
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
    data_collator=IndicDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Best validation loss: {trainer.state.best_metric:.4f} | saved {OUTPUT_DIR}")
