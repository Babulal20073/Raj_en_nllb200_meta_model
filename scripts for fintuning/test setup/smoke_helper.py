"""Shared test harness for smoke testing fine-tuning approaches and language pairs."""
import os
import shutil
import sys
import builtins
from pathlib import Path

if not hasattr(builtins, "OnnxSeq2SeqConfigWithPast"):
    builtins.OnnxSeq2SeqConfigWithPast = object

# Add parent directory for bayesian_search import
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import IndicTransToolkit.collator as indic_collator
indic_collator.pad_without_fast_tokenizer_warning = pad_without_fast_tokenizer_warning
from IndicTransToolkit import IndicDataCollator, IndicProcessor

from bayesian_search import bayesian_search_cv

SAMPLE_DATA = {
    "raj": [
        ("राम घर गयो।", "Ram went home."),
        ("सीता पानी ल्याई।", "Sita brought water."),
        ("म्हारो नाम के है?", "What is my name?"),
        ("तू कठे जावे?", "Where are you going?"),
    ],
    "har": [
        ("राम घर गया।", "Ram went home."),
        ("सीता पानी ल्याई।", "Sita brought water."),
        ("तेरा नाम के सै?", "What is your name?"),
        ("तू कित जा सै?", "Where are you going?"),
    ],
    "odia": [
        ("ରାମ ଘରକୁ ଗଲା।", "Ram went home."),
        ("ସୀତା ପାଣି ଆଣିଲା।", "Sita brought water."),
        ("ତୁମର ନାମ କଣ?", "What is your name?"),
        ("ତୁମେ କୁଆଡ଼େ ଯାଉଛ?", "Where are you going?"),
    ],
    "ben": [
        ("রাম বাড়ি গেল।", "Ram went home."),
        ("সীতা জল আনল।", "Sita brought water."),
        ("তোমার নাম কি?", "What is your name?"),
        ("তুমি কোথায় যাচ্ছ?", "Where are you going?"),
    ],
}

LANG_CODE_MAP = {
    "raj": {"nllb": "hin_Deva", "indic": "hin_Deva"},
    "har": {"nllb": "hin_Deva", "indic": "hin_Deva"},
    "odia": {"nllb": "ory_Orya", "indic": "ory_Orya"},
    "ben": {"nllb": "ben_Beng", "indic": "ben_Beng"},
    "eng": {"nllb": "eng_Latn", "indic": "eng_Latn"},
}


def load_pairs_or_fallback(src_lang, tgt_lang):
    """Load pairs from data_split if present, else fallback to benchmark samples."""
    train_src_path = base_dir.parent / "data_split" / f"train.{src_lang}"
    train_tgt_path = base_dir.parent / "data_split" / f"train.{tgt_lang}"
    val_src_path = base_dir.parent / "data_split" / f"val.{src_lang}"
    val_tgt_path = base_dir.parent / "data_split" / f"val.{tgt_lang}"

    if train_src_path.exists() and train_tgt_path.exists():
        with open(train_src_path, encoding="utf-8") as fs, open(train_tgt_path, encoding="utf-8") as ft:
            train_pairs = [(s.strip(), t.strip()) for s, t in zip(fs, ft) if s.strip() and t.strip()][:4]
    else:
        train_pairs = SAMPLE_DATA.get(src_lang, SAMPLE_DATA["raj"])

    if val_src_path.exists() and val_tgt_path.exists():
        with open(val_src_path, encoding="utf-8") as fs, open(val_tgt_path, encoding="utf-8") as ft:
            val_pairs = [(s.strip(), t.strip()) for s, t in zip(fs, ft) if s.strip() and t.strip()][:2]
    else:
        val_pairs = train_pairs[-2:]

    return train_pairs, val_pairs


def run_smoke_test(model_type, pivot_mode, src_lang="raj", tgt_lang="eng"):
    """Run an automated smoke test for a specific model approach and language pair."""
    approach_str = f"{model_type}_{'pivot' if pivot_mode else 'not'}_{src_lang}_{tgt_lang}"
    output_dir = base_dir.parent / "scratch" / f"smoke_test_run_{approach_str}"

    print("\n" + "=" * 70)
    print(f" RUNNING SMOKE TEST: {approach_str}")
    print(f" Model: {model_type.upper()} | Pivot: {pivot_mode} | Pair: {src_lang} -> {tgt_lang}")
    print(f" Test Output Directory: {output_dir}")
    print("=" * 70)

    train_pairs, val_pairs = load_pairs_or_fallback(src_lang, tgt_lang)
    tgt_code = LANG_CODE_MAP[tgt_lang]["nllb"]

    if model_type == "nllb":
        _snapshots = Path.home() / ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots"
        _cached = [p for p in _snapshots.glob("*") if (p / "pytorch_model.bin").is_file()]
        model_name = str(_cached[0]) if _cached else "facebook/nllb-200-distilled-600M"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if pivot_mode:
            tokenizer.src_lang = LANG_CODE_MAP[src_lang]["nllb"]
        tokenizer.tgt_lang = tgt_code

        def tokenize_batch(batch):
            if pivot_mode:
                return tokenizer(batch["src"], text_target=batch["tgt"], max_length=32, truncation=True)
            else:
                m_in = tokenizer(batch["src"], max_length=32, truncation=True)
                labels = tokenizer(batch["tgt"], max_length=32, truncation=True)
                m_in["labels"] = labels["input_ids"]
                return m_in

        def to_dataset(pairs):
            ds = Dataset.from_dict({"src": [s for s, _ in pairs], "tgt": [t for _, t in pairs]})
            return ds.map(tokenize_batch, batched=True, remove_columns=["src", "tgt"])

        train_ds = to_dataset(train_pairs)
        val_ds = to_dataset(val_pairs)

        def create_model(r=8, lora_alpha=16, lora_dropout=0.05):
            m = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
            m.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
            return get_peft_model(
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

        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    else:  # indictrans2
        model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            processor = IndicProcessor(inference=False)
        except Exception as e:
            print(f"\n[Notice] IndicTrans2 model/tokenizer is not fully downloaded in offline cache.")
            print(f"Details: {e}")
            print(f"When internet connection is available or once weights are downloaded, this test will execute.")
            print(f"Skipping IndicTrans2 smoke test execution for now.\n")
            return

        def tokenize_indic(batch):
            if pivot_mode:
                src_code = LANG_CODE_MAP[src_lang]["indic"]
                sources = processor.preprocess_batch(batch["src"], src_lang=src_code, tgt_lang=tgt_code, is_target=False)
            else:
                sources = batch["src"]

            targets = processor.preprocess_batch(batch["tgt"], src_lang=tgt_code, tgt_lang=tgt_code, is_target=True)
            m_in = tokenizer(sources, max_length=32, truncation=True, padding=False)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=32, truncation=True, padding=False)
            m_in["labels"] = labels["input_ids"]
            return m_in

        def to_dataset_indic(pairs):
            ds = Dataset.from_dict({"src": [s for s, _ in pairs], "tgt": [t for _, t in pairs]})
            return ds.map(tokenize_indic, batched=True, remove_columns=["src", "tgt"])

        train_ds = to_dataset_indic(train_pairs)
        val_ds = to_dataset_indic(val_pairs)

        def create_model(r=8, lora_alpha=16, lora_dropout=0.05):
            m = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            return get_peft_model(
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

        collator = None  # Instantiated per trial with fresh model

    def train_trial(params, trial_dir):
        m = create_model(
            r=int(params.get("r", 8)),
            lora_alpha=int(params.get("lora_alpha", 16)),
            lora_dropout=float(params.get("lora_dropout", 0.05)),
        )
        trial_collator = collator if collator is not None else IndicDataCollator(
            tokenizer=tokenizer, model=m, padding="longest", pad_to_multiple_of=8, label_pad_token_id=-100
        )
        args = Seq2SeqTrainingArguments(
            output_dir=trial_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=1,
            eval_strategy="steps",
            eval_steps=1,
            learning_rate=float(params.get("lr", 1e-4)),
            logging_steps=1,
            report_to="none",
            use_cpu=True,
        )
        trainer = Seq2SeqTrainer(
            model=m,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=trial_collator,
        )
        trainer.train()
        eval_res = trainer.evaluate()
        val_loss = float(eval_res.get("eval_loss", 2.0))
        return val_loss, m, trainer.state.log_history

    space = {
        "lr": (1e-5, 5e-4, "log"),
        "r": [4, 8],
    }

    # Run quick 2-trial smoke test
    bayesian_search_cv(
        train_fn=train_trial,
        search_space=space,
        n_trials=2,
        initial_params={"lr": 1e-4, "r": 8},
        output_dir=str(output_dir),
        tokenizer=tokenizer,
    )

    # Verification of produced artifacts
    assert output_dir.exists(), f"Output dir {output_dir} does not exist"
    out_files = os.listdir(output_dir)
    assert "loss_curve.png" in out_files, "Missing loss_curve.png"
    assert "loss_curve.pdf" in out_files, "Missing loss_curve.pdf"
    assert "loss_curve_history.csv" in out_files, "Missing loss_curve_history.csv"
    assert "bayesian_optimization_convergence.png" in out_files, "Missing convergence plot"
    assert "best_params.json" in out_files, "Missing best_params.json"

    # Cleanup test output
    shutil.rmtree(output_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f" SUCCESS: Smoke test passed for {approach_str}!")
    print("=" * 70 + "\n")
