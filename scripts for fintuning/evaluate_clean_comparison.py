"""Evaluate base and LoRA NLLB/IndicTrans2 on the untouched clean test split."""
import gc
from pathlib import Path

import torch
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor


TEST_SRC = "data_split/test.raj"
TEST_TGT = "data_split/test.eng"
OUT_DIR = Path("comparison_outputs_clean")
BATCH_SIZE = 1
MAX_LEN = 128
SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"
NLLB_NAME = "facebook/nllb-200-distilled-600M"
INDIC_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE = (
    torch.bfloat16
    if USE_BF16
    else (torch.float16 if DEVICE.type == "cuda" else torch.float32)
)


def read_parallel():
    src = [line.strip() for line in Path(TEST_SRC).read_text(encoding="utf-8").splitlines()]
    refs = [line.strip() for line in Path(TEST_TGT).read_text(encoding="utf-8").splitlines()]
    if len(src) != len(refs):
        raise ValueError(f"Test line mismatch: {len(src)} source vs {len(refs)} references")
    pairs = [(s, t) for s, t in zip(src, refs) if s and t]
    return [s for s, _ in pairs], [t for _, t in pairs]


def score(name, hypotheses, references):
    bleu = BLEU().corpus_score(hypotheses, [references]).score
    chrf = CHRF(word_order=2).corpus_score(hypotheses, [references]).score
    path = OUT_DIR / f"{name}.txt"
    path.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    print(f"{name:28s} BLEU={bleu:6.2f}  chrF++={chrf:6.2f}  predictions={path}")
    return {"model": name, "BLEU": bleu, "chrF++": chrf}


def generate_nllb(src, adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(NLLB_NAME)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    base = AutoModelForSeq2SeqLM.from_pretrained(NLLB_NAME, torch_dtype=DTYPE).to(DEVICE)
    base.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    model.eval()
    output = []
    for start in range(0, len(src), BATCH_SIZE):
        batch = tokenizer(
            src[start : start + BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.inference_mode():
            tokens = model.generate(
                **batch,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
                max_new_tokens=MAX_LEN,
                num_beams=1,
            )
        output.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))
        print(f"NLLB {min(start + BATCH_SIZE, len(src))}/{len(src)}", end="\r")
    print()
    del model, base, tokenizer
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return output


def generate_indic(src, adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(INDIC_NAME, trust_remote_code=True)
    processor = IndicProcessor(inference=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        INDIC_NAME,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    model.eval()
    output = []
    for start in range(0, len(src), BATCH_SIZE):
        batch_text = processor.preprocess_batch(
            src[start : start + BATCH_SIZE], src_lang=SRC_LANG, tgt_lang=TGT_LANG
        )
        batch = tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.inference_mode():
            # The model's remote generation code assumes tuple-based KV caches;
            # newer Transformers passes cache entries this code cannot handle.
            # Disable KV caching to keep generation compatible.
            tokens = model.generate(
                **batch, max_new_tokens=MAX_LEN, num_beams=1, use_cache=False
            )
        decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        output.extend(processor.postprocess_batch(decoded, lang=TGT_LANG))
        print(f"IndicTrans2 {min(start + BATCH_SIZE, len(src))}/{len(src)}", end="\r")
    print()
    del model, base, tokenizer, processor
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return output


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src, refs = read_parallel()
    print(f"Untouched test examples: {len(src)} | device={DEVICE} | dtype={DTYPE}")
    rows = []
    rows.append(score("nllb_base", generate_nllb(src), refs))
    rows.append(score("nllb_lora", generate_nllb(src, "nllb_lora_rajasthani_clean"), refs))
    rows.append(score("indictrans2_base", generate_indic(src), refs))
    rows.append(score("indictrans2_lora", generate_indic(src, "indictrans2_lora_rajasthani_clean"), refs))
    report = OUT_DIR / "metrics.csv"
    report.write_text(
        "model,BLEU,chrF++\n" + "".join(f"{r['model']},{r['BLEU']:.4f},{r['chrF++']:.4f}\n" for r in rows),
        encoding="utf-8",
    )
    print(f"Metrics saved to {report}")
    print("Review predictions with a Rajasthani speaker; the source-language code is a Hindi proxy.")


if __name__ == "__main__":
    main()
