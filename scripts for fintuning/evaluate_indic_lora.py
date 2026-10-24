"""Evaluate IndicTrans2 LoRA adapters on the held-out Rajasthani→English test set.

Three models are compared end-to-end:
  1. Zero-Shot  – base IndicTrans2 with no fine-tuning (Hindi proxy src_lang)
  2. LoRA Proxy – LoRA adapter trained via finetune_indic_lora.py  (hin_Deva proxy)
  3. LoRA Direct – LoRA adapter trained via finetune_indic_lora_direct.py (no proxy)

Metrics: BLEU (sacreBLEU), chrF (sacreBLEU), exact-match accuracy.

Usage (run from the project root, e.g. Raj_en_nllb200_meta_model/):
    python "scripts for fintuning/evaluate_indic_lora.py"
"""

import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from sacrebleu import corpus_bleu
from sacrebleu.metrics import CHRF

# IndicTransToolkit patch – the collator resolves this as a module-level name
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import IndicTransToolkit.collator as _indic_collator
_indic_collator.pad_without_fast_tokenizer_warning = pad_without_fast_tokenizer_warning
from IndicTransToolkit import IndicProcessor  # noqa: E402  (must come after patch)

# ==================== CONFIG ====================
BASE_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"

# LoRA checkpoint directories (relative to cwd / project root)
LORA_PROXY_PATH  = "indictrans2_lora_rajasthani_clean"   # finetune_indic_lora.py output
LORA_DIRECT_PATH = "indictrans2_lora_rajasthani_direct"  # finetune_indic_lora_direct.py output

# Test data (relative to cwd / project root)
TEST_SRC = "data_split/test.raj"
TEST_TGT = "data_split/test.eng"

# Language codes used for IndicProcessor preprocessing
SRC_LANG = "hin_Deva"   # Hindi proxy – same as training for the proxy adapter
TGT_LANG = "eng_Latn"

MAX_LEN    = 128
BATCH_SIZE = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ================================================


# ---------- Load test data ----------
print("Loading test data ...")
with open(TEST_SRC, encoding="utf-8") as f:
    src_sentences = [line.strip() for line in f if line.strip()]

with open(TEST_TGT, encoding="utf-8") as f:
    ref_sentences = [line.strip() for line in f if line.strip()]

assert len(src_sentences) == len(ref_sentences), (
    f"Source/reference line count mismatch: {len(src_sentences)} vs {len(ref_sentences)}"
)
print(f"Test sentences: {len(src_sentences)}")


# ---------- IndicProcessor (inference mode) ----------
processor = IndicProcessor(inference=True)


# ---------- Tokenizer ----------
def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


# ---------- Translation helpers ----------
def _translate_proxy(model, tokenizer):
    """Translate using the Hindi proxy src_lang (mirrors finetune_indic_lora.py)."""
    model.eval()
    outputs = []

    for i in range(0, len(src_sentences), BATCH_SIZE):
        batch = src_sentences[i : i + BATCH_SIZE]

        # IndicProcessor preprocessing with the hin_Deva proxy tag
        preprocessed = processor.preprocess_batch(
            batch, src_lang=SRC_LANG, tgt_lang=TGT_LANG
        )
        inputs = tokenizer(
            preprocessed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(DEVICE)

        tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=MAX_LEN,
                num_beams=4,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        postprocessed = processor.postprocess_batch(decoded, lang=TGT_LANG)
        outputs.extend(postprocessed)

    return outputs


def _translate_direct(model, tokenizer):
    """Translate without any src_lang proxy (mirrors finetune_indic_lora_direct.py)."""
    model.eval()
    outputs = []

    for i in range(0, len(src_sentences), BATCH_SIZE):
        batch = src_sentences[i : i + BATCH_SIZE]

        # No IndicProcessor on the source side – raw Rajasthani text fed directly
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(DEVICE)

        tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=MAX_LEN,
                num_beams=4,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        postprocessed = processor.postprocess_batch(decoded, lang=TGT_LANG)
        outputs.extend(postprocessed)

    return outputs


# ---------- Metric helpers ----------
def _normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def evaluate(name, hyps):
    """Compute and print BLEU / chrF / exact-match accuracy."""
    bleu = corpus_bleu(hyps, [ref_sentences]).score
    chrf = CHRF().corpus_score(hyps, [ref_sentences]).score
    acc  = 100.0 * sum(
        _normalize(h) == _normalize(r)
        for h, r in zip(hyps, ref_sentences)
    ) / len(ref_sentences)
    print(f"{name:<35} BLEU: {bleu:6.2f} | chrF: {chrf:6.2f} | Acc: {acc:6.2f}")
    return bleu, chrf, acc


# ---------- Helper: load base model ----------
def _load_base(dtype):
    return AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    )


# ============================================================
#  Main evaluation loop
# ============================================================
dtype = (
    torch.bfloat16
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

tokenizer = load_tokenizer()

# ---- 1. Zero-Shot (Hindi proxy, no fine-tuning) ----
print("\n[1/3] Evaluating Zero-Shot IndicTrans2 (proxy src_lang) ...")
zero_model = _load_base(dtype).to(DEVICE)
zero_hyps  = _translate_proxy(zero_model, tokenizer)
zero_bleu, zero_chrf, zero_acc = evaluate("Zero-Shot IndicTrans2 (proxy)", zero_hyps)
del zero_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ---- 2. LoRA Proxy adapter ----
print("\n[2/3] Evaluating LoRA Proxy IndicTrans2 ...")
proxy_base  = _load_base(dtype)
proxy_model = PeftModel.from_pretrained(proxy_base, LORA_PROXY_PATH).to(DEVICE)
proxy_hyps  = _translate_proxy(proxy_model, tokenizer)
proxy_bleu, proxy_chrf, proxy_acc = evaluate("LoRA Proxy IndicTrans2", proxy_hyps)
del proxy_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ---- 3. LoRA Direct adapter ----
print("\n[3/3] Evaluating LoRA Direct IndicTrans2 ...")
direct_base  = _load_base(dtype)
direct_model = PeftModel.from_pretrained(direct_base, LORA_DIRECT_PATH).to(DEVICE)
direct_hyps  = _translate_direct(direct_model, tokenizer)
direct_bleu, direct_chrf, direct_acc = evaluate("LoRA Direct IndicTrans2", direct_hyps)
del direct_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ---- Summary table ----
print("\n")
print("=" * 62)
print(f"{'Model':<35} {'BLEU':>6}  {'chrF':>6}  {'Acc':>6}")
print("-" * 62)
print(f"{'Zero-Shot IndicTrans2 (proxy)':<35} {zero_bleu:6.2f}  {zero_chrf:6.2f}  {zero_acc:6.2f}")
print(f"{'LoRA Proxy IndicTrans2':<35} {proxy_bleu:6.2f}  {proxy_chrf:6.2f}  {proxy_acc:6.2f}")
print(f"{'LoRA Direct IndicTrans2':<35} {direct_bleu:6.2f}  {direct_chrf:6.2f}  {direct_acc:6.2f}")
print("=" * 62)
