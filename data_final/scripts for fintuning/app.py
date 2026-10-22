"""
Multi-language → English Translation CLI
=========================================
Usage:
    python app.py

Input format:
    <tag> sentence in source language

Tags:
    <r>  Rajasthani  (hin_Deva proxy in NLLB)
    <o>  Odia        (ory_Orya in NLLB)
    <b>  Bengali     (ben_Beng in NLLB)
    <h>  Haryanvi    (hin_Deva proxy in NLLB)

Examples:
    <r> म्हारो देस राजस्थान है
    <o> ମୋ ଦେଶ ଭଲ ଅଛି
    <b> আমার দেশ সুন্দর
    <h> म्हारा देश हरियाणा सै

Type 'quit' or 'exit' to stop.
Type 'help' to show this message again.
"""

import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ─────────────────────────────────────────────
#  CONFIG — update LoRA paths once models are trained
# ─────────────────────────────────────────────

BASE_MODEL = "facebook/nllb-200-distilled-600M"

# Map: tag → (nllb_src_lang_code, path_to_lora_adapter)
LANGUAGE_CONFIG = {
    "<r>": {
        "name":      "Rajasthani",
        "src_lang":  "hin_Deva",   # Hindi proxy (Rajasthani not in NLLB-200)
        "lora_path": "Raj_en_nllb200_meta_model/nllb_lora_finetuned",
    },
    "<o>": {
        "name":      "Odia",
        "src_lang":  "ory_Orya",   # Odia is natively in NLLB-200
        "lora_path": "models/nllb_lora_odia",     # ← update to your trained checkpoint
    },
    "<b>": {
        "name":      "Bengali",
        "src_lang":  "ben_Beng",   # Bengali is natively in NLLB-200
        "lora_path": "models/nllb_lora_bengali",  # ← update to your trained checkpoint
    },
    "<h>": {
        "name":      "Haryanvi",
        "src_lang":  "hin_Deva",   # Hindi proxy (Haryanvi not in NLLB-200)
        "lora_path": "models/nllb_lora_haryanvi", # ← update to your trained checkpoint
    },
}

TGT_LANG = "eng_Latn"
MAX_LEN  = 192
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
#  MODEL CACHE — lazy-load, keep in memory
# ─────────────────────────────────────────────

_cache: dict = {}   # tag → (tokenizer, model)


def load_model(tag: str):
    """Load base NLLB + LoRA adapter for the given tag. Cache after first load."""
    if tag in _cache:
        return _cache[tag]

    cfg = LANGUAGE_CONFIG[tag]
    lora_path = cfg["lora_path"]

    import os
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(
            f"LoRA model directory not found: '{lora_path}'\n"
            f"  → Train the {cfg['name']} model first, then update LANGUAGE_CONFIG in app.py"
        )

    print(f"  [Loading] {cfg['name']} model from '{lora_path}' ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.src_lang = cfg["src_lang"]
    tokenizer.tgt_lang = TGT_LANG

    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.to(DEVICE)
    model.eval()

    _cache[tag] = (tokenizer, model)
    print(f"  [Ready]   {cfg['name']} model loaded on {DEVICE.upper()}\n", flush=True)
    return tokenizer, model


# ─────────────────────────────────────────────
#  PARSING
# ─────────────────────────────────────────────

TAG_PATTERN = re.compile(r"^(<[rohb]>)\s*(.*)", re.IGNORECASE | re.DOTALL)

def parse_input(raw: str):
    """
    Returns (tag, text) or raises ValueError.
    Example:  '<r> म्हारो देस'  →  ('<r>', 'म्हारो देस')
    """
    raw = raw.strip()
    m = TAG_PATTERN.match(raw)
    if not m:
        raise ValueError(
            "No valid language tag found.\n"
            "  Expected one of: <r> <o> <b> <h> followed by your sentence."
        )
    tag  = m.group(1).lower()
    text = m.group(2).strip()
    if not text:
        raise ValueError("Tag found but sentence is empty. Please type text after the tag.")
    return tag, text


# ─────────────────────────────────────────────
#  TRANSLATION
# ─────────────────────────────────────────────

def translate(tag: str, text: str) -> str:
    tokenizer, model = load_model(tag)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
            max_length=MAX_LEN,
            num_beams=4,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ─────────────────────────────────────────────
#  CLI LOOP
# ─────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║   Multi-Language → English Translator  (NLLB-200)   ║
║                                                      ║
║   Tags:  <r> Rajasthani  <o> Odia                   ║
║          <b> Bengali     <h> Haryanvi                ║
║                                                      ║
║   Type 'help' for usage, 'quit' to exit.             ║
╚══════════════════════════════════════════════════════╝
"""

def help_text():
    print(__doc__)


def main():
    print(BANNER)
    print(f"Device: {DEVICE.upper()}")
    print(f"Base model: {BASE_MODEL}\n")

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Bye!")
            sys.exit(0)
        if raw.lower() == "help":
            help_text()
            continue

        # ── Parse tag + sentence ──
        try:
            tag, text = parse_input(raw)
        except ValueError as e:
            print(f"\n[Error] {e}\n")
            continue

        lang_name = LANGUAGE_CONFIG[tag]["name"]
        print(f"\n  Language : {lang_name}")
        print(f"  Input    : {text}")

        # ── Load model + translate ──
        try:
            result = translate(tag, text)
        except FileNotFoundError as e:
            print(f"\n[Model Not Found]\n  {e}\n")
            continue
        except Exception as e:
            print(f"\n[Translation Error] {type(e).__name__}: {e}\n")
            continue

        print(f"  Output   : {result}\n")


if __name__ == "__main__":
    main()
