import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SRC_DIR = "data_split"
OUT_DIR = "data_zero"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 8
MAX_LEN = 256

print("🔄 Loading NLLB model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

def translate_file(input_path, output_path, src_lang, tgt_lang):
    with open(input_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    results = []

    tokenizer.src_lang = src_lang

    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=MAX_LEN
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(decoded)

        if i % (BATCH_SIZE * 10) == 0:
            print(f"Translated {i}/{len(lines)}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"✅ Saved: {output_path}")

# ----------- Raj -> Eng (Zero-shot) -----------
print("\n🔁 Zero-shot Raj → Eng")
translate_file(
    input_path=os.path.join(SRC_DIR, "train.raj"),
    output_path=os.path.join(OUT_DIR, "raj2eng.txt"),
    src_lang="hin_Deva",
    tgt_lang="eng_Latn"
)

# ----------- Eng -> Raj (Zero-shot) -----------
print("\n🔁 Zero-shot Eng → Raj")
translate_file(
    input_path=os.path.join(SRC_DIR, "train.eng"),
    output_path=os.path.join(OUT_DIR, "eng2raj.txt"),
    src_lang="eng_Latn",
    tgt_lang="hin_Deva"
)

print("\n🎉 Zero-shot translation complete")
