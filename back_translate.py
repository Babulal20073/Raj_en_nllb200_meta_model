import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

SRC_DIR = "data_filtered"
OUT_DIR = "data_bt"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 8
MAX_LEN = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

def translate(lines, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    out = []
    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LEN
        ).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=MAX_LEN
            )
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        if i % (BATCH_SIZE * 10) == 0:
            print(f"Back-translated {i}/{len(lines)}")
    return out

# Load filtered Eng→Raj pairs
with open(os.path.join(SRC_DIR, "eng_raj.eng"), encoding="utf-8") as f:
    eng_lines = [l.strip() for l in f if l.strip()]

# Back-translate Eng → Raj
bt_raj = translate(eng_lines, src_lang="eng_Latn", tgt_lang="hin_Deva")

# Save back-translation pairs
with open(os.path.join(OUT_DIR, "bt_raj.eng"), "w", encoding="utf-8") as fe:
    fe.write("\n".join(eng_lines))
with open(os.path.join(OUT_DIR, "bt_raj.raj"), "w", encoding="utf-8") as fr:
    fr.write("\n".join(bt_raj))

print("Back-translation complete")
