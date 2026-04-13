import os
import unicodedata

RAW_DIR = "data_raw"
OUT_DIR = "data_clean"

os.makedirs(OUT_DIR, exist_ok=True)

raj_file = os.path.join(RAW_DIR, "train.raj")
eng_file = os.path.join(RAW_DIR, "train.eng")

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    return text.strip()

with open(raj_file, encoding="utf-8") as fr, open(eng_file, encoding="utf-8") as fe:
    raj_lines = fr.readlines()
    eng_lines = fe.readlines()

if len(raj_lines) != len(eng_lines):
    raise ValueError("❌ Line count mismatch between train.raj and train.eng")

clean_raj = []
clean_eng = []

for r, e in zip(raj_lines, eng_lines):
    r = clean_text(r)
    e = clean_text(e)

    if not r or not e:
        continue
    if len(r.split()) < 2 or len(e.split()) < 2:
        continue

    clean_raj.append(r)
    clean_eng.append(e)

with open(os.path.join(OUT_DIR, "clean.raj"), "w", encoding="utf-8") as fr:
    fr.write("\n".join(clean_raj))

with open(os.path.join(OUT_DIR, "clean.eng"), "w", encoding="utf-8") as fe:
    fe.write("\n".join(clean_eng))

print("✅ Dataset preparation done")
print(f"📊 Total clean sentence pairs: {len(clean_raj)}")
