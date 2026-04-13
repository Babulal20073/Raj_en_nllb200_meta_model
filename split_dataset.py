import os
import random

CLEAN_DIR = "data_clean"
OUT_DIR = "data_split"

os.makedirs(OUT_DIR, exist_ok=True)

raj_path = os.path.join(CLEAN_DIR, "clean.raj")
eng_path = os.path.join(CLEAN_DIR, "clean.eng")

with open(raj_path, encoding="utf-8") as fr, open(eng_path, encoding="utf-8") as fe:
    raj_lines = fr.readlines()
    eng_lines = fe.readlines()

assert len(raj_lines) == len(eng_lines), "❌ Parallel data mismatch"

data = list(zip(raj_lines, eng_lines))

random.seed(42)
random.shuffle(data)

total = len(data)
train_end = int(0.8 * total)
val_end = int(0.9 * total)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

def save_split(split, name):
    raj_out = os.path.join(OUT_DIR, f"{name}.raj")
    eng_out = os.path.join(OUT_DIR, f"{name}.eng")

    with open(raj_out, "w", encoding="utf-8") as fr, \
         open(eng_out, "w", encoding="utf-8") as fe:
        for r, e in split:
            fr.write(r)
            fe.write(e)

save_split(train_data, "train")
save_split(val_data, "val")
save_split(test_data, "test")

print("✅ Dataset split complete")
print(f"Train: {len(train_data)}")
print(f"Val  : {len(val_data)}")
print(f"Test : {len(test_data)}")
