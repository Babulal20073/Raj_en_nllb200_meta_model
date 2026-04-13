import os

def load(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def save(pairs, out_raj, out_eng):
    with open(out_raj, "w", encoding="utf-8") as fr, \
         open(out_eng, "w", encoding="utf-8") as fe:
        for r, e in pairs:
            fr.write(r + "\n")
            fe.write(e + "\n")

pairs = []

# 1) Clean parallel
clean_raj = load("data_clean/clean.raj")
clean_eng = load("data_clean/clean.eng")
pairs.extend(zip(clean_raj, clean_eng))

# 2) Filtered Raj→Eng
r2e_raj = load("data_filtered/raj_eng.raj")
r2e_eng = load("data_filtered/raj_eng.eng")
pairs.extend(zip(r2e_raj, r2e_eng))

# 3) Filtered Eng→Raj
e2r_eng = load("data_filtered/eng_raj.eng")
e2r_raj = load("data_filtered/eng_raj.raj")
pairs.extend(zip(e2r_raj, e2r_eng))

# 4) Back-translated
bt_eng = load("data_bt/bt_raj.eng")
bt_raj = load("data_bt/bt_raj.raj")
pairs.extend(zip(bt_raj, bt_eng))

# Deduplicate
pairs = list(dict.fromkeys(pairs))

os.makedirs("data_final", exist_ok=True)
save(pairs, "data_final/train.raj", "data_final/train.eng")

print(f"Final training pairs: {len(pairs)}")
