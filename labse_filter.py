import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.70
BATCH_SIZE = 64

SRC_SPLIT = "data_split"
SRC_ZERO = "data_zero"
OUT_DIR = "data_filtered"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading LaBSE model...")
model = SentenceTransformer("sentence-transformers/LaBSE", device=DEVICE)

def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def filter_pairs(src_lines, tgt_lines):
    assert len(src_lines) == len(tgt_lines)
    keep_src, keep_tgt = [], []

    for i in range(0, len(src_lines), BATCH_SIZE):
        src_batch = src_lines[i:i+BATCH_SIZE]
        tgt_batch = tgt_lines[i:i+BATCH_SIZE]

        src_emb = model.encode(src_batch, convert_to_tensor=True)
        tgt_emb = model.encode(tgt_batch, convert_to_tensor=True)

        sims = cosine_similarity(
            src_emb.cpu().numpy(),
            tgt_emb.cpu().numpy()
        )

        for j in range(len(src_batch)):
            if sims[j][j] >= THRESHOLD:
                keep_src.append(src_batch[j])
                keep_tgt.append(tgt_batch[j])

        if i % (BATCH_SIZE * 10) == 0:
            print(f"Processed {i}/{len(src_lines)}")

    return keep_src, keep_tgt

# -------- Raj -> Eng filtering --------
print("\nFiltering Raj → Eng synthetic pairs")
raj = load_lines(os.path.join(SRC_SPLIT, "train.raj"))
raj2eng = load_lines(os.path.join(SRC_ZERO, "raj2eng.txt"))

f_raj, f_eng = filter_pairs(raj, raj2eng)

with open(os.path.join(OUT_DIR, "raj_eng.raj"), "w", encoding="utf-8") as fr:
    fr.write("\n".join(f_raj))

with open(os.path.join(OUT_DIR, "raj_eng.eng"), "w", encoding="utf-8") as fe:
    fe.write("\n".join(f_eng))

print(f"Kept Raj→Eng pairs: {len(f_raj)}")

# -------- Eng -> Raj filtering --------
print("\nFiltering Eng → Raj synthetic pairs")
eng = load_lines(os.path.join(SRC_SPLIT, "train.eng"))
eng2raj = load_lines(os.path.join(SRC_ZERO, "eng2raj.txt"))

f_eng2, f_raj2 = filter_pairs(eng, eng2raj)

with open(os.path.join(OUT_DIR, "eng_raj.eng"), "w", encoding="utf-8") as fe:
    fe.write("\n".join(f_eng2))

with open(os.path.join(OUT_DIR, "eng_raj.raj"), "w", encoding="utf-8") as fr:
    fr.write("\n".join(f_raj2))

print(f"Kept Eng→Raj pairs: {len(f_eng2)}")

print("\nLaBSE filtering complete")
