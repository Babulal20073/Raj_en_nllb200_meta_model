import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sacrebleu import corpus_bleu
from sacrebleu.metrics import CHRF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG =================
BASE_MODEL = "facebook/nllb-200-distilled-600M"
LORA_MODEL_PATH = "nllb_lora_rajasthani_direct"    

TEST_RAJ = "../data_split/test.raj"
TEST_ENG = "../data_split/test.eng"

SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"

MAX_LEN = 160
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================


print("Loading test data...")
with open(TEST_RAJ, encoding="utf-8") as f:
    src = [l.strip() for l in f if l.strip()]

with open(TEST_ENG, encoding="utf-8") as f:
    refs = [l.strip() for l in f if l.strip()]

assert len(src) == len(refs)
print(f"Test sentences: {len(src)}")


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.src_lang = SRC_LANG
    tok.tgt_lang = TGT_LANG
    return tok


def translate(model, tokenizer):
    outputs = []
    model.eval()
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)   
    for i in range(0, len(src), BATCH_SIZE):
        batch = src[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        ).to(DEVICE)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=MAX_LEN
            )

        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def compute_semantic_accuracy(hypotheses, references):
    """Compute semantic accuracy (0-100%) via embedding or TF-IDF cosine similarity."""
    try:
        from sentence_transformers import SentenceTransformer, util
        st_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        emb_h = st_model.encode(hypotheses, convert_to_tensor=True)
        emb_r = st_model.encode(references, convert_to_tensor=True)
        cos_sim = util.cos_sim(emb_h, emb_r)
        return float(torch.diag(cos_sim).mean().item()) * 100.0
    except Exception:
        pass

    scores = []
    for h, r in zip(hypotheses, references):
        if not h.strip() or not r.strip():
            scores.append(1.0 if h.strip() == r.strip() else 0.0)
            continue
        vec = TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb").fit([h, r])
        m = vec.transform([h, r])
        scores.append(float(cosine_similarity(m[0:1], m[1:2])[0][0]))
    return float(np.mean(scores)) * 100.0 if scores else 0.0


def evaluate(name, model):
    hyps = translate(model, tokenizer)
    bleu = corpus_bleu(hyps, [refs]).score
    chrf = CHRF(word_order=0).corpus_score(hyps, [refs]).score
    chrf_plus = CHRF(word_order=2).corpus_score(hyps, [refs]).score
    acc = 100 * sum(normalize(h) == normalize(r) for h, r in zip(hyps, refs)) / len(refs)
    sem_acc = compute_semantic_accuracy(hyps, refs)
    print(f"{name:<22} BLEU: {bleu:5.2f} | chrF: {chrf:5.2f} | chrF++: {chrf_plus:5.2f} | Acc: {acc:5.2f}% | SemAcc: {sem_acc:5.2f}%")
    return bleu, chrf, chrf_plus, acc, sem_acc


# ---------- TOKENIZER ----------
tokenizer = load_tokenizer()


# ---------- ZERO-SHOT ----------
print("\nEvaluating Zero-Shot NLLB...")
zero_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

zero_bleu, zero_chrf, zero_chrf_plus, zero_acc, zero_sem = evaluate("Zero-Shot NLLB", zero_model)


# ---------- LoRA ----------
print("\nEvaluating LoRA Fine-Tuned NLLB...")
lora_base = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
)

lora_model = PeftModel.from_pretrained(lora_base, LORA_MODEL_PATH)
lora_model.to(DEVICE)

lora_bleu, lora_chrf, lora_chrf_plus, lora_acc, lora_sem = evaluate("LoRA Fine-Tuned NLLB", lora_model)


print("\n========================= FINAL COMPARISON =========================")
print("Model                   BLEU    chrF   chrF++   ExactAcc  SemanticAcc")
print("--------------------------------------------------------------------")
print(f"Zero-Shot NLLB         {zero_bleu:6.2f}  {zero_chrf:6.2f}  {zero_chrf_plus:6.2f}    {zero_acc:5.2f}%     {zero_sem:5.2f}%")
print(f"LoRA Fine-Tuned NLLB    {lora_bleu:6.2f}  {lora_chrf:6.2f}  {lora_chrf_plus:6.2f}    {lora_acc:5.2f}%     {lora_sem:5.2f}%")
print("====================================================================")