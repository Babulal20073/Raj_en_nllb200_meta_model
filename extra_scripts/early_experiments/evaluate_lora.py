import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sacrebleu import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf

# ================= CONFIG =================
BASE_MODEL = "facebook/nllb-200-distilled-600M"
LORA_MODEL_PATH = "nllb_lora_finetuned"

TEST_RAJ = "data_split/test.raj"
TEST_ENG = "data_split/test.eng"

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
            gen = model.generate(**inputs, max_length=MAX_LEN)

        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def evaluate(name, model):
    hyps = translate(model, tokenizer)
    bleu = corpus_bleu(hyps, [refs]).score
    chrf = corpus_chrf(hyps, refs)
    print(f"{name:<30} BLEU: {bleu:6.2f} | chrF: {chrf:6.2f}")
    return bleu, chrf


# ---------- TOKENIZER ----------
tokenizer = load_tokenizer()


# ---------- ZERO-SHOT ----------
print("\nEvaluating Zero-Shot NLLB...")
zero_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

zero_bleu, zero_chrf = evaluate("Zero-Shot NLLB", zero_model)


# ---------- LoRA ----------
print("\nEvaluating LoRA Fine-Tuned NLLB...")
lora_base = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
)

lora_model = PeftModel.from_pretrained(lora_base, LORA_MODEL_PATH)
lora_model.to(DEVICE)

lora_bleu, lora_chrf = evaluate("LoRA Fine-Tuned NLLB", lora_model)


print("\n================ FINAL COMPARISON ================")
print("Model                          BLEU    chrF")
print("--------------------------------------------------")
print(f"Zero-Shot NLLB                {zero_bleu:6.2f}  {zero_chrf:6.2f}")
print(f"LoRA Fine-Tuned NLLB           {lora_bleu:6.2f}  {lora_chrf:6.2f}")
print("==================================================")
