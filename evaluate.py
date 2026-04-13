import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu, corpus_chrf

# ---------------- CONFIG ----------------
TEST_RAJ = "data_split/test.raj"
TEST_ENG = "data_split/test.eng"

SRC_LANG = "hin_Deva"   # proxy for Rajasthani
TGT_LANG = "eng_Latn"

MAX_LEN = 192
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models to evaluate (table rows)
MODELS = {
    "Zero-Shot NLLB-200": "facebook/nllb-200-distilled-600M",
    "Synthetic + LaBSE Filtered": "facebook/nllb-200-distilled-600M",
    "Fine-Tuned NLLB (Final)": "nllb_finetuned"
}

# ---------------------------------------


def load_test_data():
    with open(TEST_RAJ, encoding="utf-8") as f:
        src = [l.strip() for l in f if l.strip()]
    with open(TEST_ENG, encoding="utf-8") as f:
        ref = [l.strip() for l in f if l.strip()]
    assert len(src) == len(ref)
    return src, ref


def evaluate_model(model_name, model_path, src_sentences, ref_sentences):
    print(f"\nEvaluating: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    model.eval()
    translations = []

    for i in range(0, len(src_sentences), BATCH_SIZE):
        batch = src_sentences[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LEN,
                num_beams=4
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)

    bleu = corpus_bleu(translations, [ref_sentences]).score
    chrf = corpus_chrf(translations, [ref_sentences]).score

    return round(bleu, 2), round(chrf, 2)


# ---------------- MAIN ----------------
print("Loading test data...")
src_sentences, ref_sentences = load_test_data()

results = []

for model_name, model_path in MODELS.items():
    bleu, chrf = evaluate_model(
        model_name,
        model_path,
        src_sentences,
        ref_sentences
    )
    results.append((model_name, bleu, chrf))

# ---------------- RESULTS ----------------
print("\n================ FINAL RESULTS =================")
print("Model\t\t\t\tBLEU\tchrF")
print("------------------------------------------------")

for name, bleu, chrf in results:
    print(f"{name:<30} {bleu:<6} {chrf}")

print("================================================")
