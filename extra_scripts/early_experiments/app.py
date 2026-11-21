import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "nllb_lora_finetuned"

SRC_LANG = "hin_Deva"   # proxy for Rajasthani
TGT_LANG = "eng_Latn"

MAX_LEN = 192

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device)

model.eval()


def translate(text):
    if not text.strip():
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LEN,
            num_beams=4
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


interface = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(
        label="Rajasthani (Devanagari)",
        placeholder="यहां राजस्थानी वाक्य लिखें",
        lines=3
    ),
    outputs=gr.Textbox(
        label="English Translation",
        lines=3
    ),
    title="Rajasthani → English Translator",
    description="Fine-tuned NLLB-200 model (low-resource setup)"
)

if __name__ == "__main__":
    interface.launch()
