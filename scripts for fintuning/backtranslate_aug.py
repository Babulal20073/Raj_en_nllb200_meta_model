"""Create optional back-translation augmentation for a training split.

This script translates English training sentences into Devanagari with NLLB's
Hindi setting and appends those pseudo-Hindi inputs as synthetic pairs. It is
not a Rajasthani/Haryanvi generator: review the generated text with speakers
before treating it as dialect data.

Run from the project root:
    python "scripts for fintuning/backtranslate_aug.py"

The script reads only the configured TRAIN files and writes a new output
directory. It never changes the input files or validation/test splits.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ================= CONFIG =================
DATASET = "rajasthani"  # "rajasthani" | "haryanvi"
DATASETS = {
    "rajasthani": {
        "dialect_file": "data_final/train.raj",
        "english_file": "data_final/train.eng",
        "output_dir": "data_augmented/rajasthani",
        "dialect_ext": "raj",
    },
    "haryanvi": {
        "dialect_file": "PATH/TO/train.hry",
        "english_file": "PATH/TO/train.eng",
        "output_dir": "data_augmented/haryanvi",
        "dialect_ext": "hry",
    },
}
MODEL_NAME = "facebook/nllb-200-distilled-600M"
PROXY_LANG = "hin_Deva"  # NLLB has no Rajasthani/Haryanvi language code
ENGLISH_LANG = "eng_Latn"
BATCH_SIZE = 8
MAX_LEN = 256
NUM_BEAMS = 4
MAX_NEW_TOKENS = 256
SEED = 42
# =========================================


def read_aligned_pairs(dialect_file: Path, english_file: Path):
    dialect_lines = dialect_file.read_text(encoding="utf-8").splitlines()
    english_lines = english_file.read_text(encoding="utf-8").splitlines()
    if len(dialect_lines) != len(english_lines):
        raise ValueError(
            f"Parallel files have different line counts: "
            f"{dialect_file} ({len(dialect_lines)}) vs "
            f"{english_file} ({len(english_lines)})"
        )

    pairs, seen = [], set()
    for dialect, english in zip(dialect_lines, english_lines):
        dialect, english = dialect.strip(), english.strip()
        if not dialect or not english or (dialect, english) in seen:
            continue
        seen.add((dialect, english))
        pairs.append((dialect, english))
    if not pairs:
        raise ValueError("No non-empty aligned training pairs found")
    return pairs


def save_pairs(path_dialect: Path, path_english: Path, pairs):
    path_dialect.parent.mkdir(parents=True, exist_ok=True)
    path_dialect.write_text(
        "".join(f"{dialect}\n" for dialect, _ in pairs), encoding="utf-8"
    )
    path_english.write_text(
        "".join(f"{english}\n" for _, english in pairs), encoding="utf-8"
    )


def main():
    if DATASET not in DATASETS:
        raise ValueError(f"Unknown DATASET={DATASET!r}; choose from {', '.join(DATASETS)}")
    cfg = DATASETS[DATASET]
    dialect_path, english_path = Path(cfg["dialect_file"]), Path(cfg["english_file"])
    for path in (dialect_path, english_path):
        if "PATH/TO" in str(path) or not path.is_file():
            raise FileNotFoundError(f"Set a real training file path in DATASETS: {path}")

    original_pairs = read_aligned_pairs(dialect_path, english_path)
    print(f"Dataset: {DATASET} | original training pairs: {len(original_pairs)}")
    print(
        f"WARNING: generating Hindi-proxy Devanagari with {MODEL_NAME}; "
        "synthetic inputs are not verified Rajasthani/Haryanvi."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (
        torch.float16 if device == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = ENGLISH_LANG
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype
    ).to(device)
    model.eval()
    forced_bos_id = tokenizer.convert_tokens_to_ids(PROXY_LANG)
    english_lines = [english for _, english in original_pairs]
    generated = []

    for start in range(0, len(english_lines), BATCH_SIZE):
        batch = english_lines[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
            )
        generated.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        done = min(start + len(batch), len(english_lines))
        print(f"Generated {done}/{len(english_lines)} synthetic inputs")

    # Keep only non-empty, distinct pseudo-source/English pairs and reject
    # extreme length mismatches. Preserve original pairs unconditionally.
    existing_pairs = set(original_pairs)
    synthetic_pairs, synthetic_seen = [], set()
    for pseudo_dialect, english in zip(generated, english_lines):
        pseudo_dialect = " ".join(pseudo_dialect.split())
        if not pseudo_dialect:
            continue
        ratio = (len(pseudo_dialect.split()) + 1) / (len(english.split()) + 1)
        pair = (pseudo_dialect, english)
        if ratio < 1 / 3 or ratio > 3 or pair in existing_pairs or pair in synthetic_seen:
            continue
        synthetic_seen.add(pair)
        synthetic_pairs.append(pair)

    combined_pairs = original_pairs + synthetic_pairs
    output_dir = Path(cfg["output_dir"])
    dialect_ext = cfg["dialect_ext"]
    save_pairs(output_dir / f"original.{dialect_ext}", output_dir / "original.eng", original_pairs)
    save_pairs(output_dir / f"synthetic.{dialect_ext}", output_dir / "synthetic.eng", synthetic_pairs)
    # This train.* naming lets either existing fine-tuning script use this
    # directory after its DATASET file paths are pointed here.
    save_pairs(output_dir / f"train.{dialect_ext}", output_dir / "train.eng", combined_pairs)
    manifest = {
        "dataset": DATASET,
        "base_model": MODEL_NAME,
        "source_language_code": PROXY_LANG,
        "target_language_code": ENGLISH_LANG,
        "original_pairs": len(original_pairs),
        "synthetic_pairs_kept": len(synthetic_pairs),
        "combined_pairs": len(combined_pairs),
        "note": "Synthetic source text uses NLLB Hindi proxy and is not dialect-verified.",
        "seed": SEED,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Original pairs:  {len(original_pairs)}")
    print(f"Synthetic pairs: {len(synthetic_pairs)}")
    print(f"Combined pairs:  {len(combined_pairs)}")
    print(f"Saved to: {output_dir.resolve()}")
    print("Keep validation/test files unchanged and compare with/without augmentation.")


if __name__ == "__main__":
    main()
