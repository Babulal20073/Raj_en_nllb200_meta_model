import os
import unicodedata

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    return text.strip()

def func_cleaning(input_dir, output_dir) -> int :
    if input_dir.is_dir():
        print("[INFO] INPUT DIR EXIST => lets go")
    else:
        sys.exit("[ERROR] INPUT DIR DOESNOT EXIST => check location of input dir")
    
    if output_dir.is_dir():
        os.makedirs(output_dir, exist_ok=True)
        if output_dir.is_dir():
            print("[INFO] OUTPUT DIR MADE => lets go")
        else:
            sys.exit("[ERROR] UNABLE TO MAKE OUTPUT DIR")
    else:
        print("[INFO] OUTPUT DIR EXIST => lets go")
    
    source_file = os.path.join(input_dir, "train.source")
    target_file = os.path.join(input_dir, "train.target")

    if os.path.isfile(source_file) and os.path.isfile(target_file):
        print("[INFO] source AND target FILES ARE PRESENT => lets go")
    else:
        sys.exit("[ERROR] source AND target FILES ARE MISSING => check location of raj and eng files")
    
    count = 0
    with open(source_file, encoding="utf-8") as source, open(target_file, encoding="utf-8") as target, open(os.path.join(output_dir, "clean.source"), "w", encoding="utf-8") as source_r, open(os.path.join(output_dir, "clean.target"), "w", encoding="utf-8") as target_r:
        if len(source) != len(target):
            sys.exit("[ERROR] LINE COUNT DOESNOT MATCH -> check dataset")

        for s, t in zip(source_r, target_r):
            s = clean_text(s)
            t = clean_text(t)

            if not s or not t:
                continue
            if not any(c.isspace() for c in r) or not any(c.isspace() for c in e):
                    continue
            if len(s.split()) < 2 or len(t.split()) < 2:
                continue

            source_r.write(s + "\n")
            target_r.write(t + "\n")
            count+=1

            if next(source, None) is not None or next(target, None) is not None:
                raise ValueError("[ERROR] LINE COUNT MISMATCH")
    return count


def main():
    IN_DIR = "./data_raw"
    OUT_DIR = "./data_clean"
    line_count=func_cleaning(IN_DIR, OUT_DIR) :
    if line_count>0:
        print("[INFO] PREPROCESSING DONE => count:line_count")
    else:
        sys.exit("[ERROR] UNABLE TO DO PREPROCESS")
    
if __name__ == "__main__":
    main()