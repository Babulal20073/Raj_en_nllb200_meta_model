import os
import unicodedata
import random 

SEED = 80085 # "BOOBS" 

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    return text.strip()

def save_split(data, start, end, output_dir, name) :
    source_out = os.path.join(output_dir, f"{name}.source")
    target_out = os.path.join(output_dir, f"{name}.target")

    with open(source_out, "w", encoding="utf-8",buffering=1024 * 1024) as source_w,open(target_out, "w", encoding="utf-8", buffering=1024 * 1024) as target_w:
        for source_line, target_line in data[start:end]:
            source_w.write(source_line)
            target_w.write(target_line)

    return source_out, target_out

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

def func_train_test_val_split(clean_dir, output_dir) -> tuple :
    
    if clean_dir.is_dir():
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

    with open(source_file, encoding="utf-8") as source, open(target_file, encoding="utf-8") as target:
        total_data = list(zip_longest(source, target, fillvalue=None))
        if next(source, None) is not None or next(target, None) is not None:
                raise ValueError("[ERROR] LINE COUNT MISMATCH")

    total_data = len(data)

    random.Random(SEED).shuffle(total_data)

    train_end = int(0.75 * total_data)
    val_end = int(0.9 * total_data)

    train_paths = save_split(data, 0, train_end, output_dir, "train")
    val_paths = save_split(data, train_end, val_end, output_dir, "val")
    test_paths = save_split(data, val_end, total_data, output_dir, "test")

    
    return total_data,train_end,val_end - train_end,total_data - val_end,train_paths,val_paths,test_paths



def main():
    IN_DIR = "./data_raw"
    OUT_DIR = "./data_clean"
    TRAINING_DATE = "./data_split"
    
    line_count=func_cleaning(IN_DIR, OUT_DIR) 
    if line_count>0:
            print(f"[INFO] PREPROCESSING DONE => count:{line_count}")
    else:
        sys.exit("[ERROR] UNABLE TO DO PREPROCESS")
    
    
    total, train, val, test, train_paths, val_paths, test_paths = func_train_test_val_split(OUT_DIR, TRAINING_DATE)

    print(
        f"[INFO] DATASET SPLIT COMPLETE => "
        f"total:{total}, train:{train}, val:{val}, test:{test}"
    )

    print("[INFO] TRAIN PATHS:", train_paths)
    print("[INFO] VALIDATION PATHS:", val_paths)
    print("[INFO] TEST PATHS:", test_paths)


if __name__ == "__main__":
    main()