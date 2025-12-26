import json
import random
from pathlib import Path

INPUT = "data/raw/train.fixed.jsonl"
TRAIN_OUT = "data/processed/train.jsonl"
VAL_OUT = "data/processed/val.jsonl"

VAL_RATIO = 0.05   # 5% cho validation

def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.shuffle(data)
    split = int(len(data) * (1 - VAL_RATIO))

    train_data = data[:split]
    val_data = data[split:]

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for x in train_data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(VAL_OUT, "w", encoding="utf-8") as f:
        for x in val_data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"✅ Train: {len(train_data)} samples")
    print(f"✅ Val  : {len(val_data)} samples")

if __name__ == "__main__":
    main()
