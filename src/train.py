import json
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


# ---------------------------
# Helpers: ép kiểu an toàn
# ---------------------------
def to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        if default is None:
            raise
        return default

def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        # hỗ trợ kiểu "2e-4" nếu YAML đọc thành string
        try:
            return float(str(x).strip())
        except Exception:
            if default is None:
                raise
            return default

def to_bool(x, default=None):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
    if default is None:
        raise ValueError(f"Cannot parse bool from {x!r}")
    return default


# ---------------------------
# Load config
# ---------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_path = cfg["paths"]["model_name"]

# ép kiểu các hyperparams để tránh "float vs str"
batch_size = to_int(cfg["training"]["batch_size"])
grad_acc = to_int(cfg["training"]["gradient_accumulation_steps"])
num_epochs = to_float(cfg["training"]["num_epochs"])  # Trainer accept float/int
lr = to_float(cfg["training"]["learning_rate"])
fp16 = to_bool(cfg["training"].get("fp16", False))

max_seq_length = to_int(cfg["training"].get("max_seq_length", 2048), default=2048)

# ---------------------------
# Load tokenizer + model (8bit)
# (Kaggle hay VRAM ít: nếu lỗi, đổi sang BitsAndBytesConfig 4bit)
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True,
)

# chuẩn cho k-bit training
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=to_int(cfg["lora"]["r"]),
    lora_alpha=to_int(cfg["lora"]["alpha"]),
    lora_dropout=to_float(cfg["lora"]["dropout"]),
    target_modules=cfg["lora"]["target_modules"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)


# ---------------------------
# Load dataset JSONL -> HuggingFace Dataset
# Expect each line: {"messages":[...]}  (chat format)
# ---------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

raw_train = read_jsonl(cfg["paths"]["processed_train"])

def safe_cast(obj):
    if isinstance(obj, dict):
        return {k: safe_cast(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_cast(x) for x in obj]
    if isinstance(obj, (int, float)):
        return str(obj)  # 🔥 ÉP TOÀN BỘ SỐ → STRING
    return obj

safe_data = [safe_cast(x) for x in raw_train]

train_ds = Dataset.from_list(safe_data)

# Convert messages -> a single text prompt (simple, stable)
def messages_to_text(ex):
    msgs = ex["messages"]
    # format rất basic để model học "user -> assistant JSON"
    # (Bạn có thể thay bằng tokenizer.apply_chat_template nếu muốn)
    text = ""
    for m in msgs:
        role = m["role"]
        content = m["content"]
        text += f"<|{role}|>\n{content}\n"
    text += "<|end|>\n"
    return {"text": text}

train_ds = train_ds.map(messages_to_text, remove_columns=train_ds.column_names)

def tokenize(ex):
    out = tokenizer(
        ex["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    # Causal LM: labels = input_ids
    out["labels"] = out["input_ids"].copy()
    return out

train_ds = train_ds.map(tokenize, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ---------------------------
# TrainingArguments (đã ép kiểu lr)
# ---------------------------
args = TrainingArguments(
    output_dir=cfg["paths"]["output_dir"],
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc,
    num_train_epochs=num_epochs,
    learning_rate=lr,                 # ✅ chắc chắn float
    fp16=fp16,
    logging_dir=cfg["paths"]["log_dir"],
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=data_collator,
)

trainer.train()
