from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import yaml, json

cfg = yaml.safe_load(open("config.yaml"))

model = AutoModelForCausalLM.from_pretrained(
    cfg["paths"]["model_name"],
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model_name"])
tokenizer.pad_token = tokenizer.eos_token

lora_cfg = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"],
    target_modules=cfg["lora"]["target_modules"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_cfg)

def load_dataset(path):
    with open(path) as f:
        return [json.loads(x) for x in f]

train_data = load_dataset(cfg["paths"]["processed_train"])

args = TrainingArguments(
    output_dir=cfg["paths"]["output_dir"],
    per_device_train_batch_size=cfg["training"]["batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    num_train_epochs=cfg["training"]["num_epochs"],
    learning_rate=cfg["training"]["learning_rate"],
    fp16=cfg["training"]["fp16"],
    logging_dir=cfg["paths"]["log_dir"],
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data
)

trainer.train()
