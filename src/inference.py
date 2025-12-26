import yaml, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

cfg = yaml.safe_load(open("config.yaml"))

tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["output_dir"])
model = AutoModelForCausalLM.from_pretrained(
    cfg["paths"]["output_dir"],
    device_map="auto"
)

prompt = """Convert the following math word problem into InfinityMATH format:
Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes.
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    temperature=cfg["generation"]["temperature"],
    max_new_tokens=cfg["generation"]["max_new_tokens"]
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
