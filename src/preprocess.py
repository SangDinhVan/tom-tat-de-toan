import json
import yaml
from pathlib import Path

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW = cfg["paths"]["raw_data"]
OUT = cfg["paths"]["processed_train"]

SYSTEM = Path("prompts/system.txt").read_text()
INST = Path("prompts/instruction.txt").read_text()

out = []

with open(RAW) as f:
    for line in f:
        sample = json.loads(line)
        out.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": INST + "\n\n" + sample["input"]},
                {"role": "assistant", "content": json.dumps(sample["output"], ensure_ascii=False)}
            ]
        })

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
