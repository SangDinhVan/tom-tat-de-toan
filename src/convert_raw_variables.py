import re, json
from pathlib import Path

IN_PATH = "data/raw/train.jsonl"          # file hiện tại của bạn
OUT_PATH = "data/raw/train.fixed.jsonl"   # file sau khi convert

assign_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$")

def parse_vars(s: str) -> dict:
    out = {}
    for line in s.splitlines():
        line = line.strip()
        if not line or line.startswith("print("):
            continue
        m = assign_re.match(line)
        if m:
            k, v = m.group(1), m.group(2)
            out[k] = float(v) if "." in v else int(v)
    return out

def main():
    in_file = Path(IN_PATH)
    out_file = Path(OUT_PATH)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with in_file.open("r", encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)
            vars_field = ex.get("variables", {})
            if isinstance(vars_field, str):
                ex["variables"] = parse_vars(vars_field)
            # nếu đã là dict thì giữ nguyên
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Wrote fixed file -> {OUT_PATH}")

if __name__ == "__main__":
    main()
