import json, re, pathlib, yaml
from tqdm import tqdm

KEYWORDS = {
    "knn": ["nearest", r"\bk-?nn\b"],
    "linear_regression": ["least squares", r"\bnormal equation\b"],
    "lda_qda": [r"\blda\b", r"\bqda\b", "discriminant"],
    "neural_nets": ["backprop", "activation", "gradient descent"],
}

def match_topic(text: str):
    for topic, pats in KEYWORDS.items():
        for p in pats:
            if re.search(p, text, re.I):
                return topic
    return "misc"

out = []
for line in pathlib.Path("data/processed/chunks.jsonl").read_text().splitlines():
    chunk = json.loads(line)
    chunk["topic"] = match_topic(chunk["text"])
    out.append(chunk)

with open("data/processed/chunks_tagged.jsonl", "w") as f:
    for c in out:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
