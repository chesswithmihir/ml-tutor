# src/ingest/llm_chunk.py

import json
import os
import pathlib
import re
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAW_CHUNKS = pathlib.Path("data/processed/chunks.jsonl")
OUT_FILE = pathlib.Path("data/processed/chunks_llm.jsonl")
TOPICS = [
    # Classification
    "perceptron", "hard_margin_svm", "soft_margin_svm",
    "gaussian_discriminant", "lda", "qda",
    "logistic_reg", "decision_tree", "random_forest",
    "adaboost", "knn",

    # Regression
    "linear_reg", "polynomial_reg",
    "ridge_reg", "lasso",

    # Neural nets
    "neural_nets", "backpropagation",
    "cnn", "batch_norm", "resnet",
    "adam", "sgd", "vanishing_gradient",

    # Unsupervised & density
    "pca", "svd", "k_means", "hierarchical_clustering",
    "density_estimation", "mle",

    # Topics that cut across
    "bias_variance", "bayes_decision", "map",
    "gradient_descent", "optimization",

    # Kernels & dimensionality
    "kernels", "kernel_trick", "dimensionality_reduction",

    # Catch-all
    "misc"
]
# make a quoted, comma-separated list for the prompt
TOPIC_CHOICES = ", ".join(f"'{t}'" for t in TOPICS)

def parse_json(text: str) -> List[Dict[str, str]]:
    """Extract JSON array from an LLM response, normalize quotes and remove trailing commas."""
    # 1) Extract from first '[' to last ']' if present
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON array found in LLM output")
    js = m.group(0)

    # 2) Replace single-quoted keys in JSON objects with double quotes
    js = re.sub(r"(?P<pre>[\{\[,]\s*)'(.*?)'(?=\s*:)", r"\1\"\2\"", js)
    # 3) Replace any remaining single-quoted strings with double quotes
    js = re.sub(r"(?<!\")'(.*?)'(?!\")", r"\"\1\"", js)

    # 4) Remove trailing commas before ] or }
    js = re.sub(r",(\s*[\]\}])", r"\1", js)

    # 5) Parse
    return json.loads(js)


def clean_and_chunk(page_text: str, page_id: str) -> List[Dict[str, str]]:
    """Call GPT to clean OCR text and split into knowledge chunks."""
    prompt = f"""
IMPORTANT: Respond with ONLY a JSON array of objects, no markdown or extra text.
Each object should have keys: id, topic, text.

Allowed topics: {TOPIC_CHOICES}


I'm building a tutor. Here's raw OCR from page {page_id}:

```
{page_text}
```

1) Clean up the text: fix broken lines, remove junk characters, restore simple math like x_1, x^2.
2) Split it into 3–5 'knowledge chunks' (each <=250 words).
3) For each chunk, provide JSON:
{{
  "id": "{page_id}_chunk{{n}}",
  "topic": <one of the allowed topics above>,
  "text": "<cleaned chunk text>"
}}

Return ONLY the JSON array of these objects.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content
    print(f"\n\n===== RAW LLM OUTPUT ({page_id}) =====\n{raw}\n========================================\n")
    return parse_json(raw)


def is_garbled(text: str) -> bool:
    if not text:
        return True
    bad = sum(1 for c in text if c in {'?', '\x00'})
    return bad / len(text) > 0.2


def main() -> None:
    with RAW_CHUNKS.open() as src, OUT_FILE.open('w') as dst:
        for line in src:
            page = json.loads(line)
            try:
                cleaned = clean_and_chunk(page["text"], page["id"])
            except ValueError as e:
                print(f"Failed to parse JSON for page {page['id']}: {e}")
                continue
            for chunk in cleaned:
                if is_garbled(chunk.get("text", "")):
                    print(f"Garbled chunk {chunk.get('id')}")
                    continue
                chunk.update({
                    "pdf": page["pdf"],
                    "page": page["page"],
                    "source": page["source"],
                })
                print(f"Writing chunk {chunk['id']} topic={chunk['topic']}")
                dst.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
