# src/ingest/label_exam_topics.py

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Input + output
IN_FILE  = Path("data/processed/exam_chunks_llm.jsonl")
OUT_FILE = Path("data/processed/exam_chunks_topical.jsonl")

# Reuse your existing CS-concept labels:
TOPICS = [
    "perceptron", "hard_margin_svm", "soft_margin_svm",
    "gaussian_discriminant", "lda", "qda",
    "logistic_reg", "decision_tree", "random_forest",
    "adaboost", "knn",
    "linear_reg", "polynomial_reg", "ridge_reg", "lasso",
    "neural_nets", "backpropagation", "cnn", "batch_norm",
    "resnet", "adam", "sgd", "vanishing_gradient",
    "pca", "svd", "k_means", "hierarchical_clustering",
    "density_estimation", "mle",
    "bias_variance", "bayes_decision", "map",
    "gradient_descent", "optimization",
    "kernels", "kernel_trick", "dimensionality_reduction",
    "misc"
]
TOPIC_CHOICES = ", ".join(f"'{t}'" for t in TOPICS)

def parse_json_object(text: str) -> dict:
    """Extract a single JSON object from LLM output."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found")
    js = m.group(0)
    # normalize quotes
    js = re.sub(r"(?P<pre>[\{\[,]\s*)'([^']*)'(?=\s*:)", r'\1"\2"', js)
    js = re.sub(r":\s*'([^']*)'", r': "\1"', js)
    js = re.sub(r",(\s*[\]\}])", r"\1", js)
    return json.loads(js)

def label_topic(text: str, chunk_id: str) -> str:
    prompt = f"""
IMPORTANT: respond with ONLY a JSON object with keys "id" and "topic", no extra text.

Allowed topics: {TOPIC_CHOICES}

Here is an exam question (id={chunk_id}):

```
{text}
```

Pick the single most relevant topic from the allowed list that best describes the *content* of this question.
Return exactly:
{{
  "id": "{chunk_id}",
  "topic": "<one of the allowed topics>"
}}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    obj = parse_json_object(resp.choices[0].message.content)
    return obj.get("topic", "misc")

def main():
    if not IN_FILE.exists():
        print(f"❌ Input file not found: {IN_FILE}")
        return

    with IN_FILE.open() as src, OUT_FILE.open("w") as dst:
        for line in src:
            chunk = json.loads(line)
            try:
                topic = label_topic(chunk["text"], chunk["id"])
            except Exception as e:
                print(f"⚠️  failed to label {chunk['id']}, defaulting to misc: {e}")
                topic = "misc"

            chunk["topic"] = topic
            dst.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            print(f"📝 {chunk['id']} → {topic}")

    print(f"Done — wrote topics to {OUT_FILE}")

if __name__ == "__main__":
    main()
