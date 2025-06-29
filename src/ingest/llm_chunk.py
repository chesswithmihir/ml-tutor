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
TOPICS = "'knn','linear_reg','neural_nets','kernels','misc'"


def parse_json(text: str) -> List[Dict[str, str]]:
    """Extract JSON from an LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fence = re.search(r"```json(.*?)```", text, re.DOTALL)
        if fence:
            return json.loads(fence.group(1).strip())
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def clean_and_chunk(page_text: str, page_id: str) -> List[Dict[str, str]]:
    """Call GPT to clean OCR text and split into knowledge chunks."""
    prompt = f"""
I’m building a tutor.  Here's raw OCR from page {page_id}:

```
{page_text}
```

1) Clean up the text: fix broken lines, remove junk characters, restore simple math like x_1, x^2.
2) Split it into 3–5 “knowledge chunks” (each <=250 words).
3) For each chunk, provide JSON:
{{
  "id": "{page_id}_chunk{{n}}",
  "topic": <one of {TOPICS}>,
  "text": "<cleaned chunk>"
}}
Return a JSON array of these objects.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return parse_json(resp.choices[0].message.content)


def is_garbled(text: str) -> bool:
    if not text:
        return True
    bad = sum(1 for c in text if c in {'?', '\x00'})
    return bad / len(text) > 0.2


def main() -> None:
    with RAW_CHUNKS.open() as src, OUT_FILE.open('w') as dst:
        for line in src:
            page = json.loads(line)
            cleaned = clean_and_chunk(page["text"], page["id"])
            for chunk in cleaned:
                if is_garbled(chunk["text"]):
                    print(f"Garbled chunk {chunk['id']}")
                    continue
                chunk.update({
                    "pdf": page["pdf"],
                    "page": page["page"],
                    "source": page["source"],
                })
                dst.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
