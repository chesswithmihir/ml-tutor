# src/ingest/llm_chunk_exam.py

import json
import os
import pathlib
import re
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ← point this at the exam questions you just split
RAW_CHUNKS = pathlib.Path("data/processed/exam_chunks.jsonl")
OUT_FILE   = pathlib.Path("data/processed/exam_chunks_llm.jsonl")

# We'll tag blank‐pages as "exam_problem" and solution pages as "exam_solution"
TOPICS = ["exam_problem", "exam_solution"]
TOPIC_CHOICES = ", ".join(f"'{t}'" for t in TOPICS)

def parse_json(text: str) -> Dict[str, str]:
    """Parse JSON response from LLM, expecting a single object."""
    # Look for JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found")
    js = m.group(0)
    
    # Fix common JSON formatting issues from LLM responses
    # Replace single quotes with double quotes for keys
    js = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', js)
    # Replace single quotes with double quotes for string values
    js = re.sub(r":\s*'([^']*)'", r': "\1"', js)
    # Remove trailing commas
    js = re.sub(r",(\s*[\]\}])", r"\1", js)
    
    return json.loads(js)

def clean_question_text(question_text: str, question_id: str, chunk_id: str) -> Dict[str, str]:
    """
    Clean and process a single question using LLM.
    Returns a single cleaned chunk object.
    """
    prompt = f"""
IMPORTANT: Return ONLY a JSON object with keys id, topic, text.
Allowed topics: {TOPIC_CHOICES}

Here is the OCR text for question {question_id}:
```
{question_text}
```

Clean this question text by:
1) Fixing broken lines, junk chars, fix math exponents/subscripts
2) Ensuring the text is coherent and readable
3) Keeping the question as a single unit (don't split into multiple parts)

Return exactly this format:
{{
  "id": "{chunk_id}",
  "topic": "<one of the allowed topics>",
  "text": "<cleaned question text>"
}}
"""
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return parse_json(resp.choices[0].message.content)

def is_garbled(text: str) -> bool:
    """Check if text appears to be garbled OCR."""
    if not text:
        return True
    bad = sum(1 for c in text if c in {"?", "\x00"})
    return bad / len(text) > 0.2

def main():
    with RAW_CHUNKS.open() as src, OUT_FILE.open("w") as dst:
        for line in src:
            question = json.loads(line)
            
            # Choose default topic by source
            default_topic = (
                "exam_problem"
                if question["source"] == "exam_blank"
                else "exam_solution"
            )
            
            # Skip if text is obviously garbled
            if is_garbled(question.get("text", "")):
                print(f"🗑 skipping garbled question {question['id']}")
                continue
            
            try:
                # Clean the question text using LLM
                cleaned = clean_question_text(
                    question["text"], 
                    question["question_id"], 
                    question["id"]
                )
                
                # Override topic with our default
                cleaned["topic"] = default_topic
                
                # Carry through all the original metadata
                cleaned.update({
                    "exam":        question["exam"],        # e.g. "midf15"
                    "question_id": question["question_id"], # e.g. "Q3"
                    "source":      question["source"],      # "exam_blank" or "exam_solution"
                    "pdf":         question["pdf"],         # filename
                    "page":        question["page"],        # page number
                    "page_image":  question["page_image"],  # image path
                    "linked_pdf":  question["linked_pdf"],  # solution ↔ blank link
                })
                
                dst.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                print(f"✅ wrote {cleaned['id']} [{cleaned['topic']}] - {cleaned['question_id']}")
                
            except Exception as e:
                print(f"❌ processing failed for {question['id']}: {e}")
                # Write the original as fallback
                fallback_chunk = {
                    "id": question["id"],
                    "topic": default_topic,
                    "text": question["text"],
                    "exam": question["exam"],
                    "question_id": question["question_id"],
                    "source": question["source"],
                    "pdf": question["pdf"],
                    "page": question["page"],
                    "page_image": question["page_image"],
                    "linked_pdf": question["linked_pdf"],
                }
                dst.write(json.dumps(fallback_chunk, ensure_ascii=False) + "\n")
                print(f"⚠️  wrote fallback for {question['id']}")

    print("Done — exam question chunks written to", OUT_FILE)

if __name__ == "__main__":
    main()