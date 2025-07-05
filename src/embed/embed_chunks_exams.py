# src/embed/embed_chunks_exams.py

import json, os, pathlib
from openai import OpenAI
from dotenv import load_dotenv

# ← switch to the persistent client
from chromadb import PersistentClient

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAW = pathlib.Path("data/processed/exam_chunks_topical.jsonl")
PERSIST_DIR     = "data/db"              # ← same folder as notes
COLLECTION_NAME = "ml_tutor_exams"

# ← re-open on-disk Chroma and get (or create) the exams collection
db = PersistentClient(path=PERSIST_DIR)
collection = db.get_or_create_collection(COLLECTION_NAME)

for line in RAW.open():
    chunk = json.loads(line)
    resp = openai_client.embeddings.create(
      input=[chunk["text"]],
      model="text-embedding-ada-002"
    )
    vec = resp.data[0].embedding

    collection.add(
      ids=[chunk["id"]],
      embeddings=[vec],
      metadatas=[{
        "exam":        chunk["exam"],
        "question_id": chunk["question_id"],
        "topic":       chunk["topic"],
        "source":      chunk["source"],
        "pdf":         chunk["pdf"],
        "page":        chunk["page"],
      }],
      documents=[chunk["text"]]
    )

print("✅ All exam chunks embedded into persistent collection 'ml_tutor_exams'")
