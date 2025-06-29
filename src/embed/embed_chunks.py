import json
import os
from typing import List, Dict

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

db = chromadb.Client().get_or_create_collection("cs189")
SOURCE_FILE = "data/processed/chunks_llm.jsonl"
BATCH_SIZE = 100


def embed_batch(batch: List[Dict[str, str]]) -> None:
    ids = [c["id"] for c in batch]
    texts = [c["text"] for c in batch]
    meta = [
        {"topic": c["topic"], **{k: c[k] for k in ("source", "pdf", "page")}}
        for c in batch
    ]
    embs = client.embeddings.create(input=texts, model="text-embedding-3-small").data
    db.add(documents=texts, metadatas=meta, ids=ids, embeddings=[e.embedding for e in embs])


def main() -> None:
    batch: List[Dict[str, str]] = []
    with open(SOURCE_FILE) as f:
        for line in f:
            chunk = json.loads(line)
            batch.append(chunk)
            if len(batch) == BATCH_SIZE:
                embed_batch(batch)
                batch.clear()
    if batch:
        embed_batch(batch)


if __name__ == "__main__":
    main()
