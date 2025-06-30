# src/embed/embed_chunks.py
import json
import os
from typing import List, Dict

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient

# Load OpenAI API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Persistent directory for Chroma
PERSIST_DIR = "data/db"

# Initialize Chroma persistent client (on-disk storage)
db_client = PersistentClient(PERSIST_DIR)
collection = db_client.get_or_create_collection("cs189")

# Source of cleaned chunks and batching
SOURCE_FILE = "data/processed/chunks_llm.jsonl"
BATCH_SIZE = 100

def embed_batch(batch: List[Dict[str, str]]) -> None:
    """Embed a list of chunks and store them in Chroma."""
    ids = [c["id"] for c in batch]
    texts = [c["text"] for c in batch]
    metadatas = [
        {"topic": c["topic"], **{k: c[k] for k in ("source", "pdf", "page")}}
        for c in batch
    ]
    # Generate embeddings via OpenAI
    resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = [d.embedding for d in resp.data]

    # Add to Chroma collection (auto-persisted by PersistentClient)
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )


def main() -> None:
    total = 0
    batch: List[Dict[str, str]] = []
    with open(SOURCE_FILE, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            batch.append(chunk)
            if len(batch) >= BATCH_SIZE:
                embed_batch(batch)
                total += len(batch)
                batch.clear()
    # Handle final batch
    if batch:
        embed_batch(batch)
        total += len(batch)
    print(f"Embedded {total} chunks into collection 'cs189' (persisted at {PERSIST_DIR})")


if __name__ == "__main__":
    main()
