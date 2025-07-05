#!/usr/bin/env python3
# src/retrieval/vector_store_exams.py

import os
import json
from chromadb import PersistentClient
from openai import OpenAI
from dotenv import load_dotenv

# ───── optional: silence Chroma telemetry warnings ─────
os.environ["CHROMA_TELEMETRY"] = "false"

# ───── load your API key ─────
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───── change these if you moved your exams into a separate folder ─────
PERSIST_DIR     = "data/db"
COLLECTION_NAME = "ml_tutor_exams"

# ───── open the on-disk store and collection ─────
client     = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

# ───── quick sanity checks ─────
print(f"📚 Collection name: {COLLECTION_NAME}")
try:
    total = collection.count()
    print(f"🔢 Total exam chunks in collection: {total}")
except Exception:
    print("⚠️  Couldn't read collection.count()")

# fetch embeddings array and print its dimension
store = collection.get(include=["embeddings"])
embs  = store.get("embeddings", [])

try:
    if len(embs) > 0:
        # embeddings might be a numpy array or list-of-lists
        first = embs[0]
        print(f"🎨 Stored embedding dimension = {len(first)}")
    else:
        print("⚠️  No embeddings found in collection!")
except Exception as e:
    print(f"⚠️  Error inspecting embeddings: {e}")

def query(text: str, k: int = 3):
    """Embed the query, then ask Chroma for top-k nearest exam chunks."""
    resp = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"   # same model you used for exams
    )
    q_emb = resp.data[0].embedding
    return collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )

if __name__ == "__main__":
    tests = [
        "Show me the ridge regression question",
        "Which question asks about kernelized SVM?",
        "What is the bootstrap method question?",
        "Give me Q3 from midf15",
    ]
    for t in tests:
        print(f"\n=== Query: {t!r}")
        res = query(t, k=3)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        if not docs:
            print("⚠️  No results—did you embed into the right collection?")
            continue

        for doc, meta in zip(docs, metas):
            snippet = doc.replace("\n", " ")[:200]
            print(f" • [{meta['exam']}.{meta['question_id']} | {meta['source']}] → {snippet}…")
