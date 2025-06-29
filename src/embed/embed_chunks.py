import json, os, chromadb
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

db = chromadb.Client().get_or_create_collection("cs189")

batch = []
with open("data/processed/chunks_tagged.jsonl") as f:
    for line in f:
        chunk = json.loads(line)
        batch.append(chunk)
        if len(batch) == 100:
            ids = [c["id"] for c in batch]
            texts = [c["text"] for c in batch]
            meta  = [{"topic": c["topic"], **{k: c[k] for k in ("source","pdf","page")}}
                     for c in batch]
            embs = client.embeddings.create(input=texts,
                                            model="text-embedding-3-small").data
            db.add(documents=texts, metadatas=meta, ids=ids, embeddings=[e.embedding for e in embs])
            batch.clear()
