# src/retrieval/vector_store.py

import os
import json
from chromadb import PersistentClient
from openai import OpenAI
from dotenv import load_dotenv

# (Optional) silence those telemetry warnings
os.environ["CHROMA_TELEMETRY"] = "false"

# Load OpenAI API key
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Must match embed_chunks.py's persist directory
PERSIST_DIR    = "data/db"
COLLECTION_NAME = "cs189"

# Re-open the exact same on-disk store you just populated
client     = PersistentClient(PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

# DEBUG: print stored embedding dimension
store = collection.get(include=["embeddings"])
embs  = store["embeddings"]

try:
    first_vec = embs[0]
    print(f"👉 Stored embedding dimension = {len(first_vec)}")
except Exception as e:
    # Will catch empty lists or numpy ambiguity
    print(f"⚠️ Could not read embeddings (type {type(embs)}, error: {e})")

def query(text: str, k: int = 3):
    """Return top-k documents and metadata for the query text."""
    # Generate embedding for the query using the SAME model as storage
    resp = openai_client.embeddings.create(input=[text], model="text-embedding-3-small")
    query_embedding = resp.data[0].embedding
    
    return collection.query(
        query_embeddings=[query_embedding],  # Use query_embeddings instead of query_texts
        n_results=k,
    )

if __name__ == "__main__":
    queries = [
        "What is path compression in Union-Find?",
        "Explain the vanishing gradient problem",
        "How do ReLUs help in deep neural networks?",
    ]
    for q in queries:
        print(f"\n=== Query: {q}")
        result = query(q, k=3)

        print("Full result:")
        print(json.dumps(result, indent=2))

        print("\nTop hits:")
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        if not docs:
            print(
                "No results returned. Make sure you:\n"
                "  • Ran embed_chunks.py after `rm -rf data/db`\n"
                f"  • Are using the SAME PERSIST_DIR ({PERSIST_DIR}) and collection name."
            )
            continue

        for doc, meta in zip(docs, metas):
            snippet = doc.replace("\n", " ")[:200]
            print(f" • {meta['pdf']} p{meta['page']} [topic={meta['topic']}] → {snippet}...")