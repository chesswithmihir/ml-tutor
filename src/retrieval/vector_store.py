import chromadb

_db = chromadb.Client().get_or_create_collection("cs189")


def query(text: str, k: int = 3):
    """Return top-k documents and metadata for the query text."""
    return _db.query(query_texts=[text], n_results=k)


if __name__ == "__main__":
    result = query("What is path compression in Union-Find?", k=3)
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        snippet = doc[:200].replace("\n", " ")
        print(f"{meta['pdf']} p{meta['page']} :: {snippet}...")
