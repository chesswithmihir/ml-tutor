#!/usr/bin/env python3
# src/tutor/broker.py

import os
import argparse
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient

# Silence Chroma telemetry warnings
os.environ["CHROMA_TELEMETRY"] = "false"

# Load API key
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths & collection names must match your embedding scripts
PERSIST_DIR       = "data/db"
NOTES_COLLECTION  = "cs189"
EXAMS_COLLECTION  = "ml_tutor_exams"

class MLTutorBroker:
    def __init__(self):
        self.client           = PersistentClient(path=PERSIST_DIR)
        self.notes_collection = self.client.get_collection(NOTES_COLLECTION)
        self.exams_collection = self.client.get_collection(EXAMS_COLLECTION)

    def _embed(self, text: str, model: str) -> List[float]:
        resp = openai_client.embeddings.create(input=[text], model=model)
        return resp.data[0].embedding

    def query_notes(self, query: str, k: int = 3) -> List[Dict]:
        emb = self._embed(query, "text-embedding-3-small")
        r   = self.notes_collection.query(
            query_embeddings=[emb], n_results=k
        )
        docs = r["documents"][0]
        metas = r["metadatas"][0]
        dists = r["distances"][0]
        return [
            {"text": d, "meta": m, "dist": dist}
            for d, m, dist in zip(docs, metas, dists)
        ]

    def query_exams(self, query: str, k: int = 5, topic: Optional[str] = None) -> List[Dict]:
        emb = self._embed(query, "text-embedding-ada-002")
        where = {"topic": topic} if topic else None
        r = self.exams_collection.query(
            query_embeddings=[emb],
            n_results=k,
            where=where
        )
        docs = r["documents"][0]
        metas = r["metadatas"][0]
        dists = r["distances"][0]
        return [
            {"text": d, "meta": m, "dist": dist}
            for d, m, dist in zip(docs, metas, dists)
        ]


def main():
    p = argparse.ArgumentParser(
        description="ML-Tutor: retrieve notes definitions & exam questions via RAG"
    )
    p.add_argument("query", help="What the student wants (e.g. “ridge regression”).")
    p.add_argument(
        "--no-exams", action="store_true",
        help="Only show notes; skip exam lookup"
    )
    p.add_argument(
        "--topic", default=None,
        help="Filter exam questions by topic (e.g. ridge_reg, svm, etc.)"
    )
    args = p.parse_args()

    broker = MLTutorBroker()

    print(f"\n🔍 Query: “{args.query}”\n")

    print("📚  Top definitions from notes:")
    notes = broker.query_notes(args.query, k=3)
    for hit in notes:
        m = hit["meta"]
        print(f" • {m['pdf']} p{m['page']} [topic={m['topic']}] → {hit['text'][:200]}…")

    if not args.no_exams:
        print("\n📝  Top exam questions:")
        exams = broker.query_exams(args.query, k=5, topic=args.topic)
        for hit in exams:
            m = hit["meta"]
            # label question vs solution
            src = "Q" + m["question_id"] if m["source"]=="exam_blank" else "A" + m["question_id"]
            print(f" • [{m['exam']}.{src}] [topic={m['topic']}] → {hit['text'][:200]}…")

    print()

if __name__ == "__main__":
    main()
