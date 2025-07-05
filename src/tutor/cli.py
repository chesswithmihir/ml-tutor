#!/usr/bin/env python3
# src/tutor/cli.py

import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient

# ───── suppress telemetry noise ─────
os.environ["CHROMA_TELEMETRY"] = "false"

def embed(openai_client, text, model):
    resp = openai_client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding

def print_hits(label, docs, metas):
    print(f"\n{label}")
    if not docs:
        print("  (none)\n")
        return
    for doc, m in zip(docs, metas):
        snippet = doc.replace("\n", " ")[:200] + "…"
        print(f" • {m.get('pdf','?')} p{m.get('page','?')} [topic={m.get('topic','?')}] → {snippet}")
    print()

def main():
    parser = argparse.ArgumentParser(description="RAG lookup for notes & exams")
    parser.add_argument("query", help="Your question or request")
    parser.add_argument("--topic", help="(optional) exam-topic to filter on, e.g. ridge_reg")
    parser.add_argument("--k_notes", type=int, default=3, help="how many definitions to fetch")
    parser.add_argument("--k_exams", type=int, default=5, help="how many exam chunks to fetch")
    args = parser.parse_args()

    load_dotenv()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # open your on-disk Chroma
    DB = "data/db"
    notes_col = PersistentClient(path=DB).get_or_create_collection("cs189")
    exams_col = PersistentClient(path=DB).get_or_create_collection("ml_tutor_exams")

    # embed + query notes
    emb_notes = embed(openai_client, args.query, "text-embedding-3-small")
    res_notes = notes_col.query(query_embeddings=[emb_notes], n_results=args.k_notes)

    # embed + query exams (with optional metadata filter)
    emb_exams = embed(openai_client, args.query, "text-embedding-ada-002")
    where = {"topic": args.topic} if args.topic else None
    res_exams = exams_col.query(
        query_embeddings=[emb_exams],
        n_results=args.k_exams,
        where=where
    )

    # pull them out
    docs_n = res_notes.get("documents", [[]])[0]
    metas_n = res_notes.get("metadatas", [[]])[0]
    docs_e = res_exams.get("documents", [[]])[0]
    metas_e = res_exams.get("metadatas", [[]])[0]

    print(f"\n🔍 RAG lookup for “{args.query}”\n")
    print_hits("📚 Top definitions from notes:", docs_n, metas_n)
    print_hits("📝 Top exam questions/solutions:", docs_e, metas_e)

if __name__ == "__main__":
    main()
