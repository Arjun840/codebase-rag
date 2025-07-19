#!/usr/bin/env python3
"""
Query ChromaDB Vector Index for Codebase Search

This script embeds a user query (natural language, code, or error message)
and retrieves the most similar code/doc chunks from the ChromaDB collection.

Usage:
    python scripts/query_chromadb.py --query "How do I authenticate users?" --collection-name codebase --top-k 5
    python scripts/query_chromadb.py --query "def foo(x): return x*x" --query-type code --collection-name codebase
"""

import argparse
from sentence_transformers import SentenceTransformer
import chromadb

# Helper to add context to query
def prepare_query(query, query_type):
    if query_type == "code":
        return f"What does this code do: {query}"
    elif query_type == "error":
        return f"Error message: {query}"
    elif query_type == "doc":
        return f"Documentation question: {query}"
    else:
        return query


def main():
    parser = argparse.ArgumentParser(description="Query ChromaDB for code/doc search.")
    parser.add_argument('--query', type=str, required=True, help='User query (natural language, code, or error)')
    parser.add_argument('--query-type', type=str, choices=['general', 'code', 'error', 'doc'], default='general', help='Type of query (improves embedding relevance)')
    parser.add_argument('--collection-name', type=str, default='codebase', help='ChromaDB collection name')
    parser.add_argument('--model', type=str, default='flax-sentence-embeddings/st-codesearch-distilroberta-base', help='Embedding model to use')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top results to return')
    args = parser.parse_args()

    # 1. Load embedding model
    print(f"🔹 Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # 2. Prepare query
    query_text = prepare_query(args.query, args.query_type)
    print(f"🔹 Embedding query: {query_text}")
    query_embedding = model.encode([query_text])[0]

    # Connect to ChromaDB
    print(f"🔹 Connecting to ChromaDB collection: {args.collection_name}")
    client = chromadb.PersistentClient(path="./chromadb_data")
    collection = client.get_collection(args.collection_name)

    # 4. Query for similar vectors
    print(f"🔹 Searching for top {args.top_k} most similar chunks...")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=args.top_k
    )

    # 5. Print results
    print("\n=== Top Results ===")
    for rank, (doc, meta, score) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), 1):
        file_name = meta.get('file_name', 'unknown')
        chunk_index = meta.get('chunk_index', '?')
        print(f"\n#{rank} | Score: {score:.4f} | File: {file_name} | Chunk: {chunk_index}")
        print("-"*60)
        print(doc[:500])
        print("-"*60)
        # Optionally print more metadata
        # print(meta)

if __name__ == "__main__":
    main() 