#!/usr/bin/env python3
"""
Advanced ChromaDB Query with Filtering and Reranking

This script performs similarity search with optional filtering and reranking
for more accurate code/doc retrieval.

Usage:
    python scripts/advanced_query_chromadb.py --query "authentication" --filter-file "auth.py" --rerank
    python scripts/advanced_query_chromadb.py --query "database connection" --filter-language python --top-k 10
"""

import argparse
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from typing import List, Dict, Any, Tuple


def prepare_query(query: str, query_type: str) -> str:
    """Add context to query based on type."""
    if query_type == "code":
        return f"What does this code do: {query}"
    elif query_type == "error":
        return f"Error message: {query}"
    elif query_type == "doc":
        return f"Documentation question: {query}"
    else:
        return query


def filter_results(results: Dict[str, Any], 
                  filter_file: str = None,
                  filter_language: str = None,
                  filter_function: str = None) -> Tuple[List, List, List]:
    """Filter results based on criteria."""
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    filtered_docs, filtered_metas, filtered_scores = [], [], []
    
    for doc, meta, score in zip(documents, metadatas, distances):
        # File name filter
        if filter_file and filter_file.lower() not in meta.get('file_name', '').lower():
            continue
            
        # Language filter
        if filter_language and filter_language.lower() != meta.get('language', '').lower():
            continue
            
        # Function name filter
        if filter_function and filter_function.lower() not in doc.lower():
            continue
            
        filtered_docs.append(doc)
        filtered_metas.append(meta)
        filtered_scores.append(score)
    
    return filtered_docs, filtered_metas, filtered_scores


def rerank_results(query: str, 
                  documents: List[str], 
                  metadatas: List[Dict], 
                  scores: List[float],
                  reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Tuple[List, List, List]:
    """Rerank results using a cross-encoder model."""
    print(f"🔹 Reranking with {reranker_model}...")
    
    try:
        # Load cross-encoder for reranking
        cross_encoder = CrossEncoder(reranker_model)
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Get reranking scores
        rerank_scores = cross_encoder.predict(pairs)
        
        # Sort by reranking scores
        ranked_indices = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
        
        # Reorder results
        reranked_docs = [documents[i] for i in ranked_indices]
        reranked_metas = [metadatas[i] for i in ranked_indices]
        reranked_scores = [rerank_scores[i] for i in ranked_indices]
        
        return reranked_docs, reranked_metas, reranked_scores
        
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}. Using original ranking.")
        return documents, metadatas, scores


def print_results(documents: List[str], 
                 metadatas: List[Dict], 
                 scores: List[float], 
                 show_metadata: bool = False):
    """Print search results in a formatted way."""
    print(f"\n=== Top {len(documents)} Results ===")
    
    for rank, (doc, meta, score) in enumerate(zip(documents, metadatas, scores), 1):
        file_name = meta.get('file_name', 'unknown')
        chunk_index = meta.get('chunk_index', '?')
        language = meta.get('language', 'unknown')
        
        print(f"\n#{rank} | Score: {score:.4f} | File: {file_name} | Chunk: {chunk_index} | Lang: {language}")
        print("-" * 80)
        
        # Truncate and format document content
        content = doc[:600] + "..." if len(doc) > 600 else doc
        print(content)
        
        if show_metadata:
            print(f"\nMetadata: {meta}")
        
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Advanced ChromaDB query with filtering and reranking.")
    parser.add_argument('--query', type=str, required=True, help='User query')
    parser.add_argument('--query-type', type=str, choices=['general', 'code', 'error', 'doc'], 
                       default='general', help='Type of query')
    parser.add_argument('--collection-name', type=str, default='codebase', help='ChromaDB collection name')
    parser.add_argument('--model', type=str, 
                       default='flax-sentence-embeddings/st-codesearch-distilroberta-base', 
                       help='Embedding model')
    parser.add_argument('--top-k', type=int, default=10, help='Initial number of results to retrieve')
    parser.add_argument('--filter-file', type=str, help='Filter by file name (partial match)')
    parser.add_argument('--filter-language', type=str, help='Filter by programming language')
    parser.add_argument('--filter-function', type=str, help='Filter by function name in content')
    parser.add_argument('--rerank', action='store_true', help='Use cross-encoder for reranking')
    parser.add_argument('--reranker-model', type=str, 
                       default='cross-encoder/ms-marco-MiniLM-L-6-v2', 
                       help='Cross-encoder model for reranking')
    parser.add_argument('--show-metadata', action='store_true', help='Show full metadata for each result')
    args = parser.parse_args()

    # 1. Load embedding model
    print(f"🔹 Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # 2. Prepare and embed query
    query_text = prepare_query(args.query, args.query_type)
    print(f"🔹 Embedding query: {query_text}")
    query_embedding = model.encode([query_text])[0]

    # 3. Connect to ChromaDB and search
    print(f"🔹 Connecting to ChromaDB collection: {args.collection_name}")
    client = chromadb.PersistentClient(path="./chromadb_data")
    collection = client.get_collection(args.collection_name)

    # 4. Perform initial similarity search
    print(f"🔹 Searching for top {args.top_k} most similar chunks...")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=args.top_k
    )

    # 5. Apply filters if specified
    if any([args.filter_file, args.filter_language, args.filter_function]):
        print("🔹 Applying filters...")
        documents, metadatas, scores = filter_results(
            results, 
            filter_file=args.filter_file,
            filter_language=args.filter_language,
            filter_function=args.filter_function
        )
        print(f"   Filtered to {len(documents)} results")
    else:
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        scores = results['distances'][0]

    # 6. Apply reranking if requested
    if args.rerank and documents:
        documents, metadatas, scores = rerank_results(
            args.query, documents, metadatas, scores, args.reranker_model
        )

    # 7. Print results
    if documents:
        print_results(documents, metadatas, scores, args.show_metadata)
    else:
        print("❌ No results found matching your criteria.")


if __name__ == "__main__":
    main() 