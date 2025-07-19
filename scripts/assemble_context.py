#!/usr/bin/env python3
"""
Assemble Context for LLM from Retrieved Chunks

This script takes retrieved code/doc chunks and assembles them into a well-formatted
context string for LLM consumption, with proper source indicators and token management.

Usage:
    python scripts/assemble_context.py --query "authentication" --max-tokens 4000
    python scripts/assemble_context.py --query "database connection" --format detailed --max-chunks 8
"""

import argparse
import json
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Tuple
import tiktoken


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def format_chunk_source(metadata: Dict[str, Any]) -> str:
    """Format source information for a chunk."""
    file_name = metadata.get('file_name', 'unknown')
    chunk_index = metadata.get('chunk_index', '?')
    language = metadata.get('language', 'unknown')
    function_name = metadata.get('function_name', '')
    class_name = metadata.get('class_name', '')
    
    source_parts = [f"File: {file_name}"]
    
    if function_name:
        source_parts.append(f"function: {function_name}")
    elif class_name:
        source_parts.append(f"class: {class_name}")
    
    source_parts.append(f"chunk: {chunk_index}")
    source_parts.append(f"language: {language}")
    
    return f"[{' | '.join(source_parts)}]"


def format_chunk_content(content: str, format_type: str = "simple") -> str:
    """Format chunk content based on type."""
    if format_type == "detailed":
        return content
    else:
        # Simple format: truncate long content
        max_length = 800
        if len(content) > max_length:
            return content[:max_length] + "..."
        return content


def assemble_context(chunks: List[Dict[str, Any]], 
                    format_type: str = "simple",
                    max_tokens: int = 4000,
                    max_chunks: int = None) -> Tuple[str, Dict[str, Any]]:
    """
    Assemble retrieved chunks into LLM context.
    
    Args:
        chunks: List of chunk dictionaries with 'content', 'metadata', 'score'
        format_type: 'simple' or 'detailed'
        max_tokens: Maximum tokens in final context
        max_chunks: Maximum number of chunks to include
    
    Returns:
        Tuple of (context_string, stats_dict)
    """
    if not chunks:
        return "", {"total_chunks": 0, "total_tokens": 0, "truncated": False}
    
    # Sort chunks by score (highest first)
    sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
    
    # Limit by max_chunks if specified
    if max_chunks:
        sorted_chunks = sorted_chunks[:max_chunks]
    
    # Build context
    context_parts = []
    total_tokens = 0
    chunks_used = 0
    truncated = False
    
    for chunk in sorted_chunks:
        content = chunk['content']
        metadata = chunk['metadata']
        score = chunk['score']
        
        # Format source
        source = format_chunk_source(metadata)
        
        # Format content
        formatted_content = format_chunk_content(content, format_type)
        
        # Create chunk block
        chunk_block = f"{source}\n{formatted_content}\n"
        
        # Check token limit
        chunk_tokens = count_tokens(chunk_block)
        
        if total_tokens + chunk_tokens > max_tokens:
            truncated = True
            break
        
        context_parts.append(chunk_block)
        total_tokens += chunk_tokens
        chunks_used += 1
    
    # Join with separators
    context = "\n---\n".join(context_parts)
    
    stats = {
        "total_chunks": chunks_used,
        "total_tokens": total_tokens,
        "truncated": truncated,
        "max_tokens": max_tokens,
        "format_type": format_type
    }
    
    return context, stats


def retrieve_chunks(query: str, 
                   collection_name: str = "codebase",
                   top_k: int = 10,
                   model_name: str = "flax-sentence-embeddings/st-codesearch-distilroberta-base") -> List[Dict[str, Any]]:
    """Retrieve chunks from ChromaDB."""
    # Load model and embed query
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])[0]
    
    # Query ChromaDB
    client = chromadb.PersistentClient(path="./chromadb_data")
    collection = client.get_collection(collection_name)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    # Format results
    chunks = []
    for doc, meta, score in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        chunks.append({
            'content': doc,
            'metadata': meta,
            'score': score
        })
    
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Assemble context from retrieved chunks for LLM.")
    parser.add_argument('--query', type=str, required=True, help='Search query')
    parser.add_argument('--collection-name', type=str, default='codebase', help='ChromaDB collection name')
    parser.add_argument('--top-k', type=int, default=10, help='Number of chunks to retrieve')
    parser.add_argument('--format', type=str, choices=['simple', 'detailed'], default='simple', 
                       help='Format type for chunk content')
    parser.add_argument('--max-tokens', type=int, default=4000, help='Maximum tokens in context')
    parser.add_argument('--max-chunks', type=int, help='Maximum number of chunks to include')
    parser.add_argument('--output-file', type=str, help='Save context to file')
    parser.add_argument('--show-stats', action='store_true', help='Show detailed statistics')
    args = parser.parse_args()

    print(f"🔍 Retrieving chunks for query: '{args.query}'")
    
    # Retrieve chunks
    chunks = retrieve_chunks(args.query, args.collection_name, args.top_k)
    print(f"📄 Retrieved {len(chunks)} chunks")
    
    # Assemble context
    context, stats = assemble_context(
        chunks, 
        format_type=args.format,
        max_tokens=args.max_tokens,
        max_chunks=args.max_chunks
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("ASSEMBLED CONTEXT FOR LLM")
    print(f"{'='*60}")
    
    if args.show_stats:
        print(f"\n📊 Statistics:")
        print(f"   Chunks used: {stats['total_chunks']}")
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Max tokens: {stats['max_tokens']}")
        print(f"   Format: {stats['format_type']}")
        print(f"   Truncated: {stats['truncated']}")
        print(f"   Token usage: {stats['total_tokens']/stats['max_tokens']*100:.1f}%")
    
    print(f"\n📝 Context ({stats['total_tokens']} tokens):")
    print("-" * 60)
    print(context)
    print("-" * 60)
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(context)
        print(f"\n💾 Context saved to: {args.output_file}")
    
    # Print usage instructions
    print(f"\n💡 Usage:")
    print(f"   Copy the context above and paste it into your LLM prompt.")
    print(f"   Example prompt: 'Based on this codebase context: {context[:100]}...'")
    print(f"   Question: '{args.query}'")


if __name__ == "__main__":
    main() 