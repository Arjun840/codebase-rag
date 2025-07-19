#!/usr/bin/env python3
"""
Build ChromaDB Vector Index for Codebase Embeddings

This script loads saved embeddings and metadata, initializes a ChromaDB collection,
and adds all vectors and metadata for fast similarity search.

Usage:
    python scripts/build_chromadb_index.py --embeddings-path ./embeddings/embeddings.npy --metadata-path ./embeddings/metadata.json --collection-name codebase
"""

import argparse
import numpy as np
import json
from pathlib import Path
import chromadb


def clean_metadata_for_chromadb(metadata):
    """Clean metadata to be compatible with ChromaDB's strict type requirements."""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            # Skip None values
            continue
        elif isinstance(value, (int, float, str, bool)):
            # These types are directly supported
            cleaned[key] = value
        elif isinstance(value, list):
            # Convert lists to strings
            cleaned[key] = str(value)
        elif isinstance(value, dict):
            # Convert dicts to strings
            cleaned[key] = str(value)
        else:
            # Convert everything else to string
            cleaned[key] = str(value)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index from codebase embeddings.")
    parser.add_argument('--embeddings-path', type=Path, default=Path('./embeddings/embeddings.npy'), help='Path to embeddings.npy')
    parser.add_argument('--metadata-path', type=Path, default=Path('./embeddings/metadata.json'), help='Path to metadata.json')
    parser.add_argument('--collection-name', type=str, default='codebase', help='ChromaDB collection name')
    args = parser.parse_args()

    # Load embeddings and metadata
    print(f"🔹 Loading embeddings from {args.embeddings_path}")
    embeddings = np.load(args.embeddings_path)
    print(f"🔹 Loading metadata from {args.metadata_path}")
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)

    if len(embeddings) != len(metadata):
        raise ValueError(f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) count mismatch!")

    # Initialize ChromaDB client and collection
    print(f"🔹 Initializing ChromaDB collection '{args.collection_name}'")
    # Use persistent storage instead of in-memory
    client = chromadb.PersistentClient(path="./chromadb_data")
    collection = client.create_collection(args.collection_name)

    # Add all embeddings to the collection
    print(f"🔹 Adding {len(embeddings)} vectors to ChromaDB...")
    for i, (meta, vector) in enumerate(zip(metadata, embeddings)):
        # Clean metadata for ChromaDB compatibility
        cleaned_meta = clean_metadata_for_chromadb(meta)
        
        # ChromaDB expects lists, not numpy arrays
        # Use a unique id for each document
        doc_id = cleaned_meta.get('id', f'doc_{i}')
        # Use content if available, else file name
        content = cleaned_meta.get('content', '')
        if not content and 'file_name' in cleaned_meta:
            content = f"[FILE: {cleaned_meta['file_name']}]"
        
        collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[vector.tolist()],
            metadatas=[cleaned_meta]
        )
        if (i + 1) % 1000 == 0 or (i + 1) == len(embeddings):
            print(f"   Added {i + 1}/{len(embeddings)}", end='\r')

    print(f"\n✅ Added {len(embeddings)} vectors to ChromaDB collection '{args.collection_name}'.")
    print("You can now run similarity queries using this collection!")

if __name__ == "__main__":
    main() 