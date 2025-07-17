#!/usr/bin/env python3
"""
Simple Embedding Demo

This script demonstrates the basic usage of the EmbeddingGenerator
with sample code snippets and queries.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebase_rag.core.embedding_generator import EmbeddingGenerator
from src.codebase_rag.utils.document import Document


async def simple_embedding_demo():
    """Demonstrate basic embedding functionality."""
    print("🚀 Simple Embedding Demo")
    print("=" * 40)
    
    # Sample code snippets
    code_snippets = [
        """
def authenticate_user(username, password):
    '''Authenticate user with username and password'''
    if not username or not password:
        raise ValueError("Username and password required")
    
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return user
    return None
""",
        """
def sort_array(arr):
    '''Sort an array in ascending order'''
    return sorted(arr)
""",
        """
import requests
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    '''API endpoint to get all users'''
    try:
        users = User.query.all()
        return jsonify([user.to_dict() for user in users])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
""",
        """
def connect_database():
    '''Connect to the database'''
    try:
        connection = sqlite3.connect('app.db')
        return connection
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise
""",
        """
def handle_error(error):
    '''Handle application errors'''
    logger.error(f"Error occurred: {error}")
    if isinstance(error, ValidationError):
        return {'error': 'Invalid input'}, 400
    elif isinstance(error, AuthenticationError):
        return {'error': 'Authentication failed'}, 401
    else:
        return {'error': 'Internal server error'}, 500
"""
    ]
    
    # Sample queries
    queries = [
        "How do I authenticate users?",
        "Find a function that sorts an array",
        "Show me API endpoint examples",
        "How to connect to database?",
        "Error handling examples"
    ]
    
    # Initialize embedding generator
    print("🔧 Initializing embedding generator...")
    embedding_generator = EmbeddingGenerator(
        model_name='flax-sentence-embeddings/st-codesearch-distilroberta-base',
        batch_size=4,
        show_progress_bar=True
    )
    
    await embedding_generator.initialize()
    
    # Create Document objects
    documents = []
    for i, code in enumerate(code_snippets):
        doc = Document(
            content=code.strip(),
            metadata={
                'source': f'sample_code_{i}',
                'file_name': f'sample_{i}.py',
                'file_type': 'python',
                'language': 'python',
                'chunk_index': i,
                'function_name': f'function_{i}'
            }
        )
        documents.append(doc)
    
    print(f"📝 Created {len(documents)} sample documents")
    
    # Generate embeddings for code snippets
    print("\n🔍 Generating embeddings for code snippets...")
    embeddings, metadata = await embedding_generator.embed_documents_batch(documents)
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"📏 Embedding dimension: {embeddings.shape[1]}")
    
    # Test queries
    print("\n🔍 Testing queries...")
    for query in queries:
        print(f"\n❓ Query: {query}")
        
        # Generate query embedding
        query_embedding = await embedding_generator.embed_query(query, "code_search")
        
        # Calculate similarities
        similarities = []
        for i, code_embedding in enumerate(embeddings):
            similarity = embedding_generator.calculate_similarity(query_embedding, code_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 2 matches
        print("   Top matches:")
        for rank, (idx, sim) in enumerate(similarities[:2], 1):
            code_preview = code_snippets[idx].strip().split('\n')[1][:50] + "..."
            print(f"   {rank}. Score: {sim:.4f} - {code_preview}")
    
    # Test different query types
    print("\n🧪 Testing different query types...")
    test_query = "authentication function"
    
    query_types = ["general", "code_search", "error_analysis", "documentation"]
    
    for query_type in query_types:
        query_embedding = await embedding_generator.embed_query(test_query, query_type)
        
        # Find best match
        similarities = []
        for i, code_embedding in enumerate(embeddings):
            similarity = embedding_generator.calculate_similarity(query_embedding, code_embedding)
            similarities.append((i, similarity))
        
        best_match = max(similarities, key=lambda x: x[1])
        print(f"   {query_type:15}: Score {best_match[1]:.4f} (snippet {best_match[0]})")
    
    # Show model information
    print("\n📊 Model Information:")
    model_info = embedding_generator.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Clean up
    embedding_generator.cleanup()
    
    print("\n✅ Demo completed successfully!")


async def batch_processing_demo():
    """Demonstrate batch processing capabilities."""
    print("\n🔄 Batch Processing Demo")
    print("=" * 40)
    
    # Create a larger set of documents
    large_documents = []
    for i in range(50):  # 50 documents
        content = f"""
def function_{i}():
    '''This is function number {i}'''
    result = process_data_{i}()
    return result

def process_data_{i}():
    '''Process data for function {i}'''
    return f"processed_data_{i}"
"""
        
        doc = Document(
            content=content.strip(),
            metadata={
                'source': f'batch_demo_{i}',
                'file_name': f'batch_file_{i}.py',
                'file_type': 'python',
                'language': 'python',
                'chunk_index': i,
                'function_name': f'function_{i}'
            }
        )
        large_documents.append(doc)
    
    print(f"📝 Created {len(large_documents)} documents for batch processing")
    
    # Initialize with smaller batch size
    embedding_generator = EmbeddingGenerator(
        model_name='flax-sentence-embeddings/st-codesearch-distilroberta-base',
        batch_size=8,  # Smaller batch size for demo
        show_progress_bar=True
    )
    
    await embedding_generator.initialize()
    
    # Process in batches
    print("🔍 Processing documents in batches...")
    embeddings, metadata = await embedding_generator.embed_documents_batch(large_documents)
    
    print(f"✅ Processed {len(embeddings)} documents")
    print(f"📏 Embedding shape: {embeddings.shape}")
    
    # Test a query
    query = "function that processes data"
    query_embedding = await embedding_generator.embed_query(query, "code_search")
    
    # Find similar documents
    similarities = []
    for i, embedding in enumerate(embeddings):
        similarity = embedding_generator.calculate_similarity(query_embedding, embedding)
        similarities.append((i, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🔍 Query: {query}")
    print("   Top 3 matches:")
    for rank, (idx, sim) in enumerate(similarities[:3], 1):
        function_name = large_documents[idx].metadata['function_name']
        print(f"   {rank}. Score: {sim:.4f} - {function_name}")
    
    # Clean up
    embedding_generator.cleanup()
    
    print("\n✅ Batch processing demo completed!")


async def main():
    """Main function."""
    try:
        await simple_embedding_demo()
        await batch_processing_demo()
        
        print("\n🎉 All demos completed successfully!")
        print("\n💡 Key takeaways:")
        print("   - Code-aware models understand both code and natural language")
        print("   - Batch processing handles large codebases efficiently")
        print("   - Query types can be used to improve search relevance")
        print("   - Memory management prevents OOM errors")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 