#!/usr/bin/env python3
"""
Quick test script to verify code-aware embedding models are working.

This script tests if the models can be loaded and used for basic embedding tasks.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebase_rag.core.embeddings import EmbeddingManager


async def test_model(model_name: str, model_display_name: str) -> dict:
    """Test a single model."""
    print(f"\n🔍 Testing {model_display_name}...")
    
    try:
        start_time = time.time()
        
        # Initialize the embedding manager
        manager = EmbeddingManager(model_name=model_name)
        await manager.initialize()
        
        # Test with a simple query
        test_query = "find authentication function"
        query_embedding = await manager.embed_query(test_query)
        
        # Test with sample code
        test_code = """
def login(username, password):
    if authenticate(username, password):
        return create_session(username)
    return None
"""
        code_embedding = await manager.embed_query(test_code)
        
        # Calculate similarity
        similarity = manager.similarity(query_embedding, code_embedding)
        
        end_time = time.time()
        
        result = {
            'status': 'success',
            'embedding_dim': len(query_embedding),
            'similarity': similarity,
            'time_taken': end_time - start_time
        }
        
        print(f"  ✅ Success!")
        print(f"  📏 Embedding dimension: {result['embedding_dim']}")
        print(f"  🎯 Query-code similarity: {result['similarity']:.4f}")
        print(f"  ⏱️ Time taken: {result['time_taken']:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }


async def main():
    """Main function to test all models."""
    print("🚀 Testing Code-Aware Embedding Models")
    print("=" * 50)
    
    models_to_test = [
        ("flax-sentence-embeddings/st-codesearch-distilroberta-base", "CodeSearch DistilRoBERTa"),
        ("microsoft/codebert-base", "CodeBERT"),
        ("microsoft/graphcodebert-base", "GraphCodeBERT"),
        ("huggingface/CodeBERTa-small-v1", "CodeBERTa"),
        ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM (General)"),
    ]
    
    results = {}
    
    for model_id, display_name in models_to_test:
        results[display_name] = await test_model(model_id, display_name)
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"{'Model':<25} {'Status':<10} {'Dim':<6} {'Similarity':<12} {'Time(s)'}")
    print("-" * 70)
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            print(f"{model_name:<25} {'✅ OK':<10} {result['embedding_dim']:<6} {result['similarity']:<12.4f} {result['time_taken']:.2f}")
        else:
            print(f"{model_name:<25} {'❌ FAIL':<10} {'N/A':<6} {'N/A':<12} {'N/A'}")
    
    # Check if default model works
    print("\n🎯 Default Model Test")
    print("-" * 25)
    
    from src.codebase_rag.config import config
    default_model = config.embedding_model
    print(f"Default model: {default_model}")
    
    if default_model in [model_id for model_id, _ in models_to_test]:
        print("✅ Default model is included in tests above")
    else:
        print("🔍 Testing default model separately...")
        await test_model(default_model, "Default Model")
    
    print("\n✨ All tests completed!")
    print("\nIf you see ✅ for your chosen model, it's ready to use!")
    print("If you see ❌, check your internet connection and try again.")


if __name__ == "__main__":
    asyncio.run(main()) 