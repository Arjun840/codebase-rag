#!/usr/bin/env python3
"""
Demo script showcasing code-aware embedding models.

This script demonstrates the different embedding models available in the RAG system,
with a focus on code-aware models that understand both code and natural language.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeAwareModelsDemo:
    """Demo class for testing code-aware embedding models."""
    
    def __init__(self):
        """Initialize the demo."""
        self.models = {
            # Code-aware models
            "CodeSearch DistilRoBERTa": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
            "CodeBERT": "microsoft/codebert-base",
            "GraphCodeBERT": "microsoft/graphcodebert-base",
            "CodeBERTa": "huggingface/CodeBERTa-small-v1",
            
            # General purpose models for comparison
            "MiniLM (General)": "sentence-transformers/all-MiniLM-L6-v2",
            "MPNet (General)": "sentence-transformers/all-mpnet-base-v2",
            "DistilRoBERTa (General)": "sentence-transformers/all-distilroberta-v1",
        }
        
        self.test_queries = [
            "How do I authenticate users?",
            "Find a function that sorts an array",
            "What does this error mean: ImportError",
            "Show me examples of API endpoints",
            "How to implement a login system?",
            "Find database connection code",
            "What is the purpose of this function?",
            "Show me error handling examples"
        ]
        
        self.sample_code = [
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
    
    async def run_demo(self):
        """Run the complete demo."""
        print("🚀 Code-Aware Embedding Models Demo")
        print("=" * 50)
        
        # Demo 1: Model comparison
        await self.demo_model_comparison()
        
        # Demo 2: Code search performance
        await self.demo_code_search_performance()
        
        # Demo 3: Model recommendations
        await self.demo_model_recommendations()
        
        print("\n🎉 Demo completed!")
        print("Choose the model that best fits your use case:")
        print("- CodeSearch DistilRoBERTa: Best for code search tasks")
        print("- CodeBERT: Good balance of code and documentation understanding")
        print("- GraphCodeBERT: Advanced code structure understanding")
        print("- General models: Use when code understanding is not critical")
    
    async def demo_model_comparison(self):
        """Demo 1: Compare different models."""
        print("\n📊 Demo 1: Model Comparison")
        print("-" * 30)
        
        # Import here to avoid circular imports
        from codebase_rag.core.embeddings import EmbeddingManager
        
        results = {}
        
        for model_name, model_id in self.models.items():
            print(f"\n🔍 Testing {model_name}...")
            
            try:
                # Initialize embedding manager
                embedding_manager = EmbeddingManager(model_name=model_id)
                await embedding_manager.initialize()
                
                # Test embedding generation
                start_time = time.time()
                
                # Embed a sample query
                query = "Find authentication function"
                query_embedding = await embedding_manager.embed_query(query)
                
                # Embed sample code
                code_embeddings = await embedding_manager.embed_batch(self.sample_code[:2])
                
                end_time = time.time()
                
                results[model_name] = {
                    'model_id': model_id,
                    'embedding_dim': len(query_embedding),
                    'processing_time': end_time - start_time,
                    'status': 'success'
                }
                
                print(f"  ✅ Embedding dimension: {len(query_embedding)}")
                print(f"  ⏱️ Processing time: {end_time - start_time:.2f}s")
                
            except Exception as e:
                results[model_name] = {
                    'model_id': model_id,
                    'status': 'error',
                    'error': str(e)
                }
                print(f"  ❌ Error: {str(e)}")
        
        # Display comparison table
        print("\n📋 Model Comparison Summary:")
        print("=" * 70)
        print(f"{'Model':<25} {'Dimensions':<12} {'Time(s)':<10} {'Status'}")
        print("-" * 70)
        
        for model_name, info in results.items():
            if info['status'] == 'success':
                print(f"{model_name:<25} {info['embedding_dim']:<12} {info['processing_time']:<10.2f} ✅")
            else:
                print(f"{model_name:<25} {'N/A':<12} {'N/A':<10} ❌")
    
    async def demo_code_search_performance(self):
        """Demo 2: Code search performance with different models."""
        print("\n🔍 Demo 2: Code Search Performance")
        print("-" * 35)
        
        from codebase_rag.core.embeddings import EmbeddingManager
        import numpy as np
        
        # Test with code-aware models only
        code_aware_models = {
            "CodeSearch DistilRoBERTa": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
            "CodeBERT": "microsoft/codebert-base",
            "MiniLM (General)": "sentence-transformers/all-MiniLM-L6-v2",
        }
        
        test_query = "function that authenticates users"
        
        print(f"🔍 Query: '{test_query}'")
        print(f"📝 Searching through {len(self.sample_code)} code snippets...")
        
        for model_name, model_id in code_aware_models.items():
            print(f"\n🧪 Testing {model_name}...")
            
            try:
                # Initialize embedding manager
                embedding_manager = EmbeddingManager(model_name=model_id)
                await embedding_manager.initialize()
                
                # Embed query and code
                query_embedding = await embedding_manager.embed_query(test_query)
                code_embeddings = await embedding_manager.embed_batch(self.sample_code)
                
                # Calculate similarities
                similarities = []
                for i, code_embedding in enumerate(code_embeddings):
                    similarity = embedding_manager.similarity(query_embedding, code_embedding)
                    similarities.append((i, similarity))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                print(f"  📊 Top 3 matches:")
                for rank, (idx, sim) in enumerate(similarities[:3], 1):
                    code_preview = self.sample_code[idx].strip().split('\n')[1][:50] + "..."
                    print(f"    {rank}. Score: {sim:.3f} - {code_preview}")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
    
    async def demo_model_recommendations(self):
        """Demo 3: Model recommendations for different use cases."""
        print("\n💡 Demo 3: Model Recommendations")
        print("-" * 35)
        
        recommendations = {
            "Code Search & Retrieval": {
                "model": "CodeSearch DistilRoBERTa",
                "model_id": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
                "reason": "Specifically trained on CodeSearchNet for code search tasks",
                "use_cases": ["Finding functions by description", "Code snippet search", "API discovery"]
            },
            "Code & Documentation": {
                "model": "CodeBERT",
                "model_id": "microsoft/codebert-base",
                "reason": "Pre-trained on both code and documentation",
                "use_cases": ["Code-documentation alignment", "Code summarization", "Mixed content search"]
            },
            "Advanced Code Analysis": {
                "model": "GraphCodeBERT",
                "model_id": "microsoft/graphcodebert-base",
                "reason": "Considers code structure and data flow",
                "use_cases": ["Code clone detection", "Complex code analysis", "Structural code understanding"]
            },
            "General Purpose": {
                "model": "MiniLM",
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "reason": "Fast and efficient for general text",
                "use_cases": ["Documentation search", "General text retrieval", "Fast processing"]
            }
        }
        
        for use_case, info in recommendations.items():
            print(f"\n🎯 {use_case}:")
            print(f"  📦 Recommended Model: {info['model']}")
            print(f"  🔧 Model ID: {info['model_id']}")
            print(f"  💡 Reason: {info['reason']}")
            print(f"  📋 Use Cases: {', '.join(info['use_cases'])}")
    
    def show_configuration_examples(self):
        """Show configuration examples for different models."""
        print("\n⚙️ Configuration Examples")
        print("-" * 25)
        
        print("\n1. Environment Variable:")
        print("   export EMBEDDING_MODEL=flax-sentence-embeddings/st-codesearch-distilroberta-base")
        
        print("\n2. .env File:")
        print("   EMBEDDING_MODEL=flax-sentence-embeddings/st-codesearch-distilroberta-base")
        
        print("\n3. Python Code:")
        print("""
from codebase_rag.core import RAGSystem

# Initialize with CodeSearch DistilRoBERTa
rag_system = RAGSystem(config_override={
    'embedding_model': 'flax-sentence-embeddings/st-codesearch-distilroberta-base'
})

# Initialize with CodeBERT
rag_system = RAGSystem(config_override={
    'embedding_model': 'microsoft/codebert-base'
})
""")
        
        print("\n4. CLI Usage:")
        print("   codebase-rag --embedding-model flax-sentence-embeddings/st-codesearch-distilroberta-base")
        
        print("\n5. Web Interface:")
        print("   Select the model from the dropdown in the sidebar")


async def main():
    """Main function to run the demo."""
    demo = CodeAwareModelsDemo()
    
    try:
        await demo.run_demo()
        demo.show_configuration_examples()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 