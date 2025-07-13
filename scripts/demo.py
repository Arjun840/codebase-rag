#!/usr/bin/env python3
"""
Demo script for the Codebase RAG system.
This script demonstrates the main features of the RAG system.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codebase_rag.core import RAGSystem
from codebase_rag.utils.logging_utils import setup_logging


async def main():
    """Run the demo."""
    print("🔍 Codebase RAG Demo")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize the RAG system
    print("🚀 Initializing RAG system...")
    rag_system = RAGSystem()
    await rag_system.initialize()
    
    # Demo with this project's codebase
    project_root = Path(__file__).parent.parent
    print(f"📁 Indexing demo codebase: {project_root}")
    
    try:
        await rag_system.index_codebase(project_root)
        print("✅ Indexing completed!")
        
        # Demo queries
        demo_queries = [
            "How does the RAG system work?",
            "What is the Document class used for?",
            "How do I configure the embedding model?",
            "What file processors are available?",
            "How to start the web interface?"
        ]
        
        print("\n🤖 Demo Queries:")
        print("-" * 30)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n[{i}] Query: {query}")
            print("🔍 Searching...")
            
            result = await rag_system.ask(query, top_k=3)
            
            print(f"💬 Answer: {result['answer'][:200]}...")
            print(f"📊 Sources: {result['metadata']['num_sources']}")
            print(f"🎯 Top Score: {result['metadata']['top_score']:.4f}")
            
            if result['sources']:
                print("📝 Top Source:")
                source = result['sources'][0]
                print(f"   File: {source.metadata.get('file_name', 'Unknown')}")
                print(f"   Language: {source.metadata.get('language', 'Unknown')}")
                print(f"   Type: {source.metadata.get('type', 'Unknown')}")
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Try the web interface: codebase-rag web")
        print("2. Index your own codebase: codebase-rag index /path/to/your/code")
        print("3. Use the CLI: codebase-rag interactive")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    
    finally:
        # Cleanup
        await rag_system.cleanup()
        print("🧹 Cleanup completed")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 