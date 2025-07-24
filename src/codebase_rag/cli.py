"""Command-line interface for the RAG system."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .core import RAGSystem
from .config import config
from .utils.logging_utils import setup_logging


def setup_cli_logging(verbose: bool = False):
    """Set up logging for CLI."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level, use_loguru=True)


async def initialize_system(
    db_type: Optional[str] = None,
    embedding_model: Optional[str] = None,
    generation_model: Optional[str] = None
) -> RAGSystem:
    """Initialize the RAG system with optional overrides."""
    config_override = {}
    
    if db_type:
        config_override['vector_db_type'] = db_type
    if embedding_model:
        config_override['embedding_model'] = embedding_model
    if generation_model:
        config_override['generation_model'] = generation_model
    
    rag_system = RAGSystem(config_override)
    await rag_system.initialize()
    
    return rag_system


async def index_command(args):
    """Handle the index command."""
    logger = logging.getLogger(__name__)
    
    codebase_path = Path(args.path)
    
    if not codebase_path.exists():
        logger.error(f"Path does not exist: {codebase_path}")
        return 1
    
    if not codebase_path.is_dir():
        logger.error(f"Path is not a directory: {codebase_path}")
        return 1
    
    try:
        logger.info(f"Indexing codebase: {codebase_path}")
        
        # Initialize RAG system
        rag_system = await initialize_system(
            db_type=args.db_type,
            embedding_model=args.embedding_model,
            generation_model=args.generation_model
        )
        
        # Index the codebase
        await rag_system.index_codebase(codebase_path, args.force)
        
        logger.info("Indexing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return 1


async def search_command(args):
    """Handle the search command."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize RAG system
        rag_system = await initialize_system(
            db_type=args.db_type,
            embedding_model=args.embedding_model,
            generation_model=args.generation_model
        )
        
        # Perform search
        result = await rag_system.ask(args.query, args.top_k)
        
        # Display results
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result['answer'])
        
        print("\n" + "="*60)
        print("SOURCES:")
        print("="*60)
        
        for i, source in enumerate(result['sources']):
            print(f"\n--- Source {i+1} (Score: {source.score:.4f}) ---")
            print(f"File: {source.source}")
            print(f"Type: {source.metadata.get('type', 'Unknown')}")
            print(f"Language: {source.metadata.get('language', 'Unknown')}")
            print(f"\nContent:\n{source.content}")
        
        print(f"\nMetadata: {result['metadata']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1


async def interactive_command(args):
    """Handle the interactive command."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize RAG system
        rag_system = await initialize_system(
            db_type=args.db_type,
            embedding_model=args.embedding_model,
            generation_model=args.generation_model
        )
        
        print("Welcome to Codebase RAG Interactive Mode!")
        print("Type 'quit' or 'exit' to exit.")
        print("Type 'help' for available commands.")
        print("-" * 50)
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help - Show this help message")
                    print("  quit/exit - Exit interactive mode")
                    print("  Any other text - Search query")
                    continue
                
                if not query:
                    continue
                
                print("Searching...")
                result = await rag_system.ask(query, args.top_k)
                
                print(f"\nAnswer: {result['answer']}")
                
                if args.show_sources:
                    print(f"\nSources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'][:3]):  # Show top 3
                        print(f"  {i+1}. {source.source} (Score: {source.score:.4f})")
                
            except KeyboardInterrupt:
                print("\nUse 'quit' or 'exit' to exit.")
                continue
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                continue
        
        return 0
        
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Codebase RAG: Retrieval-Augmented Generation for Code",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--db-type", choices=["chromadb", "faiss"], help="Vector database type")
    parser.add_argument("--embedding-model", help="Embedding model name")
    parser.add_argument("--generation-model", help="Generation model name")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a codebase")
    index_parser.add_argument("path", help="Path to the codebase to index")
    index_parser.add_argument("--force", action="store_true", help="Force reindexing")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the indexed codebase")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")
    interactive_parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    interactive_parser.add_argument("--show-sources", action="store_true", help="Show source information")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_cli_logging(args.verbose)
    
    # Handle commands
    if args.command == "index":
        return asyncio.run(index_command(args))
    elif args.command == "search":
        return asyncio.run(search_command(args))
    elif args.command == "interactive":
        return asyncio.run(interactive_command(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 