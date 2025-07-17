#!/usr/bin/env python3
"""
Codebase Embedding Script

This script demonstrates how to use the EmbeddingGenerator to process large codebases
with proper memory management, batch processing, and metadata tracking.

Usage:
    python scripts/embed_codebase.py --codebase-path /path/to/codebase --output-path ./embeddings
    python scripts/embed_codebase.py --codebase-path /path/to/codebase --model microsoft/codebert-base --batch-size 32
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
import json

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebase_rag.core.embedding_generator import EmbeddingGenerator
from src.codebase_rag.processors.code_processor import CodeProcessor
from src.codebase_rag.processors.document_processor import DocumentProcessor
from src.codebase_rag.utils.document import Document
from src.codebase_rag.config import config


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('embedding_process.log')
        ]
    )


async def collect_documents(codebase_path: Path) -> List[Document]:
    """Collect all documents from the codebase."""
    logger = logging.getLogger(__name__)
    
    documents = []
    
    # Initialize processors
    code_processor = CodeProcessor(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    document_processor = DocumentProcessor(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # Find all files
    code_files = []
    doc_files = []
    
    for ext in config.supported_extensions:
        if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.rb', '.php']:
            code_files.extend(codebase_path.rglob(f"*{ext}"))
        else:
            doc_files.extend(codebase_path.rglob(f"*{ext}"))
    
    logger.info(f"Found {len(code_files)} code files and {len(doc_files)} documentation files")
    
    # Process code files
    logger.info("Processing code files...")
    for file_path in code_files:
        try:
            # Skip files that should be ignored
            if any(pattern in str(file_path) for pattern in config.ignore_patterns):
                continue
            
            file_docs = await code_processor.process_file(file_path)
            documents.extend(file_docs)
            
        except Exception as e:
            logger.warning(f"Error processing code file {file_path}: {e}")
    
    # Process documentation files
    logger.info("Processing documentation files...")
    for file_path in doc_files:
        try:
            # Skip files that should be ignored
            if any(pattern in str(file_path) for pattern in config.ignore_patterns):
                continue
            
            file_docs = await document_processor.process_file(file_path)
            documents.extend(file_docs)
            
        except Exception as e:
            logger.warning(f"Error processing doc file {file_path}: {e}")
    
    logger.info(f"Total documents collected: {len(documents)}")
    return documents


async def embed_codebase(
    codebase_path: Path,
    output_path: Path,
    model_name: str = None,
    batch_size: int = 16,
    max_sequence_length: int = 512,
    device: str = None,
    save_embeddings: bool = True,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Embed a codebase using the specified model.
    
    Args:
        codebase_path: Path to the codebase
        output_path: Path to save embeddings
        model_name: Name of the embedding model
        batch_size: Batch size for processing
        max_sequence_length: Maximum sequence length
        device: Device to use (cuda/cpu)
        save_embeddings: Whether to save embeddings to disk
        show_progress: Whether to show progress bars
        
    Returns:
        Dictionary with processing statistics
    """
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    # Step 1: Collect documents
    logger.info(f"Collecting documents from {codebase_path}")
    documents = await collect_documents(codebase_path)
    
    if not documents:
        logger.warning("No documents found in the codebase")
        return {'status': 'no_documents'}
    
    # Step 2: Initialize embedding generator
    logger.info(f"Initializing embedding generator with model: {model_name}")
    embedding_generator = EmbeddingGenerator(
        model_name=model_name,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        device=device,
        show_progress_bar=show_progress
    )
    
    # Step 3: Generate embeddings
    logger.info("Generating embeddings...")
    embeddings, metadata = await embedding_generator.embed_documents_batch(documents)
    
    # Step 4: Save embeddings if requested
    if save_embeddings:
        logger.info(f"Saving embeddings to {output_path}")
        embedding_generator.save_embeddings(embeddings, metadata, output_path)
    
    # Step 5: Generate statistics
    processing_time = time.time() - start_time
    
    # Analyze metadata
    file_types = {}
    languages = {}
    total_content_length = 0
    
    for meta in metadata:
        file_type = meta.get('file_type', 'unknown')
        language = meta.get('language', 'unknown')
        content_length = meta.get('content_length', 0)
        
        file_types[file_type] = file_types.get(file_type, 0) + 1
        languages[language] = languages.get(language, 0) + 1
        total_content_length += content_length
    
    statistics = {
        'status': 'success',
        'total_documents': len(documents),
        'total_embeddings': len(embeddings),
        'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
        'processing_time_seconds': processing_time,
        'documents_per_second': len(documents) / processing_time if processing_time > 0 else 0,
        'file_types': file_types,
        'languages': languages,
        'total_content_length': total_content_length,
        'average_content_length': total_content_length / len(documents) if documents else 0,
        'model_info': embedding_generator.get_model_info(),
        'output_path': str(output_path) if save_embeddings else None
    }
    
    # Clean up
    embedding_generator.cleanup()
    
    return statistics


def print_statistics(stats: Dict[str, Any]):
    """Print processing statistics in a nice format."""
    print("\n" + "="*60)
    print("📊 EMBEDDING PROCESSING STATISTICS")
    print("="*60)
    
    if stats['status'] == 'no_documents':
        print("❌ No documents found in the codebase")
        return
    
    print(f"✅ Status: {stats['status']}")
    print(f"📄 Total Documents: {stats['total_documents']:,}")
    print(f"🔢 Embedding Dimension: {stats['embedding_dimension']}")
    print(f"⏱️ Processing Time: {stats['processing_time_seconds']:.2f} seconds")
    print(f"🚀 Speed: {stats['documents_per_second']:.2f} docs/second")
    print(f"📏 Total Content: {stats['total_content_length']:,} characters")
    print(f"📊 Average Content: {stats['average_content_length']:.0f} characters")
    
    print(f"\n🔧 Model Information:")
    model_info = stats['model_info']
    print(f"   Model: {model_info['model_name']}")
    print(f"   Device: {model_info['device']}")
    print(f"   Batch Size: {model_info['batch_size']}")
    print(f"   Max Sequence Length: {model_info['max_sequence_length']}")
    
    print(f"\n📁 File Types:")
    for file_type, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {file_type}: {count:,}")
    
    print(f"\n🌐 Languages:")
    for language, count in sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {language}: {count:,}")
    
    if stats['output_path']:
        print(f"\n💾 Output saved to: {stats['output_path']}")
    
    print("="*60)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Embed a codebase using code-aware models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default model
  python scripts/embed_codebase.py --codebase-path /path/to/codebase --output-path ./embeddings
  
  # Use CodeBERT with custom batch size
  python scripts/embed_codebase.py --codebase-path /path/to/codebase --model microsoft/codebert-base --batch-size 32
  
  # Use GPU with GraphCodeBERT
  python scripts/embed_codebase.py --codebase-path /path/to/codebase --model microsoft/graphcodebert-base --device cuda
  
  # Process without saving (for testing)
  python scripts/embed_codebase.py --codebase-path /path/to/codebase --no-save
        """
    )
    
    parser.add_argument(
        '--codebase-path',
        type=Path,
        required=True,
        help='Path to the codebase to embed'
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path('./embeddings'),
        help='Path to save embeddings (default: ./embeddings)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Embedding model to use (default: from config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )
    
    parser.add_argument(
        '--max-sequence-length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to use (default: auto-detect)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Process embeddings without saving to disk'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not args.codebase_path.exists():
        logger.error(f"Codebase path does not exist: {args.codebase_path}")
        return 1
    
    if not args.codebase_path.is_dir():
        logger.error(f"Codebase path is not a directory: {args.codebase_path}")
        return 1
    
    # Create output directory if saving
    if not args.no_save:
        args.output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Starting codebase embedding process")
        logger.info(f"Codebase: {args.codebase_path}")
        logger.info(f"Model: {args.model or 'default'}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Device: {args.device or 'auto'}")
        
        # Process the codebase
        statistics = await embed_codebase(
            codebase_path=args.codebase_path,
            output_path=args.output_path,
            model_name=args.model,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            device=args.device,
            save_embeddings=not args.no_save,
            show_progress=not args.no_progress
        )
        
        # Print results
        print_statistics(statistics)
        
        if statistics['status'] == 'success':
            logger.info("Codebase embedding completed successfully!")
            return 0
        else:
            logger.error("Codebase embedding failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 