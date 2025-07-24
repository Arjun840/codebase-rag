"""Optimized configuration for better RAG system performance."""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field


class OptimizedConfig(BaseModel):
    """Performance-optimized configuration settings for the RAG system."""
    
    # Optimized Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",  # Faster, smaller model (384d)
        description="Fast embedding model with good performance"
    )
    generation_model: str = Field(
        default="ollama/codellama:7b",  # Local Ollama model for speed
        description="Local generation model for faster response"
    )
    max_sequence_length: int = Field(
        default=256,  # Reduced for faster processing
        description="Reduced sequence length for speed"
    )
    
    # Performance-Optimized Retrieval Settings  
    default_top_k: int = Field(
        default=5,  # Reduced from default 10
        description="Number of documents to retrieve (reduced for speed)"
    )
    fast_top_k: int = Field(
        default=3,  # For ultra-fast responses
        description="Even smaller retrieval for fastest responses"
    )
    
    # Vector Database Configuration (optimized)
    vector_db_type: str = Field(
        default="chromadb",
        description="ChromaDB for good performance"
    )
    vector_db_path: Path = Field(
        default=Path("./data/vector_db"),
        description="Path to vector database storage"
    )
    collection_name: str = Field(
        default="fast_codebase_embeddings",
        description="Name of the optimized vector collection"
    )
    
    # Optimized Processing Configuration
    chunk_size: int = Field(
        default=800,  # Slightly smaller chunks
        description="Smaller chunks for faster processing"
    )
    chunk_overlap: int = Field(
        default=100,  # Reduced overlap
        description="Reduced overlap for speed"
    )
    max_workers: int = Field(
        default=6,  # Slightly increased for parallel processing
        description="More workers for parallel processing"
    )
    
    # Generation Optimization
    max_new_tokens: int = Field(
        default=150,  # Reduced from typical 200-300
        description="Fewer tokens for faster generation"
    )
    temperature: float = Field(
        default=0.1,  # Lower temperature for more deterministic, faster responses
        description="Lower temperature for faster, more focused responses"
    )
    
    # Performance Thresholds
    similarity_threshold: float = Field(
        default=0.3,  # Higher threshold to reduce irrelevant results
        description="Higher threshold for better quality results"
    )
    
    # Caching Configuration
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching for repeated queries"
    )
    cache_size: int = Field(
        default=100,
        description="Number of cached responses"
    )
    
    # File Processing (optimized)
    supported_extensions: List[str] = Field(
        default=[".py", ".js", ".ts", ".java", ".go", ".rs", ".md"],  # Reduced list
        description="Core file types for faster indexing"
    )
    
    # Timeout settings
    query_timeout: int = Field(
        default=30,  # 30 second timeout
        description="Query timeout in seconds"
    )
    embedding_timeout: int = Field(
        default=10,  # 10 second timeout for embeddings
        description="Embedding timeout in seconds"
    )
    generation_timeout: int = Field(
        default=20,  # 20 second timeout for generation
        description="Generation timeout in seconds"
    )


def get_fast_config() -> OptimizedConfig:
    """Get configuration optimized for speed."""
    return OptimizedConfig(
        default_top_k=3,
        max_new_tokens=100,
        chunk_size=600,
        temperature=0.05
    )


def get_balanced_config() -> OptimizedConfig:
    """Get configuration balancing speed and quality."""
    return OptimizedConfig(
        default_top_k=5,
        max_new_tokens=150,
        chunk_size=800,
        temperature=0.1
    )


def get_quality_config() -> OptimizedConfig:
    """Get configuration optimized for quality (slower)."""
    return OptimizedConfig(
        default_top_k=8,
        max_new_tokens=250,
        chunk_size=1000,
        temperature=0.2
    )


# Export optimized config instance
optimized_config = OptimizedConfig() 