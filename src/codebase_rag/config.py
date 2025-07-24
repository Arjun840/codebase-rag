"""Configuration management for the RAG system."""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """Configuration settings for the RAG system."""
    
    # Model Configuration
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "flax-sentence-embeddings/st-codesearch-distilroberta-base"),
        description="The embedding model to use"
    )
    generation_model: str = Field(
        default=os.getenv("GENERATION_MODEL", "codellama/CodeLlama-7b-hf"),
        description="The generation model to use"
    )
    max_sequence_length: int = Field(
        default=int(os.getenv("MAX_SEQUENCE_LENGTH", "512")),
        description="Maximum sequence length for models"
    )
    
    # Available Code-Aware Embedding Models
    code_embedding_models: List[str] = Field(
        default=[
            "flax-sentence-embeddings/st-codesearch-distilroberta-base",  # CodeSearch DistilRoBERTa (768d)
            "microsoft/codebert-base",  # CodeBERT (768d)
            "microsoft/graphcodebert-base",  # GraphCodeBERT (768d)
            "huggingface/CodeBERTa-small-v1",  # CodeBERTa (768d)
            "sentence-transformers/all-MiniLM-L6-v2",  # General purpose (384d)
            "sentence-transformers/all-mpnet-base-v2",  # General purpose (768d)
            "sentence-transformers/all-distilroberta-v1",  # General purpose (768d)
        ],
        description="Available embedding models (code-aware and general purpose)"
    )
    
    # Available Code-Savvy Generation Models
    code_generation_models: List[str] = Field(
        default=[
            "codellama/CodeLlama-7b-hf",  # Code Llama 7B (recommended for most systems)
            "codellama/CodeLlama-13b-hf",  # Code Llama 13B (high-end systems)
            "bigcode/starcoder2-7b",  # StarCoder2 7B (good code understanding)
            "bigcode/starcoder2-15b",  # StarCoder2 15B (excellent, requires good GPU)
            "microsoft/DialoGPT-medium",  # General conversational (current default)
            "microsoft/DialoGPT-large",  # Larger conversational model
            "openai/gpt-3.5-turbo",  # OpenAI API (excellent code understanding)
            "openai/gpt-4",  # OpenAI API (best code understanding)
        ],
        description="Available generation models (code-aware and general purpose)"
    )
    
    # Vector Database Configuration
    vector_db_type: str = Field(
        default=os.getenv("VECTOR_DB_TYPE", "chromadb"),
        description="Type of vector database to use"
    )
    vector_db_path: Path = Field(
        default=Path(os.getenv("VECTOR_DB_PATH", "./data/vector_db")),
        description="Path to vector database storage"
    )
    collection_name: str = Field(
        default=os.getenv("COLLECTION_NAME", "codebase_embeddings"),
        description="Name of the vector collection"
    )
    
    # Processing Configuration
    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "1000")),
        description="Size of text chunks for processing"
    )
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "200")),
        description="Overlap between chunks"
    )
    max_workers: int = Field(
        default=int(os.getenv("MAX_WORKERS", "4")),
        description="Maximum number of worker processes"
    )
    
    # Web App Configuration
    debug: bool = Field(
        default=os.getenv("DEBUG", "false").lower() == "true",
        description="Enable debug mode"
    )
    
    # API Keys
    hugging_face_api_key: Optional[str] = Field(
        default=os.getenv("HUGGING_FACE_API_KEY"),
        description="Hugging Face API key"
    )
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )
    
    # File Processing
    supported_extensions: List[str] = Field(
        default=[".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".md", ".txt", ".yml", ".yaml", ".json", ".xml", ".html", ".css", ".scss", ".sql"],
        description="Supported file extensions for processing"
    )
    ignore_patterns: List[str] = Field(
        default=["*.pyc", "__pycache__", ".git", "node_modules", ".env", "*.log", "*.tmp", "venv", ".venv", "env", ".pytest_cache", ".coverage", "dist", "build", "*.egg-info"],
        description="Patterns to ignore during file processing"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
config = Config() 