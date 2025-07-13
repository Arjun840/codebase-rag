"""Embedding management for text and code."""

import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from ..utils import Document

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_length: int = 512):
        """Initialize the embedding manager."""
        self.model_name = model_name
        self.max_length = max_length
        self.model: Optional[SentenceTransformer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
    
    async def initialize(self):
        """Initialize the embedding model."""
        if self.model is not None:
            return
        
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=str(self.device))
            self.model.max_seq_length = self.max_length
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """Embed a list of documents."""
        if self.model is None:
            await self.initialize()
        
        # Extract text content
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        if self.model is None:
            await self.initialize()
        
        # Generate embedding
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding[0]
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return await self.embed_query(text)
    
    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        if self.model is None:
            await self.initialize()
        
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self.model is None:
            # Default dimension for all-MiniLM-L6-v2
            return 384
        
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        normalized1 = embedding1 / norm1
        normalized2 = embedding2 / norm2
        
        # Calculate cosine similarity
        return float(np.dot(normalized1, normalized2))
    
    def batch_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate similarity between two batches of embeddings."""
        # Normalize embeddings
        norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        normalized1 = embeddings1 / (norms1 + 1e-8)
        normalized2 = embeddings2 / (norms2 + 1e-8)
        
        # Calculate cosine similarity
        return np.dot(normalized1, normalized2.T)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            # Clear model from memory
            del self.model
            self.model = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("Embedding manager cleanup completed") 