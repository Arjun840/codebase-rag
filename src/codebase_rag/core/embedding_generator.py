"""Advanced embedding generator with memory management and batch processing."""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Generator
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import gc
import time
from pathlib import Path
import pickle
import json

from ..utils import Document
from ..config import config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Advanced embedding generator with memory management and batch processing."""
    
    def __init__(
        self,
        model_name: str = None,
        batch_size: int = 16,
        max_sequence_length: int = 512,
        device: str = None,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True
    ):
        """Initialize the embedding generator."""
        self.model_name = model_name or config.embedding_model
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dimension: Optional[int] = None
        
        logger.info(f"Initializing EmbeddingGenerator with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
    
    async def initialize(self):
        """Initialize the embedding model."""
        if self.model is not None:
            return
        
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            start_time = time.time()
            
            # Load the model
            self.model = SentenceTransformer(
                self.model_name,
                device=str(self.device)
            )
            
            # Set model parameters
            self.model.max_seq_length = self.max_sequence_length
            
            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _prepare_texts_for_embedding(self, documents: List[Document]) -> List[str]:
        """Prepare texts for embedding with metadata context."""
        prepared_texts = []
        
        for doc in documents:
            # Create context-aware text representation
            context_parts = []
            
            # Add file type context
            if doc.metadata.get('file_type'):
                context_parts.append(f"[{doc.metadata['file_type'].upper()}]")
            
            # Add language context
            if doc.metadata.get('language'):
                context_parts.append(f"[{doc.metadata['language'].upper()}]")
            
            # Add function/class context if available
            if doc.metadata.get('function_name'):
                context_parts.append(f"Function: {doc.metadata['function_name']}")
            elif doc.metadata.get('class_name'):
                context_parts.append(f"Class: {doc.metadata['class_name']}")
            
            # Add the actual content
            context_parts.append(doc.content)
            
            # Combine with separator
            prepared_text = " ".join(context_parts)
            prepared_texts.append(prepared_text)
        
        return prepared_texts
    
    def _create_batches(
        self, 
        items: List[Any], 
        batch_size: int
    ) -> Generator[List[Any], None, None]:
        """Create batches from a list of items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    async def embed_documents_batch(
        self, 
        documents: List[Document],
        batch_size: int = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Embed documents in batches with memory management.
        
        Args:
            documents: List of Document objects to embed
            batch_size: Override default batch size
            
        Returns:
            Tuple of (embeddings, metadata_list)
        """
        if self.model is None:
            await self.initialize()
        
        batch_size = batch_size or self.batch_size
        total_docs = len(documents)
        
        logger.info(f"Embedding {total_docs} documents in batches of {batch_size}")
        
        # Prepare texts for embedding
        prepared_texts = self._prepare_texts_for_embedding(documents)
        
        # Initialize storage
        all_embeddings = []
        all_metadata = []
        
        # Process in batches
        num_batches = (total_docs + batch_size - 1) // batch_size
        
        with tqdm(
            total=num_batches,
            desc="Generating embeddings",
            disable=not self.show_progress_bar
        ) as pbar:
            
            for batch_idx, batch_docs in enumerate(self._create_batches(documents, batch_size)):
                batch_texts = prepared_texts[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                
                try:
                    # Generate embeddings for this batch
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize_embeddings
                    )
                    
                    # Store embeddings and metadata
                    all_embeddings.append(batch_embeddings)
                    
                    # Extract metadata for this batch
                    batch_metadata = []
                    for doc in batch_docs:
                        metadata = {
                            'source': doc.metadata.get('source', 'unknown'),
                            'file_name': doc.metadata.get('file_name', 'unknown'),
                            'file_type': doc.metadata.get('file_type', 'unknown'),
                            'language': doc.metadata.get('language', 'unknown'),
                            'chunk_index': doc.metadata.get('chunk_index', 0),
                            'line_start': doc.metadata.get('line_start', 0),
                            'line_end': doc.metadata.get('line_end', 0),
                            'function_name': doc.metadata.get('function_name'),
                            'class_name': doc.metadata.get('class_name'),
                            'content_length': len(doc.content),
                            'embedding_dimension': self.embedding_dimension
                        }
                        batch_metadata.append(metadata)
                    
                    all_metadata.extend(batch_metadata)
                    
                    # Memory management
                    if batch_idx % 10 == 0:  # Every 10 batches
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    # Continue with next batch
                    continue
        
        # Combine all embeddings
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
        else:
            final_embeddings = np.array([])
        
        logger.info(f"Successfully embedded {len(final_embeddings)} documents")
        logger.info(f"Embedding shape: {final_embeddings.shape}")
        
        return final_embeddings, all_metadata
    
    async def embed_single_document(self, document: Document) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed a single document."""
        if self.model is None:
            await self.initialize()
        
        # Prepare text
        prepared_text = self._prepare_texts_for_embedding([document])[0]
        
        # Generate embedding
        embedding = self.model.encode(
            [prepared_text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )[0]
        
        # Create metadata
        metadata = {
            'source': document.metadata.get('source', 'unknown'),
            'file_name': document.metadata.get('file_name', 'unknown'),
            'file_type': document.metadata.get('file_type', 'unknown'),
            'language': document.metadata.get('language', 'unknown'),
            'chunk_index': document.metadata.get('chunk_index', 0),
            'line_start': document.metadata.get('line_start', 0),
            'line_end': document.metadata.get('line_end', 0),
            'function_name': document.metadata.get('function_name'),
            'class_name': document.metadata.get('class_name'),
            'content_length': len(document.content),
            'embedding_dimension': self.embedding_dimension
        }
        
        return embedding, metadata
    
    async def embed_query(self, query: str, query_type: str = "general") -> np.ndarray:
        """Embed a query with type context."""
        if self.model is None:
            await self.initialize()
        
        # Add query type context
        if query_type == "code_search":
            enhanced_query = f"[CODE_SEARCH] {query}"
        elif query_type == "error_analysis":
            enhanced_query = f"[ERROR_ANALYSIS] {query}"
        elif query_type == "documentation":
            enhanced_query = f"[DOCUMENTATION] {query}"
        else:
            enhanced_query = f"[GENERAL] {query}"
        
        # Generate embedding
        embedding = self.model.encode(
            [enhanced_query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )[0]
        
        return embedding
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Ensure embeddings are normalized
        if not self.normalize_embeddings:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1 = embedding1 / norm1
            embedding2 = embedding2 / norm2
        
        # Calculate cosine similarity
        return float(np.dot(embedding1, embedding2))
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        output_path: Path
    ):
        """Save embeddings and metadata to disk."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_file = output_path / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        # Save metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save model info
        model_info = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'batch_size': self.batch_size,
            'max_sequence_length': self.max_sequence_length,
            'normalize_embeddings': self.normalize_embeddings,
            'total_documents': len(embeddings),
            'created_at': time.time()
        }
        
        info_file = output_path / "model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Saved embeddings to {output_path}")
        logger.info(f"Embeddings: {embeddings_file}")
        logger.info(f"Metadata: {metadata_file}")
        logger.info(f"Model info: {info_file}")
    
    def load_embeddings(self, input_path: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings and metadata from disk."""
        input_path = Path(input_path)
        
        # Load embeddings
        embeddings_file = input_path / "embeddings.npy"
        embeddings = np.load(embeddings_file)
        
        # Load metadata
        metadata_file = input_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load model info
        info_file = input_path / "model_info.json"
        with open(info_file, 'r') as f:
            model_info = json.load(f)
        
        logger.info(f"Loaded embeddings from {input_path}")
        logger.info(f"Model: {model_info['model_name']}")
        logger.info(f"Documents: {len(embeddings)}")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        
        return embeddings, metadata
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            return {'status': 'not_initialized'}
        
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'max_sequence_length': self.max_sequence_length,
            'normalize_embeddings': self.normalize_embeddings,
            'status': 'initialized'
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("EmbeddingGenerator resources cleaned up") 