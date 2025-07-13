"""Vector store implementations for document storage and retrieval."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
import pickle
import json

from ..utils import Document

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with their embeddings."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup resources."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store."""
    
    def __init__(self, db_path: Path, collection_name: str):
        """Initialize ChromaDB vector store."""
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
    async def initialize(self):
        """Initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create data directory
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized at {self.db_path}")
            
        except ImportError:
            logger.error("ChromaDB not installed. Please install with: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents to ChromaDB."""
        if self.collection is None:
            await self.initialize()
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents in ChromaDB."""
        if self.collection is None:
            await self.initialize()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Convert results to Document objects
        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i]
            )
            score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
            documents.append((doc, score))
        
        return documents
    
    async def cleanup(self):
        """Cleanup ChromaDB resources."""
        if self.client:
            # ChromaDB automatically persists data
            pass
        
        logger.info("ChromaDB cleanup completed")


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store."""
    
    def __init__(self, db_path: Path, collection_name: str):
        """Initialize FAISS vector store."""
        self.db_path = db_path
        self.collection_name = collection_name
        self.index = None
        self.documents = []
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        # File paths
        self.index_file = self.db_path / f"{collection_name}.index"
        self.docs_file = self.db_path / f"{collection_name}.docs"
        
    async def initialize(self):
        """Initialize FAISS index."""
        try:
            import faiss
            
            # Create data directory
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing index if available
            if self.index_file.exists() and self.docs_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                with open(self.docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {len(self.documents)} documents")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)
                self.documents = []
                logger.info("Created new FAISS index")
                
        except ImportError:
            logger.error("FAISS not installed. Please install with: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise
    
    async def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents to FAISS index."""
        if self.index is None:
            await self.initialize()
        
        # Update dimension if needed
        if embeddings.shape[1] != self.dimension:
            self.dimension = embeddings.shape[1]
            import faiss
            # Create new index with correct dimension
            new_index = faiss.IndexFlatIP(self.dimension)
            if self.index.ntotal > 0:
                # Transfer existing vectors
                existing_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                new_index.add(existing_vectors)
            self.index = new_index
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents.extend(documents)
        
        # Save index and documents
        await self._save_index()
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents in FAISS index."""
        if self.index is None:
            await self.initialize()
        
        if self.index.ntotal == 0:
            return []
        
        # Search
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Convert results to Document objects
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                results.append((doc, float(score)))
        
        return results
    
    async def _save_index(self):
        """Save FAISS index and documents to disk."""
        try:
            import faiss
            faiss.write_index(self.index, str(self.index_file))
            
            with open(self.docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
                
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    async def cleanup(self):
        """Cleanup FAISS resources."""
        await self._save_index()
        logger.info("FAISS cleanup completed")


class VectorStore:
    """Unified vector store interface."""
    
    def __init__(self, db_type: str, db_path: Path, collection_name: str):
        """Initialize vector store."""
        self.db_type = db_type.lower()
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # Create appropriate store
        if self.db_type == "chromadb":
            self.store = ChromaVectorStore(self.db_path, collection_name)
        elif self.db_type == "faiss":
            self.store = FAISSVectorStore(self.db_path, collection_name)
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")
        
        # Index tracking
        self.index_file = self.db_path / "indexed_codebases.json"
        self.indexed_codebases = self._load_indexed_codebases()
    
    async def initialize(self):
        """Initialize the vector store."""
        await self.store.initialize()
    
    async def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents to the vector store."""
        await self.store.add_documents(documents, embeddings)
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        return await self.store.search(query_embedding, top_k)
    
    def is_indexed(self, codebase_path: str) -> bool:
        """Check if a codebase is already indexed."""
        return codebase_path in self.indexed_codebases
    
    def mark_indexed(self, codebase_path: str):
        """Mark a codebase as indexed."""
        self.indexed_codebases.add(codebase_path)
        self._save_indexed_codebases()
    
    def _load_indexed_codebases(self) -> set:
        """Load indexed codebases from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load indexed codebases: {e}")
        return set()
    
    def _save_indexed_codebases(self):
        """Save indexed codebases to disk."""
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_file, 'w') as f:
                json.dump(list(self.indexed_codebases), f)
        except Exception as e:
            logger.error(f"Failed to save indexed codebases: {e}")
    
    async def cleanup(self):
        """Cleanup vector store resources."""
        await self.store.cleanup() 