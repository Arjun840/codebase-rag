"""Main RAG system orchestrator."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..config import config
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator
from ..processors import CodeProcessor, DocumentProcessor

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a search query."""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str


class RAGSystem:
    """Main RAG system for codebase search and assistance."""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize the RAG system."""
        self.config = config
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.embedding_model,
            max_length=self.config.max_sequence_length
        )
        
        self.vector_store = VectorStore(
            db_type=self.config.vector_db_type,
            db_path=self.config.vector_db_path,
            collection_name=self.config.collection_name
        )
        
        self.retriever = Retriever(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store
        )
        
        self.generator = Generator(
            model_name=self.config.generation_model,
            max_length=self.config.max_sequence_length
        )
        
        # Initialize processors
        self.code_processor = CodeProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize all components."""
        if self._is_initialized:
            return
        
        logger.info("Initializing RAG system...")
        
        # Initialize embedding manager
        await self.embedding_manager.initialize()
        
        # Initialize vector store
        await self.vector_store.initialize()
        
        # Initialize generator
        await self.generator.initialize()
        
        self._is_initialized = True
        logger.info("RAG system initialized successfully")
    
    async def index_codebase(self, codebase_path: Path, force_reindex: bool = False):
        """Index a codebase for search."""
        if not self._is_initialized:
            await self.initialize()
        
        logger.info(f"Indexing codebase: {codebase_path}")
        
        # Check if already indexed
        if not force_reindex and self.vector_store.is_indexed(str(codebase_path)):
            logger.info("Codebase already indexed, skipping...")
            return
        
        # Process files
        documents = []
        
        # Process code files
        code_files = self._find_code_files(codebase_path)
        for file_path in code_files:
            try:
                file_documents = await self.code_processor.process_file(file_path)
                documents.extend(file_documents)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Process documentation files
        doc_files = self._find_doc_files(codebase_path)
        for file_path in doc_files:
            try:
                file_documents = await self.document_processor.process_file(file_path)
                documents.extend(file_documents)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Generate embeddings and store
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = await self.embedding_manager.embed_documents(documents)
        
        # Store in vector database
        await self.vector_store.add_documents(documents, embeddings)
        
        # Mark as indexed
        self.vector_store.mark_indexed(str(codebase_path))
        
        logger.info(f"Successfully indexed {len(documents)} documents")
    
    async def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for relevant documents."""
        if not self._is_initialized:
            await self.initialize()
        
        # Embed query
        query_embedding = await self.embedding_manager.embed_query(query)
        
        # Search in vector store
        results = await self.vector_store.search(query_embedding, top_k)
        
        # Convert to SearchResult objects
        search_results = []
        for doc, score in results:
            search_results.append(SearchResult(
                content=doc.content,
                metadata=doc.metadata,
                score=score,
                source=doc.metadata.get('source', 'unknown')
            ))
        
        return search_results
    
    async def generate_answer(self, query: str, context_documents: List[SearchResult]) -> str:
        """Generate an answer using retrieved context."""
        if not self._is_initialized:
            await self.initialize()
        
        # Prepare context
        context = self._prepare_context(context_documents)
        
        # Generate answer
        answer = await self.generator.generate_answer(query, context)
        
        return answer
    
    async def ask(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Ask a question and get a comprehensive answer."""
        if not self._is_initialized:
            await self.initialize()
        
        # Search for relevant documents
        search_results = await self.search(query, top_k)
        
        # Generate answer
        answer = await self.generate_answer(query, search_results)
        
        return {
            'query': query,
            'answer': answer,
            'sources': search_results,
            'metadata': {
                'num_sources': len(search_results),
                'top_score': search_results[0].score if search_results else 0.0
            }
        }
    
    def _find_code_files(self, path: Path) -> List[Path]:
        """Find code files in the given path."""
        code_files = []
        
        for ext in self.config.supported_extensions:
            if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.go', '.rs', '.rb', '.php']:
                code_files.extend(path.rglob(f'*{ext}'))
        
        # Filter out ignored patterns
        filtered_files = []
        for file_path in code_files:
            if not any(pattern in str(file_path) for pattern in self.config.ignore_patterns):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _find_doc_files(self, path: Path) -> List[Path]:
        """Find documentation files in the given path."""
        doc_files = []
        
        doc_extensions = ['.md', '.txt', '.yml', '.yaml', '.json', '.xml', '.html', '.css', '.scss']
        for ext in doc_extensions:
            if ext in self.config.supported_extensions:
                doc_files.extend(path.rglob(f'*{ext}'))
        
        # Filter out ignored patterns
        filtered_files = []
        for file_path in doc_files:
            if not any(pattern in str(file_path) for pattern in self.config.ignore_patterns):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _prepare_context(self, documents: List[SearchResult]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(documents):
            context_parts.append(f"[Source {i+1}] {doc.source}")
            context_parts.append(f"Score: {doc.score:.4f}")
            context_parts.append(f"Content: {doc.content}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'vector_store'):
            await self.vector_store.cleanup()
        
        if hasattr(self, 'generator'):
            await self.generator.cleanup()
        
        logger.info("RAG system cleanup completed") 