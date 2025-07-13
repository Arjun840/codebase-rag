"""Document retriever for the RAG system."""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ..utils import Document
from .embeddings import EmbeddingManager
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant documents for a given query."""
    
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStore):
        """Initialize the retriever."""
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
    
    async def retrieve(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for a query."""
        # Embed the query
        query_embedding = await self.embedding_manager.embed_query(query)
        
        # Search in vector store
        results = await self.vector_store.search(query_embedding, top_k)
        
        # Filter by threshold
        filtered_results = []
        for doc, score in results:
            if score >= threshold:
                filtered_results.append((doc, score))
        
        return filtered_results
    
    async def retrieve_by_similarity(self, reference_text: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Retrieve documents similar to a reference text."""
        # This is essentially the same as retrieve but with different semantics
        return await self.retrieve(reference_text, top_k, threshold)
    
    async def retrieve_by_code(self, code_snippet: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Retrieve documents similar to a code snippet."""
        # Add code-specific context to the query
        enhanced_query = f"Code: {code_snippet}"
        return await self.retrieve(enhanced_query, top_k, threshold)
    
    async def retrieve_by_error(self, error_message: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Retrieve documents relevant to an error message."""
        # Add error-specific context to the query
        enhanced_query = f"Error: {error_message}"
        return await self.retrieve(enhanced_query, top_k, threshold)
    
    async def retrieve_multi_query(self, queries: List[str], top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Retrieve documents for multiple queries and combine results."""
        all_results = []
        seen_doc_ids = set()
        
        for query in queries:
            query_results = await self.retrieve(query, top_k, threshold)
            
            # Add unique results
            for doc, score in query_results:
                if doc.id not in seen_doc_ids:
                    all_results.append((doc, score))
                    seen_doc_ids.add(doc.id)
        
        # Sort by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return all_results[:top_k]
    
    async def retrieve_with_context(self, query: str, context: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Retrieve documents with additional context."""
        # Combine query and context
        enhanced_query = f"Context: {context}\nQuery: {query}"
        return await self.retrieve(enhanced_query, top_k, threshold)
    
    def rerank_results(self, results: List[Tuple[Document, float]], query: str) -> List[Tuple[Document, float]]:
        """Rerank results based on additional criteria."""
        # Simple reranking based on document metadata
        def rerank_score(doc_score_pair):
            doc, score = doc_score_pair
            
            # Boost code documents for code-related queries
            if any(keyword in query.lower() for keyword in ['function', 'class', 'method', 'code', 'implement']):
                if doc.metadata.get('type') == 'code':
                    score *= 1.2
            
            # Boost recent documents
            if 'timestamp' in doc.metadata:
                # This would require timestamp in metadata
                pass
            
            # Boost documents with exact keyword matches
            query_words = set(query.lower().split())
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                score *= (1 + overlap * 0.1)
            
            return score
        
        # Apply reranking
        reranked = [(doc, rerank_score((doc, score))) for doc, score in results]
        
        # Sort by new scores
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    async def retrieve_with_filters(self, query: str, filters: Dict[str, Any], top_k: int = 10, threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """Retrieve documents with metadata filters."""
        # Get all results first
        all_results = await self.retrieve(query, top_k * 2, threshold)  # Get more to filter
        
        # Apply filters
        filtered_results = []
        for doc, score in all_results:
            if self._matches_filters(doc, filters):
                filtered_results.append((doc, score))
        
        # Return top_k filtered results
        return filtered_results[:top_k]
    
    def _matches_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """Check if document matches the given filters."""
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            
            if isinstance(value, list):
                if doc.metadata[key] not in value:
                    return False
            else:
                if doc.metadata[key] != value:
                    return False
        
        return True
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get a specific document by ID."""
        # This would require the vector store to support ID-based retrieval
        # For now, we'll search with a dummy query and filter by ID
        try:
            # Search with empty query to get all documents (not efficient, but works for small datasets)
            results = await self.vector_store.search(np.zeros(384), 1000)  # Assuming 384-dim embeddings
            
            for doc, _ in results:
                if doc.id == doc_id:
                    return doc
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving document by ID {doc_id}: {e}")
            return None 