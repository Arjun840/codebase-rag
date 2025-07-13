"""Streamlit web application for the RAG system."""

import streamlit as st
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import traceback

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from codebase_rag.core import RAGSystem
from codebase_rag.config import config
from codebase_rag.utils.logging_utils import setup_logging

# Setup logging
setup_logging(level="INFO", use_loguru=True)
logger = logging.getLogger(__name__)


class StreamlitApp:
    """Streamlit web application for the RAG system."""
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.rag_system = None
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Codebase RAG Assistant",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()
    
    def render_header(self):
        """Render the application header."""
        st.title("🔍 Codebase RAG Assistant")
        st.markdown("Ask questions about your codebase using natural language!")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.title("Configuration")
        
        # RAG System Status
        st.sidebar.subheader("System Status")
        if self.rag_system is None:
            st.sidebar.error("RAG system not initialized")
        else:
            st.sidebar.success("RAG system ready")
        
        # Configuration options
        st.sidebar.subheader("Settings")
        
        # Vector database selection
        db_type = st.sidebar.selectbox(
            "Vector Database",
            ["chromadb", "faiss"],
            index=0 if config.vector_db_type == "chromadb" else 1
        )
        
        # Embedding model selection
        embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L12-v2"],
            index=0
        )
        
        # Generation model selection
        generation_model = st.sidebar.selectbox(
            "Generation Model",
            ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-small", "microsoft/DialoGPT-large"],
            index=0
        )
        
        # Search parameters
        st.sidebar.subheader("Search Parameters")
        top_k = st.sidebar.slider("Number of results", 1, 20, 10)
        threshold = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.1)
        
        # Initialize system button
        if st.sidebar.button("Initialize RAG System"):
            self.initialize_rag_system(db_type, embedding_model, generation_model)
        
        # Codebase indexing
        st.sidebar.subheader("Codebase Indexing")
        codebase_path = st.sidebar.text_input(
            "Codebase Path",
            placeholder="Enter path to your codebase..."
        )
        
        force_reindex = st.sidebar.checkbox("Force reindex", value=False)
        
        if st.sidebar.button("Index Codebase"):
            if codebase_path and self.rag_system:
                self.index_codebase(codebase_path, force_reindex)
            else:
                st.sidebar.error("Please provide a codebase path and ensure RAG system is initialized")
        
        # Store parameters in session state
        st.session_state.update({
            'db_type': db_type,
            'embedding_model': embedding_model,
            'generation_model': generation_model,
            'top_k': top_k,
            'threshold': threshold
        })
    
    def render_main_content(self):
        """Render the main content area."""
        if self.rag_system is None:
            st.warning("Please initialize the RAG system first using the sidebar.")
            return
        
        # Query input
        st.subheader("Ask a Question")
        
        # Query examples
        with st.expander("Example Queries"):
            st.markdown("""
            - "How do I use the authentication system?"
            - "What does this error mean: ImportError: No module named 'requests'?"
            - "Show me examples of API endpoint definitions"
            - "How to implement user registration?"
            - "What is the database schema for users?"
            """)
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            placeholder="How do I implement user authentication?",
            height=100
        )
        
        # Query type selection
        query_type = st.selectbox(
            "Query Type",
            ["General", "Code Search", "Error Help", "Documentation"],
            index=0
        )
        
        # Search button
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("Search", type="primary")
        with col2:
            if st.button("Clear"):
                st.experimental_rerun()
        
        # Execute search
        if search_button and query:
            with st.spinner("Searching..."):
                self.execute_search(query, query_type)
    
    def initialize_rag_system(self, db_type: str, embedding_model: str, generation_model: str):
        """Initialize the RAG system with given parameters."""
        try:
            with st.spinner("Initializing RAG system..."):
                config_override = {
                    'vector_db_type': db_type,
                    'embedding_model': embedding_model,
                    'generation_model': generation_model
                }
                
                self.rag_system = RAGSystem(config_override)
                
                # Run async initialization
                asyncio.run(self.rag_system.initialize())
                
                st.success("RAG system initialized successfully!")
                logger.info("RAG system initialized")
                
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {str(e)}")
            logger.error(f"Failed to initialize RAG system: {e}")
            st.exception(e)
    
    def index_codebase(self, codebase_path: str, force_reindex: bool = False):
        """Index a codebase."""
        try:
            codebase_path = Path(codebase_path)
            
            if not codebase_path.exists():
                st.error(f"Path does not exist: {codebase_path}")
                return
            
            if not codebase_path.is_dir():
                st.error(f"Path is not a directory: {codebase_path}")
                return
            
            with st.spinner("Indexing codebase..."):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress (this is a simple simulation)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Indexing... {i + 1}%")
                
                # Run async indexing
                asyncio.run(self.rag_system.index_codebase(codebase_path, force_reindex))
                
                progress_bar.progress(100)
                status_text.text("Indexing complete!")
                st.success(f"Successfully indexed codebase at {codebase_path}")
                logger.info(f"Indexed codebase: {codebase_path}")
                
        except Exception as e:
            st.error(f"Failed to index codebase: {str(e)}")
            logger.error(f"Failed to index codebase: {e}")
            st.exception(e)
    
    def execute_search(self, query: str, query_type: str):
        """Execute a search query."""
        try:
            # Get parameters from session state
            top_k = st.session_state.get('top_k', 10)
            threshold = st.session_state.get('threshold', 0.0)
            
            # Execute search
            result = asyncio.run(self.rag_system.ask(query, top_k))
            
            # Display results
            self.display_search_results(result, query_type)
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            logger.error(f"Search failed: {e}")
            st.exception(e)
    
    def display_search_results(self, result: Dict[str, Any], query_type: str):
        """Display search results."""
        # Answer section
        st.subheader("Answer")
        
        if result['answer']:
            st.markdown(result['answer'])
        else:
            st.warning("No answer generated. Please try rephrasing your question.")
        
        # Metadata
        st.subheader("Search Metadata")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sources Found", result['metadata']['num_sources'])
        
        with col2:
            st.metric("Top Similarity Score", f"{result['metadata']['top_score']:.4f}")
        
        # Sources section
        st.subheader("Sources")
        
        if result['sources']:
            for i, source in enumerate(result['sources']):
                with st.expander(f"Source {i+1}: {source.source} (Score: {source.score:.4f})"):
                    st.code(source.content, language=source.metadata.get('language', 'text'))
                    
                    # Show metadata
                    st.markdown("**Metadata:**")
                    metadata_cols = st.columns(2)
                    
                    with metadata_cols[0]:
                        st.text(f"File: {source.metadata.get('file_name', 'Unknown')}")
                        st.text(f"Type: {source.metadata.get('type', 'Unknown')}")
                        st.text(f"Language: {source.metadata.get('language', 'Unknown')}")
                    
                    with metadata_cols[1]:
                        st.text(f"Size: {source.metadata.get('chunk_size', 'Unknown')} chars")
                        st.text(f"Chunk: {source.metadata.get('chunk_index', 'Unknown')}")
                        st.text(f"Line count: {source.metadata.get('line_count', 'Unknown')}")
        else:
            st.warning("No sources found. The codebase might not be indexed yet.")
    
    def render_footer(self):
        """Render the application footer."""
        st.markdown("---")
        st.markdown(
            "Built with ❤️ using Streamlit and Sentence Transformers | "
            "Powered by RAG (Retrieval-Augmented Generation)"
        )


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main() 