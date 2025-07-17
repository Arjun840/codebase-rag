"""Streamlit web interface for the RAG system."""

import streamlit as st
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from ..config import config
from ..core.rag_system import RAGSystem
from ..utils.logging_utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


class StreamlitApp:
    """Streamlit web application for the RAG system."""
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.rag_system = None
        self.setup_page()
    
    def setup_page(self):
        """Set up the Streamlit page configuration."""
        st.set_page_config(
            page_title="Codebase RAG System",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔍 Codebase RAG System")
        st.markdown("**Ask questions about your codebase using natural language**")
    
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
        
        # Embedding model selection with descriptions
        st.sidebar.markdown("**Embedding Model:**")
        
        # Model descriptions
        model_descriptions = {
            "flax-sentence-embeddings/st-codesearch-distilroberta-base": "🏆 CodeSearch DistilRoBERTa (768d) - Trained on CodeSearchNet for code search",
            "microsoft/codebert-base": "🔧 CodeBERT (768d) - Pre-trained on code & documentation",
            "microsoft/graphcodebert-base": "📊 GraphCodeBERT (768d) - Considers code structure and data flow",
            "huggingface/CodeBERTa-small-v1": "⚡ CodeBERTa (768d) - Lightweight code model",
            "sentence-transformers/all-MiniLM-L6-v2": "🔄 MiniLM (384d) - General purpose, fast",
            "sentence-transformers/all-mpnet-base-v2": "🎯 MPNet (768d) - General purpose, high quality",
            "sentence-transformers/all-distilroberta-v1": "🚀 DistilRoBERTa (768d) - General purpose, balanced"
        }
        
        # Find current model index
        current_model_index = 0
        if hasattr(config, 'embedding_model') and config.embedding_model in config.code_embedding_models:
            current_model_index = config.code_embedding_models.index(config.embedding_model)
        
        # Create selectbox options with descriptions
        model_options = [model_descriptions.get(model, model) for model in config.code_embedding_models]
        
        selected_model_desc = st.sidebar.selectbox(
            "Choose embedding model:",
            model_options,
            index=current_model_index,
            help="Code-aware models are recommended for better code understanding"
        )
        
        # Get the actual model name from the description
        embedding_model = config.code_embedding_models[model_options.index(selected_model_desc)]
        
        # Show model info
        if "codesearch" in embedding_model.lower():
            st.sidebar.info("💡 Recommended: This model is specifically trained for code search tasks")
        elif "codebert" in embedding_model.lower():
            st.sidebar.info("🔧 Code-aware: This model understands both code and documentation")
        elif "graphcodebert" in embedding_model.lower():
            st.sidebar.info("📊 Advanced: This model considers code structure and data flow")
        
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
        
        if st.sidebar.button("Index Codebase") and codebase_path:
            self.index_codebase(codebase_path, force_reindex)
        
        return {
            'db_type': db_type,
            'embedding_model': embedding_model,
            'generation_model': generation_model,
            'top_k': top_k,
            'threshold': threshold
        }
    
    def render_main_content(self, config_params):
        """Render the main content area."""
        # Query input
        st.subheader("Ask a Question")
        
        # Query type selection
        query_type = st.selectbox(
            "Query Type",
            ["General", "Code Search", "Error Analysis", "Documentation"],
            help="Select the type of query for better results"
        )
        
        # Query input
        query = st.text_area(
            "Your Question:",
            placeholder="e.g., 'How do I authenticate users?' or 'What does this error mean?'",
            height=100
        )
        
        # Search button
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        with col2:
            if st.button("🔄 Clear"):
                st.rerun()
        
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
                
                st.success(f"RAG system initialized successfully with {embedding_model}!")
                logger.info("RAG system initialized")
                
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {str(e)}")
            logger.error(f"Failed to initialize RAG system: {e}")
            st.exception(e)
    
    def index_codebase(self, codebase_path: str, force_reindex: bool = False):
        """Index a codebase."""
        if self.rag_system is None:
            st.error("Please initialize the RAG system first")
            return
        
        try:
            path = Path(codebase_path)
            if not path.exists():
                st.error(f"Path does not exist: {codebase_path}")
                return
            
            with st.spinner(f"Indexing codebase: {codebase_path}"):
                # Run async indexing
                asyncio.run(self.rag_system.index_codebase(path, force_reindex))
                
                st.success(f"Successfully indexed: {codebase_path}")
                logger.info(f"Indexed codebase: {codebase_path}")
                
        except Exception as e:
            st.error(f"Failed to index codebase: {str(e)}")
            logger.error(f"Failed to index codebase: {e}")
            st.exception(e)
    
    def execute_search(self, query: str, query_type: str):
        """Execute a search query."""
        if self.rag_system is None:
            st.error("Please initialize the RAG system first")
            return
        
        try:
            # Enhance query based on type
            enhanced_query = self.enhance_query(query, query_type)
            
            # Run async search
            results = asyncio.run(self.rag_system.search(enhanced_query))
            
            if not results:
                st.warning("No results found. Try a different query or check if the codebase is indexed.")
                return
            
            # Display results
            st.subheader(f"Search Results ({len(results)} found)")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i} - {result.source} (Score: {result.score:.3f})"):
                    st.code(result.content, language="python")
                    
                    # Show metadata
                    if result.metadata:
                        st.markdown("**Metadata:**")
                        for key, value in result.metadata.items():
                            st.text(f"{key}: {value}")
            
            # Generate answer
            with st.spinner("Generating answer..."):
                answer = asyncio.run(self.rag_system.generate_answer(enhanced_query, results))
                
                st.subheader("Generated Answer")
                st.markdown(answer)
                
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            logger.error(f"Search failed: {e}")
            st.exception(e)
    
    def enhance_query(self, query: str, query_type: str) -> str:
        """Enhance query based on type."""
        if query_type == "Code Search":
            return f"Find code that: {query}"
        elif query_type == "Error Analysis":
            return f"Analyze error: {query}"
        elif query_type == "Documentation":
            return f"Documentation about: {query}"
        else:
            return query
    
    def run(self):
        """Run the Streamlit app."""
        # Render sidebar
        config_params = self.render_sidebar()
        
        # Render main content
        self.render_main_content(config_params)
        
        # Footer
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit and RAG")


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main() 