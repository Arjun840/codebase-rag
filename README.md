# Codebase RAG: Retrieval-Augmented Generation for Developer Assistance

A powerful Retrieval-Augmented Generation (RAG) system that helps developers find answers within large codebases, documentation, and error logs. Ask natural language questions about your code and get intelligent responses powered by state-of-the-art AI models.

## 🚀 Features

- **Natural Language Queries**: Ask questions about your codebase in plain English
- **Multi-language Support**: Works with Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more
- **Intelligent Code Understanding**: Parses and understands code structure, functions, classes, and documentation
- **Error Analysis**: Help debug errors by analyzing error messages and tracebacks
- **Multiple Vector Stores**: Support for ChromaDB and FAISS for efficient similarity search
- **Web Interface**: Beautiful Streamlit web UI for easy interaction
- **CLI Interface**: Command-line tools for automation and scripting
- **Configurable Models**: Choose your preferred embedding and generation models

## 📋 Requirements

- Python 3.9 or higher
- 4GB+ RAM (8GB+ recommended for larger codebases)
- Optional: NVIDIA GPU for faster inference

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/codebase-rag.git
cd codebase-rag

# Install the package
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 🚀 Quick Start

### 1. Initialize the System

```bash
# Index your codebase
codebase-rag index /path/to/your/codebase

# Start the web interface
codebase-rag web
```

### 2. Ask Questions

Open your browser to `http://localhost:8501` and start asking questions:

- "How do I use the authentication system?"
- "What does this error mean: ImportError: No module named 'requests'?"
- "Show me examples of API endpoint definitions"
- "How to implement user registration?"

### 3. Command Line Usage

```bash
# Search from command line
codebase-rag search "How to create a new user?"

# Interactive mode
codebase-rag interactive
```

## 📖 Usage Guide

### Configuration

Create a `.env` file in your project root:

```env
# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
GENERATION_MODEL=microsoft/DialoGPT-medium
MAX_SEQUENCE_LENGTH=512

# Vector Database Configuration
VECTOR_DB_TYPE=chromadb
VECTOR_DB_PATH=./data/vector_db
COLLECTION_NAME=codebase_embeddings

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_WORKERS=4
```

### Python API

```python
import asyncio
from pathlib import Path
from codebase_rag.core import RAGSystem

async def main():
    # Initialize the RAG system
    rag_system = RAGSystem()
    await rag_system.initialize()
    
    # Index your codebase
    await rag_system.index_codebase(Path("/path/to/your/codebase"))
    
    # Ask a question
    result = await rag_system.ask("How do I authenticate users?")
    
    print("Answer:", result['answer'])
    print("Sources:", len(result['sources']))
    
    # Cleanup
    await rag_system.cleanup()

# Run the example
asyncio.run(main())
```

### Command Line Interface

```bash
# Index a codebase
codebase-rag index /path/to/codebase --force

# Search with specific parameters
codebase-rag search "authentication" --top-k 5 --db-type chromadb

# Interactive mode with custom settings
codebase-rag interactive --show-sources --top-k 10

# Start web interface on custom port
codebase-rag web --port 8502
```

## 🏗️ Architecture

```
codebase-rag/
├── src/codebase_rag/
│   ├── core/                 # Core RAG components
│   │   ├── rag_system.py     # Main orchestrator
│   │   ├── embeddings.py     # Embedding management
│   │   ├── vector_store.py   # Vector database interface
│   │   ├── retriever.py      # Document retrieval
│   │   └── generator.py      # Answer generation
│   ├── processors/           # File processors
│   │   ├── code_processor.py # Code file processing
│   │   ├── document_processor.py # Documentation processing
│   │   └── error_processor.py # Error log processing
│   ├── utils/               # Utilities
│   │   ├── document.py      # Document representation
│   │   ├── text_splitter.py # Text chunking
│   │   ├── file_utils.py    # File operations
│   │   └── logging_utils.py # Logging configuration
│   ├── web/                 # Web interface
│   │   └── streamlit_app.py # Streamlit application
│   ├── config.py            # Configuration management
│   └── cli.py               # Command-line interface
├── requirements.txt         # Dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

## 🔧 Configuration Options

### Vector Databases

**ChromaDB (Recommended)**
- Persistent storage
- Built-in filtering
- Easy setup

**FAISS**
- High performance
- Memory efficient
- Good for large datasets

### Embedding Models

- `all-MiniLM-L6-v2`: Fast, good quality (384 dimensions)
- `all-mpnet-base-v2`: Higher quality (768 dimensions)
- `all-MiniLM-L12-v2`: Balanced (384 dimensions)

### Generation Models

- `microsoft/DialoGPT-medium`: Balanced size/quality
- `microsoft/DialoGPT-small`: Faster inference
- `microsoft/DialoGPT-large`: Higher quality

## 🎯 Use Cases

### 1. Code Understanding
```
Query: "How does the user authentication work?"
Response: Based on the codebase, user authentication uses JWT tokens...
```

### 2. Error Debugging
```
Query: "What does this error mean: AttributeError: 'NoneType' object has no attribute 'get'"
Response: This error occurs when you try to call the 'get' method on a None object...
```

### 3. API Documentation
```
Query: "Show me how to create a new API endpoint"
Response: To create a new API endpoint, you need to define a route in the Flask app...
```

### 4. Code Examples
```
Query: "Give me examples of database queries in this codebase"
Response: Here are some database query examples from your codebase...
```

## 🚀 Advanced Features

### Custom Processors

```python
from codebase_rag.processors import CodeProcessor

class CustomProcessor(CodeProcessor):
    def process_file(self, file_path):
        # Custom processing logic
        pass
```

### Custom Embeddings

```python
from codebase_rag.core import EmbeddingManager

# Use custom embedding model
embedding_manager = EmbeddingManager(
    model_name="your-custom-model",
    max_length=512
)
```

### Filtering and Search

```python
# Search with filters
results = await rag_system.search(
    query="authentication",
    filters={"language": "python", "type": "code"}
)
```

## 🔍 Monitoring and Logging

The system provides comprehensive logging:

```python
from codebase_rag.utils.logging_utils import setup_logging

# Development logging
setup_logging(level="DEBUG", log_file="logs/debug.log")

# Production logging
setup_logging(level="INFO", log_file="logs/app.log")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/codebase-rag.git
cd codebase-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src/
flake8 src/
```

## 📊 Performance Tips

1. **Use ChromaDB for persistence**: Better for long-term storage
2. **Optimize chunk size**: Smaller chunks for specific queries, larger for context
3. **Use GPU acceleration**: Set `CUDA_VISIBLE_DEVICES` for faster inference
4. **Index incrementally**: Use `force_reindex=False` for updates
5. **Monitor memory usage**: Large codebases may require more RAM

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce chunk size
export CHUNK_SIZE=500

# Use smaller embedding model
export EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**2. Slow Indexing**
```bash
# Increase worker count
export MAX_WORKERS=8

# Use FAISS for large datasets
export VECTOR_DB_TYPE=faiss
```

**3. Poor Search Results**
```bash
# Lower similarity threshold
export SIMILARITY_THRESHOLD=0.3

# Increase chunk overlap
export CHUNK_OVERLAP=300
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface

## 🗺️ Roadmap

- [ ] Support for more programming languages
- [ ] Integration with popular IDEs
- [ ] Real-time codebase synchronization
- [ ] Advanced query understanding
- [ ] Multi-modal support (images, diagrams)
- [ ] Team collaboration features
- [ ] Cloud deployment options

## 📞 Support

- 📧 Email: [your-email@example.com](mailto:your-email@example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/codebase-rag/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/codebase-rag/discussions)

---

⭐ If you find this project helpful, please give it a star on GitHub!