# Embedding Generation Guide

This guide explains how to use the `EmbeddingGenerator` class to create embeddings for your codebase with proper memory management and batch processing.

## 🎯 Overview

The `EmbeddingGenerator` is designed to handle large codebases efficiently by:
- **Batch Processing**: Process documents in configurable batches to manage memory
- **Memory Management**: Automatic garbage collection and GPU memory cleanup
- **Context-Aware Embeddings**: Add metadata context to improve search relevance
- **Query Type Support**: Different query types for specialized searches
- **Persistence**: Save and load embeddings for reuse

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from src.codebase_rag.core.embedding_generator import EmbeddingGenerator
from src.codebase_rag.utils.document import Document

async def basic_example():
    # Initialize the generator
    generator = EmbeddingGenerator(
        model_name='flax-sentence-embeddings/st-codesearch-distilroberta-base',
        batch_size=16
    )
    
    # Create sample documents
    documents = [
        Document(
            content="def hello_world(): return 'Hello, World!'",
            metadata={'file_name': 'main.py', 'language': 'python'}
        )
    ]
    
    # Generate embeddings
    embeddings, metadata = await generator.embed_documents_batch(documents)
    
    # Clean up
    generator.cleanup()
    
    return embeddings, metadata
```

### Command Line Usage

```bash
# Embed a codebase with default settings
python scripts/embed_codebase.py --codebase-path /path/to/codebase --output-path ./embeddings

# Use CodeBERT with custom batch size
python scripts/embed_codebase.py --codebase-path /path/to/codebase --model microsoft/codebert-base --batch-size 32

# Use GPU acceleration
python scripts/embed_codebase.py --codebase-path /path/to/codebase --device cuda
```

## 📋 Available Models

### Code-Aware Models (Recommended)

| Model | ID | Dimensions | Best For |
|-------|----|------------|----------|
| **CodeSearch DistilRoBERTa** | `flax-sentence-embeddings/st-codesearch-distilroberta-base` | 768 | Code search, function finding |
| **CodeBERT** | `microsoft/codebert-base` | 768 | Code and documentation understanding |
| **GraphCodeBERT** | `microsoft/graphcodebert-base` | 768 | Advanced code structure with data flow |
| **CodeBERTa** | `huggingface/CodeBERTa-small-v1` | 768 | Fast code understanding |

### General Purpose Models

| Model | ID | Dimensions | Use Case |
|-------|----|------------|----------|
| **all-MiniLM-L6-v2** | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast processing, limited memory |
| **all-mpnet-base-v2** | `sentence-transformers/all-mpnet-base-v2` | 768 | High quality, general purpose |
| **all-distilroberta-v1** | `sentence-transformers/all-distilroberta-v1` | 768 | Balanced performance |

## 🔧 Configuration Options

### EmbeddingGenerator Parameters

```python
generator = EmbeddingGenerator(
    model_name='flax-sentence-embeddings/st-codesearch-distilroberta-base',  # Model to use
    batch_size=16,                    # Documents per batch
    max_sequence_length=512,          # Max token length
    device='auto',                    # 'cpu', 'cuda', or 'auto'
    normalize_embeddings=True,        # Normalize vectors for cosine similarity
    show_progress_bar=True           # Show progress during processing
)
```

### Memory Management

The generator automatically handles memory management:

```python
# Automatic memory cleanup every 10 batches
if batch_idx % 10 == 0:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Manual cleanup
generator.cleanup()
```

## 📊 Batch Processing

### Large Codebase Processing

```python
async def process_large_codebase(codebase_path: Path):
    # Collect documents (using your existing processors)
    documents = await collect_documents_from_codebase(codebase_path)
    
    # Initialize generator with appropriate batch size
    generator = EmbeddingGenerator(
        batch_size=32,  # Adjust based on available memory
        show_progress_bar=True
    )
    
    # Process in batches
    embeddings, metadata = await generator.embed_documents_batch(documents)
    
    # Save results
    generator.save_embeddings(embeddings, metadata, Path('./embeddings'))
    
    return embeddings, metadata
```

### Batch Size Guidelines

| Memory Available | Recommended Batch Size | Use Case |
|------------------|----------------------|----------|
| 4GB RAM | 8-16 | Small codebases, testing |
| 8GB RAM | 16-32 | Medium codebases |
| 16GB+ RAM | 32-64 | Large codebases |
| GPU (8GB+ VRAM) | 64-128 | GPU acceleration |

## 🔍 Query Embedding

### Query Types

The generator supports different query types for specialized searches:

```python
# General query
query_embedding = await generator.embed_query("How do I authenticate users?")

# Code search query
query_embedding = await generator.embed_query("authentication function", "code_search")

# Error analysis query
query_embedding = await generator.embed_query("database connection error", "error_analysis")

# Documentation query
query_embedding = await generator.embed_query("API documentation", "documentation")
```

### Similarity Calculation

```python
# Calculate cosine similarity
similarity = generator.calculate_similarity(query_embedding, document_embedding)

# Find most similar documents
similarities = []
for i, doc_embedding in enumerate(document_embeddings):
    sim = generator.calculate_similarity(query_embedding, doc_embedding)
    similarities.append((i, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)
```

## 💾 Persistence

### Saving Embeddings

```python
# Save embeddings with metadata
generator.save_embeddings(
    embeddings=embeddings,
    metadata=metadata,
    output_path=Path('./embeddings')
)

# This creates:
# - embeddings.npy (numpy array)
# - metadata.json (document metadata)
# - model_info.json (model configuration)
```

### Loading Embeddings

```python
# Load previously saved embeddings
embeddings, metadata = generator.load_embeddings(Path('./embeddings'))

# The generator will automatically use the correct model
# based on the saved model_info.json
```

## 🎯 Context-Aware Embeddings

The generator automatically adds context to improve search relevance:

```python
# Input document
doc = Document(
    content="def authenticate_user(username, password): ...",
    metadata={
        'file_type': 'python',
        'language': 'python',
        'function_name': 'authenticate_user',
        'class_name': 'AuthManager'
    }
)

# Generated embedding includes context:
# "[PYTHON] [PYTHON] Function: authenticate_user def authenticate_user(username, password): ..."
```

## 📈 Performance Optimization

### GPU Acceleration

```python
# Use GPU if available
generator = EmbeddingGenerator(
    device='cuda',  # or 'auto' for auto-detection
    batch_size=64   # Larger batches on GPU
)
```

### Memory Optimization

```python
# For very large codebases
generator = EmbeddingGenerator(
    batch_size=8,   # Smaller batches
    max_sequence_length=256  # Shorter sequences
)

# Process in chunks
for chunk in document_chunks:
    embeddings, metadata = await generator.embed_documents_batch(chunk)
    # Save chunk embeddings
    generator.save_embeddings(embeddings, metadata, f'./embeddings/chunk_{i}')
```

### Progress Monitoring

```python
# Enable progress bars
generator = EmbeddingGenerator(show_progress_bar=True)

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## 🔧 Integration with RAG System

### Update Existing EmbeddingManager

```python
# In your existing EmbeddingManager
from .embedding_generator import EmbeddingGenerator

class EmbeddingManager:
    def __init__(self, model_name: str = None):
        self.generator = EmbeddingGenerator(model_name=model_name)
    
    async def embed_documents(self, documents: List[Document]):
        return await self.generator.embed_documents_batch(documents)
    
    async def embed_query(self, query: str, query_type: str = "general"):
        return await self.generator.embed_query(query, query_type)
```

## 🧪 Testing and Validation

### Run Demo Scripts

```bash
# Simple demo with sample code
python scripts/simple_embedding_demo.py

# Test different models
python scripts/test_models.py

# Process a real codebase
python scripts/embed_codebase.py --codebase-path ./your-project --output-path ./embeddings
```

### Validation Examples

```python
# Test query-document matching
query = "authentication function"
query_embedding = await generator.embed_query(query, "code_search")

# Find best matches
best_matches = find_similar_documents(query_embedding, document_embeddings, top_k=5)

# Verify results make sense
for match in best_matches:
    print(f"Score: {match['score']:.4f}")
    print(f"Content: {match['content'][:100]}...")
    print("---")
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```python
   # Reduce batch size
   generator = EmbeddingGenerator(batch_size=4)
   
   # Use CPU instead of GPU
   generator = EmbeddingGenerator(device='cpu')
   ```

2. **Model Download Issues**
   ```python
   # Check internet connection
   # Models are downloaded automatically on first use
   # Check ~/.cache/huggingface/hub/ for cached models
   ```

3. **Slow Processing**
   ```python
   # Increase batch size (if memory allows)
   generator = EmbeddingGenerator(batch_size=64)
   
   # Use GPU acceleration
   generator = EmbeddingGenerator(device='cuda')
   ```

### Performance Monitoring

```python
import time
import psutil

start_time = time.time()
start_memory = psutil.virtual_memory().used

# Process embeddings
embeddings, metadata = await generator.embed_documents_batch(documents)

end_time = time.time()
end_memory = psutil.virtual_memory().used

print(f"Processing time: {end_time - start_time:.2f}s")
print(f"Memory used: {(end_memory - start_memory) / 1024**3:.2f}GB")
print(f"Documents per second: {len(documents) / (end_time - start_time):.2f}")
```

## 📚 Advanced Usage

### Custom Text Preparation

```python
# Override text preparation for custom context
def custom_text_preparation(doc: Document) -> str:
    # Add custom context
    context = f"[{doc.metadata.get('project', 'unknown')}]"
    context += f"[{doc.metadata.get('module', 'unknown')}]"
    context += f" {doc.content}"
    return context

# Use in your own embedding pipeline
prepared_texts = [custom_text_preparation(doc) for doc in documents]
embeddings = model.encode(prepared_texts)
```

### Multi-Model Comparison

```python
models = [
    'flax-sentence-embeddings/st-codesearch-distilroberta-base',
    'microsoft/codebert-base',
    'microsoft/graphcodebert-base'
]

results = {}
for model_name in models:
    generator = EmbeddingGenerator(model_name=model_name)
    embeddings, metadata = await generator.embed_documents_batch(documents)
    
    # Test query performance
    query_embedding = await generator.embed_query("authentication", "code_search")
    similarities = calculate_similarities(query_embedding, embeddings)
    
    results[model_name] = {
        'embeddings': embeddings,
        'top_similarity': max(similarities)
    }
```

This guide covers all the essential aspects of using the `EmbeddingGenerator` for creating high-quality embeddings of your codebase. The system is designed to be both powerful and easy to use, with automatic memory management and batch processing to handle codebases of any size. 