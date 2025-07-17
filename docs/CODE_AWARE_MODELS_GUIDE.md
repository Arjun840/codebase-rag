# Code-Aware Embedding Models Guide

This guide explains the code-aware embedding models available in the RAG system and how to use them effectively for better code understanding and retrieval.

## 🎯 Overview

Code-aware embedding models are specifically designed to understand both programming languages and natural language, making them ideal for code search, documentation retrieval, and developer assistance tasks. Unlike general-purpose models, these models are trained on code repositories and understand programming concepts.

## 📋 Available Models

### 🏆 Recommended: CodeSearch DistilRoBERTa
- **Model ID**: `flax-sentence-embeddings/st-codesearch-distilroberta-base`
- **Dimensions**: 768
- **Training**: Trained on CodeSearchNet dataset
- **Best for**: Code search, finding functions by description, API discovery
- **Paper**: Based on DistilRoBERTa architecture

```python
# Example usage
from codebase_rag.core import RAGSystem

rag_system = RAGSystem(config_override={
    'embedding_model': 'flax-sentence-embeddings/st-codesearch-distilroberta-base'
})
```

**Strengths**:
- Specifically trained for code search tasks
- Excellent at mapping natural language queries to code
- Good balance of performance and speed
- Maps code and queries to same 768-dimensional space

**Use Cases**:
- "Find a function that sorts an array"
- "Show me authentication code"
- "How to connect to database?"

### 🔧 CodeBERT
- **Model ID**: `microsoft/codebert-base`
- **Dimensions**: 768
- **Training**: Pre-trained on code and documentation pairs
- **Best for**: Code-documentation alignment, mixed content search
- **Paper**: [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)

```python
rag_system = RAGSystem(config_override={
    'embedding_model': 'microsoft/codebert-base'
})
```

**Strengths**:
- Bi-modal training on code and documentation
- Good for tasks involving both code and text
- Robust performance across multiple programming languages
- MLM + RTD training objectives

**Use Cases**:
- Code summarization
- Documentation generation
- Code-text alignment tasks

### 📊 GraphCodeBERT
- **Model ID**: `microsoft/graphcodebert-base`
- **Dimensions**: 768
- **Training**: Considers code structure and data flow
- **Best for**: Complex code analysis, code clone detection
- **Paper**: [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://arxiv.org/abs/2009.08366)

```python
rag_system = RAGSystem(config_override={
    'embedding_model': 'microsoft/graphcodebert-base'
})
```

**Strengths**:
- Understands code structure and data flow
- Advanced semantic understanding of code
- Excellent for code clone detection
- Structure-aware pre-training tasks

**Use Cases**:
- Code clone detection
- Complex code analysis
- Understanding variable dependencies
- Structural code understanding

### ⚡ CodeBERTa
- **Model ID**: `huggingface/CodeBERTa-small-v1`
- **Dimensions**: 768
- **Training**: RoBERTa-like model trained on CodeSearchNet
- **Best for**: Lightweight code processing, code completion
- **Architecture**: 6 layers, 84M parameters

```python
rag_system = RAGSystem(config_override={
    'embedding_model': 'huggingface/CodeBERTa-small-v1'
})
```

**Strengths**:
- Smaller model size (84M parameters)
- Faster inference
- Good for code completion tasks
- Efficient byte-level BPE tokenizer

**Use Cases**:
- Code completion
- Lightweight code analysis
- Resource-constrained environments

## 🚀 Getting Started

### 1. Configuration Methods

#### Environment Variable
```bash
export EMBEDDING_MODEL=flax-sentence-embeddings/st-codesearch-distilroberta-base
```

#### .env File
```env
EMBEDDING_MODEL=flax-sentence-embeddings/st-codesearch-distilroberta-base
```

#### Python Code
```python
from codebase_rag.core import RAGSystem

# Initialize with CodeSearch DistilRoBERTa
rag_system = RAGSystem(config_override={
    'embedding_model': 'flax-sentence-embeddings/st-codesearch-distilroberta-base'
})

await rag_system.initialize()
```

#### CLI Usage
```bash
codebase-rag --embedding-model flax-sentence-embeddings/st-codesearch-distilroberta-base index /path/to/code
```

#### Web Interface
1. Start the web app: `codebase-rag web`
2. Select the model from the sidebar dropdown
3. Click "Initialize RAG System"

### 2. Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Code Search | CodeSearch DistilRoBERTa | Specifically trained for this task |
| Documentation | CodeBERT | Good with code-text pairs |
| Code Analysis | GraphCodeBERT | Understands code structure |
| Fast Processing | CodeBERTa | Lightweight and fast |
| General Purpose | MiniLM | Good fallback option |

### 3. Performance Comparison

| Model | Dimensions | Speed | Code Understanding | Documentation | Memory Usage |
|-------|------------|-------|-------------------|---------------|-------------|
| CodeSearch DistilRoBERTa | 768 | Fast | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |
| CodeBERT | 768 | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium |
| GraphCodeBERT | 768 | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Medium |
| CodeBERTa | 768 | Fast | ⭐⭐⭐ | ⭐⭐⭐ | Low |
| MiniLM (General) | 384 | Very Fast | ⭐⭐ | ⭐⭐⭐ | Low |

## 🎯 Best Practices

### 1. Model Selection
- **Start with CodeSearch DistilRoBERTa** for most code search tasks
- **Use CodeBERT** when you need balanced code and documentation understanding
- **Choose GraphCodeBERT** for complex code analysis requiring structural understanding
- **Consider CodeBERTa** for resource-constrained environments

### 2. Query Optimization
- Use descriptive natural language queries
- Include programming language context when helpful
- Be specific about what you're looking for

**Good queries**:
- "Find a Python function that authenticates users"
- "Show me error handling code in JavaScript"
- "How to connect to PostgreSQL database?"

**Less effective queries**:
- "auth"
- "function"
- "code"

### 3. Performance Tips
- **Batch processing**: Process multiple queries together for better throughput
- **Caching**: Enable result caching for repeated queries
- **Indexing**: Use appropriate chunk sizes for your codebase
- **Hardware**: Use GPU when available for faster inference

## 🔧 Advanced Usage

### Custom Model Configuration
```python
from codebase_rag.core.embeddings import EmbeddingManager

# Initialize with custom settings
embedding_manager = EmbeddingManager(
    model_name='flax-sentence-embeddings/st-codesearch-distilroberta-base',
    max_length=512
)

await embedding_manager.initialize()

# Embed code snippets
code_embeddings = await embedding_manager.embed_documents(documents)
```

### Comparing Multiple Models
```python
import asyncio
from codebase_rag.core.embeddings import EmbeddingManager

async def compare_models():
    models = [
        'flax-sentence-embeddings/st-codesearch-distilroberta-base',
        'microsoft/codebert-base',
        'microsoft/graphcodebert-base'
    ]
    
    query = "authentication function"
    
    for model_name in models:
        manager = EmbeddingManager(model_name=model_name)
        await manager.initialize()
        
        embedding = await manager.embed_query(query)
        print(f"{model_name}: {len(embedding)} dimensions")

asyncio.run(compare_models())
```

## 📊 Benchmarks

### CodeSearchNet Evaluation
Based on the CodeSearchNet benchmark:

| Model | Ruby | JavaScript | Go | Python | Java | PHP | Overall |
|-------|------|------------|----|---------|----- |-----|---------|
| CodeSearch DistilRoBERTa | 0.703 | 0.644 | 0.897 | 0.692 | 0.691 | 0.649 | 0.713 |
| CodeBERT | 0.679 | 0.620 | 0.882 | 0.672 | 0.676 | 0.628 | 0.693 |
| RoBERTa (General) | 0.587 | 0.517 | 0.850 | 0.587 | 0.599 | 0.560 | 0.617 |

*Higher scores indicate better performance (MRR metric)*

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Errors**
   - Ensure stable internet connection
   - Check Hugging Face API key if using private models
   - Verify model name is correct

2. **Memory Issues**
   - Reduce batch size
   - Use smaller models like CodeBERTa
   - Enable model caching

3. **Slow Performance**
   - Use GPU if available
   - Reduce sequence length
   - Consider batch processing

### Performance Optimization
```python
# Optimize for speed
config_override = {
    'embedding_model': 'huggingface/CodeBERTa-small-v1',  # Smaller model
    'max_sequence_length': 256,  # Shorter sequences
    'batch_size': 16  # Smaller batches
}

# Optimize for accuracy
config_override = {
    'embedding_model': 'microsoft/graphcodebert-base',  # Advanced model
    'max_sequence_length': 512,  # Longer sequences
    'batch_size': 8  # Larger batches
}
```

## 📚 References

- [CodeSearchNet Dataset](https://github.com/github/CodeSearchNet)
- [CodeBERT Paper](https://arxiv.org/abs/2002.08155)
- [GraphCodeBERT Paper](https://arxiv.org/abs/2009.08366)
- [SentenceTransformers Documentation](https://www.sbert.net/)

## 🤝 Contributing

To add new code-aware models:

1. Add model ID to `config.py`
2. Update web interface options
3. Add model to demo script
4. Update this documentation
5. Test model compatibility

## 📄 License

This guide is part of the Codebase RAG system. See project LICENSE for details. 