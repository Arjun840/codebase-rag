# Code-Savvy LLM Selection Guide

This guide helps you choose the best code-aware language model for your RAG system based on your hardware capabilities and requirements.

## Quick Start

1. **Test your system and available models:**
   ```bash
   python scripts/test_generation_models.py
   ```

2. **Demo different models:**
   ```bash
   python scripts/demo_generation_models.py
   ```

3. **Set your chosen model in `.env`:**
   ```bash
   GENERATION_MODEL=your_chosen_model_name
   ```

## Model Categories

### 🏆 Code-Specific Models (Recommended)

These models are specifically trained for code understanding and generation:

#### Code Llama Models
- **`codellama/CodeLlama-7b-hf`** (Recommended for most systems)
  - Size: ~14 GB
  - Excellent code understanding
  - Good balance of performance and resource usage
  - Works well on mid-range systems

- **`codellama/CodeLlama-13b-hf`** (High-end systems)
  - Size: ~26 GB
  - Superior code understanding
  - Requires 16GB+ RAM and good GPU
  - Best for high-end workstations

#### StarCoder Models
- **`bigcode/starcoder2-7b`** (Good alternative to Code Llama 7B)
  - Size: ~14 GB
  - Strong code generation capabilities
  - Good for code completion and explanation

- **`bigcode/starcoder2-15b`** (High-end systems)
  - Size: ~30 GB
  - Excellent code understanding
  - Requires powerful GPU and 32GB+ RAM
  - Best performance for code tasks

### 🌐 API-Based Models (Convenience)

These models run on OpenAI's servers and require an API key:

#### OpenAI Models
- **`openai/gpt-3.5-turbo`** (Recommended API option)
  - Size: 0 GB (runs on OpenAI servers)
  - Excellent code understanding
  - Requires `OPENAI_API_KEY`
  - Pay-per-use pricing

- **`openai/gpt-4`** (Best API option)
  - Size: 0 GB (runs on OpenAI servers)
  - Superior code understanding and reasoning
  - Requires `OPENAI_API_KEY`
  - Higher cost but best performance

### 💬 General Conversational Models

These models are good for general conversation but less specialized for code:

- **`microsoft/DialoGPT-medium`** (Current default)
  - Size: ~0.8 GB
  - Good for general conversation
  - Limited code understanding
  - Works on any system

- **`microsoft/DialoGPT-large`**
  - Size: ~1.5 GB
  - Better conversation quality
  - Still limited for code tasks

## System Requirements

### High-End Systems (16GB+ RAM, Good GPU)
**Recommended:** `codellama/CodeLlama-13b-hf`, `bigcode/starcoder2-15b`
- GPU: 8GB+ VRAM
- RAM: 16GB+
- Storage: 50GB+ free space

### Mid-Range Systems (8-16GB RAM, Any GPU)
**Recommended:** `codellama/CodeLlama-7b-hf`, `bigcode/starcoder2-7b`
- GPU: 4GB+ VRAM (optional but recommended)
- RAM: 8GB+
- Storage: 30GB+ free space

### Lightweight Systems (4-8GB RAM)
**Recommended:** `microsoft/DialoGPT-medium`, API models
- RAM: 4GB+
- Storage: 5GB+ free space
- Internet connection (for API models)

### Very Lightweight Systems (<4GB RAM)
**Recommended:** API models only
- RAM: 2GB+
- Internet connection required
- API key needed

## Model Comparison

| Model | Code Understanding | Resource Usage | Speed | Cost | Best For |
|-------|-------------------|----------------|-------|------|----------|
| Code Llama 7B | ⭐⭐⭐⭐⭐ | Medium | Fast | Free | Most users |
| Code Llama 13B | ⭐⭐⭐⭐⭐ | High | Medium | Free | High-end systems |
| StarCoder2 7B | ⭐⭐⭐⭐ | Medium | Fast | Free | Code generation |
| StarCoder2 15B | ⭐⭐⭐⭐⭐ | Very High | Slow | Free | Best performance |
| GPT-3.5 Turbo | ⭐⭐⭐⭐ | None | Very Fast | $0.002/1K tokens | Convenience |
| GPT-4 | ⭐⭐⭐⭐⭐ | None | Fast | $0.03/1K tokens | Best quality |
| DialoGPT Medium | ⭐⭐ | Low | Very Fast | Free | General chat |

## Installation and Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Your System
```bash
python scripts/test_generation_models.py
```

### 3. Choose Your Model
Based on the test results, choose the best model for your system.

### 4. Configure Environment
Create a `.env` file based on `env.example`:
```bash
cp env.example .env
```

Edit `.env` and set your chosen model:
```bash
GENERATION_MODEL=codellama/CodeLlama-7b-hf
```

### 5. For API Models
If using OpenAI models, add your API key to `.env`:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage Examples

### Basic Usage
```python
from src.codebase_rag.core.generator import Generator

# Initialize with your chosen model
generator = Generator(model_name="codellama/CodeLlama-7b-hf")

# Generate code explanation
answer = await generator.generate_code_explanation(
    code_snippet="def quicksort(arr): ...",
    context="This is a sorting algorithm"
)
```

### With RAG System
```python
from src.codebase_rag.core.rag_system import RAGSystem

# Initialize RAG system with custom model
rag = RAGSystem({
    'generation_model': 'codellama/CodeLlama-7b-hf'
})

# Ask questions about your codebase
result = await rag.ask("How does the authentication system work?")
```

## Performance Tips

### For Local Models
1. **Use GPU acceleration** when available
2. **Close other applications** to free up RAM
3. **Use smaller models** if you experience memory issues
4. **Enable model quantization** for memory efficiency

### For API Models
1. **Cache responses** to reduce API calls
2. **Use appropriate token limits** to control costs
3. **Batch requests** when possible
4. **Monitor usage** to avoid unexpected charges

## Troubleshooting

### Common Issues

#### Out of Memory Errors
- **Solution:** Use a smaller model or API-based model
- **Alternative:** Reduce `max_sequence_length` in config

#### Slow Performance
- **Solution:** Use GPU acceleration if available
- **Alternative:** Switch to API-based model

#### Model Loading Failures
- **Solution:** Check internet connection for model download
- **Alternative:** Use a different model

#### API Errors
- **Solution:** Verify API key is set correctly
- **Alternative:** Check API quota and billing

### Getting Help

1. Run the test script to diagnose issues:
   ```bash
   python scripts/test_generation_models.py
   ```

2. Check the demo script for examples:
   ```bash
   python scripts/demo_generation_models.py
   ```

3. Review system requirements and choose appropriate model

## Best Practices

1. **Start with Code Llama 7B** - it's the best balance of performance and resource usage
2. **Test before committing** - always test models on your specific hardware
3. **Consider API models** for convenience and consistent performance
4. **Monitor resource usage** - especially for large models
5. **Keep models updated** - newer versions often have better performance

## Advanced Configuration

### Custom Model Parameters
You can customize model behavior in the generator:

```python
generator = Generator(
    model_name="codellama/CodeLlama-7b-hf",
    max_length=1024  # Increase for longer responses
)

# Generate with custom temperature
answer = await generator.generate_with_temperature(
    query="Explain this code",
    context=code_context,
    temperature=0.3  # Lower for more focused responses
)
```

### Model Quantization
For memory efficiency, you can use quantized models:

```python
# Use 4-bit quantized model (requires bitsandbytes)
generator = Generator(model_name="codellama/CodeLlama-7b-hf-4bit")
```

This guide should help you select and configure the best code-savvy LLM for your RAG system! 