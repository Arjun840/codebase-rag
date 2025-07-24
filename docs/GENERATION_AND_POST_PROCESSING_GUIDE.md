# Generation and Post-Processing Guide

## Overview

This guide explains how the RAG system handles generation and post-processing using Ollama as the language model backend. The system is designed to provide high-quality, code-aware responses with proper formatting and references.

## Generation Pipeline

### 1. Prompt Construction

The system constructs prompts in the following format:

```
You are a helpful coding assistant that answers questions about code and programming.
Use the following context to answer the question. Include code examples when relevant.

Question:
[User's question]

Relevant context:
[Retrieved code snippets and documentation]

Answer:
```

### 2. Ollama API Integration

The system uses Ollama's local API with optimized parameters for code generation:

```python
payload = {
    "model": "codellama:7b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "num_predict": max_new_tokens,
        "temperature": 0.3,  # Lower temperature for focused code responses
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1  # Reduce repetition
    }
}
```

**Key Parameters:**
- **Temperature**: 0.3 (lower for more deterministic code responses)
- **Top-p**: 0.9 (nucleus sampling)
- **Top-k**: 40 (limit vocabulary diversity)
- **Repeat Penalty**: 1.1 (prevent repetitive outputs)

### 3. Enhanced Post-Processing

The system includes comprehensive post-processing to improve response quality:

#### A. Response Cleaning
- Removes common artifacts (`<|endoftext|>`, `</s>`)
- Stops at certain tokens that indicate end of response
- Handles short responses gracefully

#### B. File Reference Extraction
Automatically extracts file references from context:
```python
# Looks for patterns like:
# - `filename.py`
# - `path/to/file.py`
# - "In file `filename.py`"
```

#### C. Code Block Formatting
Automatically formats code that should be in code blocks:
```python
# Detects patterns like:
# - def function_name():
# - class ClassName:
# - import statements
# - return statements
```

#### D. Markdown Formatting
- Ensures proper line breaks
- Formats code blocks with syntax highlighting
- Cleans up extra whitespace

#### E. Reference Addition
Adds a references section to responses:
```
**References:**
- `models.py`
- `utils/sorting.py`
- `examples/sort_demo.py`
```

## Example Workflow

### Input
**User Question**: "How can I sort a list of custom objects by an attribute using this library?"

**Retrieved Context**:
```python
# models.py
class Item:
    def __init__(self, name, value):
        self.name = name
        self.value = value

# utils/sorting.py
def sort_items_by_value(items: List[Item]) -> List[Item]:
    """Sort a list of Item objects by their value attribute."""
    return sorted(items, key=lambda item: item.value)
```

### Generated Response
```
To sort a list of custom objects by an attribute using the `sorted` function from the `utils.sorting` module, you can use the `key` parameter to specify the attribute that you want to sort by. In this case, you would pass in the name of the attribute as a string, like this:

```python
sorted_items = sort_items_by_value(items, key='value')
```

This will sort the list of `Item` objects by their `value` attribute.

**References:**
- `models.py`
- `utils/sorting.py`
```

## Key Features

### 1. Code-Aware Responses
- Understands code structure and syntax
- Provides practical code examples
- References specific functions and classes

### 2. Error Debugging
- Identifies attribute errors and missing methods
- Suggests fixes based on codebase context
- Explains why errors occur

### 3. Code Generation
- Generates new functions based on existing patterns
- Maintains consistency with codebase style
- Includes proper type hints and documentation

### 4. Automatic Formatting
- Formats code blocks with syntax highlighting
- Adds file references automatically
- Ensures proper markdown structure

## Integration with RAG Pipeline

The generation component integrates seamlessly with the full RAG pipeline:

1. **Retrieval**: Vector search finds relevant code snippets
2. **Generation**: Ollama generates code-aware responses
3. **Post-Processing**: Automatic formatting and reference addition
4. **Output**: Clean, formatted response with citations

## Performance Optimizations

### 1. Temperature Tuning
- Lower temperature (0.3) for code generation
- Higher temperature available for creative tasks
- Custom temperature support for different use cases

### 2. Token Management
- Configurable max tokens (default: 200)
- Context truncation for long inputs
- Efficient prompt construction

### 3. Error Handling
- Graceful API error handling
- Fallback responses for failures
- Detailed logging for debugging

## Usage Examples

### Basic Generation
```python
generator = Generator(model_name="codellama/CodeLlama-7b-hf")
answer = await generator.generate_answer(question, context)
```

### Custom Parameters
```python
answer = await generator.generate_answer(
    question, 
    context, 
    max_new_tokens=300
)
```

### Error Debugging
```python
answer = await generator.generate_error_solution(
    error_message, 
    context
)
```

### Code Generation
```python
answer = await generator.generate_code_suggestion(
    description, 
    context
)
```

## Best Practices

1. **Context Quality**: Ensure retrieved context is relevant and complete
2. **Question Clarity**: Ask specific, well-formed questions
3. **Token Limits**: Balance response length with quality
4. **Error Handling**: Always handle potential API failures
5. **Testing**: Test with various question types and contexts

## Troubleshooting

### Common Issues

1. **Ollama Not Running**
   - Ensure Ollama service is started: `ollama serve`
   - Check if model is downloaded: `ollama list`

2. **Poor Response Quality**
   - Check context relevance
   - Adjust temperature settings
   - Verify model is code-aware

3. **Formatting Issues**
   - Check post-processing pipeline
   - Verify markdown formatting
   - Review file reference extraction

### Debug Mode
Enable detailed logging to troubleshoot issues:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

The enhanced generation and post-processing pipeline provides:
- High-quality, code-aware responses
- Automatic formatting and references
- Robust error handling
- Seamless RAG integration

This system is ready for production use with real codebases and provides excellent developer experience for code-related questions and tasks. 