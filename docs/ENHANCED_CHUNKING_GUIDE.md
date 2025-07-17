# Enhanced Chunking and Metadata Guide

This guide documents the enhanced chunking strategies and metadata extraction features implemented in the RAG system for better document processing and retrieval.

## Overview

The enhanced chunking system provides:
- **Topic-focused chunking** (100-300 words per chunk)
- **Section-aware splitting** for documentation
- **Distinct issue chunking** for error logs
- **Rich metadata extraction** with context
- **Smart filtering** and categorization

## Key Features

### 1. Smart Text Splitting

#### WordCount-Based Chunking
- **Min chunk size**: 100 words
- **Max chunk size**: 300 words
- **Overlap**: 20 words
- **Focus**: Single topic per chunk

#### Content-Aware Splitting
- **Markdown**: Section and header boundaries
- **Documentation**: Paragraph and topic boundaries
- **Error logs**: Distinct error/issue boundaries
- **Q&A**: Question-answer pairs

### 2. Enhanced Metadata Extraction

#### File Context
```python
{
    "file_path": "/path/to/file.py",
    "file_name": "auth.py",
    "file_extension": ".py",
    "programming_language": "python",
    "repository_name": "my-project",
    "branch": "main"
}
```

#### Content Context
```python
{
    "content_type": "code",
    "word_count": 245,
    "functions": ["authenticate", "hash_password"],
    "classes": ["User", "AuthManager"],
    "imports": ["hashlib", "jwt"],
    "complexity": 8,
    "has_docstring": true
}
```

#### Section Context
```python
{
    "section_title": "Authentication Guide",
    "section_path": "API > Authentication > OAuth",
    "section_level": 2,
    "topics": ["authentication", "oauth", "security"]
}
```

### 3. Context Generation

The system generates context strings for better LLM understanding:

```
File: auth.py | Repository: my-project | Language: python | Functions: authenticate, hash_password | Classes: User
```

This context helps the LLM provide more accurate responses like:
> "According to `auth.py` in the `my-project` repository, the `authenticate` function..."

## Implementation Details

### SmartTextSplitter Class

```python
from codebase_rag.utils import SmartTextSplitter

splitter = SmartTextSplitter(
    min_chunk_words=100,
    max_chunk_words=300,
    overlap_words=20,
    preserve_structure=True
)

# Split markdown content
chunks = splitter.split_markdown(content, file_path)

# Split error logs
error_chunks = splitter.split_error_logs(error_log, file_path)

# Split Q&A content
qa_chunks = splitter.split_qa_content(qa_text, file_path)
```

### MetadataExtractor Class

```python
from codebase_rag.utils import MetadataExtractor

extractor = MetadataExtractor()

# Extract comprehensive metadata
metadata = extractor.extract_metadata(
    content=chunk_content,
    file_path="/path/to/file.py",
    chunk_info={
        "section_title": "Authentication",
        "chunk_type": "function",
        "complexity": 5
    }
)

# Generate context string
context = extractor.create_context_string(metadata)
```

### Enhanced Processors

#### DocumentProcessor
```python
from codebase_rag.processors import DocumentProcessor

processor = DocumentProcessor(use_smart_chunking=True)
documents = await processor.process_file(file_path)

# Each document now has rich metadata
for doc in documents:
    print(f"Section: {doc.metadata['section_title']}")
    print(f"Context: {doc.metadata['context']}")
    print(f"Tags: {doc.metadata['tags']}")
```

#### ErrorProcessor
```python
from codebase_rag.processors import ErrorProcessor

processor = ErrorProcessor(use_smart_chunking=True)
error_docs = await processor.process_error_log(error_log)

# Each error is now a separate chunk
for doc in error_docs:
    print(f"Error: {doc.metadata['chunk_title']}")
    print(f"Severity: {doc.metadata['severity']}")
```

## Chunking Strategies by Content Type

### 1. Markdown Documentation

**Strategy**: Section-aware chunking with header hierarchy

```markdown
# Main Title
Content for main section...

## Section 1
Content for section 1...

### Subsection 1.1
Content for subsection...
```

**Result**: 3 chunks with proper section paths:
- Chunk 1: `Main Title`
- Chunk 2: `Main Title > Section 1`
- Chunk 3: `Main Title > Section 1 > Subsection 1.1`

### 2. Error Logs

**Strategy**: Distinct error boundary detection

```
2023-12-01 10:30:15 ERROR: Authentication failed
Traceback (most recent call last):
  File "auth.py", line 45, in login
    user = authenticate(username, password)
AuthenticationError: Invalid credentials

2023-12-01 10:31:02 ERROR: Database connection timeout
Traceback (most recent call last):
  File "db.py", line 78, in connect
    conn = psycopg2.connect(...)
psycopg2.OperationalError: timeout expired
```

**Result**: 2 separate chunks, each containing a complete error context

### 3. Q&A Content

**Strategy**: Question-answer pair grouping

```
Q: How do I implement authentication?
A: You can use OAuth 2.0 or JWT tokens...

Q: What's the difference between authentication and authorization?
A: Authentication verifies identity, authorization controls access...
```

**Result**: 2 chunks, each containing a complete Q&A pair

### 4. API Documentation

**Strategy**: Endpoint-focused chunking

```markdown
### GET /users
Retrieve a list of users.

Parameters:
- limit: Maximum number of users
- offset: Number of users to skip

Response:
{
  "users": [...],
  "total": 100
}
```

**Result**: Single chunk containing complete endpoint documentation

## Metadata Tags and Categories

### Automatic Tagging

The system automatically generates tags for filtering:

```python
{
    "tags": [
        "lang:python",           # Programming language
        "type:code",             # Content type
        "ext:py",                # File extension
        "repo:my-project",       # Repository name
        "documented",            # Has docstring
        "tested",                # Has tests
        "complex"                # High complexity
    ]
}
```

### Categories

Content is automatically categorized:

```python
{
    "categories": [
        "code",                  # Primary category
        "python",                # Language
        "functions",             # Contains functions
        "authentication"         # Domain
    ]
}
```

## Filtering and Search

### By Content Type
```python
# Find all code chunks
code_chunks = [doc for doc in documents if doc.metadata['content_type'] == 'code']

# Find all documentation
docs = [doc for doc in documents if doc.metadata['content_type'] == 'documentation']
```

### By Programming Language
```python
# Find Python code
python_code = [doc for doc in documents if 'lang:python' in doc.metadata['tags']]

# Find JavaScript code
js_code = [doc for doc in documents if 'lang:javascript' in doc.metadata['tags']]
```

### By Repository
```python
# Find chunks from specific repository
repo_chunks = [doc for doc in documents if 'repo:my-project' in doc.metadata['tags']]
```

### By Quality Indicators
```python
# Find documented code
documented = [doc for doc in documents if 'documented' in doc.metadata['tags']]

# Find tested code
tested = [doc for doc in documents if 'tested' in doc.metadata['tags']]

# Find high confidence chunks
high_confidence = [doc for doc in documents if doc.metadata['confidence_score'] > 0.7]
```

## Context-Aware Responses

### Before Enhancement
```
Query: "How do I authenticate users?"
Response: "You can use the authenticate function to verify credentials..."
```

### After Enhancement
```
Query: "How do I authenticate users?"
Response: "According to auth.py in the my-project repository, the authenticate function in the AuthManager class provides user authentication using password hashing and JWT tokens..."
```

## Benefits

### 1. Better Retrieval Quality
- **Logical boundaries**: No mid-function splits
- **Complete context**: Functions with their docstrings
- **Semantic coherence**: Topic-focused chunks

### 2. Enhanced LLM Understanding
- **Rich metadata**: Programming language, complexity, etc.
- **Context strings**: Clear file and section context
- **Proper attribution**: "According to file.py..."

### 3. Improved Filtering
- **Multi-dimensional**: Language, type, repository, quality
- **Confidence scoring**: Prioritize high-quality chunks
- **Smart categorization**: Automatic content classification

### 4. Better Error Handling
- **Distinct issues**: Each error as separate chunk
- **Complete context**: Full stack traces preserved
- **Severity classification**: Prioritize critical errors

## Usage Examples

### Basic Usage
```python
from codebase_rag.processors import DocumentProcessor

processor = DocumentProcessor(use_smart_chunking=True)
documents = await processor.process_file("README.md")

for doc in documents:
    print(f"Section: {doc.metadata['section_title']}")
    print(f"Words: {doc.metadata['word_count']}")
    print(f"Context: {doc.metadata['context']}")
```

### Advanced Filtering
```python
# Find authentication-related code
auth_chunks = []
for doc in documents:
    if any(keyword in doc.content.lower() for keyword in ['auth', 'login', 'password']):
        if doc.metadata['content_type'] == 'code':
            auth_chunks.append(doc)

# Sort by confidence
auth_chunks.sort(key=lambda x: x.metadata['confidence_score'], reverse=True)
```

### Error Analysis
```python
from codebase_rag.processors import ErrorProcessor

processor = ErrorProcessor(use_smart_chunking=True)
error_docs = await processor.process_error_log(error_log)

# Group by severity
critical_errors = [doc for doc in error_docs if doc.metadata['severity'] == 'critical']
high_errors = [doc for doc in error_docs if doc.metadata['severity'] == 'high']
```

## Configuration

### Smart Text Splitter
```python
splitter = SmartTextSplitter(
    min_chunk_words=100,        # Minimum words per chunk
    max_chunk_words=300,        # Maximum words per chunk
    overlap_words=20,           # Overlap between chunks
    preserve_structure=True     # Maintain logical boundaries
)
```

### Document Processor
```python
processor = DocumentProcessor(
    chunk_size=1000,            # Legacy chunk size (if smart chunking disabled)
    chunk_overlap=200,          # Legacy overlap
    use_smart_chunking=True     # Enable smart chunking
)
```

### Error Processor
```python
processor = ErrorProcessor(
    chunk_size=1000,            # Legacy chunk size
    chunk_overlap=200,          # Legacy overlap
    use_smart_chunking=True     # Enable smart chunking
)
```

## Demo Scripts

### Enhanced Chunking Demo
```bash
cd scripts
python3 demo_enhanced_chunking.py
```

Shows:
- Smart markdown chunking
- Documentation processing
- Error log analysis
- Metadata extraction
- Context generation

### Collection Demo
```bash
cd scripts
python3 demo_collection.py
```

Shows:
- GitHub repository collection
- AST-based code chunking
- Documentation gathering
- Chunking strategy comparison

## Best Practices

### 1. Chunk Size Guidelines
- **Code**: 100-300 words (function-level)
- **Documentation**: 100-300 words (topic-level)
- **Error logs**: 50-200 words (issue-level)
- **Q&A**: Variable (question-answer pairs)

### 2. Metadata Utilization
- **Use context strings** for LLM prompts
- **Filter by confidence** for quality control
- **Leverage tags** for precise filtering
- **Consider complexity** for difficulty assessment

### 3. Error Handling
- **Enable smart chunking** for better error separation
- **Use severity filtering** for priority handling
- **Preserve stack traces** for debugging context

### 4. Performance Optimization
- **Batch processing** for large codebases
- **Confidence filtering** to reduce noise
- **Repository-specific** indexing for better organization

## Migration Guide

### From Basic to Enhanced Chunking

1. **Update processor initialization**:
```python
# Before
processor = DocumentProcessor()

# After
processor = DocumentProcessor(use_smart_chunking=True)
```

2. **Update metadata access**:
```python
# Before
file_path = doc.metadata['file_path']

# After
file_path = doc.metadata['file_path']
section = doc.metadata['section_title']
context = doc.metadata['context']
tags = doc.metadata['tags']
```

3. **Update filtering logic**:
```python
# Before
python_files = [doc for doc in docs if doc.metadata['file_path'].endswith('.py')]

# After
python_files = [doc for doc in docs if 'lang:python' in doc.metadata['tags']]
```

## Troubleshooting

### Common Issues

1. **Chunks too small/large**:
   - Adjust `min_chunk_words` and `max_chunk_words`
   - Check content structure (headers, paragraphs)

2. **Missing metadata**:
   - Ensure `use_smart_chunking=True`
   - Check file path and content type detection

3. **Poor context generation**:
   - Verify metadata extraction is working
   - Check file path resolution

4. **Filtering not working**:
   - Ensure tags are generated correctly
   - Check tag format (e.g., "lang:python")

### Debug Tips

1. **Enable logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check metadata**:
```python
print(doc.metadata)
```

3. **Verify chunk boundaries**:
```python
for i, doc in enumerate(documents):
    print(f"Chunk {i}: {doc.metadata['section_title']}")
    print(f"Words: {doc.metadata['word_count']}")
```

## Future Enhancements

### Planned Features
- **Multi-language AST** support for more languages
- **Semantic similarity** for chunk relevance
- **Dynamic chunk sizing** based on content complexity
- **Custom metadata extractors** for domain-specific content
- **Real-time chunking** for streaming content

### API Improvements
- **Async chunking** for better performance
- **Streaming results** for large documents
- **Progress tracking** for long operations
- **Memory optimization** for large codebases

---

This enhanced chunking system provides the foundation for more accurate and context-aware RAG responses, making it easier for developers to find relevant information within large codebases and documentation. 