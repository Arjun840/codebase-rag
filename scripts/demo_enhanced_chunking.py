#!/usr/bin/env python3
"""
Demo script showcasing enhanced chunking and metadata features.

This script demonstrates:
1. Smart text splitting for documentation (100-300 words per chunk)
2. Enhanced metadata extraction with rich context
3. Section-aware chunking for better topic focus
4. Error log processing with distinct issue chunking
5. Improved filtering and context generation
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import tempfile

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codebase_rag.processors import DocumentProcessor, ErrorProcessor
from codebase_rag.utils import SmartTextSplitter, MetadataExtractor, SmartChunk, ChunkFocus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedChunkingDemo:
    """Demonstrates enhanced chunking and metadata features."""
    
    def __init__(self):
        """Initialize the demo."""
        self.output_dir = Path("enhanced_chunking_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize processors with smart chunking enabled
        self.doc_processor = DocumentProcessor(use_smart_chunking=True)
        self.error_processor = ErrorProcessor(use_smart_chunking=True)
        
        # Initialize components directly
        self.smart_splitter = SmartTextSplitter(
            min_chunk_words=100,
            max_chunk_words=300,
            overlap_words=20,
            preserve_structure=True
        )
        
        self.metadata_extractor = MetadataExtractor()
    
    async def run_demo(self):
        """Run the complete demo."""
        print("🚀 Starting Enhanced Chunking and Metadata Demo")
        print("=" * 60)
        
        # Demo 1: Smart markdown chunking
        await self.demo_smart_markdown_chunking()
        
        # Demo 2: Documentation chunking with sections
        await self.demo_documentation_chunking()
        
        # Demo 3: Error log processing
        await self.demo_error_log_processing()
        
        # Demo 4: Enhanced metadata extraction
        await self.demo_metadata_extraction()
        
        # Demo 5: Context generation and filtering
        await self.demo_context_filtering()
        
        print("\n✅ Demo completed successfully!")
        print(f"📁 Output files saved to: {self.output_dir}")
    
    async def demo_smart_markdown_chunking(self):
        """Demonstrate smart markdown chunking with topic focus."""
        print("\n📝 Demo 1: Smart Markdown Chunking")
        print("-" * 40)
        
        # Create sample markdown documentation
        markdown_content = '''
# RAG System Documentation

This documentation covers the Retrieval-Augmented Generation system for developer assistance.

## Overview

The RAG system helps developers find relevant information within large codebases by combining vector search with language models. It provides natural language queries about code, documentation, and error logs.

### Key Features

- **Natural Language Queries**: Ask questions in plain English about your codebase
- **Multi-source Search**: Search across code, documentation, and error logs
- **Context-Aware Results**: Get results with proper context and metadata
- **Smart Chunking**: Content is split at logical boundaries for better relevance

## Architecture

### Core Components

The system consists of several key components:

#### Vector Store
The vector store manages document embeddings and similarity search. It supports multiple backends:

- **ChromaDB**: Persistent storage with good performance
- **FAISS**: Fast similarity search for large datasets
- **Pinecone**: Cloud-based vector database (optional)

#### Document Processors
Document processors handle different file types:

- **Code Processor**: Handles Python, JavaScript, Java, and more
- **Documentation Processor**: Processes Markdown, RST, and text files
- **Error Processor**: Analyzes error logs and stack traces

#### Retrieval System
The retrieval system finds relevant documents using:

1. **Semantic Search**: Vector similarity for conceptual matches
2. **Keyword Search**: Traditional text search for exact matches
3. **Hybrid Search**: Combines both approaches for best results

## Getting Started

### Installation

To install the RAG system, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-org/rag-system.git
cd rag-system

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
```

### Basic Usage

Here's how to use the system:

```python
from rag_system import RAGSystem

# Initialize the system
rag = RAGSystem()

# Index a codebase
await rag.index_codebase("/path/to/your/code")

# Ask questions
response = await rag.query("How do I implement authentication?")
print(response)
```

### Configuration

The system can be configured through environment variables:

- `EMBEDDING_MODEL`: The embedding model to use (default: all-MiniLM-L6-v2)
- `VECTOR_STORE`: Vector store backend (default: chromadb)
- `CHUNK_SIZE`: Maximum chunk size in characters (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## Advanced Features

### Custom Processors

You can create custom processors for specific file types:

```python
from rag_system.processors import BaseProcessor

class CustomProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.custom']
    
    async def process_file(self, file_path):
        # Your custom processing logic
        pass
```

### Query Optimization

To optimize queries:

1. **Use Specific Terms**: Include relevant keywords in your queries
2. **Add Context**: Mention the programming language or framework
3. **Filter Results**: Use metadata filters to narrow down results

### Performance Tuning

For better performance:

- **Batch Processing**: Process multiple files at once
- **Caching**: Enable result caching for repeated queries
- **Indexing**: Use appropriate indexing strategies

## API Reference

### RAGSystem Class

The main class for interacting with the system.

#### Methods

- `index_codebase(path)`: Index a codebase directory
- `query(question, filters=None)`: Query the system
- `add_document(doc)`: Add a single document
- `get_stats()`: Get system statistics

### Document Class

Represents a document in the system.

#### Properties

- `content`: The document content
- `metadata`: Document metadata
- `embedding`: Vector embedding (if available)

## Troubleshooting

### Common Issues

**Q: The system is slow to respond**
A: Check your vector store configuration and consider using FAISS for better performance.

**Q: Results are not relevant**
A: Try adjusting the chunk size or using more specific queries.

**Q: Out of memory errors**
A: Reduce the batch size or use a more efficient embedding model.

### Error Codes

- `E001`: Invalid configuration
- `E002`: Vector store connection failed
- `E003`: Document processing error
- `E004`: Query execution failed

## Contributing

We welcome contributions! Please read our contributing guidelines before submitting pull requests.

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Run tests before submitting

### Code Style

We follow PEP 8 for Python code. Please ensure your code passes linting checks.

## License

This project is licensed under the MIT License. See LICENSE file for details.
'''
        
        # Save sample markdown
        markdown_file = self.output_dir / "sample_documentation.md"
        with open(markdown_file, "w") as f:
            f.write(markdown_content)
        
        print(f"📄 Created sample documentation: {markdown_file}")
        
        # Process with smart chunking
        documents = await self.doc_processor._process_markdown(markdown_file)
        
        print(f"📊 Generated {len(documents)} smart chunks")
        
        # Analyze chunks
        chunk_analysis = []
        for i, doc in enumerate(documents):
            chunk_info = {
                "chunk_index": i,
                "section_title": doc.metadata.get('section_title'),
                "section_path": doc.metadata.get('section_path'),
                "word_count": doc.metadata.get('word_count'),
                "content_type": doc.metadata.get('content_type'),
                "topics": doc.metadata.get('topics', []),
                "confidence_score": doc.metadata.get('confidence_score'),
                "tags": doc.metadata.get('tags', []),
                "context": doc.metadata.get('context'),
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            }
            chunk_analysis.append(chunk_info)
            
            print(f"  {i+1}. {chunk_info['section_title'] or 'Untitled'}")
            print(f"     📊 {chunk_info['word_count']} words | Context: {chunk_info['context']}")
            print(f"     🏷️  Tags: {', '.join(chunk_info['tags'][:5])}")
        
        # Save analysis
        analysis_file = self.output_dir / "smart_markdown_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(chunk_analysis, f, indent=2)
        
        print(f"💾 Saved analysis to: {analysis_file}")
    
    async def demo_documentation_chunking(self):
        """Demonstrate documentation chunking with different content types."""
        print("\n📚 Demo 2: Documentation Chunking")
        print("-" * 35)
        
        # Create different types of documentation
        docs = {
            "api_documentation.md": '''
# API Documentation

## Authentication

### OAuth 2.0

Our API uses OAuth 2.0 for authentication. You need to obtain an access token before making requests.

To get an access token:

1. Register your application
2. Redirect users to the authorization URL
3. Exchange the authorization code for an access token
4. Use the access token in your API requests

### API Keys

For server-to-server communication, you can use API keys:

```bash
curl -H "Authorization: Bearer your-api-key" https://api.example.com/data
```

## Endpoints

### GET /users

Retrieve a list of users.

**Parameters:**
- `limit` (optional): Maximum number of users to return
- `offset` (optional): Number of users to skip

**Response:**
```json
{
  "users": [
    {
      "id": 1,
      "name": "John Doe",
      "email": "john@example.com"
    }
  ],
  "total": 100
}
```

### POST /users

Create a new user.

**Request Body:**
```json
{
  "name": "Jane Smith",
  "email": "jane@example.com"
}
```

**Response:**
```json
{
  "id": 2,
  "name": "Jane Smith",
  "email": "jane@example.com",
  "created_at": "2023-01-01T00:00:00Z"
}
```
''',
            "faq.md": '''
# Frequently Asked Questions

## General Questions

### What is this system?

This system is a Retrieval-Augmented Generation (RAG) platform designed to help developers find information within large codebases. It combines vector search with language models to provide intelligent answers to natural language queries.

### How does it work?

The system works by:
1. Indexing your codebase and documentation
2. Converting content into vector embeddings
3. Searching for relevant content based on your queries
4. Generating contextual answers using language models

### What file types are supported?

We support a wide range of file types including:
- Code files (Python, JavaScript, Java, C++, Go, Rust, etc.)
- Documentation (Markdown, RST, plain text)
- Configuration files (JSON, YAML, TOML)
- Log files and error reports

## Setup Questions

### How do I install the system?

Follow these steps:
1. Clone the repository
2. Install Python dependencies
3. Set up your environment variables
4. Run the setup script

### What are the system requirements?

Minimum requirements:
- Python 3.8 or higher
- 4GB RAM
- 2GB disk space
- Internet connection for downloading models

### Can I use custom models?

Yes! You can configure custom embedding models and language models through environment variables or configuration files.

## Usage Questions

### How do I query the system?

You can query the system using natural language:
- "How do I implement authentication?"
- "Show me examples of database queries"
- "What are the API endpoints?"

### How do I improve search results?

To get better results:
- Use specific technical terms
- Include context about the programming language
- Mention relevant frameworks or libraries
- Be specific about what you're looking for

### Can I filter results?

Yes, you can filter results by:
- File type or extension
- Programming language
- Repository or project
- Date modified
- Content type (code, documentation, etc.)

## Technical Questions

### How accurate are the results?

Accuracy depends on several factors:
- Quality of the indexed content
- Specificity of your queries
- Relevance of the training data
- Configuration of the embedding model

### How fast is the system?

Query response times typically range from 100ms to 2 seconds, depending on:
- Size of the indexed content
- Complexity of the query
- Hardware specifications
- Vector store configuration

### Can I use it offline?

Partial offline support is available:
- Local vector stores work offline
- Pre-downloaded models work offline
- Online features require internet connection

## Troubleshooting

### The system is slow

Try these solutions:
1. Check your vector store configuration
2. Reduce the chunk size
3. Use a faster embedding model
4. Increase hardware resources

### Results are not relevant

To improve relevance:
1. Adjust the chunk size and overlap
2. Update your embedding model
3. Refine your queries
4. Check the indexed content quality

### Out of memory errors

Reduce memory usage by:
1. Decreasing batch sizes
2. Using smaller embedding models
3. Implementing pagination
4. Cleaning up old indices
'''
        }
        
        # Process each document
        for filename, content in docs.items():
            print(f"\n📄 Processing: {filename}")
            
            # Save file
            file_path = self.output_dir / filename
            with open(file_path, "w") as f:
                f.write(content)
            
            # Process with smart chunking
            documents = await self.doc_processor._process_markdown(file_path)
            
            print(f"📊 Generated {len(documents)} chunks")
            
            # Show chunk summaries
            for i, doc in enumerate(documents[:3]):  # Show first 3 chunks
                section = doc.metadata.get('section_title', 'Untitled')
                words = doc.metadata.get('word_count', 0)
                focus = doc.metadata.get('chunk_type', 'unknown')
                
                print(f"  {i+1}. {section} ({words} words, {focus})")
                print(f"     Preview: {doc.content[:100]}...")
    
    async def demo_error_log_processing(self):
        """Demonstrate error log processing with distinct issue chunking."""
        print("\n🚨 Demo 3: Error Log Processing")
        print("-" * 33)
        
        # Create sample error log
        error_log = '''
2023-12-01 10:30:15 ERROR: Authentication failed for user john@example.com
Traceback (most recent call last):
  File "/app/auth/views.py", line 45, in login
    user = authenticate(username=username, password=password)
  File "/app/auth/backend.py", line 23, in authenticate
    raise AuthenticationError("Invalid credentials")
AuthenticationError: Invalid credentials

2023-12-01 10:31:02 ERROR: Database connection timeout
Traceback (most recent call last):
  File "/app/database/connection.py", line 78, in connect
    conn = psycopg2.connect(database=db_name, user=db_user, password=db_password, host=db_host)
  File "/usr/lib/python3.8/site-packages/psycopg2/__init__.py", line 127, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwds)
psycopg2.OperationalError: timeout expired

2023-12-01 10:32:18 CRITICAL: Payment processing failed
Traceback (most recent call last):
  File "/app/payments/processor.py", line 134, in process_payment
    response = stripe.Charge.create(amount=amount, currency=currency, source=token)
  File "/usr/lib/python3.8/site-packages/stripe/api_resources/charge.py", line 49, in create
    return cls._static_request('post', cls.class_url(), params)
  File "/usr/lib/python3.8/site-packages/stripe/api_requestor.py", line 153, in _static_request
    response, api_key = requestor.request(method, url, params, headers)
stripe.error.CardError: Your card was declined.

2023-12-01 10:33:45 ERROR: File not found
Traceback (most recent call last):
  File "/app/uploads/handler.py", line 67, in process_upload
    with open(file_path, 'rb') as f:
FileNotFoundError: [Errno 2] No such file or directory: '/uploads/user_123/document.pdf'

2023-12-01 10:34:12 ERROR: Invalid JSON in configuration file
Traceback (most recent call last):
  File "/app/config/loader.py", line 34, in load_config
    config = json.loads(config_content)
  File "/usr/lib/python3.8/json/decoder.py", line 357, in decode
    return _decode_strdict(s, idx, _w, _wh)
json.JSONDecodeError: Expecting property name in double quotes: line 15 column 3 (char 456)

2023-12-01 10:35:28 ERROR: Redis connection failed
Traceback (most recent call last):
  File "/app/cache/redis_client.py", line 45, in get_connection
    conn = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
  File "/usr/lib/python3.8/site-packages/redis/client.py", line 547, in __init__
    self.connection_pool = ConnectionPool(**kwargs)
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.
'''
        
        # Save error log
        error_file = self.output_dir / "sample_errors.log"
        with open(error_file, "w") as f:
            f.write(error_log)
        
        print(f"📄 Created sample error log: {error_file}")
        
        # Process with smart chunking
        documents = await self.error_processor.process_error_log(error_log, str(error_file))
        
        print(f"📊 Generated {len(documents)} error issue chunks")
        
        # Analyze error chunks
        for i, doc in enumerate(documents):
            error_title = doc.metadata.get('chunk_title', 'Unknown Error')
            error_type = doc.metadata.get('chunk_type', 'unknown')
            severity = doc.metadata.get('severity', 'unknown')
            language = doc.metadata.get('programming_language', 'unknown')
            
            print(f"  {i+1}. {error_title}")
            print(f"     🔴 Type: {error_type} | Severity: {severity} | Language: {language}")
            print(f"     📝 Preview: {doc.content[:100]}...")
    
    async def demo_metadata_extraction(self):
        """Demonstrate comprehensive metadata extraction."""
        print("\n🔍 Demo 4: Enhanced Metadata Extraction")
        print("-" * 40)
        
        # Create sample Python code
        python_code = '''
"""
User authentication module with OAuth 2.0 support.

This module provides functions and classes for handling user authentication
using OAuth 2.0 protocol with support for multiple providers.
"""

import hashlib
import jwt
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class User:
    """User data class."""
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

class AuthenticationError(Exception):
    """Custom authentication error."""
    pass

class OAuth2Provider:
    """OAuth 2.0 provider implementation."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize OAuth provider."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_url = "https://oauth.example.com/token"
        self.auth_url = "https://oauth.example.com/authorize"
    
    def get_auth_url(self, state: str) -> str:
        """Generate authorization URL."""
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'read write',
            'state': state
        }
        
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.auth_url}?{query_string}"
    
    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post(self.token_url, data=data)
        
        if response.status_code != 200:
            raise AuthenticationError(f"Token exchange failed: {response.text}")
        
        return response.json()

def hash_password(password: str, salt: str = None) -> tuple:
    """Hash a password with salt."""
    if salt is None:
        salt = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]
    
    # Use PBKDF2 for secure hashing
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return hashed.hex(), salt

def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """Verify a password against its hash."""
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == hashed_password

@jwt.jwt_required
def create_jwt_token(user_id: int, expires_in: int = 3600) -> str:
    """Create JWT token for user."""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(seconds=expires_in),
        'iat': datetime.utcnow()
    }
    
    return jwt.encode(payload, 'secret_key', algorithm='HS256')

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password."""
    # In a real app, this would query the database
    stored_password = "hashed_password_from_db"
    stored_salt = "salt_from_db"
    
    if verify_password(password, stored_password, stored_salt):
        return User(
            id=1,
            username=username,
            email=f"{username}@example.com",
            created_at=datetime.now()
        )
    
    return None

class AuthenticationManager:
    """Main authentication manager."""
    
    def __init__(self):
        """Initialize authentication manager."""
        self.oauth_providers = {}
        self.active_sessions = {}
    
    def add_oauth_provider(self, name: str, provider: OAuth2Provider):
        """Add OAuth provider."""
        self.oauth_providers[name] = provider
    
    def login(self, username: str, password: str) -> Optional[str]:
        """Login user and return JWT token."""
        user = authenticate_user(username, password)
        
        if user:
            token = create_jwt_token(user.id)
            self.active_sessions[token] = user
            return token
        
        raise AuthenticationError("Invalid credentials")
    
    def oauth_login(self, provider_name: str, code: str) -> Optional[str]:
        """OAuth login flow."""
        if provider_name not in self.oauth_providers:
            raise AuthenticationError(f"Unknown provider: {provider_name}")
        
        provider = self.oauth_providers[provider_name]
        
        try:
            token_data = provider.exchange_code_for_token(code)
            # Process token data and create user session
            # This is simplified - real implementation would be more complex
            
            user_id = token_data.get('user_id', 1)
            jwt_token = create_jwt_token(user_id)
            
            return jwt_token
            
        except Exception as e:
            raise AuthenticationError(f"OAuth login failed: {str(e)}")
    
    def logout(self, token: str):
        """Logout user."""
        if token in self.active_sessions:
            del self.active_sessions[token]
    
    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from JWT token."""
        return self.active_sessions.get(token)
'''
        
        # Save sample code
        code_file = self.output_dir / "auth_module.py"
        with open(code_file, "w") as f:
            f.write(python_code)
        
        print(f"📄 Created sample Python code: {code_file}")
        
        # Extract metadata
        metadata_context = self.metadata_extractor.extract_metadata(python_code, str(code_file))
        
        print(f"📊 Extracted comprehensive metadata:")
        print(f"  📝 File: {metadata_context.file_name}")
        print(f"  🐍 Language: {metadata_context.programming_language}")
        print(f"  📏 Size: {metadata_context.word_count} words, {metadata_context.line_count} lines")
        print(f"  🔧 Functions: {', '.join(metadata_context.functions[:5])}")
        print(f"  🏗️  Classes: {', '.join(metadata_context.classes)}")
        print(f"  📦 Imports: {', '.join(metadata_context.imports[:3])}")
        print(f"  🎯 Complexity: {metadata_context.complexity}")
        print(f"  🏷️  Tags: {', '.join(metadata_context.tags)}")
        print(f"  📂 Categories: {', '.join(metadata_context.categories)}")
        print(f"  🎯 Confidence: {metadata_context.confidence_score:.2f}")
        
        # Create context string
        context_string = self.metadata_extractor.create_context_string(metadata_context)
        print(f"  📋 Context: {context_string}")
        
        # Save metadata
        metadata_file = self.output_dir / "metadata_analysis.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata_context.to_dict(), f, indent=2)
        
        print(f"💾 Saved metadata analysis to: {metadata_file}")
    
    async def demo_context_filtering(self):
        """Demonstrate context generation and filtering capabilities."""
        print("\n🔍 Demo 5: Context Generation and Filtering")
        print("-" * 42)
        
        # Create sample documents with different contexts
        sample_docs = [
            {
                "content": "def authenticate_user(username, password):\n    return verify_credentials(username, password)",
                "file_path": "/app/auth/views.py",
                "language": "python",
                "functions": ["authenticate_user"],
                "complexity": 3
            },
            {
                "content": "# Authentication Guide\n\nThis guide explains how to implement authentication in your application.",
                "file_path": "/docs/auth_guide.md",
                "language": None,
                "section_title": "Authentication Guide",
                "topics": ["authentication", "security"]
            },
            {
                "content": "ERROR: Authentication failed for user john@example.com\nInvalid credentials provided",
                "file_path": "/logs/auth_errors.log",
                "language": None,
                "error_type": "authentication_error",
                "severity": "high"
            }
        ]
        
        print("📋 Processing sample documents with different contexts:")
        
        processed_docs = []
        for i, doc_data in enumerate(sample_docs):
            # Extract metadata
            metadata_context = self.metadata_extractor.extract_metadata(
                doc_data["content"], 
                doc_data["file_path"],
                chunk_info=doc_data
            )
            
            # Generate context string
            context_string = self.metadata_extractor.create_context_string(metadata_context)
            
            processed_doc = {
                "index": i,
                "content_preview": doc_data["content"][:100],
                "file_path": doc_data["file_path"],
                "content_type": metadata_context.content_type.value,
                "context": context_string,
                "tags": metadata_context.tags,
                "categories": metadata_context.categories,
                "confidence": metadata_context.confidence_score
            }
            
            processed_docs.append(processed_doc)
            
            print(f"\n  {i+1}. {Path(doc_data['file_path']).name}")
            print(f"     📄 Type: {processed_doc['content_type']}")
            print(f"     📋 Context: {processed_doc['context']}")
            print(f"     🏷️  Tags: {', '.join(processed_doc['tags'][:3])}")
            print(f"     🎯 Confidence: {processed_doc['confidence']:.2f}")
        
        # Demonstrate filtering
        print("\n🔍 Filtering Examples:")
        
        # Filter by language
        python_docs = [doc for doc in processed_docs if "lang:python" in doc["tags"]]
        print(f"  📝 Python documents: {len(python_docs)}")
        
        # Filter by content type
        code_docs = [doc for doc in processed_docs if doc["content_type"] == "code"]
        print(f"  💻 Code documents: {len(code_docs)}")
        
        # Filter by confidence
        high_confidence_docs = [doc for doc in processed_docs if doc["confidence"] > 0.5]
        print(f"  🎯 High confidence documents: {len(high_confidence_docs)}")
        
        # Save processing results
        results_file = self.output_dir / "context_filtering_results.json"
        with open(results_file, "w") as f:
            json.dump(processed_docs, f, indent=2)
        
        print(f"\n💾 Saved processing results to: {results_file}")
    
    def print_summary(self):
        """Print demo summary."""
        print("\n" + "=" * 60)
        print("📋 Enhanced Chunking and Metadata Demo Summary")
        print("=" * 60)
        print("✅ Smart markdown chunking with topic focus (100-300 words)")
        print("✅ Section-aware documentation processing")
        print("✅ Error log chunking with distinct issue separation")
        print("✅ Comprehensive metadata extraction with context")
        print("✅ Advanced filtering and categorization")
        print("✅ Rich context generation for better LLM understanding")
        
        print(f"\n📁 All demo outputs saved to: {self.output_dir}")
        
        # List generated files
        output_files = list(self.output_dir.glob("*"))
        if output_files:
            print("\n📄 Generated files:")
            for file_path in sorted(output_files):
                print(f"  {file_path.name}")
        
        print("\n🎯 Key Benefits:")
        print("  • Better retrieval relevance with topic-focused chunks")
        print("  • Rich metadata for improved filtering and context")
        print("  • Logical boundaries preserve semantic meaning")
        print("  • Enhanced LLM understanding through context strings")
        print("  • Distinct error issue separation for better debugging")


async def main():
    """Main demo function."""
    demo = EnhancedChunkingDemo()
    
    try:
        await demo.run_demo()
        demo.print_summary()
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 