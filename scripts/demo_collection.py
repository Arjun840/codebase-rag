#!/usr/bin/env python3
"""
Demo script showing enhanced codebase collection and AST-based chunking capabilities.

This script demonstrates:
1. Collecting codebases from various sources
2. Enhanced AST-based chunking for Python code
3. Documentation collection from Stack Overflow, GitHub, and APIs
4. Different chunking strategies and their effects
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import tempfile
import os

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codebase_rag.utils import (
    CodebaseCollector, 
    DocumentationCollector,
    EnhancedASTChunker, 
    CodeChunk, 
    ChunkType
)
from codebase_rag.processors import CodeProcessor
from codebase_rag.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CollectionDemo:
    """Demonstrates codebase collection and chunking capabilities."""
    
    def __init__(self):
        """Initialize the demo."""
        self.config = Config()
        self.output_dir = Path("demo_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize collectors
        self.codebase_collector = CodebaseCollector(
            max_file_size=1024 * 1024,  # 1MB
            github_token=os.getenv("GITHUB_TOKEN")
        )
        
        self.doc_collector = DocumentationCollector(
            max_requests_per_minute=30,
            user_agent="RAG-Demo/1.0"
        )
        
        self.ast_chunker = EnhancedASTChunker(
            max_chunk_size=1000,
            min_chunk_size=100,
            preserve_context=True,
            include_imports=True
        )
        
        self.code_processor = CodeProcessor(
            chunk_size=1000,
            chunk_overlap=200,
            use_ast_chunking=True
        )
    
    async def run_demo(self):
        """Run the complete demo."""
        print("🚀 Starting RAG Codebase Collection Demo")
        print("=" * 50)
        
        # Demo 1: AST-based chunking
        await self.demo_ast_chunking()
        
        # Demo 2: Codebase collection from GitHub
        await self.demo_github_collection()
        
        # Demo 3: Local directory collection
        await self.demo_local_collection()
        
        # Demo 4: Documentation collection
        await self.demo_documentation_collection()
        
        # Demo 5: Compare chunking strategies
        await self.demo_chunking_comparison()
        
        print("\n✅ Demo completed successfully!")
        print(f"📁 Output files saved to: {self.output_dir}")
    
    async def demo_ast_chunking(self):
        """Demonstrate AST-based chunking with sample Python code."""
        print("\n📄 Demo 1: AST-based Chunking")
        print("-" * 30)
        
        # Create sample Python code
        sample_code = '''
"""
Sample Python module for demonstrating AST-based chunking.
"""

import os
import sys
from typing import List, Dict, Any
from datetime import datetime

# Module-level constants
VERSION = "1.0.0"
DEBUG = True

class DataProcessor:
    """Processes data with various methods."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the processor."""
        self.name = name
        self.config = config
        self.processed_count = 0
    
    def process_data(self, data: List[str]) -> List[str]:
        """Process a list of data items."""
        processed = []
        for item in data:
            if self.validate_item(item):
                processed.append(self.transform_item(item))
            else:
                logger.warning(f"Invalid item skipped: {item}")
        
        self.processed_count += len(processed)
        return processed
    
    def validate_item(self, item: str) -> bool:
        """Validate a single item."""
        if not item or not isinstance(item, str):
            return False
        
        # Complex validation logic
        if len(item) < 3:
            return False
        
        if item.startswith("invalid"):
            return False
        
        return True
    
    def transform_item(self, item: str) -> str:
        """Transform a single item."""
        return item.upper().strip()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_count": self.processed_count,
            "timestamp": datetime.now().isoformat()
        }

async def async_processor(data: List[str]) -> List[str]:
    """Asynchronous data processor."""
    results = []
    
    for item in data:
        # Simulate async processing
        await asyncio.sleep(0.01)
        results.append(f"async_{item}")
    
    return results

def main():
    """Main function."""
    processor = DataProcessor("demo", {"debug": True})
    
    sample_data = ["item1", "item2", "invalid_item", "item3"]
    results = processor.process_data(sample_data)
    
    print(f"Processed {len(results)} items")
    print(f"Stats: {processor.stats}")

if __name__ == "__main__":
    main()
'''
        
        # Save sample code
        sample_file = self.output_dir / "sample_code.py"
        with open(sample_file, "w") as f:
            f.write(sample_code)
        
        print(f"📝 Created sample Python file: {sample_file}")
        
        # Perform AST-based chunking
        chunks = self.ast_chunker.chunk_python_code(sample_code)
        
        print(f"🔪 Generated {len(chunks)} AST-based chunks:")
        
        # Save chunks and display summary
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunk_info = {
                "index": i,
                "type": chunk.chunk_type.value,
                "name": chunk.name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "size": len(chunk.content),
                "complexity": chunk.complexity,
                "parameters": chunk.parameters,
                "decorators": chunk.decorators,
                "content": chunk.content
            }
            chunks_data.append(chunk_info)
            
            print(f"  {i+1}. {chunk.chunk_type.value}: {chunk.name or 'unnamed'} "
                  f"(lines {chunk.start_line}-{chunk.end_line}, {len(chunk.content)} chars)")
            
            if chunk.complexity > 1:
                print(f"     📊 Complexity: {chunk.complexity}")
            
            if chunk.parameters:
                print(f"     📋 Parameters: {', '.join(chunk.parameters)}")
        
        # Save chunks to file
        chunks_file = self.output_dir / "ast_chunks.json"
        with open(chunks_file, "w") as f:
            json.dump(chunks_data, f, indent=2)
        
        print(f"💾 Saved chunk analysis to: {chunks_file}")
    
    async def demo_github_collection(self):
        """Demonstrate GitHub codebase collection."""
        print("\n🐙 Demo 2: GitHub Codebase Collection")
        print("-" * 35)
        
        # Use a small, well-known repository for demonstration
        repo_url = "https://github.com/octocat/Hello-World"
        
        try:
            print(f"📥 Collecting codebase from: {repo_url}")
            
            documents = await self.codebase_collector.collect_from_github(
                repo_url=repo_url,
                branch="master",
                include_issues=True,
                include_prs=False,
                include_wiki=False
            )
            
            print(f"📚 Collected {len(documents)} documents")
            
            # Analyze collected documents
            doc_types = {}
            for doc in documents:
                doc_type = doc.metadata.get('file_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            print("📊 Document types:")
            for doc_type, count in sorted(doc_types.items()):
                print(f"  {doc_type}: {count}")
            
            # Save sample documents
            sample_docs = []
            for doc in documents[:5]:  # First 5 documents
                sample_docs.append({
                    "id": doc.id,
                    "metadata": doc.metadata,
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                })
            
            github_file = self.output_dir / "github_collection.json"
            with open(github_file, "w") as f:
                json.dump(sample_docs, f, indent=2)
            
            print(f"💾 Saved sample documents to: {github_file}")
            
        except Exception as e:
            logger.error(f"Error collecting from GitHub: {e}")
            print(f"❌ GitHub collection failed: {e}")
            print("💡 This might be due to rate limits or missing GitHub token")
    
    async def demo_local_collection(self):
        """Demonstrate local directory collection."""
        print("\n📁 Demo 3: Local Directory Collection")
        print("-" * 34)
        
        # Create a temporary directory with sample files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample project structure
            (temp_path / "src").mkdir()
            (temp_path / "docs").mkdir()
            (temp_path / "tests").mkdir()
            
            # Create sample files
            files_to_create = {
                "src/main.py": '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
''',
                "src/utils.py": '''
import math

def calculate_area(radius: float) -> float:
    """Calculate circle area."""
    return math.pi * radius ** 2

class Calculator:
    """Simple calculator."""
    
    def add(self, a: float, b: float) -> float:
        return a + b
    
    def multiply(self, a: float, b: float) -> float:
        return a * b
''',
                "docs/README.md": '''
# Sample Project

This is a sample project for demonstrating codebase collection.

## Features

- Hello world functionality
- Mathematical utilities
- Circle area calculation
''',
                "tests/test_utils.py": '''
import pytest
from src.utils import calculate_area, Calculator

def test_calculate_area():
    assert abs(calculate_area(1.0) - 3.14159) < 0.001

def test_calculator():
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.multiply(4, 5) == 20
''',
                "requirements.txt": '''
pytest>=7.0.0
numpy>=1.20.0
''',
                ".gitignore": '''
__pycache__/
*.pyc
.pytest_cache/
'''
            }
            
            # Create files
            for file_path, content in files_to_create.items():
                full_path = temp_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content)
            
            print(f"📂 Created sample project in: {temp_path}")
            
            # Collect from local directory
            documents = await self.codebase_collector.collect_from_directory(temp_path)
            
            print(f"📚 Collected {len(documents)} documents")
            
            # Analyze by file type
            file_types = {}
            for doc in documents:
                file_type = doc.metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            print("📊 File types collected:")
            for file_type, count in sorted(file_types.items()):
                print(f"  {file_type}: {count}")
            
            # Process Python files with AST chunking
            python_docs = [doc for doc in documents if doc.metadata.get('file_type') == 'python']
            
            if python_docs:
                print(f"\n🐍 Processing {len(python_docs)} Python files with AST chunking:")
                
                ast_results = []
                for doc in python_docs:
                    chunks = self.ast_chunker.chunk_python_code(doc.content)
                    
                    file_name = doc.metadata.get('file_name', 'unknown')
                    print(f"  {file_name}: {len(chunks)} chunks")
                    
                    for chunk in chunks:
                        if chunk.chunk_type == ChunkType.FUNCTION:
                            print(f"    🔧 Function: {chunk.name} (complexity: {chunk.complexity})")
                        elif chunk.chunk_type == ChunkType.CLASS:
                            print(f"    🏗️  Class: {chunk.name}")
                        elif chunk.chunk_type == ChunkType.METHOD:
                            print(f"    ⚙️  Method: {chunk.name}")
                    
                    ast_results.append({
                        "file_name": file_name,
                        "chunks": len(chunks),
                        "chunk_types": [chunk.chunk_type.value for chunk in chunks]
                    })
                
                # Save AST results
                ast_file = self.output_dir / "local_ast_analysis.json"
                with open(ast_file, "w") as f:
                    json.dump(ast_results, f, indent=2)
                
                print(f"💾 Saved AST analysis to: {ast_file}")
    
    async def demo_documentation_collection(self):
        """Demonstrate documentation collection from various sources."""
        print("\n📖 Demo 4: Documentation Collection")
        print("-" * 33)
        
        # Demo Stack Overflow collection
        print("📚 Collecting Stack Overflow questions...")
        try:
            stackoverflow_docs = await self.doc_collector.collect_stackoverflow_questions(
                tags=["python", "machine-learning"],
                max_questions=5,
                include_answers=True,
                sort="votes"
            )
            
            print(f"📊 Collected {len(stackoverflow_docs)} Stack Overflow documents")
            
            # Sample the first few documents
            sample_so_docs = []
            for doc in stackoverflow_docs[:3]:
                sample_so_docs.append({
                    "type": doc.metadata.get('type'),
                    "title": doc.metadata.get('title'),
                    "score": doc.metadata.get('score'),
                    "tags": doc.metadata.get('tags'),
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                })
            
            so_file = self.output_dir / "stackoverflow_collection.json"
            with open(so_file, "w") as f:
                json.dump(sample_so_docs, f, indent=2)
            
            print(f"💾 Saved Stack Overflow documents to: {so_file}")
            
        except Exception as e:
            logger.error(f"Error collecting Stack Overflow data: {e}")
            print(f"❌ Stack Overflow collection failed: {e}")
            print("💡 This might be due to API rate limits")
        
        # Demo GitHub README collection
        print("\n📄 Collecting GitHub README files...")
        try:
            readme_docs = await self.doc_collector.collect_readme_files(
                repo_urls=[
                    "https://github.com/python/cpython",
                    "https://github.com/pytorch/pytorch"
                ],
                github_token=os.getenv("GITHUB_TOKEN")
            )
            
            print(f"📊 Collected {len(readme_docs)} README documents")
            
            # Save README info
            readme_info = []
            for doc in readme_docs:
                readme_info.append({
                    "repository": doc.metadata.get('repository'),
                    "file_name": doc.metadata.get('file_name'),
                    "size": len(doc.content),
                    "content_preview": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
                })
            
            readme_file = self.output_dir / "readme_collection.json"
            with open(readme_file, "w") as f:
                json.dump(readme_info, f, indent=2)
            
            print(f"💾 Saved README documents to: {readme_file}")
            
        except Exception as e:
            logger.error(f"Error collecting README files: {e}")
            print(f"❌ README collection failed: {e}")
            print("💡 This might be due to rate limits or missing GitHub token")
    
    async def demo_chunking_comparison(self):
        """Compare different chunking strategies."""
        print("\n⚖️  Demo 5: Chunking Strategy Comparison")
        print("-" * 38)
        
        # Create a complex Python file for comparison
        complex_code = '''
"""
Complex Python module for chunking comparison.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Status(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    name: str
    status: Status
    created_at: datetime
    data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}

class TaskManager:
    """Manages asynchronous tasks with complex workflow."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.tasks: Dict[str, Task] = {}
        self.running_tasks: List[str] = []
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self._task_counter = 0
    
    async def create_task(self, name: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new task."""
        task_id = f"task_{self._task_counter}"
        self._task_counter += 1
        
        task = Task(
            id=task_id,
            name=name,
            status=Status.PENDING,
            created_at=datetime.now(),
            data=data or {}
        )
        
        self.tasks[task_id] = task
        logger.info(f"Created task {task_id}: {name}")
        
        return task_id
    
    async def execute_task(self, task_id: str) -> bool:
        """Execute a specific task."""
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        task = self.tasks[task_id]
        
        if task.status != Status.PENDING:
            logger.warning(f"Task {task_id} is not pending")
            return False
        
        try:
            task.status = Status.PROCESSING
            self.running_tasks.append(task_id)
            
            # Simulate complex processing
            await self._process_task(task)
            
            task.status = Status.COMPLETED
            self.running_tasks.remove(task_id)
            self.completed_tasks.append(task_id)
            
            logger.info(f"Task {task_id} completed successfully")
            return True
            
        except Exception as e:
            task.status = Status.FAILED
            if task_id in self.running_tasks:
                self.running_tasks.remove(task_id)
            self.failed_tasks.append(task_id)
            
            logger.error(f"Task {task_id} failed: {e}")
            return False
    
    async def _process_task(self, task: Task):
        """Internal task processing logic."""
        # Simulate different types of processing based on task data
        processing_type = task.data.get('type', 'default')
        
        if processing_type == 'heavy':
            await self._heavy_processing(task)
        elif processing_type == 'io':
            await self._io_processing(task)
        elif processing_type == 'network':
            await self._network_processing(task)
        else:
            await self._default_processing(task)
    
    async def _heavy_processing(self, task: Task):
        """CPU-intensive processing."""
        await asyncio.sleep(2)  # Simulate heavy computation
        
        # Simulate some computation
        result = 0
        for i in range(1000000):
            result += i * 2
        
        task.data['result'] = result
    
    async def _io_processing(self, task: Task):
        """I/O-intensive processing."""
        await asyncio.sleep(1)  # Simulate I/O operations
        
        # Simulate file operations
        task.data['files_processed'] = 100
        task.data['bytes_processed'] = 1024 * 1024
    
    async def _network_processing(self, task: Task):
        """Network-intensive processing."""
        await asyncio.sleep(1.5)  # Simulate network calls
        
        # Simulate API calls
        task.data['api_calls'] = 50
        task.data['response_time'] = 0.25
    
    async def _default_processing(self, task: Task):
        """Default processing."""
        await asyncio.sleep(0.5)
        task.data['processed'] = True
    
    async def execute_all_pending(self):
        """Execute all pending tasks with concurrency control."""
        pending_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.status == Status.PENDING
        ]
        
        if not pending_tasks:
            logger.info("No pending tasks to execute")
            return
        
        logger.info(f"Executing {len(pending_tasks)} pending tasks")
        
        # Execute tasks with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def execute_with_semaphore(task_id: str):
            async with semaphore:
                return await self.execute_task(task_id)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(task_id) for task_id in pending_tasks],
            return_exceptions=True
        )
        
        successful = sum(1 for result in results if result is True)
        logger.info(f"Executed {successful}/{len(pending_tasks)} tasks successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        return {
            'total_tasks': len(self.tasks),
            'pending': len([t for t in self.tasks.values() if t.status == Status.PENDING]),
            'running': len(self.running_tasks),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks),
            'success_rate': len(self.completed_tasks) / len(self.tasks) if self.tasks else 0
        }
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific task."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            'id': task.id,
            'name': task.name,
            'status': task.status.value,
            'created_at': task.created_at.isoformat(),
            'data': task.data
        }

async def main():
    """Main demonstration function."""
    manager = TaskManager(max_concurrent=5)
    
    # Create various types of tasks
    tasks = [
        ("Heavy Processing", {"type": "heavy"}),
        ("I/O Processing", {"type": "io"}),
        ("Network Processing", {"type": "network"}),
        ("Default Processing", {"type": "default"}),
        ("Another Heavy Task", {"type": "heavy"}),
    ]
    
    # Create tasks
    for name, data in tasks:
        await manager.create_task(name, data)
    
    # Execute all tasks
    await manager.execute_all_pending()
    
    # Print statistics
    stats = manager.get_stats()
    print(f"Task execution completed: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Save complex code
        complex_file = self.output_dir / "complex_code.py"
        with open(complex_file, "w") as f:
            f.write(complex_code)
        
        print(f"📝 Created complex Python file: {complex_file}")
        
        # Compare AST chunking vs regular chunking
        print("\n🔍 Comparing chunking strategies:")
        
        # AST-based chunking
        ast_chunks = self.ast_chunker.chunk_python_code(complex_code)
        print(f"📊 AST Chunking: {len(ast_chunks)} chunks")
        
        # Regular text-based chunking
        from codebase_rag.utils import CodeTextSplitter
        text_splitter = CodeTextSplitter(chunk_size=1000, chunk_overlap=200, language="python")
        text_chunks = text_splitter.split_text(complex_code)
        print(f"📊 Text Chunking: {len(text_chunks)} chunks")
        
        # Analyze AST chunks
        ast_analysis = {
            "total_chunks": len(ast_chunks),
            "chunk_types": {},
            "avg_complexity": 0,
            "chunks_with_functions": 0,
            "chunks_with_classes": 0
        }
        
        total_complexity = 0
        for chunk in ast_chunks:
            chunk_type = chunk.chunk_type.value
            ast_analysis["chunk_types"][chunk_type] = ast_analysis["chunk_types"].get(chunk_type, 0) + 1
            
            if chunk.chunk_type == ChunkType.FUNCTION:
                ast_analysis["chunks_with_functions"] += 1
            elif chunk.chunk_type == ChunkType.CLASS:
                ast_analysis["chunks_with_classes"] += 1
            
            total_complexity += chunk.complexity
        
        ast_analysis["avg_complexity"] = total_complexity / len(ast_chunks) if ast_chunks else 0
        
        # Analyze text chunks
        text_analysis = {
            "total_chunks": len(text_chunks),
            "avg_chunk_size": sum(len(chunk) for chunk in text_chunks) / len(text_chunks) if text_chunks else 0,
            "size_distribution": {}
        }
        
        # Categorize text chunks by size
        for chunk in text_chunks:
            size = len(chunk)
            if size < 300:
                category = "small"
            elif size < 700:
                category = "medium"
            else:
                category = "large"
            
            text_analysis["size_distribution"][category] = text_analysis["size_distribution"].get(category, 0) + 1
        
        # Display comparison
        print("\n📊 AST Chunking Analysis:")
        print(f"  Total chunks: {ast_analysis['total_chunks']}")
        print(f"  Average complexity: {ast_analysis['avg_complexity']:.2f}")
        print(f"  Function chunks: {ast_analysis['chunks_with_functions']}")
        print(f"  Class chunks: {ast_analysis['chunks_with_classes']}")
        print("  Chunk types:")
        for chunk_type, count in ast_analysis["chunk_types"].items():
            print(f"    {chunk_type}: {count}")
        
        print("\n📊 Text Chunking Analysis:")
        print(f"  Total chunks: {text_analysis['total_chunks']}")
        print(f"  Average chunk size: {text_analysis['avg_chunk_size']:.0f} characters")
        print("  Size distribution:")
        for size_category, count in text_analysis["size_distribution"].items():
            print(f"    {size_category}: {count}")
        
        # Save comparison results
        comparison_data = {
            "ast_chunking": ast_analysis,
            "text_chunking": text_analysis,
            "sample_ast_chunks": [
                {
                    "type": chunk.chunk_type.value,
                    "name": chunk.name,
                    "lines": f"{chunk.start_line}-{chunk.end_line}",
                    "size": len(chunk.content),
                    "complexity": chunk.complexity,
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                } for chunk in ast_chunks[:5]
            ],
            "sample_text_chunks": [
                {
                    "index": i,
                    "size": len(chunk),
                    "content_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
                } for i, chunk in enumerate(text_chunks[:5])
            ]
        }
        
        comparison_file = self.output_dir / "chunking_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n💾 Saved comparison results to: {comparison_file}")
    
    def print_summary(self):
        """Print demo summary."""
        print("\n" + "=" * 50)
        print("📋 Demo Summary")
        print("=" * 50)
        print("✅ Demonstrated AST-based chunking with logical boundaries")
        print("✅ Showed codebase collection from GitHub repositories")
        print("✅ Demonstrated local directory collection and processing")
        print("✅ Explored documentation collection from various sources")
        print("✅ Compared different chunking strategies")
        print(f"\n📁 All demo outputs saved to: {self.output_dir}")
        
        # List generated files
        output_files = list(self.output_dir.glob("*"))
        if output_files:
            print("\n📄 Generated files:")
            for file_path in sorted(output_files):
                print(f"  {file_path.name}")


async def main():
    """Main demo function."""
    demo = CollectionDemo()
    
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