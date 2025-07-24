"""Code file processor for the RAG system."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import ast
from pygments import highlight
from pygments.lexers import get_lexer_for_filename, get_lexer_by_name
from pygments.formatters import NullFormatter
from pygments.util import ClassNotFound

from ..utils import Document, TextSplitter, CodeTextSplitter, EnhancedASTChunker, CodeChunk

logger = logging.getLogger(__name__)


class CodeProcessor:
    """Processes code files for indexing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, use_ast_chunking: bool = True):
        """Initialize the code processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ast_chunking = use_ast_chunking
        
        # Language mapping
        self.language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bash': 'bash',
            '.zsh': 'zsh',
            '.css': 'css',
            '.scss': 'scss',
            '.html': 'html',
            '.xml': 'xml',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml',
        }
    
    async def process_file(self, file_path: Path) -> List[Document]:
        """Process a single code file."""
        logger.info(f"Processing code file: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Get language
            language = self._detect_language(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, content, language)
            
            # Use enhanced AST chunking for Python files if enabled
            if self.use_ast_chunking and language == 'python':
                documents = await self._process_python_with_ast(file_path, content, metadata)
            else:
                # Create text splitter for the language
                splitter = CodeTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    language=language
                )
                
                # Split into chunks
                chunks = splitter.split_text(content)
                
                # Create documents
                documents = []
                for i, chunk in enumerate(chunks):
                    doc_metadata = {
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk)
                    }
                    
                    # Final cleanup: convert any remaining lists to strings
                    for key, value in doc_metadata.items():
                        if isinstance(value, list):
                            doc_metadata[key] = ', '.join(str(item) for item in value)
                    
                    doc = Document.from_code(
                        file_path=str(file_path),
                        content=chunk,
                        language=language,
                        **doc_metadata
                    )
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    async def _process_python_with_ast(self, file_path: Path, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process Python file using AST-based chunking."""
        try:
            # Create AST chunker
            ast_chunker = EnhancedASTChunker(
                max_chunk_size=self.chunk_size,
                min_chunk_size=self.chunk_size // 4,
                preserve_context=True,
                include_imports=True
            )
            
            # Get AST-based chunks
            code_chunks = ast_chunker.chunk_python_code(content)
            
            # Create documents from chunks
            documents = []
            for i, chunk in enumerate(code_chunks):
                doc_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(code_chunks),
                    'chunk_size': len(chunk.content),
                    'chunk_type': chunk.chunk_type.value,
                    'chunk_name': chunk.name,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'complexity': chunk.complexity,
                    'parameters': ', '.join(chunk.parameters) if isinstance(chunk.parameters, list) else str(chunk.parameters),
                    'decorators': ', '.join(chunk.decorators) if isinstance(chunk.decorators, list) else str(chunk.decorators),
                    'dependencies': ', '.join(chunk.dependencies) if isinstance(chunk.dependencies, list) else str(chunk.dependencies)
                }
                
                # Add docstring if available
                if chunk.docstring:
                    doc_metadata['docstring'] = chunk.docstring
                
                # Final cleanup: convert any remaining lists to strings
                for key, value in doc_metadata.items():
                    if isinstance(value, list):
                        doc_metadata[key] = ', '.join(str(item) for item in value)
                
                doc = Document.from_code(
                    file_path=str(file_path),
                    content=chunk.content,
                    language='python',
                    **doc_metadata
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} AST-based documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error in AST processing for {file_path}: {e}")
            # Fall back to regular text splitting
            splitter = CodeTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                language='python'
            )
            
            chunks = splitter.split_text(content)
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                }
                
                # Final cleanup: convert any remaining lists to strings
                for key, value in doc_metadata.items():
                    if isinstance(value, list):
                        doc_metadata[key] = ', '.join(str(item) for item in value)
                
                doc = Document.from_code(
                    file_path=str(file_path),
                    content=chunk,
                    language='python',
                    **doc_metadata
                )
                documents.append(doc)
            
            return documents
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect the programming language of a file."""
        extension = file_path.suffix.lower()
        
        # Check direct mapping
        if extension in self.language_map:
            return self.language_map[extension]
        
        # Try to use pygments to detect language
        try:
            lexer = get_lexer_for_filename(str(file_path))
            return lexer.name.lower()
        except ClassNotFound:
            pass
        
        # Check filename patterns
        filename = file_path.name.lower()
        if filename in ['makefile', 'dockerfile', 'rakefile']:
            return filename
        
        # Default to text
        return 'text'
    
    def _extract_metadata(self, file_path: Path, content: str, language: str) -> Dict[str, Any]:
        """Extract metadata from code file."""
        metadata = {
            # 'language': language,  # Removed to avoid duplicate argument
            'file_name': file_path.name,
            'file_size': len(content),
            'extension': file_path.suffix,
            'line_count': content.count('\n') + 1,
        }
        
        # Language-specific metadata extraction
        if language == 'python':
            metadata.update(self._extract_python_metadata(content))
        elif language == 'javascript':
            metadata.update(self._extract_javascript_metadata(content))
        elif language == 'java':
            metadata.update(self._extract_java_metadata(content))
        
        return metadata
    
    def _extract_python_metadata(self, content: str) -> Dict[str, Any]:
        """Extract Python-specific metadata."""
        metadata = {
            'functions': [],
            'classes': [],
            'imports': [],
            'docstrings': []
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metadata['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    metadata['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node)
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        metadata['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            metadata['imports'].append(f"{node.module}.{alias.name}")
        
        except SyntaxError:
            logger.warning(f"Could not parse Python file for metadata extraction")
        
        # Convert lists to strings for ChromaDB compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = ', '.join(str(item) if not isinstance(item, dict) else f"{item.get('name', str(item))}" for item in value)
        
        return metadata
    
    def _extract_javascript_metadata(self, content: str) -> Dict[str, Any]:
        """Extract JavaScript-specific metadata."""
        metadata = {
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': []
        }
        
        # Simple regex-based extraction (could be improved with proper parsing)
        import re
        
        # Extract function declarations
        func_pattern = r'function\s+(\w+)\s*\('
        metadata['functions'] = re.findall(func_pattern, content)
        
        # Extract class declarations
        class_pattern = r'class\s+(\w+)'
        metadata['classes'] = re.findall(class_pattern, content)
        
        # Extract imports
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        metadata['imports'] = re.findall(import_pattern, content)
        
        # Extract exports
        export_pattern = r'export\s+.*?(\w+)'
        metadata['exports'] = re.findall(export_pattern, content)
        
        # Convert lists to strings for ChromaDB compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = ', '.join(str(item) for item in value)
        
        return metadata
    
    def _extract_java_metadata(self, content: str) -> Dict[str, Any]:
        """Extract Java-specific metadata."""
        metadata = {
            'classes': [],
            'methods': [],
            'imports': [],
            'package': None
        }
        
        # Simple regex-based extraction
        import re
        
        # Extract package
        package_pattern = r'package\s+([^;]+);'
        package_match = re.search(package_pattern, content)
        if package_match:
            metadata['package'] = package_match.group(1)
        
        # Extract imports
        import_pattern = r'import\s+([^;]+);'
        metadata['imports'] = re.findall(import_pattern, content)
        
        # Extract class declarations
        class_pattern = r'(?:public|private|protected)?\s*class\s+(\w+)'
        metadata['classes'] = re.findall(class_pattern, content)
        
        # Extract method declarations
        method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\('
        metadata['methods'] = re.findall(method_pattern, content)
        
        # Convert lists to strings for ChromaDB compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = ', '.join(str(item) for item in value)
        
        return metadata
    
    def _validate_syntax(self, content: str, language: str) -> bool:
        """Validate syntax of code content."""
        if language == 'python':
            try:
                ast.parse(content)
                return True
            except SyntaxError:
                return False
        
        # For other languages, we'd need specific parsers
        # For now, return True
        return True
    
    async def process_directory(self, directory_path: Path) -> List[Document]:
        """Process all code files in a directory."""
        documents = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.language_map:
                file_documents = await self.process_file(file_path)
                documents.extend(file_documents)
        
        return documents
    
    def get_code_structure(self, content: str, language: str) -> Dict[str, Any]:
        """Get the structure of code content."""
        if language == 'python':
            return self._get_python_structure(content)
        elif language == 'javascript':
            return self._get_javascript_structure(content)
        elif language == 'java':
            return self._get_java_structure(content)
        
        return {'structure': 'unknown'}
    
    def _get_python_structure(self, content: str) -> Dict[str, Any]:
        """Get Python code structure."""
        structure = {
            'type': 'python',
            'functions': [],
            'classes': [],
            'imports': []
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.ClassDef):
                    structure['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
                    })
        
        except SyntaxError:
            structure['error'] = 'Syntax error in Python code'
        
        return structure
    
    def _get_javascript_structure(self, content: str) -> Dict[str, Any]:
        """Get JavaScript code structure."""
        # This would benefit from a proper JavaScript parser
        # For now, return basic structure
        return {
            'type': 'javascript',
            'functions': [],
            'classes': [],
            'exports': []
        }
    
    def _get_java_structure(self, content: str) -> Dict[str, Any]:
        """Get Java code structure."""
        # This would benefit from a proper Java parser
        # For now, return basic structure
        return {
            'type': 'java',
            'classes': [],
            'methods': [],
            'package': None
        } 