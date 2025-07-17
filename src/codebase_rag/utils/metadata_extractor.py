"""Comprehensive metadata extraction for enhanced context and filtering."""

import re
import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content types for metadata extraction."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    ERROR_LOG = "error_log"
    QA = "qa"
    ISSUE = "issue"
    README = "readme"
    CONFIG = "config"
    TEST = "test"
    UNKNOWN = "unknown"


@dataclass
class MetadataContext:
    """Rich metadata context for documents."""
    
    # File information
    file_path: str
    file_name: str
    file_extension: str
    file_size: int
    file_hash: Optional[str] = None
    last_modified: Optional[datetime] = None
    
    # Content information
    content_type: ContentType = ContentType.UNKNOWN
    programming_language: Optional[str] = None
    word_count: int = 0
    line_count: int = 0
    
    # Structure information
    section_title: Optional[str] = None
    section_path: Optional[str] = None  # e.g., "API > Authentication > OAuth"
    section_level: int = 0
    
    # Code-specific metadata
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    complexity: int = 0
    
    # Documentation-specific metadata
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    
    # Repository/project context
    repository_name: Optional[str] = None
    repository_path: Optional[str] = None
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    
    # Quality indicators
    has_docstring: bool = False
    has_tests: bool = False
    has_examples: bool = False
    confidence_score: float = 0.0
    
    # Filtering tags
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Additional custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_extension': self.file_extension,
            'file_size': self.file_size,
            'file_hash': self.file_hash,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'content_type': self.content_type.value,
            'programming_language': self.programming_language,
            'word_count': self.word_count,
            'line_count': self.line_count,
            'section_title': self.section_title,
            'section_path': self.section_path,
            'section_level': self.section_level,
            'functions': self.functions,
            'classes': self.classes,
            'imports': self.imports,
            'exports': self.exports,
            'decorators': self.decorators,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'complexity': self.complexity,
            'topics': self.topics,
            'keywords': self.keywords,
            'references': self.references,
            'code_examples': self.code_examples,
            'repository_name': self.repository_name,
            'repository_path': self.repository_path,
            'branch': self.branch,
            'commit_hash': self.commit_hash,
            'has_docstring': self.has_docstring,
            'has_tests': self.has_tests,
            'has_examples': self.has_examples,
            'confidence_score': self.confidence_score,
            'tags': self.tags,
            'categories': self.categories,
            'custom': self.custom
        }


class MetadataExtractor:
    """Comprehensive metadata extractor."""
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cxx': 'cpp',
            '.cc': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bash': 'bash',
            '.zsh': 'zsh',
            '.fish': 'fish',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.html': 'html',
            '.htm': 'html',
            '.xml': 'xml',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'ini',
            '.md': 'markdown',
            '.rst': 'rst',
            '.tex': 'latex',
            '.r': 'r',
            '.R': 'r',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.clj': 'clojure',
            '.hs': 'haskell',
            '.elm': 'elm',
            '.vue': 'vue',
            '.svelte': 'svelte',
            '.dart': 'dart',
            '.m': 'objective-c',
            '.mm': 'objective-c',
            '.cs': 'csharp',
            '.fs': 'fsharp',
            '.vb': 'vb.net',
            '.pl': 'perl',
            '.pm': 'perl',
            '.lua': 'lua',
            '.vim': 'vim',
        }
        
        self.content_patterns = {
            'test_file': re.compile(r'test_|_test\.py$|\.test\.js$|spec\.js$|\.spec\.py$', re.IGNORECASE),
            'config_file': re.compile(r'config|settings|\.env|\.ini|\.toml|\.cfg|\.conf$', re.IGNORECASE),
            'readme': re.compile(r'readme|getting.started|introduction', re.IGNORECASE),
            'error_log': re.compile(r'\.log$|error|exception|traceback|failed', re.IGNORECASE),
            'api_doc': re.compile(r'api|endpoint|swagger|openapi', re.IGNORECASE),
            'tutorial': re.compile(r'tutorial|guide|howto|example', re.IGNORECASE),
        }
        
        self.code_patterns = {
            'function_def': re.compile(r'def\s+(\w+)|function\s+(\w+)|func\s+(\w+)', re.IGNORECASE),
            'class_def': re.compile(r'class\s+(\w+)|struct\s+(\w+)|interface\s+(\w+)', re.IGNORECASE),
            'import_stmt': re.compile(r'import\s+(\w+)|from\s+(\w+)\s+import|#include\s*<(\w+)>', re.IGNORECASE),
            'decorator': re.compile(r'@(\w+)', re.IGNORECASE),
            'docstring': re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|/\*\*[\s\S]*?\*/'),
            'comment': re.compile(r'#.*|//.*|/\*.*?\*/'),
            'todo': re.compile(r'TODO|FIXME|XXX|HACK|NOTE', re.IGNORECASE),
        }
    
    def extract_metadata(self, 
                        content: str, 
                        file_path: Optional[str] = None,
                        chunk_info: Optional[Dict[str, Any]] = None) -> MetadataContext:
        """Extract comprehensive metadata from content."""
        
        # Initialize metadata context
        metadata = MetadataContext(
            file_path=file_path or "unknown",
            file_name=Path(file_path).name if file_path else "unknown",
            file_extension=Path(file_path).suffix if file_path else "",
            file_size=len(content),
            word_count=len(content.split()),
            line_count=len(content.splitlines())
        )
        
        # Extract file-based metadata
        if file_path:
            metadata = self._extract_file_metadata(metadata, file_path)
        
        # Extract content-based metadata
        metadata = self._extract_content_metadata(metadata, content)
        
        # Extract programming language metadata
        if metadata.programming_language:
            metadata = self._extract_language_metadata(metadata, content)
        
        # Extract documentation metadata
        if metadata.content_type in [ContentType.DOCUMENTATION, ContentType.README]:
            metadata = self._extract_documentation_metadata(metadata, content)
        
        # Apply chunk-specific metadata
        if chunk_info:
            metadata = self._apply_chunk_metadata(metadata, chunk_info)
        
        # Calculate confidence score
        metadata.confidence_score = self._calculate_confidence_score(metadata)
        
        # Generate tags and categories
        metadata.tags = self._generate_tags(metadata)
        metadata.categories = self._generate_categories(metadata)
        
        return metadata
    
    def _extract_file_metadata(self, metadata: MetadataContext, file_path: str) -> MetadataContext:
        """Extract metadata from file information."""
        path = Path(file_path)
        
        # File stats
        if path.exists():
            stat = path.stat()
            metadata.file_size = stat.st_size
            metadata.last_modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Calculate file hash
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                    metadata.file_hash = hashlib.md5(content).hexdigest()
            except Exception as e:
                logger.debug(f"Could not calculate file hash: {e}")
        
        # Programming language
        metadata.programming_language = self.language_map.get(path.suffix.lower())
        
        # Content type detection
        metadata.content_type = self._detect_content_type(path)
        
        # Repository context
        metadata = self._extract_repository_context(metadata, path)
        
        return metadata
    
    def _extract_content_metadata(self, metadata: MetadataContext, content: str) -> MetadataContext:
        """Extract metadata from content."""
        
        # Basic content metrics
        metadata.word_count = len(content.split())
        metadata.line_count = len(content.splitlines())
        
        # Extract references and links
        metadata.references = self._extract_references(content)
        
        # Extract code examples
        metadata.code_examples = self._extract_code_examples(content)
        
        # Extract keywords
        metadata.keywords = self._extract_keywords(content)
        
        # Quality indicators
        metadata.has_docstring = bool(self.code_patterns['docstring'].search(content))
        metadata.has_examples = len(metadata.code_examples) > 0
        metadata.has_tests = bool(self.content_patterns['test_file'].search(metadata.file_path))
        
        return metadata
    
    def _extract_language_metadata(self, metadata: MetadataContext, content: str) -> MetadataContext:
        """Extract programming language-specific metadata."""
        
        if metadata.programming_language == 'python':
            metadata = self._extract_python_metadata(metadata, content)
        elif metadata.programming_language == 'javascript':
            metadata = self._extract_javascript_metadata(metadata, content)
        elif metadata.programming_language == 'java':
            metadata = self._extract_java_metadata(metadata, content)
        else:
            metadata = self._extract_generic_code_metadata(metadata, content)
        
        return metadata
    
    def _extract_documentation_metadata(self, metadata: MetadataContext, content: str) -> MetadataContext:
        """Extract documentation-specific metadata."""
        
        # Extract topics from headers
        headers = re.findall(r'^#+\s*(.+)$', content, re.MULTILINE)
        metadata.topics = [h.strip() for h in headers]
        
        # Extract section information
        if metadata.topics:
            metadata.section_title = metadata.topics[0]
            metadata.section_path = ' > '.join(metadata.topics[:3])  # Top 3 levels
        
        # Extract API endpoints
        api_patterns = [
            r'GET\s+([/\w\-{}]+)',
            r'POST\s+([/\w\-{}]+)',
            r'PUT\s+([/\w\-{}]+)',
            r'DELETE\s+([/\w\-{}]+)',
            r'PATCH\s+([/\w\-{}]+)',
        ]
        
        for pattern in api_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            metadata.references.extend(matches)
        
        return metadata
    
    def _extract_python_metadata(self, metadata: MetadataContext, content: str) -> MetadataContext:
        """Extract Python-specific metadata."""
        
        # Function definitions
        func_matches = re.findall(r'def\s+(\w+)\s*\(([^)]*)\)', content)
        metadata.functions = [match[0] for match in func_matches]
        
        # Class definitions
        class_matches = re.findall(r'class\s+(\w+)', content)
        metadata.classes = class_matches
        
        # Import statements
        import_matches = re.findall(r'(?:from\s+(\w+(?:\.\w+)*)\s+)?import\s+([^\n]+)', content)
        imports = []
        for from_module, import_items in import_matches:
            if from_module:
                imports.append(f"{from_module}.{import_items}")
            else:
                imports.append(import_items)
        metadata.imports = imports
        
        # Decorators
        decorator_matches = re.findall(r'@(\w+)', content)
        metadata.decorators = list(set(decorator_matches))
        
        # Calculate complexity (simple metric)
        complexity_patterns = [
            r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\btry\b',
            r'\bexcept\b', r'\bwith\b', r'\band\b', r'\bor\b'
        ]
        
        complexity = 1  # Base complexity
        for pattern in complexity_patterns:
            complexity += len(re.findall(pattern, content, re.IGNORECASE))
        
        metadata.complexity = complexity
        
        return metadata
    
    def _extract_javascript_metadata(self, metadata: MetadataContext, content: str) -> MetadataContext:
        """Extract JavaScript-specific metadata."""
        
        # Function definitions
        func_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'(\w+)\s*=>\s*',
            r'const\s+(\w+)\s*=\s*\(',
            r'let\s+(\w+)\s*=\s*\(',
            r'var\s+(\w+)\s*=\s*\(',
        ]
        
        functions = []
        for pattern in func_patterns:
            matches = re.findall(pattern, content)
            functions.extend(matches)
        
        metadata.functions = list(set(functions))
        
        # Class definitions
        class_matches = re.findall(r'class\s+(\w+)', content)
        metadata.classes = class_matches
        
        # Import/Export statements
        import_matches = re.findall(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', content)
        export_matches = re.findall(r'export\s+(?:default\s+)?(\w+)', content)
        
        metadata.imports = import_matches
        metadata.exports = export_matches
        
        return metadata
    
    def _extract_java_metadata(self, metadata: MetadataContext, content: str) -> MetadataContext:
        """Extract Java-specific metadata."""
        
        # Class definitions
        class_matches = re.findall(r'(?:public\s+)?class\s+(\w+)', content)
        metadata.classes = class_matches
        
        # Method definitions
        method_matches = re.findall(r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:\w+\s+)?(\w+)\s*\(', content)
        metadata.functions = [m for m in method_matches if m not in ['if', 'for', 'while', 'switch']]
        
        # Import statements
        import_matches = re.findall(r'import\s+([^\s;]+)', content)
        metadata.imports = import_matches
        
        # Annotations
        annotation_matches = re.findall(r'@(\w+)', content)
        metadata.decorators = list(set(annotation_matches))
        
        return metadata
    
    def _extract_generic_code_metadata(self, metadata: MetadataContext, content: str) -> MetadataContext:
        """Extract generic code metadata."""
        
        # Generic function patterns
        func_matches = re.findall(self.code_patterns['function_def'], content)
        metadata.functions = [match for match in func_matches if match]
        
        # Generic class patterns
        class_matches = re.findall(self.code_patterns['class_def'], content)
        metadata.classes = [match for match in class_matches if match]
        
        # Generic import patterns
        import_matches = re.findall(self.code_patterns['import_stmt'], content)
        metadata.imports = [match for match in import_matches if match]
        
        return metadata
    
    def _detect_content_type(self, path: Path) -> ContentType:
        """Detect content type from file path."""
        
        file_name = path.name.lower()
        
        if self.content_patterns['test_file'].search(file_name):
            return ContentType.TEST
        elif self.content_patterns['config_file'].search(file_name):
            return ContentType.CONFIG
        elif self.content_patterns['readme'].search(file_name):
            return ContentType.README
        elif self.content_patterns['error_log'].search(file_name):
            return ContentType.ERROR_LOG
        elif path.suffix.lower() in ['.md', '.rst', '.txt']:
            return ContentType.DOCUMENTATION
        elif path.suffix.lower() in self.language_map:
            return ContentType.CODE
        else:
            return ContentType.UNKNOWN
    
    def _extract_repository_context(self, metadata: MetadataContext, path: Path) -> MetadataContext:
        """Extract repository context."""
        
        # Look for .git directory
        current_path = path if path.is_dir() else path.parent
        
        while current_path != current_path.parent:
            git_dir = current_path / '.git'
            if git_dir.exists():
                metadata.repository_path = str(current_path)
                metadata.repository_name = current_path.name
                
                # Try to get branch info
                try:
                    head_file = git_dir / 'HEAD'
                    if head_file.exists():
                        with open(head_file, 'r') as f:
                            head_content = f.read().strip()
                            if head_content.startswith('ref: refs/heads/'):
                                metadata.branch = head_content.split('/')[-1]
                except Exception:
                    pass
                
                break
            
            current_path = current_path.parent
        
        return metadata
    
    def _extract_references(self, content: str) -> List[str]:
        """Extract references and links from content."""
        references = []
        
        # URLs
        url_pattern = r'https?://[^\s\)]+|www\.[^\s\)]+'
        urls = re.findall(url_pattern, content)
        references.extend(urls)
        
        # File references
        file_pattern = r'[\'"]([^\'"\s]+\.[a-zA-Z0-9]+)[\'"]'
        files = re.findall(file_pattern, content)
        references.extend(files)
        
        # Function/method references
        func_pattern = r'(\w+)\s*\('
        functions = re.findall(func_pattern, content)
        references.extend(functions[:10])  # Limit to first 10
        
        return list(set(references))
    
    def _extract_code_examples(self, content: str) -> List[str]:
        """Extract code examples from content."""
        examples = []
        
        # Code blocks
        code_block_pattern = r'```[\w]*\n(.*?)\n```'
        code_blocks = re.findall(code_block_pattern, content, re.DOTALL)
        examples.extend(code_blocks)
        
        # Inline code
        inline_code_pattern = r'`([^`]+)`'
        inline_codes = re.findall(inline_code_pattern, content)
        examples.extend(inline_codes)
        
        return examples[:5]  # Limit to first 5
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter out common words
        common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'end', 'why', 'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        filtered_words = [word for word in words if word not in common_words]
        
        # Get most frequent words
        from collections import Counter
        word_counts = Counter(filtered_words)
        
        return [word for word, count in word_counts.most_common(10)]
    
    def _apply_chunk_metadata(self, metadata: MetadataContext, chunk_info: Dict[str, Any]) -> MetadataContext:
        """Apply chunk-specific metadata."""
        
        # Section information
        if 'section_title' in chunk_info:
            metadata.section_title = chunk_info['section_title']
        
        if 'section_path' in chunk_info:
            metadata.section_path = chunk_info['section_path']
        
        if 'section_level' in chunk_info:
            metadata.section_level = chunk_info['section_level']
        
        # Function/class information
        if 'chunk_name' in chunk_info:
            if chunk_info.get('chunk_type') == 'function':
                metadata.functions = [chunk_info['chunk_name']]
            elif chunk_info.get('chunk_type') == 'class':
                metadata.classes = [chunk_info['chunk_name']]
        
        # Additional chunk metadata
        metadata.custom.update(chunk_info)
        
        return metadata
    
    def _calculate_confidence_score(self, metadata: MetadataContext) -> float:
        """Calculate confidence score based on metadata completeness."""
        score = 0.0
        
        # File information (30%)
        if metadata.file_path != "unknown":
            score += 0.1
        if metadata.programming_language:
            score += 0.1
        if metadata.file_hash:
            score += 0.1
        
        # Content information (40%)
        if metadata.functions:
            score += 0.1
        if metadata.classes:
            score += 0.1
        if metadata.has_docstring:
            score += 0.1
        if metadata.topics:
            score += 0.1
        
        # Context information (30%)
        if metadata.repository_name:
            score += 0.1
        if metadata.section_title:
            score += 0.1
        if metadata.references:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_tags(self, metadata: MetadataContext) -> List[str]:
        """Generate tags for filtering and search."""
        tags = []
        
        # Programming language
        if metadata.programming_language:
            tags.append(f"lang:{metadata.programming_language}")
        
        # Content type
        tags.append(f"type:{metadata.content_type.value}")
        
        # File type
        if metadata.file_extension:
            tags.append(f"ext:{metadata.file_extension[1:]}")  # Remove dot
        
        # Repository
        if metadata.repository_name:
            tags.append(f"repo:{metadata.repository_name}")
        
        # Quality indicators
        if metadata.has_docstring:
            tags.append("documented")
        if metadata.has_tests:
            tags.append("tested")
        if metadata.has_examples:
            tags.append("examples")
        
        # Complexity
        if metadata.complexity > 10:
            tags.append("complex")
        elif metadata.complexity < 5:
            tags.append("simple")
        
        # Size
        if metadata.word_count > 500:
            tags.append("large")
        elif metadata.word_count < 100:
            tags.append("small")
        
        return tags
    
    def _generate_categories(self, metadata: MetadataContext) -> List[str]:
        """Generate categories for organization."""
        categories = []
        
        # Primary category
        categories.append(metadata.content_type.value)
        
        # Secondary categories
        if metadata.programming_language:
            categories.append(metadata.programming_language)
        
        if metadata.section_title:
            categories.append("documentation")
        
        if metadata.functions:
            categories.append("functions")
        
        if metadata.classes:
            categories.append("classes")
        
        if metadata.content_type == ContentType.TEST:
            categories.append("testing")
        
        if metadata.content_type == ContentType.CONFIG:
            categories.append("configuration")
        
        return categories
    
    def create_context_string(self, metadata: MetadataContext) -> str:
        """Create a context string for the LLM."""
        context_parts = []
        
        # File context
        if metadata.file_path != "unknown":
            context_parts.append(f"File: {metadata.file_name}")
            
            if metadata.repository_name:
                context_parts.append(f"Repository: {metadata.repository_name}")
        
        # Section context
        if metadata.section_path:
            context_parts.append(f"Section: {metadata.section_path}")
        
        # Code context
        if metadata.programming_language:
            context_parts.append(f"Language: {metadata.programming_language}")
        
        if metadata.functions:
            context_parts.append(f"Functions: {', '.join(metadata.functions[:3])}")
        
        if metadata.classes:
            context_parts.append(f"Classes: {', '.join(metadata.classes[:3])}")
        
        return " | ".join(context_parts) 