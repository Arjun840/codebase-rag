"""Error processor for handling error logs and tracebacks."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from ..utils import Document, RecursiveCharacterTextSplitter, SmartTextSplitter, MetadataExtractor

logger = logging.getLogger(__name__)


class ErrorProcessor:
    """Processes error logs and tracebacks for indexing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, use_smart_chunking: bool = True):
        """Initialize the error processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_smart_chunking = use_smart_chunking
        
        # Initialize smart text splitter and metadata extractor
        self.smart_splitter = SmartTextSplitter(
            min_chunk_words=50,  # Shorter for error logs
            max_chunk_words=200,
            overlap_words=10,
            preserve_structure=True
        )
        
        self.metadata_extractor = MetadataExtractor()
        
        # Common error patterns
        self.error_patterns = {
            'python': {
                'exception': r'(\w+Error|Exception): (.+)',
                'traceback': r'Traceback \(most recent call last\):',
                'file_line': r'File "([^"]+)", line (\d+)',
            },
            'javascript': {
                'exception': r'(\w+Error): (.+)',
                'traceback': r'at (.+) \((.+):(\d+):(\d+)\)',
            },
            'java': {
                'exception': r'(\w+Exception): (.+)',
                'traceback': r'at (.+)\((.+):(\d+)\)',
            }
        }
    
    async def process_error_log(self, error_text: str, source: str = "unknown") -> List[Document]:
        """Process an error log string."""
        logger.info(f"Processing error log from: {source}")
        
        try:
            if self.use_smart_chunking:
                # Use smart text splitter for issue-focused chunking
                smart_chunks = self.smart_splitter.split_error_logs(error_text, source)
                
                # Create documents from smart chunks
                documents = []
                for i, chunk in enumerate(smart_chunks):
                    # Extract comprehensive metadata
                    metadata_context = self.metadata_extractor.extract_metadata(
                        chunk.content, 
                        source,
                        chunk_info={
                            'chunk_type': chunk.focus.value,
                            'chunk_title': chunk.title,
                            'chunk_index': i,
                            'total_chunks': len(smart_chunks),
                            'word_count': chunk.word_count,
                            'error_type': 'log_entry'
                        }
                    )
                    
                    # Add error-specific metadata
                    error_metadata = self._extract_error_metadata(chunk.content)
                    metadata_context.custom.update(error_metadata)
                    
                    # Create context string for better LLM understanding
                    context_string = self.metadata_extractor.create_context_string(metadata_context)
                    
                    # Convert metadata to dict and add context
                    doc_metadata = metadata_context.to_dict()
                    doc_metadata['context'] = context_string
                    doc_metadata['source'] = source
                    doc_metadata['type'] = 'error'
                    doc_metadata['processed_at'] = datetime.now().isoformat()
                    
                    doc = Document.from_text(
                        content=chunk.content,
                        source=source,
                        **doc_metadata
                    )
                    documents.append(doc)
            else:
                # Fallback to regular text splitting
                metadata = self._extract_error_metadata(error_text)
                metadata.update({
                    'source': source,
                    'type': 'error',
                    'processed_at': datetime.now().isoformat()
                })
                
                # Create text splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
                # Split into chunks
                chunks = splitter.split_text(error_text)
                
                # Create documents
                documents = []
                for i, chunk in enumerate(chunks):
                    doc_metadata = {
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk)
                    }
                    
                    doc = Document.from_text(
                        content=chunk,
                        source=source,
                        **doc_metadata
                    )
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} documents from error log")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing error log: {e}")
            return []
    
    async def process_error_file(self, file_path: Path) -> List[Document]:
        """Process an error log file."""
        logger.info(f"Processing error file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return await self.process_error_log(content, str(file_path))
            
        except Exception as e:
            logger.error(f"Error processing error file {file_path}: {e}")
            return []
    
    def _extract_error_metadata(self, error_text: str) -> Dict[str, Any]:
        """Extract metadata from error text."""
        metadata = {
            'errors': [],
            'files': [],
            'languages': [],
            'error_types': [],
            'line_numbers': []
        }
        
        # Try to detect language and extract patterns
        for language, patterns in self.error_patterns.items():
            if self._matches_language_pattern(error_text, language):
                metadata['languages'].append(language)
                
                # Extract errors
                if 'exception' in patterns:
                    errors = re.findall(patterns['exception'], error_text)
                    for error_type, error_msg in errors:
                        metadata['errors'].append({
                            'type': error_type,
                            'message': error_msg,
                            'language': language
                        })
                        if error_type not in metadata['error_types']:
                            metadata['error_types'].append(error_type)
                
                # Extract file references
                if 'file_line' in patterns:
                    file_refs = re.findall(patterns['file_line'], error_text)
                    for file_path, line_num in file_refs:
                        metadata['files'].append({
                            'path': file_path,
                            'line': int(line_num)
                        })
                        metadata['line_numbers'].append(int(line_num))
        
        # Extract general information
        metadata.update({
            'line_count': error_text.count('\n') + 1,
            'char_count': len(error_text),
            'has_stack_trace': 'stack' in error_text.lower() or 'traceback' in error_text.lower(),
            'severity': self._estimate_severity(error_text)
        })
        
        return metadata
    
    def _matches_language_pattern(self, error_text: str, language: str) -> bool:
        """Check if error text matches a language pattern."""
        if language == 'python':
            return 'Traceback' in error_text or 'Error:' in error_text
        elif language == 'javascript':
            return 'Error:' in error_text and ('at ' in error_text or 'node_modules' in error_text)
        elif language == 'java':
            return 'Exception' in error_text and 'at ' in error_text
        
        return False
    
    def _estimate_severity(self, error_text: str) -> str:
        """Estimate error severity based on content."""
        error_text_lower = error_text.lower()
        
        # High severity indicators
        if any(keyword in error_text_lower for keyword in ['fatal', 'critical', 'segmentation fault', 'out of memory']):
            return 'high'
        
        # Medium severity indicators
        if any(keyword in error_text_lower for keyword in ['error', 'exception', 'failed', 'crashed']):
            return 'medium'
        
        # Low severity indicators
        if any(keyword in error_text_lower for keyword in ['warning', 'deprecated', 'info']):
            return 'low'
        
        return 'medium'  # Default
    
    def extract_error_context(self, error_text: str) -> Dict[str, Any]:
        """Extract contextual information from error text."""
        context = {
            'timestamp': None,
            'function_names': [],
            'variable_names': [],
            'code_snippets': [],
            'environment_info': []
        }
        
        # Extract timestamps
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
            r'\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \d{4}'
        ]
        
        for pattern in timestamp_patterns:
            matches = re.findall(pattern, error_text)
            if matches:
                context['timestamp'] = matches[0]
                break
        
        # Extract function names
        function_patterns = [
            r'def (\w+)\(',
            r'function (\w+)\(',
            r'at (\w+) \(',
            r'in (\w+) \('
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, error_text)
            context['function_names'].extend(matches)
        
        # Extract variable names from error messages
        variable_patterns = [
            r"'(\w+)' is not defined",
            r"'(\w+)' object has no attribute",
            r"Cannot read property '(\w+)' of",
            r"(\w+) is not a function"
        ]
        
        for pattern in variable_patterns:
            matches = re.findall(pattern, error_text)
            context['variable_names'].extend(matches)
        
        # Extract code snippets (lines that look like code)
        lines = error_text.split('\n')
        for line in lines:
            line = line.strip()
            if self._looks_like_code(line):
                context['code_snippets'].append(line)
        
        # Extract environment information
        env_patterns = [
            r'Python (\d+\.\d+\.\d+)',
            r'Node\.js (v\d+\.\d+\.\d+)',
            r'Java (\d+\.\d+\.\d+)',
            r'OS: (\w+)'
        ]
        
        for pattern in env_patterns:
            matches = re.findall(pattern, error_text)
            context['environment_info'].extend(matches)
        
        return context
    
    def _looks_like_code(self, line: str) -> bool:
        """Check if a line looks like code."""
        code_indicators = [
            '(', ')', '{', '}', '[', ']',
            '=', '==', '!=', '<=', '>=',
            'def ', 'class ', 'if ', 'for ', 'while ',
            'function ', 'var ', 'let ', 'const ',
            'public ', 'private ', 'protected '
        ]
        
        return any(indicator in line for indicator in code_indicators)
    
    def generate_error_summary(self, error_text: str) -> str:
        """Generate a summary of the error."""
        metadata = self._extract_error_metadata(error_text)
        context = self.extract_error_context(error_text)
        
        summary_parts = []
        
        # Main error
        if metadata['errors']:
            main_error = metadata['errors'][0]
            summary_parts.append(f"Error Type: {main_error['type']}")
            summary_parts.append(f"Message: {main_error['message']}")
        
        # Files involved
        if metadata['files']:
            file_info = metadata['files'][0]
            summary_parts.append(f"File: {file_info['path']} (line {file_info['line']})")
        
        # Language
        if metadata['languages']:
            summary_parts.append(f"Language: {metadata['languages'][0]}")
        
        # Severity
        summary_parts.append(f"Severity: {metadata['severity']}")
        
        # Context
        if context['function_names']:
            summary_parts.append(f"Function: {context['function_names'][0]}")
        
        return ' | '.join(summary_parts)
    
    async def process_directory(self, directory_path: Path) -> List[Document]:
        """Process all error log files in a directory."""
        documents = []
        
        # Common error log file patterns
        error_patterns = ['*.log', '*.error', '*.err', '*.crash']
        
        for pattern in error_patterns:
            for file_path in directory_path.rglob(pattern):
                if file_path.is_file():
                    file_documents = await self.process_error_file(file_path)
                    documents.extend(file_documents)
        
        return documents 