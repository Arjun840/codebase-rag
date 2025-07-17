"""Smart text splitter for topic-focused documentation chunking."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ChunkFocus(Enum):
    """Types of chunk focus for different content types."""
    PARAGRAPH = "paragraph"
    SECTION = "section"
    TOPIC = "topic"
    ISSUE = "issue"
    QUESTION = "question"
    CODE_BLOCK = "code_block"
    LIST = "list"
    TABLE = "table"


@dataclass
class SmartChunk:
    """Represents a semantically meaningful chunk of text."""
    content: str
    focus: ChunkFocus
    title: Optional[str] = None
    section_path: Optional[str] = None  # e.g., "Introduction > Getting Started"
    word_count: int = 0
    metadata: Dict[str, Any] = None
    start_line: int = 0
    end_line: int = 0
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.metadata is None:
            self.metadata = {}
        
        if self.word_count == 0:
            self.word_count = len(self.content.split())


class SmartTextSplitter:
    """Smart text splitter that creates topic-focused chunks."""
    
    def __init__(self, 
                 min_chunk_words: int = 100,
                 max_chunk_words: int = 300,
                 overlap_words: int = 20,
                 preserve_structure: bool = True):
        """Initialize the smart text splitter."""
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words
        self.preserve_structure = preserve_structure
        
        # Patterns for different content types
        self.patterns = {
            'markdown_header': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
            'section_break': re.compile(r'\n\s*\n\s*\n', re.MULTILINE),
            'paragraph_break': re.compile(r'\n\s*\n', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
            'inline_code': re.compile(r'`[^`]+`'),
            'list_item': re.compile(r'^[-*+]\s+', re.MULTILINE),
            'numbered_list': re.compile(r'^\d+\.\s+', re.MULTILINE),
            'quote': re.compile(r'^>\s+', re.MULTILINE),
            'table_row': re.compile(r'\|[^|\n]+\|', re.MULTILINE),
        }
    
    def split_markdown(self, text: str, file_path: Optional[str] = None) -> List[SmartChunk]:
        """Split markdown text into topic-focused chunks."""
        chunks = []
        
        # Extract and preserve code blocks first
        code_blocks = []
        def code_placeholder(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        processed_text = self.patterns['code_block'].sub(code_placeholder, text)
        
        # Parse markdown structure
        sections = self._parse_markdown_structure(processed_text)
        
        # Create chunks from sections
        for section in sections:
            section_chunks = self._chunk_section(section, code_blocks, file_path)
            chunks.extend(section_chunks)
        
        return chunks
    
    def split_documentation(self, text: str, file_path: Optional[str] = None) -> List[SmartChunk]:
        """Split general documentation into topic-focused chunks."""
        chunks = []
        
        # Split by major paragraph breaks first
        sections = self.patterns['section_break'].split(text)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # Further split by paragraphs if section is too long
            paragraphs = self.patterns['paragraph_break'].split(section)
            
            current_chunk = ""
            current_title = None
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Check if this looks like a title/header
                if self._is_title(paragraph):
                    # Save previous chunk if exists
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            current_chunk, 
                            ChunkFocus.TOPIC, 
                            current_title,
                            file_path
                        ))
                    
                    current_chunk = paragraph
                    current_title = paragraph
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                
                # Check if chunk is getting too long
                if len(current_chunk.split()) > self.max_chunk_words:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        ChunkFocus.TOPIC,
                        current_title,
                        file_path
                    ))
                    current_chunk = ""
                    current_title = None
            
            # Add final chunk
            if current_chunk:
                chunks.append(self._create_chunk(
                    current_chunk,
                    ChunkFocus.TOPIC,
                    current_title,
                    file_path
                ))
        
        return chunks
    
    def split_error_logs(self, text: str, file_path: Optional[str] = None) -> List[SmartChunk]:
        """Split error logs into distinct issue chunks."""
        chunks = []
        
        # Common error patterns
        error_patterns = [
            r'ERROR\s*:',
            r'Exception\s*:',
            r'Traceback\s*\(most recent call last\):',
            r'FAILED\s*:',
            r'CRITICAL\s*:',
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}.*ERROR',
            r'File\s*"[^"]+",\s*line\s*\d+',
        ]
        
        # Split by error boundaries
        error_boundary = re.compile('|'.join(error_patterns), re.MULTILINE | re.IGNORECASE)
        
        current_error = ""
        error_title = None
        
        lines = text.split('\n')
        
        for line in lines:
            if error_boundary.search(line):
                # Save previous error if exists
                if current_error.strip():
                    chunks.append(self._create_chunk(
                        current_error,
                        ChunkFocus.ISSUE,
                        error_title,
                        file_path
                    ))
                
                # Start new error
                current_error = line
                error_title = self._extract_error_title(line)
            else:
                if current_error:
                    current_error += "\n" + line
                elif line.strip():  # Start collecting if we have content
                    current_error = line
        
        # Add final error
        if current_error.strip():
            chunks.append(self._create_chunk(
                current_error,
                ChunkFocus.ISSUE,
                error_title,
                file_path
            ))
        
        return chunks
    
    def split_qa_content(self, text: str, file_path: Optional[str] = None) -> List[SmartChunk]:
        """Split Q&A content into question-answer pairs."""
        chunks = []
        
        # Patterns for Q&A content
        question_patterns = [
            r'^Q\s*:',
            r'^Question\s*:',
            r'^#\s*Q\d*\s*:',
            r'^\d+\.\s*',
            r'^-\s*',
        ]
        
        answer_patterns = [
            r'^A\s*:',
            r'^Answer\s*:',
            r'^Solution\s*:',
            r'^Response\s*:',
        ]
        
        question_regex = re.compile('|'.join(question_patterns), re.MULTILINE | re.IGNORECASE)
        answer_regex = re.compile('|'.join(answer_patterns), re.MULTILINE | re.IGNORECASE)
        
        # Split by paragraphs first
        paragraphs = self.patterns['paragraph_break'].split(text)
        
        current_qa = ""
        current_question = None
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if this is a new question
            if question_regex.match(paragraph):
                # Save previous Q&A if exists
                if current_qa:
                    chunks.append(self._create_chunk(
                        current_qa,
                        ChunkFocus.QUESTION,
                        current_question,
                        file_path
                    ))
                
                # Start new Q&A
                current_qa = paragraph
                current_question = self._extract_question_title(paragraph)
            else:
                # Add to current Q&A
                if current_qa:
                    current_qa += "\n\n" + paragraph
                else:
                    current_qa = paragraph
                
                # Extract question title if not set
                if not current_question:
                    current_question = self._extract_question_title(paragraph)
        
        # Add final Q&A
        if current_qa:
            chunks.append(self._create_chunk(
                current_qa,
                ChunkFocus.QUESTION,
                current_question,
                file_path
            ))
        
        return chunks
    
    def _parse_markdown_structure(self, text: str) -> List[Dict[str, Any]]:
        """Parse markdown structure into sections."""
        sections = []
        lines = text.split('\n')
        
        current_section = {
            'title': None,
            'level': 0,
            'content': '',
            'line_start': 0,
            'line_end': 0,
            'path': []
        }
        
        section_stack = []
        
        for i, line in enumerate(lines):
            header_match = self.patterns['markdown_header'].match(line)
            
            if header_match:
                # Save previous section
                if current_section['content'].strip():
                    current_section['line_end'] = i
                    sections.append(current_section.copy())
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Update section stack
                section_stack = [s for s in section_stack if s['level'] < level]
                section_stack.append({'level': level, 'title': title})
                
                current_section = {
                    'title': title,
                    'level': level,
                    'content': line,
                    'line_start': i,
                    'line_end': i,
                    'path': [s['title'] for s in section_stack]
                }
            else:
                current_section['content'] += '\n' + line
        
        # Add final section
        if current_section['content'].strip():
            current_section['line_end'] = len(lines)
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(self, section: Dict[str, Any], code_blocks: List[str], file_path: Optional[str]) -> List[SmartChunk]:
        """Chunk a single section."""
        chunks = []
        content = section['content']
        
        # Restore code blocks
        for i, code_block in enumerate(code_blocks):
            content = content.replace(f'__CODE_BLOCK_{i}__', code_block)
        
        # Split by paragraphs
        paragraphs = self.patterns['paragraph_break'].split(content)
        
        current_chunk = ""
        current_words = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_words = len(paragraph.split())
            
            # Check if adding this paragraph would exceed max words
            if current_words + paragraph_words > self.max_chunk_words and current_chunk:
                # Create chunk
                chunk = SmartChunk(
                    content=current_chunk,
                    focus=ChunkFocus.SECTION,
                    title=section['title'],
                    section_path=' > '.join(section['path']),
                    word_count=current_words,
                    start_line=section['line_start'],
                    end_line=section['line_end'],
                    metadata={
                        'file_path': file_path,
                        'section_level': section['level'],
                        'section_title': section['title'],
                        'full_path': section['path']
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                current_words = len(current_chunk.split())
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                
                current_words += paragraph_words
        
        # Add final chunk
        if current_chunk and current_words >= self.min_chunk_words:
            chunk = SmartChunk(
                content=current_chunk,
                focus=ChunkFocus.SECTION,
                title=section['title'],
                section_path=' > '.join(section['path']),
                word_count=current_words,
                start_line=section['line_start'],
                end_line=section['line_end'],
                metadata={
                    'file_path': file_path,
                    'section_level': section['level'],
                    'section_title': section['title'],
                    'full_path': section['path']
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, focus: ChunkFocus, title: Optional[str], file_path: Optional[str]) -> SmartChunk:
        """Create a smart chunk with metadata."""
        return SmartChunk(
            content=content,
            focus=focus,
            title=title,
            word_count=len(content.split()),
            metadata={
                'file_path': file_path,
                'focus_type': focus.value,
                'title': title
            }
        )
    
    def _is_title(self, text: str) -> bool:
        """Check if text looks like a title or header."""
        # Short line, no ending punctuation (except :), possibly capitalized
        if len(text) > 100:
            return False
        
        if text.endswith('.') or text.endswith(',') or text.endswith(';'):
            return False
        
        # Check for title patterns
        title_patterns = [
            r'^[A-Z][^.!?]*:?\s*$',  # Starts with capital, no sentence ending
            r'^[A-Z\s]+$',           # All caps
            r'^\d+\.\s*[A-Z]',       # Numbered heading
            r'^[A-Z][^.!?]*\s*$',    # Single sentence without ending punctuation
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _extract_error_title(self, line: str) -> str:
        """Extract error title from error line."""
        # Try to extract meaningful error description
        error_patterns = [
            r'ERROR\s*:\s*(.+)',
            r'Exception\s*:\s*(.+)',
            r'FAILED\s*:\s*(.+)',
            r'CRITICAL\s*:\s*(.+)',
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to first 50 characters
        return line[:50].strip()
    
    def _extract_question_title(self, text: str) -> str:
        """Extract question title from Q&A text."""
        # Remove Q&A prefixes
        clean_text = re.sub(r'^(Q\s*:|Question\s*:|#\s*Q\d*\s*:|\d+\.\s*|-\s*)', '', text, flags=re.IGNORECASE)
        
        # Take first sentence or first 50 characters
        sentences = clean_text.split('.')
        if sentences:
            return sentences[0].strip()
        
        return clean_text[:50].strip()
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        words = text.split()
        if len(words) <= self.overlap_words:
            return text
        
        return ' '.join(words[-self.overlap_words:])
    
    def split_by_content_type(self, text: str, content_type: str, file_path: Optional[str] = None) -> List[SmartChunk]:
        """Split text based on content type."""
        if content_type == 'markdown':
            return self.split_markdown(text, file_path)
        elif content_type == 'documentation':
            return self.split_documentation(text, file_path)
        elif content_type == 'error_log':
            return self.split_error_logs(text, file_path)
        elif content_type == 'qa':
            return self.split_qa_content(text, file_path)
        else:
            return self.split_documentation(text, file_path)  # Default
    
    def get_content_type(self, text: str, file_path: Optional[str] = None) -> str:
        """Automatically detect content type."""
        if file_path:
            file_path = Path(file_path)
            if file_path.suffix.lower() == '.md':
                return 'markdown'
            elif file_path.suffix.lower() in ['.log', '.txt'] and 'error' in file_path.name.lower():
                return 'error_log'
        
        # Check content patterns
        if self.patterns['markdown_header'].search(text):
            return 'markdown'
        elif re.search(r'(ERROR|Exception|Traceback)', text, re.IGNORECASE):
            return 'error_log'
        elif re.search(r'(Q\s*:|Question\s*:|A\s*:|Answer\s*:)', text, re.IGNORECASE):
            return 'qa'
        else:
            return 'documentation' 