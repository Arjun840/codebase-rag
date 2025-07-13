"""Text splitting utilities for document processing."""

import re
from typing import List, Optional
from abc import ABC, abstractmethod


class TextSplitter(ABC):
    """Abstract base class for text splitters."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the text splitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass
    
    def create_chunks_with_overlap(self, text: str, separators: List[str]) -> List[str]:
        """Create overlapping chunks from text using separators."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # If we're not at the end, try to break at a separator
            if end < len(text):
                # Look for the best separator within the chunk
                best_split = end
                for separator in separators:
                    # Search backwards from the end of the chunk
                    for i in range(end - 1, start, -1):
                        if text[i:i+len(separator)] == separator:
                            best_split = i + len(separator)
                            break
                    if best_split < end:
                        break
                
                end = best_split
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks


class CharacterTextSplitter(TextSplitter):
    """Split text by characters with overlapping chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = "\n"):
        """Initialize character text splitter."""
        super().__init__(chunk_size, chunk_overlap)
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        """Split text by characters."""
        separators = [self.separator, " ", ""]
        return self.create_chunks_with_overlap(text, separators)


class RecursiveCharacterTextSplitter(TextSplitter):
    """Split text recursively by trying multiple separators."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        """Initialize recursive character text splitter."""
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text recursively."""
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text] if text else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Character-level splitting
            return self.create_chunks_with_overlap(text, [""])
        
        # Split by current separator
        parts = text.split(separator)
        
        # Rejoin parts that are too small
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If part is still too large, split it further
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_text_recursive(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class CodeTextSplitter(TextSplitter):
    """Split code text with awareness of code structure."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, language: str = "python"):
        """Initialize code text splitter."""
        super().__init__(chunk_size, chunk_overlap)
        self.language = language
        self.separators = self._get_separators_for_language(language)
    
    def _get_separators_for_language(self, language: str) -> List[str]:
        """Get appropriate separators for the programming language."""
        separators = {
            "python": ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
            "javascript": ["\nfunction ", "\nclass ", "\n\n", "\n", " ", ""],
            "java": ["\nclass ", "\npublic ", "\nprivate ", "\nprotected ", "\n\n", "\n", " ", ""],
            "cpp": ["\nclass ", "\nvoid ", "\nint ", "\n\n", "\n", " ", ""],
            "c": ["\nvoid ", "\nint ", "\n\n", "\n", " ", ""],
            "go": ["\nfunc ", "\ntype ", "\n\n", "\n", " ", ""],
            "rust": ["\nfn ", "\nstruct ", "\nimpl ", "\n\n", "\n", " ", ""],
        }
        
        return separators.get(language, ["\n\n", "\n", " ", ""])
    
    def split_text(self, text: str) -> List[str]:
        """Split code text."""
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split code text using separators."""
        if not separators:
            return [text] if text else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Character-level splitting
            return self.create_chunks_with_overlap(text, [""])
        
        # Split by current separator
        if separator.startswith("\n"):
            # For code separators, we want to keep the separator with the following content
            pattern = f"({re.escape(separator)})"
            parts = re.split(pattern, text)
            
            # Rejoin separator with following content
            rejoined_parts = []
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    rejoined_parts.append(parts[i] + parts[i + 1])
                else:
                    rejoined_parts.append(parts[i])
        else:
            parts = text.split(separator)
            rejoined_parts = parts
        
        # Process parts
        chunks = []
        current_chunk = ""
        
        for part in rejoined_parts:
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                if current_chunk and not separator.startswith("\n"):
                    current_chunk += separator + part
                else:
                    current_chunk += part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If part is still too large, split it further
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_text_recursive(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class MarkdownTextSplitter(TextSplitter):
    """Split markdown text with awareness of markdown structure."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize markdown text splitter."""
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ", "\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split markdown text."""
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split markdown text using separators."""
        if not separators:
            return [text] if text else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Character-level splitting
            return self.create_chunks_with_overlap(text, [""])
        
        # Split by current separator
        if separator.startswith("\n#"):
            # For markdown headers, we want to keep the separator with the following content
            pattern = f"({re.escape(separator)})"
            parts = re.split(pattern, text)
            
            # Rejoin separator with following content
            rejoined_parts = []
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    rejoined_parts.append(parts[i] + parts[i + 1])
                else:
                    rejoined_parts.append(parts[i])
        else:
            parts = text.split(separator)
            rejoined_parts = parts
        
        # Process parts
        chunks = []
        current_chunk = ""
        
        for part in rejoined_parts:
            if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                if current_chunk and not separator.startswith("\n#"):
                    current_chunk += separator + part
                else:
                    current_chunk += part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If part is still too large, split it further
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_text_recursive(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks 