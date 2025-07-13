"""Document representation for the RAG system."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import uuid
import hashlib


@dataclass
class Document:
    """Represents a document in the RAG system."""
    
    content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if self.id is None:
            # Generate a unique ID based on content hash
            content_hash = hashlib.md5(self.content.encode()).hexdigest()
            self.id = f"doc_{content_hash[:8]}"
    
    @classmethod
    def from_file(cls, file_path: str, content: str, **metadata) -> "Document":
        """Create a document from a file."""
        return cls(
            content=content,
            metadata={
                "source": file_path,
                "type": "file",
                **metadata
            }
        )
    
    @classmethod
    def from_code(cls, file_path: str, content: str, language: str, **metadata) -> "Document":
        """Create a document from code content."""
        return cls(
            content=content,
            metadata={
                "source": file_path,
                "type": "code",
                "language": language,
                **metadata
            }
        )
    
    @classmethod
    def from_text(cls, content: str, source: str = "unknown", **metadata) -> "Document":
        """Create a document from text content."""
        return cls(
            content=content,
            metadata={
                "source": source,
                "type": "text",
                **metadata
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary representation."""
        return cls(
            id=data.get("id"),
            content=data["content"],
            metadata=data["metadata"]
        )
    
    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Document(id={self.id}, source={self.metadata.get('source', 'unknown')}, length={len(self.content)})"
    
    def __repr__(self) -> str:
        """Repr representation."""
        return self.__str__() 