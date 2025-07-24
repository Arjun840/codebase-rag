"""Document processor for handling documentation files."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import yaml
from xml.etree import ElementTree as ET

from ..utils import Document, TextSplitter, MarkdownTextSplitter, RecursiveCharacterTextSplitter, SmartTextSplitter, MetadataExtractor, MetadataContext

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documentation files for indexing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, use_smart_chunking: bool = True):
        """Initialize the document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_smart_chunking = use_smart_chunking
        
        # Initialize smart text splitter and metadata extractor
        self.smart_splitter = SmartTextSplitter(
            min_chunk_words=100,
            max_chunk_words=300,
            overlap_words=20,
            preserve_structure=True
        )
        
        self.metadata_extractor = MetadataExtractor()
        
        # File type mapping
        self.processor_map = {
            '.md': self._process_markdown,
            '.txt': self._process_text,
            '.json': self._process_json,
            '.yml': self._process_yaml,
            '.yaml': self._process_yaml,
            '.xml': self._process_xml,
            '.html': self._process_html,
            '.css': self._process_css,
            '.scss': self._process_scss,
            '.rst': self._process_rst,
        }
    
    async def process_file(self, file_path: Path) -> List[Document]:
        """Process a single documentation file."""
        logger.info(f"Processing document file: {file_path}")
        
        try:
            extension = file_path.suffix.lower()
            
            # Get appropriate processor
            processor = self.processor_map.get(extension, self._process_text)
            
            # Process the file
            documents = await processor(file_path)
            
            logger.info(f"Created {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    async def _process_markdown(self, file_path: Path) -> List[Document]:
        """Process a Markdown file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if self.use_smart_chunking:
            # Use smart text splitter for better topic-focused chunking
            smart_chunks = self.smart_splitter.split_markdown(content, str(file_path))
            
            # Create documents from smart chunks
            documents = []
            for i, chunk in enumerate(smart_chunks):
                # Extract comprehensive metadata
                metadata_context = self.metadata_extractor.extract_metadata(
                    chunk.content, 
                    str(file_path),
                    chunk_info={
                        'section_title': chunk.title,
                        'section_path': chunk.section_path,
                        'section_level': chunk.metadata.get('section_level', 0),
                        'chunk_type': chunk.focus.value,
                        'chunk_index': i,
                        'total_chunks': len(smart_chunks),
                        'word_count': chunk.word_count,
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line
                    }
                )
                
                # Create context string for better LLM understanding
                context_string = self.metadata_extractor.create_context_string(metadata_context)
                
                # Convert metadata to dict and add context
                doc_metadata = metadata_context.to_dict()
                doc_metadata['context'] = context_string
                
                doc = Document.from_file(
                    file_path=str(file_path),
                    content=chunk.content,
                    **doc_metadata
                )
                documents.append(doc)
        else:
            # Fallback to regular markdown splitter
            splitter = MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Split into chunks
            chunks = splitter.split_text(content)
            
            # Extract basic metadata
            metadata = self._extract_markdown_metadata(content)
            metadata.update({
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': len(content),
                'file_type': 'markdown',
                'extension': file_path.suffix,
            })
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                }
                
                doc = Document.from_file(
                    file_path=str(file_path),
                    content=chunk,
                    **doc_metadata
                )
                documents.append(doc)
        
        return documents
    
    async def _process_text(self, file_path: Path) -> List[Document]:
        """Process a plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Split into chunks
        chunks = splitter.split_text(content)
        
        # Extract metadata
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': len(content),
            'file_type': 'text',
            'extension': file_path.suffix,
            'line_count': content.count('\n') + 1,
        }
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk)
            }
            
            doc = Document.from_file(
                file_path=str(file_path),
                content=chunk,
                **doc_metadata
            )
            documents.append(doc)
        
        return documents
    
    async def _process_json(self, file_path: Path) -> List[Document]:
        """Process a JSON file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        try:
            data = json.loads(content)
            
            # Convert JSON to readable text
            readable_content = self._json_to_text(data)
            
            # Create text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Split into chunks
            chunks = splitter.split_text(readable_content)
            
            # Extract metadata
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': len(content),
                'file_type': 'json',
                'extension': file_path.suffix,
                'json_keys': ', '.join(data.keys()) if isinstance(data, dict) else '',
                'json_type': type(data).__name__
            }
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                }
                
                doc = Document.from_file(
                    file_path=str(file_path),
                    content=chunk,
                    **doc_metadata
                )
                documents.append(doc)
            
            return documents
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in file {file_path}, processing as text")
            return await self._process_text(file_path)
    
    async def _process_yaml(self, file_path: Path) -> List[Document]:
        """Process a YAML file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        try:
            data = yaml.safe_load(content)
            
            # Convert YAML to readable text
            readable_content = self._yaml_to_text(data)
            
            # Create text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Split into chunks
            chunks = splitter.split_text(readable_content)
            
            # Extract metadata
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': len(content),
                'file_type': 'yaml',
                'extension': file_path.suffix,
                'yaml_keys': ', '.join(data.keys()) if isinstance(data, dict) else '',
                'yaml_type': type(data).__name__
            }
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                }
                
                doc = Document.from_file(
                    file_path=str(file_path),
                    content=chunk,
                    **doc_metadata
                )
                documents.append(doc)
            
            return documents
            
        except yaml.YAMLError:
            logger.warning(f"Invalid YAML in file {file_path}, processing as text")
            return await self._process_text(file_path)
    
    async def _process_xml(self, file_path: Path) -> List[Document]:
        """Process an XML file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        try:
            root = ET.fromstring(content)
            
            # Convert XML to readable text
            readable_content = self._xml_to_text(root)
            
            # Create text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Split into chunks
            chunks = splitter.split_text(readable_content)
            
            # Extract metadata
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': len(content),
                'file_type': 'xml',
                'extension': file_path.suffix,
                'root_tag': root.tag,
                'xml_namespaces': ', '.join(root.nsmap.keys()) if hasattr(root, 'nsmap') else ''
            }
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)
                }
                
                doc = Document.from_file(
                    file_path=str(file_path),
                    content=chunk,
                    **doc_metadata
                )
                documents.append(doc)
            
            return documents
            
        except ET.ParseError:
            logger.warning(f"Invalid XML in file {file_path}, processing as text")
            return await self._process_text(file_path)
    
    async def _process_html(self, file_path: Path) -> List[Document]:
        """Process an HTML file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract text content from HTML
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text()
            
            # Extract metadata
            title = soup.title.string if soup.title else None
            meta_tags = soup.find_all('meta')
            
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': len(content),
                'file_type': 'html',
                'extension': file_path.suffix,
                'title': title,
                'meta_tags': ', '.join(tag.get('name', tag.get('property', '')) for tag in meta_tags)
            }
            
        except ImportError:
            logger.warning("BeautifulSoup not installed, processing HTML as text")
            text_content = content
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': len(content),
                'file_type': 'html',
                'extension': file_path.suffix,
            }
        
        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Split into chunks
        chunks = splitter.split_text(text_content)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk)
            }
            
            doc = Document.from_file(
                file_path=str(file_path),
                content=chunk,
                **doc_metadata
            )
            documents.append(doc)
        
        return documents
    
    async def _process_css(self, file_path: Path) -> List[Document]:
        """Process a CSS file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract CSS metadata
        metadata = self._extract_css_metadata(content)
        metadata.update({
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size': len(content),
            'file_type': 'css',
            'extension': file_path.suffix,
        })
        
        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
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
            
            doc = Document.from_file(
                file_path=str(file_path),
                content=chunk,
                **doc_metadata
            )
            documents.append(doc)
        
        return documents
    
    async def _process_scss(self, file_path: Path) -> List[Document]:
        """Process a SCSS file."""
        # Process similar to CSS
        return await self._process_css(file_path)
    
    async def _process_rst(self, file_path: Path) -> List[Document]:
        """Process a reStructuredText file."""
        # Process similar to markdown
        return await self._process_text(file_path)
    
    def _extract_markdown_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from Markdown content."""
        metadata = {
            'headers': [],
            'links': [],
            'images': [],
            'code_blocks': []
        }
        
        import re
        
        # Extract headers
        header_pattern = r'^(#{1,6})\s+(.*)'
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2)
            metadata['headers'].append({'level': level, 'text': text})
        
        # Extract links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        metadata['links'] = re.findall(link_pattern, content)
        
        # Extract images
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        metadata['images'] = re.findall(image_pattern, content)
        
        # Extract code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2)
            metadata['code_blocks'].append({'language': language, 'code': code})
        
        # Convert lists to strings for ChromaDB compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                if key == 'headers':
                    metadata[key] = ', '.join(f"H{item['level']}: {item['text']}" for item in value)
                elif key == 'links':
                    metadata[key] = ', '.join(f"{text} -> {url}" for text, url in value)
                elif key == 'images':
                    metadata[key] = ', '.join(f"{alt} ({url})" for alt, url in value)
                elif key == 'code_blocks':
                    metadata[key] = ', '.join(f"{item['language']}: {len(item['code'])} chars" for item in value)
                else:
                    metadata[key] = ', '.join(str(item) for item in value)
        
        return metadata
    
    def _extract_css_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from CSS content."""
        metadata = {
            'selectors': [],
            'properties': [],
            'media_queries': []
        }
        
        import re
        
        # Extract selectors (simple pattern)
        selector_pattern = r'([^{]+)\s*{'
        metadata['selectors'] = re.findall(selector_pattern, content)
        
        # Extract properties
        property_pattern = r'([^:]+):\s*([^;]+);'
        metadata['properties'] = re.findall(property_pattern, content)
        
        # Extract media queries
        media_pattern = r'@media\s+([^{]+)'
        metadata['media_queries'] = re.findall(media_pattern, content)
        
        # Convert lists to strings for ChromaDB compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = ', '.join(str(item) for item in value)
        
        return metadata
    
    def _json_to_text(self, data: Any, indent: int = 0) -> str:
        """Convert JSON data to readable text."""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{'  ' * indent}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{'  ' * indent}{key}: {value}")
            return '\n'.join(lines)
        elif isinstance(data, list):
            lines = []
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(self._json_to_text(item, indent))
                else:
                    lines.append(f"{'  ' * indent}- {item}")
            return '\n'.join(lines)
        else:
            return str(data)
    
    def _yaml_to_text(self, data: Any, indent: int = 0) -> str:
        """Convert YAML data to readable text."""
        # Similar to JSON conversion
        return self._json_to_text(data, indent)
    
    def _xml_to_text(self, element: ET.Element, indent: int = 0) -> str:
        """Convert XML element to readable text."""
        lines = []
        
        # Add element tag and attributes
        attrs = ' '.join([f'{k}="{v}"' for k, v in element.attrib.items()])
        tag_line = f"{'  ' * indent}{element.tag}"
        if attrs:
            tag_line += f" ({attrs})"
        lines.append(tag_line)
        
        # Add text content
        if element.text and element.text.strip():
            lines.append(f"{'  ' * (indent + 1)}{element.text.strip()}")
        
        # Add children
        for child in element:
            lines.append(self._xml_to_text(child, indent + 1))
        
        return '\n'.join(lines)
    
    async def process_directory(self, directory_path: Path) -> List[Document]:
        """Process all documentation files in a directory."""
        documents = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.processor_map:
                file_documents = await self.process_file(file_path)
                documents.extend(file_documents)
        
        return documents 