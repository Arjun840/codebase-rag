"""Enhanced AST-based chunking for code files."""

import ast
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of code chunks."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"
    CONSTANT = "constant"
    COMMENT = "comment"
    MIXED = "mixed"


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    start_line: int
    end_line: int
    chunk_type: ChunkType
    name: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = None
    parameters: List[str] = None
    return_type: Optional[str] = None
    complexity: int = 0
    dependencies: List[str] = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.decorators is None:
            self.decorators = []
        if self.parameters is None:
            self.parameters = []
        if self.dependencies is None:
            self.dependencies = []


class EnhancedASTChunker:
    """Enhanced AST-based code chunker."""
    
    def __init__(self, 
                 max_chunk_size: int = 1000,
                 min_chunk_size: int = 100,
                 preserve_context: bool = True,
                 include_imports: bool = True):
        """Initialize the AST chunker."""
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.preserve_context = preserve_context
        self.include_imports = include_imports
    
    def chunk_python_file(self, file_path: Union[str, Path]) -> List[CodeChunk]:
        """Chunk a Python file using AST analysis."""
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.chunk_python_code(content)
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
    
    def chunk_python_code(self, content: str) -> List[CodeChunk]:
        """Chunk Python code using AST analysis."""
        try:
            tree = ast.parse(content)
            lines = content.splitlines()
            
            chunks = []
            
            # Extract imports first
            if self.include_imports:
                import_chunks = self._extract_imports(tree, lines)
                chunks.extend(import_chunks)
            
            # Extract top-level elements
            top_level_chunks = self._extract_top_level_elements(tree, lines)
            chunks.extend(top_level_chunks)
            
            # Handle remaining code (module-level code)
            module_chunks = self._extract_module_level_code(tree, lines, chunks)
            chunks.extend(module_chunks)
            
            # Post-process chunks
            chunks = self._post_process_chunks(chunks, lines)
            
            return chunks
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in Python code: {e}")
            return self._fallback_chunk(content)
        except Exception as e:
            logger.error(f"Error parsing Python code: {e}")
            return self._fallback_chunk(content)
    
    def _extract_imports(self, tree: ast.AST, lines: List[str]) -> List[CodeChunk]:
        """Extract import statements."""
        chunks = []
        import_lines = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append(node.lineno)
        
        if import_lines:
            import_lines.sort()
            
            # Group consecutive imports
            groups = []
            current_group = [import_lines[0]]
            
            for line_num in import_lines[1:]:
                if line_num - current_group[-1] <= 2:  # Allow for blank lines
                    current_group.append(line_num)
                else:
                    groups.append(current_group)
                    current_group = [line_num]
            
            groups.append(current_group)
            
            # Create chunks for each group
            for group in groups:
                start_line = group[0]
                end_line = group[-1]
                
                # Expand to include any following blank lines
                while end_line < len(lines) and lines[end_line].strip() == '':
                    end_line += 1
                
                content = '\n'.join(lines[start_line-1:end_line])
                
                chunk = CodeChunk(
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=ChunkType.IMPORT,
                    name="imports"
                )
                chunks.append(chunk)
        
        return chunks
    
    def _extract_top_level_elements(self, tree: ast.AST, lines: List[str]) -> List[CodeChunk]:
        """Extract top-level functions and classes."""
        chunks = []
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                chunk = self._create_function_chunk(node, lines)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._create_class_chunk(node, lines)
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, ast.AsyncFunctionDef):
                chunk = self._create_async_function_chunk(node, lines)
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _create_function_chunk(self, node: ast.FunctionDef, lines: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for a function."""
        try:
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Include decorators
            if node.decorator_list:
                start_line = node.decorator_list[0].lineno
            
            content = '\n'.join(lines[start_line-1:end_line])
            
            # Extract metadata
            decorators = [ast.unparse(d) for d in node.decorator_list] if node.decorator_list else []
            parameters = [arg.arg for arg in node.args.args]
            docstring = ast.get_docstring(node)
            
            # Calculate complexity (simple metric)
            complexity = self._calculate_complexity(node)
            
            chunk = CodeChunk(
                content=content,
                start_line=start_line,
                end_line=end_line,
                chunk_type=ChunkType.FUNCTION,
                name=node.name,
                docstring=docstring,
                decorators=decorators,
                parameters=parameters,
                complexity=complexity
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating function chunk: {e}")
            return None
    
    def _create_class_chunk(self, node: ast.ClassDef, lines: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for a class."""
        try:
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Include decorators
            if node.decorator_list:
                start_line = node.decorator_list[0].lineno
            
            # Check if class is too large
            class_content = '\n'.join(lines[start_line-1:end_line])
            
            if len(class_content) > self.max_chunk_size:
                # Split large classes into method chunks
                return self._split_large_class(node, lines)
            
            # Extract metadata
            decorators = [ast.unparse(d) for d in node.decorator_list] if node.decorator_list else []
            docstring = ast.get_docstring(node)
            
            # Get base classes
            base_classes = [ast.unparse(base) for base in node.bases] if node.bases else []
            
            chunk = CodeChunk(
                content=class_content,
                start_line=start_line,
                end_line=end_line,
                chunk_type=ChunkType.CLASS,
                name=node.name,
                docstring=docstring,
                decorators=decorators,
                dependencies=base_classes
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating class chunk: {e}")
            return None
    
    def _create_async_function_chunk(self, node: ast.AsyncFunctionDef, lines: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for an async function."""
        try:
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Include decorators
            if node.decorator_list:
                start_line = node.decorator_list[0].lineno
            
            content = '\n'.join(lines[start_line-1:end_line])
            
            # Extract metadata
            decorators = [ast.unparse(d) for d in node.decorator_list] if node.decorator_list else []
            parameters = [arg.arg for arg in node.args.args]
            docstring = ast.get_docstring(node)
            
            # Calculate complexity
            complexity = self._calculate_complexity(node)
            
            chunk = CodeChunk(
                content=content,
                start_line=start_line,
                end_line=end_line,
                chunk_type=ChunkType.FUNCTION,
                name=f"async {node.name}",
                docstring=docstring,
                decorators=decorators,
                parameters=parameters,
                complexity=complexity
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating async function chunk: {e}")
            return None
    
    def _split_large_class(self, node: ast.ClassDef, lines: List[str]) -> CodeChunk:
        """Split a large class into method chunks."""
        chunks = []
        
        # Create class header chunk
        class_start = node.lineno
        if node.decorator_list:
            class_start = node.decorator_list[0].lineno
        
        # Find first method or end of class definition
        first_method_line = None
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_method_line = child.lineno
                break
        
        if first_method_line:
            header_end = first_method_line - 1
        else:
            header_end = node.end_lineno or node.lineno
        
        # Class header chunk
        header_content = '\n'.join(lines[class_start-1:header_end])
        
        chunks.append(CodeChunk(
            content=header_content,
            start_line=class_start,
            end_line=header_end,
            chunk_type=ChunkType.CLASS,
            name=f"{node.name} (header)",
            docstring=ast.get_docstring(node)
        ))
        
        # Method chunks
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method_chunk = self._create_method_chunk(child, lines, node.name)
                if method_chunk:
                    chunks.append(method_chunk)
            elif isinstance(child, ast.AsyncFunctionDef):
                method_chunk = self._create_async_method_chunk(child, lines, node.name)
                if method_chunk:
                    chunks.append(method_chunk)
        
        return chunks
    
    def _create_method_chunk(self, node: ast.FunctionDef, lines: List[str], class_name: str) -> Optional[CodeChunk]:
        """Create a chunk for a method."""
        try:
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Include decorators
            if node.decorator_list:
                start_line = node.decorator_list[0].lineno
            
            content = '\n'.join(lines[start_line-1:end_line])
            
            # Extract metadata
            decorators = [ast.unparse(d) for d in node.decorator_list] if node.decorator_list else []
            parameters = [arg.arg for arg in node.args.args]
            docstring = ast.get_docstring(node)
            
            # Calculate complexity
            complexity = self._calculate_complexity(node)
            
            chunk = CodeChunk(
                content=content,
                start_line=start_line,
                end_line=end_line,
                chunk_type=ChunkType.METHOD,
                name=f"{class_name}.{node.name}",
                docstring=docstring,
                decorators=decorators,
                parameters=parameters,
                complexity=complexity
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating method chunk: {e}")
            return None
    
    def _create_async_method_chunk(self, node: ast.AsyncFunctionDef, lines: List[str], class_name: str) -> Optional[CodeChunk]:
        """Create a chunk for an async method."""
        try:
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Include decorators
            if node.decorator_list:
                start_line = node.decorator_list[0].lineno
            
            content = '\n'.join(lines[start_line-1:end_line])
            
            # Extract metadata
            decorators = [ast.unparse(d) for d in node.decorator_list] if node.decorator_list else []
            parameters = [arg.arg for arg in node.args.args]
            docstring = ast.get_docstring(node)
            
            # Calculate complexity
            complexity = self._calculate_complexity(node)
            
            chunk = CodeChunk(
                content=content,
                start_line=start_line,
                end_line=end_line,
                chunk_type=ChunkType.METHOD,
                name=f"{class_name}.async {node.name}",
                docstring=docstring,
                decorators=decorators,
                parameters=parameters,
                complexity=complexity
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating async method chunk: {e}")
            return None
    
    def _extract_module_level_code(self, tree: ast.AST, lines: List[str], existing_chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Extract module-level code that's not part of functions or classes."""
        chunks = []
        
        # Get line numbers that are already covered
        covered_lines = set()
        for chunk in existing_chunks:
            covered_lines.update(range(chunk.start_line, chunk.end_line + 1))
        
        # Find uncovered lines
        current_chunk_lines = []
        current_start = None
        
        for i, line in enumerate(lines, 1):
            if i not in covered_lines and line.strip():
                if current_start is None:
                    current_start = i
                current_chunk_lines.append(i)
            else:
                if current_chunk_lines:
                    # Create chunk from accumulated lines
                    content = '\n'.join(lines[current_start-1:current_chunk_lines[-1]])
                    
                    if len(content.strip()) > 0:
                        chunk = CodeChunk(
                            content=content,
                            start_line=current_start,
                            end_line=current_chunk_lines[-1],
                            chunk_type=ChunkType.MIXED,
                            name="module_level"
                        )
                        chunks.append(chunk)
                    
                    current_chunk_lines = []
                    current_start = None
        
        # Handle remaining lines
        if current_chunk_lines:
            content = '\n'.join(lines[current_start-1:current_chunk_lines[-1]])
            
            if len(content.strip()) > 0:
                chunk = CodeChunk(
                    content=content,
                    start_line=current_start,
                    end_line=current_chunk_lines[-1],
                    chunk_type=ChunkType.MIXED,
                    name="module_level"
                )
                chunks.append(chunk)
        
        return chunks
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _post_process_chunks(self, chunks: List[CodeChunk], lines: List[str]) -> List[CodeChunk]:
        """Post-process chunks to handle overlaps and size constraints."""
        # Sort chunks by start line
        chunks.sort(key=lambda x: x.start_line)
        
        # Handle overlaps
        processed_chunks = []
        for chunk in chunks:
            if not processed_chunks:
                processed_chunks.append(chunk)
            else:
                last_chunk = processed_chunks[-1]
                
                # Check for overlap
                if chunk.start_line <= last_chunk.end_line:
                    # Merge overlapping chunks
                    merged_content = '\n'.join(lines[last_chunk.start_line-1:chunk.end_line])
                    
                    merged_chunk = CodeChunk(
                        content=merged_content,
                        start_line=last_chunk.start_line,
                        end_line=chunk.end_line,
                        chunk_type=ChunkType.MIXED,
                        name=f"{last_chunk.name or 'unknown'} + {chunk.name or 'unknown'}"
                    )
                    
                    processed_chunks[-1] = merged_chunk
                else:
                    processed_chunks.append(chunk)
        
        # Split chunks that are too large
        final_chunks = []
        for chunk in processed_chunks:
            if len(chunk.content) > self.max_chunk_size:
                split_chunks = self._split_large_chunk(chunk, lines)
                final_chunks.extend(split_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_large_chunk(self, chunk: CodeChunk, lines: List[str]) -> List[CodeChunk]:
        """Split a large chunk into smaller ones."""
        chunks = []
        
        # Simple line-based splitting
        chunk_lines = lines[chunk.start_line-1:chunk.end_line]
        
        current_lines = []
        current_start = chunk.start_line
        
        for i, line in enumerate(chunk_lines):
            current_lines.append(line)
            
            if len('\n'.join(current_lines)) > self.max_chunk_size:
                # Create chunk from current lines
                content = '\n'.join(current_lines[:-1])  # Exclude the last line
                
                if content.strip():
                    split_chunk = CodeChunk(
                        content=content,
                        start_line=current_start,
                        end_line=current_start + len(current_lines) - 2,
                        chunk_type=chunk.chunk_type,
                        name=f"{chunk.name} (part {len(chunks) + 1})"
                    )
                    chunks.append(split_chunk)
                
                # Start new chunk
                current_lines = [line]
                current_start = chunk.start_line + i
        
        # Handle remaining lines
        if current_lines:
            content = '\n'.join(current_lines)
            
            if content.strip():
                split_chunk = CodeChunk(
                    content=content,
                    start_line=current_start,
                    end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type,
                    name=f"{chunk.name} (part {len(chunks) + 1})"
                )
                chunks.append(split_chunk)
        
        return chunks if chunks else [chunk]
    
    def _fallback_chunk(self, content: str) -> List[CodeChunk]:
        """Create fallback chunks when AST parsing fails."""
        lines = content.splitlines()
        chunks = []
        
        current_lines = []
        current_start = 1
        
        for i, line in enumerate(lines, 1):
            current_lines.append(line)
            
            if len('\n'.join(current_lines)) > self.max_chunk_size:
                # Create chunk
                chunk_content = '\n'.join(current_lines[:-1])
                
                if chunk_content.strip():
                    chunk = CodeChunk(
                        content=chunk_content,
                        start_line=current_start,
                        end_line=current_start + len(current_lines) - 2,
                        chunk_type=ChunkType.MIXED,
                        name=f"fallback_chunk_{len(chunks) + 1}"
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_lines = [line]
                current_start = i
        
        # Handle remaining lines
        if current_lines:
            chunk_content = '\n'.join(current_lines)
            
            if chunk_content.strip():
                chunk = CodeChunk(
                    content=chunk_content,
                    start_line=current_start,
                    end_line=len(lines),
                    chunk_type=ChunkType.MIXED,
                    name=f"fallback_chunk_{len(chunks) + 1}"
                )
                chunks.append(chunk)
        
        return chunks 