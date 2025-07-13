"""File utilities for the RAG system."""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
import fnmatch
import magic
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """Utilities for file operations."""
    
    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        """Get MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information."""
        stat = file_path.stat()
        
        info = {
            'path': str(file_path),
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix,
            'size': stat.st_size,
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'accessed': stat.st_atime,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir(),
            'is_symlink': file_path.is_symlink(),
            'exists': file_path.exists(),
        }
        
        if file_path.is_file():
            try:
                info['hash'] = FileUtils.get_file_hash(file_path)
            except Exception as e:
                logger.warning(f"Could not compute hash for {file_path}: {e}")
                info['hash'] = None
            
            try:
                info['mime_type'] = FileUtils.get_mime_type(file_path)
            except Exception as e:
                logger.warning(f"Could not determine MIME type for {file_path}: {e}")
                info['mime_type'] = None
        
        return info
    
    @staticmethod
    def get_mime_type(file_path: Path) -> Optional[str]:
        """Get MIME type of a file."""
        try:
            return magic.from_file(str(file_path), mime=True)
        except Exception:
            # Fallback to extension-based detection
            extension_map = {
                '.py': 'text/x-python',
                '.js': 'application/javascript',
                '.ts': 'application/typescript',
                '.html': 'text/html',
                '.css': 'text/css',
                '.json': 'application/json',
                '.xml': 'application/xml',
                '.md': 'text/markdown',
                '.txt': 'text/plain',
                '.yml': 'application/x-yaml',
                '.yaml': 'application/x-yaml',
            }
            return extension_map.get(file_path.suffix.lower(), 'application/octet-stream')
    
    @staticmethod
    def is_text_file(file_path: Path) -> bool:
        """Check if a file is a text file."""
        try:
            mime_type = FileUtils.get_mime_type(file_path)
            return mime_type and mime_type.startswith('text/')
        except Exception:
            # Fallback: try to read a small portion as text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)
                return True
            except Exception:
                return False
    
    @staticmethod
    def is_binary_file(file_path: Path) -> bool:
        """Check if a file is binary."""
        return not FileUtils.is_text_file(file_path)
    
    @staticmethod
    def find_files(
        directory: Path,
        patterns: List[str],
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True
    ) -> Generator[Path, None, None]:
        """Find files matching patterns."""
        exclude_patterns = exclude_patterns or []
        
        if recursive:
            search_pattern = '**/*'
        else:
            search_pattern = '*'
        
        for file_path in directory.glob(search_pattern):
            if not file_path.is_file():
                continue
            
            # Check if file matches any include pattern
            matches_include = False
            for pattern in patterns:
                if fnmatch.fnmatch(file_path.name, pattern):
                    matches_include = True
                    break
            
            if not matches_include:
                continue
            
            # Check if file matches any exclude pattern
            matches_exclude = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(str(file_path), pattern):
                    matches_exclude = True
                    break
            
            if matches_exclude:
                continue
            
            yield file_path
    
    @staticmethod
    def find_code_files(directory: Path, exclude_patterns: Optional[List[str]] = None) -> Generator[Path, None, None]:
        """Find code files in a directory."""
        code_patterns = [
            '*.py', '*.js', '*.ts', '*.jsx', '*.tsx',
            '*.java', '*.cpp', '*.c', '*.h', '*.hpp',
            '*.go', '*.rs', '*.rb', '*.php', '*.cs',
            '*.swift', '*.kt', '*.scala', '*.clj',
            '*.sql', '*.sh', '*.bash', '*.zsh',
            '*.ps1', '*.bat', '*.cmd'
        ]
        
        default_excludes = [
            '*/node_modules/*', '*/.git/*', '*/venv/*',
            '*/__pycache__/*', '*/dist/*', '*/build/*',
            '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll',
            '*.class', '*.jar', '*.war', '*.ear'
        ]
        
        exclude_patterns = (exclude_patterns or []) + default_excludes
        
        return FileUtils.find_files(directory, code_patterns, exclude_patterns)
    
    @staticmethod
    def find_doc_files(directory: Path, exclude_patterns: Optional[List[str]] = None) -> Generator[Path, None, None]:
        """Find documentation files in a directory."""
        doc_patterns = [
            '*.md', '*.rst', '*.txt', '*.doc', '*.docx',
            '*.pdf', '*.html', '*.htm', '*.xml',
            '*.json', '*.yml', '*.yaml', '*.toml',
            '*.cfg', '*.ini', '*.conf'
        ]
        
        default_excludes = [
            '*/node_modules/*', '*/.git/*', '*/venv/*',
            '*/__pycache__/*', '*/dist/*', '*/build/*'
        ]
        
        exclude_patterns = (exclude_patterns or []) + default_excludes
        
        return FileUtils.find_files(directory, doc_patterns, exclude_patterns)
    
    @staticmethod
    def read_file_safely(file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
        """Read a file safely with error handling."""
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    @staticmethod
    def write_file_safely(file_path: Path, content: str, encoding: str = 'utf-8') -> bool:
        """Write to a file safely with error handling."""
        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return False
    
    @staticmethod
    def ensure_directory(directory_path: Path) -> bool:
        """Ensure a directory exists."""
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def copy_file(src: Path, dst: Path) -> bool:
        """Copy a file safely."""
        try:
            import shutil
            FileUtils.ensure_directory(dst.parent)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"Error copying file {src} to {dst}: {e}")
            return False
    
    @staticmethod
    def move_file(src: Path, dst: Path) -> bool:
        """Move a file safely."""
        try:
            import shutil
            FileUtils.ensure_directory(dst.parent)
            shutil.move(src, dst)
            return True
        except Exception as e:
            logger.error(f"Error moving file {src} to {dst}: {e}")
            return False
    
    @staticmethod
    def delete_file(file_path: Path) -> bool:
        """Delete a file safely."""
        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    @staticmethod
    def get_directory_size(directory: Path) -> int:
        """Get the total size of a directory."""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except Exception:
                    pass
        return total_size
    
    @staticmethod
    def get_file_count(directory: Path, pattern: str = '*') -> int:
        """Get the count of files matching a pattern."""
        count = 0
        for _ in directory.rglob(pattern):
            count += 1
        return count
    
    @staticmethod
    def is_empty_directory(directory: Path) -> bool:
        """Check if a directory is empty."""
        try:
            return not any(directory.iterdir())
        except Exception:
            return True
    
    @staticmethod
    def get_relative_path(file_path: Path, base_path: Path) -> str:
        """Get relative path from base path."""
        try:
            return str(file_path.relative_to(base_path))
        except ValueError:
            return str(file_path)
    
    @staticmethod
    def normalize_path(path: Path) -> Path:
        """Normalize a path."""
        return path.resolve()
    
    @staticmethod
    def is_hidden_file(file_path: Path) -> bool:
        """Check if a file is hidden."""
        return file_path.name.startswith('.')
    
    @staticmethod
    def get_file_extension(file_path: Path) -> str:
        """Get file extension in lowercase."""
        return file_path.suffix.lower()
    
    @staticmethod
    def change_extension(file_path: Path, new_extension: str) -> Path:
        """Change file extension."""
        return file_path.with_suffix(new_extension)
    
    @staticmethod
    def backup_file(file_path: Path, backup_suffix: str = '.bak') -> Optional[Path]:
        """Create a backup of a file."""
        if not file_path.exists():
            return None
        
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        
        if FileUtils.copy_file(file_path, backup_path):
            return backup_path
        
        return None 