"""Codebase collection utilities for gathering code and documentation."""

import logging
import asyncio
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import subprocess
import requests
from urllib.parse import urlparse
import json
import time

from .document import Document
from .file_utils import get_file_size, is_binary_file

logger = logging.getLogger(__name__)


class CodebaseCollector:
    """Collects codebases from various sources."""
    
    def __init__(self, 
                 max_file_size: int = 1024 * 1024,  # 1MB
                 supported_extensions: Optional[List[str]] = None,
                 github_token: Optional[str] = None):
        """Initialize the codebase collector."""
        self.max_file_size = max_file_size
        self.github_token = github_token
        
        # Default supported extensions
        self.supported_extensions = supported_extensions or [
            # Code files
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.go', '.rs', '.rb', '.php', '.sql', '.sh', '.bash', '.zsh',
            '.css', '.scss', '.html', '.xml', '.json', '.yml', '.yaml',
            '.cs', '.swift', '.kt', '.scala', '.clj', '.hs', '.elm',
            '.vue', '.jsx', '.tsx', '.svelte', '.dart', '.r', '.m', '.mm',
            
            # Documentation files
            '.md', '.txt', '.rst', '.adoc', '.tex', '.org',
            
            # Config files
            '.toml', '.ini', '.cfg', '.conf', '.env', '.dockerfile',
            '.dockerignore', '.gitignore', '.gitattributes'
        ]
    
    async def collect_from_github(self, 
                                repo_url: str, 
                                branch: str = "main",
                                include_issues: bool = False,
                                include_prs: bool = False,
                                include_wiki: bool = False) -> List[Document]:
        """Collect codebase from a GitHub repository."""
        logger.info(f"Collecting codebase from GitHub: {repo_url}")
        
        # Parse repository URL
        repo_info = self._parse_github_url(repo_url)
        if not repo_info:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        
        owner, repo_name = repo_info
        
        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Clone repository
            await self._clone_repository(repo_url, temp_path, branch)
            
            # Collect code files
            documents = await self.collect_from_directory(temp_path / repo_name)
            
            # Collect additional GitHub data
            if include_issues or include_prs or include_wiki:
                github_docs = await self._collect_github_metadata(
                    owner, repo_name, include_issues, include_prs, include_wiki
                )
                documents.extend(github_docs)
        
        logger.info(f"Collected {len(documents)} documents from {repo_url}")
        return documents
    
    async def collect_from_directory(self, directory_path: Union[str, Path]) -> List[Document]:
        """Collect codebase from a local directory."""
        directory_path = Path(directory_path)
        logger.info(f"Collecting codebase from directory: {directory_path}")
        
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        documents = []
        
        # Walk through directory
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                # Check if file should be processed
                if self._should_process_file(file_path):
                    doc = await self._create_document_from_file(file_path)
                    if doc:
                        documents.append(doc)
        
        # Add directory structure metadata
        structure_doc = await self._create_structure_document(directory_path)
        if structure_doc:
            documents.append(structure_doc)
        
        logger.info(f"Collected {len(documents)} documents from {directory_path}")
        return documents
    
    async def collect_documentation(self, 
                                  sources: List[Dict[str, Any]]) -> List[Document]:
        """Collect documentation from various sources."""
        logger.info(f"Collecting documentation from {len(sources)} sources")
        
        documents = []
        
        for source in sources:
            source_type = source.get('type')
            
            if source_type == 'api_docs':
                docs = await self._collect_api_documentation(source)
                documents.extend(docs)
            elif source_type == 'stackoverflow':
                docs = await self._collect_stackoverflow_data(source)
                documents.extend(docs)
            elif source_type == 'github_issues':
                docs = await self._collect_github_issues(source)
                documents.extend(docs)
            elif source_type == 'url':
                docs = await self._collect_from_url(source)
                documents.extend(docs)
            else:
                logger.warning(f"Unknown source type: {source_type}")
        
        logger.info(f"Collected {len(documents)} documentation documents")
        return documents
    
    def _parse_github_url(self, url: str) -> Optional[tuple]:
        """Parse GitHub URL to extract owner and repo."""
        parsed = urlparse(url)
        
        if parsed.hostname != 'github.com':
            return None
        
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2:
            return path_parts[0], path_parts[1]
        
        return None
    
    async def _clone_repository(self, repo_url: str, temp_path: Path, branch: str):
        """Clone a Git repository."""
        try:
            cmd = ['git', 'clone', '--depth', '1', '--branch', branch, repo_url]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Git clone failed: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            # Allow some common config files
            allowed_hidden = {'.env', '.gitignore', '.gitattributes', '.dockerignore'}
            if file_path.name not in allowed_hidden:
                return False
        
        # Skip common build/cache directories
        skip_dirs = {
            'node_modules', '__pycache__', '.git', '.svn', '.hg',
            'venv', 'env', '.env', 'build', 'dist', 'target',
            '.idea', '.vscode', '.pytest_cache', '.mypy_cache'
        }
        
        if any(part in skip_dirs for part in file_path.parts):
            return False
        
        # Check file extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Check file size
        try:
            if get_file_size(file_path) > self.max_file_size:
                logger.debug(f"Skipping large file: {file_path}")
                return False
        except Exception:
            return False
        
        # Check if binary file
        if is_binary_file(file_path):
            return False
        
        return True
    
    async def _create_document_from_file(self, file_path: Path) -> Optional[Document]:
        """Create a document from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic metadata
            metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': len(content),
                'file_type': self._get_file_type(file_path),
                'extension': file_path.suffix,
                'relative_path': str(file_path.relative_to(file_path.parents[-2] if len(file_path.parents) > 1 else file_path.parent)),
            }
            
            # Create document
            doc = Document.from_file(
                file_path=str(file_path),
                content=content,
                **metadata
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Error creating document from {file_path}: {e}")
            return None
    
    def _get_file_type(self, file_path: Path) -> str:
        """Get file type based on extension."""
        ext = file_path.suffix.lower()
        
        type_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.txt': 'text',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.xml': 'xml',
        }
        
        return type_map.get(ext, 'unknown')
    
    async def _create_structure_document(self, directory_path: Path) -> Optional[Document]:
        """Create a document containing directory structure."""
        try:
            structure = self._get_directory_structure(directory_path)
            
            metadata = {
                'directory_path': str(directory_path),
                'file_type': 'directory_structure',
                'structure_type': 'tree',
            }
            
            doc = Document.from_file(
                file_path=str(directory_path / "STRUCTURE.md"),
                content=structure,
                **metadata
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Error creating structure document: {e}")
            return None
    
    def _get_directory_structure(self, directory_path: Path, max_depth: int = 3) -> str:
        """Get directory structure as a tree."""
        structure_lines = [f"# Directory Structure: {directory_path.name}\n"]
        
        def _add_directory(path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
            
            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
                
                for i, item in enumerate(items):
                    if item.name.startswith('.'):
                        continue
                    
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    structure_lines.append(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir() and not any(skip in item.name for skip in ['node_modules', '__pycache__', '.git']):
                        extension = "    " if is_last else "│   "
                        _add_directory(item, prefix + extension, depth + 1)
                        
            except PermissionError:
                pass
        
        _add_directory(directory_path)
        return "\n".join(structure_lines)
    
    async def _collect_github_metadata(self, 
                                     owner: str, 
                                     repo: str,
                                     include_issues: bool,
                                     include_prs: bool,
                                     include_wiki: bool) -> List[Document]:
        """Collect GitHub metadata (issues, PRs, wiki)."""
        documents = []
        
        if not self.github_token:
            logger.warning("GitHub token not provided, skipping metadata collection")
            return documents
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        base_url = f"https://api.github.com/repos/{owner}/{repo}"
        
        try:
            if include_issues:
                issues_docs = await self._collect_github_issues_api(base_url, headers)
                documents.extend(issues_docs)
            
            if include_prs:
                prs_docs = await self._collect_github_prs_api(base_url, headers)
                documents.extend(prs_docs)
            
            if include_wiki:
                wiki_docs = await self._collect_github_wiki_api(owner, repo, headers)
                documents.extend(wiki_docs)
                
        except Exception as e:
            logger.error(f"Error collecting GitHub metadata: {e}")
        
        return documents
    
    async def _collect_github_issues_api(self, base_url: str, headers: Dict[str, str]) -> List[Document]:
        """Collect issues from GitHub API."""
        documents = []
        
        try:
            response = requests.get(f"{base_url}/issues", headers=headers, params={'state': 'all'})
            response.raise_for_status()
            
            for issue in response.json():
                doc = Document.from_text(
                    content=f"# Issue #{issue['number']}: {issue['title']}\n\n{issue['body'] or ''}",
                    metadata={
                        'issue_number': issue['number'],
                        'title': issue['title'],
                        'state': issue['state'],
                        'created_at': issue['created_at'],
                        'updated_at': issue['updated_at'],
                        'labels': [label['name'] for label in issue.get('labels', [])],
                        'type': 'github_issue'
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error collecting GitHub issues: {e}")
        
        return documents
    
    async def _collect_github_prs_api(self, base_url: str, headers: Dict[str, str]) -> List[Document]:
        """Collect pull requests from GitHub API."""
        documents = []
        
        try:
            response = requests.get(f"{base_url}/pulls", headers=headers, params={'state': 'all'})
            response.raise_for_status()
            
            for pr in response.json():
                doc = Document.from_text(
                    content=f"# PR #{pr['number']}: {pr['title']}\n\n{pr['body'] or ''}",
                    metadata={
                        'pr_number': pr['number'],
                        'title': pr['title'],
                        'state': pr['state'],
                        'created_at': pr['created_at'],
                        'updated_at': pr['updated_at'],
                        'merged_at': pr.get('merged_at'),
                        'type': 'github_pr'
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error collecting GitHub PRs: {e}")
        
        return documents
    
    async def _collect_github_wiki_api(self, owner: str, repo: str, headers: Dict[str, str]) -> List[Document]:
        """Collect wiki pages from GitHub."""
        documents = []
        
        try:
            # GitHub wiki is actually a separate Git repository
            wiki_url = f"https://github.com/{owner}/{repo}.wiki.git"
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Try to clone wiki repository
                try:
                    await self._clone_repository(wiki_url, temp_path, "master")
                    wiki_path = temp_path / f"{repo}.wiki"
                    
                    if wiki_path.exists():
                        wiki_docs = await self.collect_from_directory(wiki_path)
                        # Update metadata to indicate these are wiki pages
                        for doc in wiki_docs:
                            doc.metadata['type'] = 'github_wiki'
                        documents.extend(wiki_docs)
                        
                except Exception:
                    logger.debug(f"Wiki not found for {owner}/{repo}")
                    
        except Exception as e:
            logger.error(f"Error collecting GitHub wiki: {e}")
        
        return documents
    
    async def _collect_api_documentation(self, source: Dict[str, Any]) -> List[Document]:
        """Collect API documentation from various sources."""
        documents = []
        
        # This is a placeholder for API documentation collection
        # Implementation would depend on the specific API documentation format
        logger.info(f"Collecting API documentation from {source.get('url', 'unknown')}")
        
        return documents
    
    async def _collect_stackoverflow_data(self, source: Dict[str, Any]) -> List[Document]:
        """Collect Stack Overflow questions and answers."""
        documents = []
        
        # This is a placeholder for Stack Overflow data collection
        # Would use Stack Overflow API to collect relevant Q&A
        logger.info(f"Collecting Stack Overflow data for {source.get('tags', [])}")
        
        return documents
    
    async def _collect_github_issues(self, source: Dict[str, Any]) -> List[Document]:
        """Collect GitHub issues from repository."""
        documents = []
        
        # This would use the GitHub API to collect issues
        logger.info(f"Collecting GitHub issues from {source.get('repo', 'unknown')}")
        
        return documents
    
    async def _collect_from_url(self, source: Dict[str, Any]) -> List[Document]:
        """Collect documentation from a URL."""
        documents = []
        
        # This would scrape documentation from web pages
        logger.info(f"Collecting documentation from {source.get('url', 'unknown')}")
        
        return documents 