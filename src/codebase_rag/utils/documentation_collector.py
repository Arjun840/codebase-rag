"""Documentation collection utilities for gathering various types of documentation."""

import logging
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin, urlparse, parse_qs
import aiohttp
import requests
from bs4 import BeautifulSoup
import re

from .document import Document

logger = logging.getLogger(__name__)


class DocumentationCollector:
    """Collects documentation from various sources."""
    
    def __init__(self, 
                 max_requests_per_minute: int = 30,
                 user_agent: str = "RAG-System/1.0",
                 timeout: int = 30):
        """Initialize the documentation collector."""
        self.max_requests_per_minute = max_requests_per_minute
        self.user_agent = user_agent
        self.timeout = timeout
        self.request_times = []
        
        # API endpoints
        self.stackoverflow_api = "https://api.stackexchange.com/2.3"
        self.github_api = "https://api.github.com"
    
    async def collect_stackoverflow_questions(self, 
                                            tags: List[str],
                                            max_questions: int = 100,
                                            include_answers: bool = True,
                                            sort: str = "votes",
                                            site: str = "stackoverflow") -> List[Document]:
        """Collect Stack Overflow questions and answers."""
        logger.info(f"Collecting Stack Overflow questions for tags: {tags}")
        
        documents = []
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Prepare parameters
            params = {
                'tagged': ';'.join(tags),
                'sort': sort,
                'order': 'desc',
                'pagesize': min(max_questions, 100),
                'site': site,
                'filter': 'withbody'
            }
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                url = f"{self.stackoverflow_api}/questions"
                async with session.get(url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for question in data.get('items', []):
                            # Create question document
                            question_doc = await self._create_stackoverflow_question_doc(question)
                            if question_doc:
                                documents.append(question_doc)
                            
                            # Get answers if requested
                            if include_answers:
                                answer_docs = await self._get_stackoverflow_answers(question['question_id'], site)
                                documents.extend(answer_docs)
                    else:
                        logger.error(f"Stack Overflow API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error collecting Stack Overflow questions: {e}")
        
        logger.info(f"Collected {len(documents)} Stack Overflow documents")
        return documents
    
    async def collect_github_issues(self, 
                                   repo_owner: str,
                                   repo_name: str,
                                   state: str = "all",
                                   labels: Optional[List[str]] = None,
                                   max_issues: int = 100,
                                   github_token: Optional[str] = None) -> List[Document]:
        """Collect GitHub issues and their comments."""
        logger.info(f"Collecting GitHub issues for {repo_owner}/{repo_name}")
        
        documents = []
        
        try:
            # Set up headers
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/vnd.github.v3+json'
            }
            
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            # Prepare parameters
            params = {
                'state': state,
                'per_page': min(max_issues, 100),
                'sort': 'updated',
                'direction': 'desc'
            }
            
            if labels:
                params['labels'] = ','.join(labels)
            
            # Rate limiting
            await self._rate_limit()
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                url = f"{self.github_api}/repos/{repo_owner}/{repo_name}/issues"
                async with session.get(url, params=params, headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        issues = await response.json()
                        
                        for issue in issues:
                            # Create issue document
                            issue_doc = await self._create_github_issue_doc(issue, repo_owner, repo_name)
                            if issue_doc:
                                documents.append(issue_doc)
                            
                            # Get comments
                            if issue['comments'] > 0:
                                comment_docs = await self._get_github_issue_comments(
                                    repo_owner, repo_name, issue['number'], headers
                                )
                                documents.extend(comment_docs)
                    else:
                        logger.error(f"GitHub API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error collecting GitHub issues: {e}")
        
        logger.info(f"Collected {len(documents)} GitHub issue documents")
        return documents
    
    async def collect_api_documentation(self, 
                                      api_urls: List[str],
                                      doc_types: List[str] = None) -> List[Document]:
        """Collect API documentation from various sources."""
        logger.info(f"Collecting API documentation from {len(api_urls)} URLs")
        
        documents = []
        doc_types = doc_types or ['openapi', 'swagger', 'rest', 'graphql']
        
        for url in api_urls:
            try:
                # Rate limiting
                await self._rate_limit()
                
                # Detect documentation type
                doc_type = self._detect_api_doc_type(url)
                
                if doc_type == 'openapi' or doc_type == 'swagger':
                    docs = await self._collect_openapi_docs(url)
                elif doc_type == 'rest':
                    docs = await self._collect_rest_docs(url)
                elif doc_type == 'graphql':
                    docs = await self._collect_graphql_docs(url)
                else:
                    docs = await self._collect_generic_docs(url)
                
                documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Error collecting API docs from {url}: {e}")
        
        logger.info(f"Collected {len(documents)} API documentation documents")
        return documents
    
    async def collect_web_documentation(self, 
                                      urls: List[str],
                                      selectors: Optional[Dict[str, str]] = None) -> List[Document]:
        """Collect documentation from web pages."""
        logger.info(f"Collecting web documentation from {len(urls)} URLs")
        
        documents = []
        
        for url in urls:
            try:
                # Rate limiting
                await self._rate_limit()
                
                # Scrape the page
                doc = await self._scrape_documentation_page(url, selectors)
                if doc:
                    documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Error collecting web docs from {url}: {e}")
        
        logger.info(f"Collected {len(documents)} web documentation documents")
        return documents
    
    async def collect_readme_files(self, 
                                 repo_urls: List[str],
                                 github_token: Optional[str] = None) -> List[Document]:
        """Collect README files from repositories."""
        logger.info(f"Collecting README files from {len(repo_urls)} repositories")
        
        documents = []
        
        for repo_url in repo_urls:
            try:
                # Parse repository URL
                repo_info = self._parse_github_url(repo_url)
                if not repo_info:
                    continue
                
                owner, repo_name = repo_info
                
                # Rate limiting
                await self._rate_limit()
                
                # Get README content
                readme_doc = await self._get_github_readme(owner, repo_name, github_token)
                if readme_doc:
                    documents.append(readme_doc)
                    
            except Exception as e:
                logger.error(f"Error collecting README from {repo_url}: {e}")
        
        logger.info(f"Collected {len(documents)} README documents")
        return documents
    
    async def _rate_limit(self):
        """Implement rate limiting."""
        now = time.time()
        
        # Remove old requests
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Check if we need to wait
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(now)
    
    async def _create_stackoverflow_question_doc(self, question: Dict[str, Any]) -> Optional[Document]:
        """Create a document from a Stack Overflow question."""
        try:
            # Clean HTML content
            title = question.get('title', '')
            body = self._clean_html(question.get('body', ''))
            
            content = f"# {title}\n\n{body}"
            
            # Create metadata
            metadata = {
                'question_id': question['question_id'],
                'title': title,
                'score': question.get('score', 0),
                'view_count': question.get('view_count', 0),
                'answer_count': question.get('answer_count', 0),
                'tags': question.get('tags', []),
                'creation_date': question.get('creation_date'),
                'last_activity_date': question.get('last_activity_date'),
                'is_answered': question.get('is_answered', False),
                'accepted_answer_id': question.get('accepted_answer_id'),
                'link': question.get('link'),
                'type': 'stackoverflow_question'
            }
            
            doc = Document.from_text(
                content=content,
                metadata=metadata
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Error creating Stack Overflow question document: {e}")
            return None
    
    async def _get_stackoverflow_answers(self, question_id: int, site: str) -> List[Document]:
        """Get answers for a Stack Overflow question."""
        documents = []
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            params = {
                'site': site,
                'filter': 'withbody',
                'sort': 'votes',
                'order': 'desc'
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.stackoverflow_api}/questions/{question_id}/answers"
                async with session.get(url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for answer in data.get('items', []):
                            doc = await self._create_stackoverflow_answer_doc(answer, question_id)
                            if doc:
                                documents.append(doc)
                                
        except Exception as e:
            logger.error(f"Error getting Stack Overflow answers: {e}")
        
        return documents
    
    async def _create_stackoverflow_answer_doc(self, answer: Dict[str, Any], question_id: int) -> Optional[Document]:
        """Create a document from a Stack Overflow answer."""
        try:
            # Clean HTML content
            body = self._clean_html(answer.get('body', ''))
            
            content = f"# Answer to Question {question_id}\n\n{body}"
            
            # Create metadata
            metadata = {
                'answer_id': answer['answer_id'],
                'question_id': question_id,
                'score': answer.get('score', 0),
                'is_accepted': answer.get('is_accepted', False),
                'creation_date': answer.get('creation_date'),
                'last_activity_date': answer.get('last_activity_date'),
                'type': 'stackoverflow_answer'
            }
            
            doc = Document.from_text(
                content=content,
                metadata=metadata
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Error creating Stack Overflow answer document: {e}")
            return None
    
    async def _create_github_issue_doc(self, issue: Dict[str, Any], repo_owner: str, repo_name: str) -> Optional[Document]:
        """Create a document from a GitHub issue."""
        try:
            title = issue.get('title', '')
            body = issue.get('body', '') or ''
            
            content = f"# Issue #{issue['number']}: {title}\n\n{body}"
            
            # Create metadata
            metadata = {
                'issue_number': issue['number'],
                'title': title,
                'state': issue.get('state'),
                'labels': [label['name'] for label in issue.get('labels', [])],
                'assignees': [assignee['login'] for assignee in issue.get('assignees', [])],
                'milestone': issue.get('milestone', {}).get('title') if issue.get('milestone') else None,
                'created_at': issue.get('created_at'),
                'updated_at': issue.get('updated_at'),
                'closed_at': issue.get('closed_at'),
                'comments': issue.get('comments', 0),
                'html_url': issue.get('html_url'),
                'repository': f"{repo_owner}/{repo_name}",
                'type': 'github_issue'
            }
            
            doc = Document.from_text(
                content=content,
                metadata=metadata
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Error creating GitHub issue document: {e}")
            return None
    
    async def _get_github_issue_comments(self, repo_owner: str, repo_name: str, issue_number: int, headers: Dict[str, str]) -> List[Document]:
        """Get comments for a GitHub issue."""
        documents = []
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.github_api}/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"
                async with session.get(url, headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        comments = await response.json()
                        
                        for comment in comments:
                            doc = await self._create_github_comment_doc(comment, issue_number, repo_owner, repo_name)
                            if doc:
                                documents.append(doc)
                                
        except Exception as e:
            logger.error(f"Error getting GitHub issue comments: {e}")
        
        return documents
    
    async def _create_github_comment_doc(self, comment: Dict[str, Any], issue_number: int, repo_owner: str, repo_name: str) -> Optional[Document]:
        """Create a document from a GitHub comment."""
        try:
            body = comment.get('body', '') or ''
            
            content = f"# Comment on Issue #{issue_number}\n\n{body}"
            
            # Create metadata
            metadata = {
                'comment_id': comment['id'],
                'issue_number': issue_number,
                'author': comment.get('user', {}).get('login'),
                'created_at': comment.get('created_at'),
                'updated_at': comment.get('updated_at'),
                'html_url': comment.get('html_url'),
                'repository': f"{repo_owner}/{repo_name}",
                'type': 'github_comment'
            }
            
            doc = Document.from_text(
                content=content,
                metadata=metadata
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Error creating GitHub comment document: {e}")
            return None
    
    def _detect_api_doc_type(self, url: str) -> str:
        """Detect the type of API documentation."""
        url_lower = url.lower()
        
        if 'swagger' in url_lower or 'openapi' in url_lower:
            return 'openapi'
        elif 'graphql' in url_lower:
            return 'graphql'
        elif 'rest' in url_lower or 'api' in url_lower:
            return 'rest'
        else:
            return 'generic'
    
    async def _collect_openapi_docs(self, url: str) -> List[Document]:
        """Collect OpenAPI/Swagger documentation."""
        documents = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        spec = await response.json()
                        
                        # Parse OpenAPI spec
                        doc = self._parse_openapi_spec(spec, url)
                        if doc:
                            documents.append(doc)
                            
        except Exception as e:
            logger.error(f"Error collecting OpenAPI docs: {e}")
        
        return documents
    
    def _parse_openapi_spec(self, spec: Dict[str, Any], url: str) -> Optional[Document]:
        """Parse OpenAPI specification into a document."""
        try:
            title = spec.get('info', {}).get('title', 'API Documentation')
            description = spec.get('info', {}).get('description', '')
            version = spec.get('info', {}).get('version', '')
            
            content = f"# {title}\n\n"
            
            if description:
                content += f"{description}\n\n"
            
            if version:
                content += f"**Version:** {version}\n\n"
            
            # Add paths
            paths = spec.get('paths', {})
            if paths:
                content += "## Endpoints\n\n"
                
                for path, methods in paths.items():
                    content += f"### {path}\n\n"
                    
                    for method, details in methods.items():
                        if isinstance(details, dict):
                            content += f"**{method.upper()}**\n\n"
                            
                            summary = details.get('summary', '')
                            description = details.get('description', '')
                            
                            if summary:
                                content += f"Summary: {summary}\n\n"
                            
                            if description:
                                content += f"Description: {description}\n\n"
            
            # Create metadata
            metadata = {
                'api_title': title,
                'api_version': version,
                'source_url': url,
                'type': 'openapi_documentation'
            }
            
            doc = Document.from_text(
                content=content,
                metadata=metadata
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Error parsing OpenAPI spec: {e}")
            return None
    
    async def _collect_rest_docs(self, url: str) -> List[Document]:
        """Collect REST API documentation."""
        # This is a placeholder - actual implementation would depend on the specific format
        return []
    
    async def _collect_graphql_docs(self, url: str) -> List[Document]:
        """Collect GraphQL documentation."""
        # This is a placeholder - actual implementation would query the GraphQL schema
        return []
    
    async def _collect_generic_docs(self, url: str) -> List[Document]:
        """Collect generic documentation from a URL."""
        documents = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse HTML if it's an HTML page
                        if 'text/html' in response.headers.get('content-type', ''):
                            doc = self._parse_html_documentation(content, url)
                        else:
                            doc = Document.from_text(
                                content=content,
                                metadata={'source_url': url, 'type': 'generic_documentation'}
                            )
                        
                        if doc:
                            documents.append(doc)
                            
        except Exception as e:
            logger.error(f"Error collecting generic docs: {e}")
        
        return documents
    
    def _parse_html_documentation(self, html_content: str, url: str) -> Optional[Document]:
        """Parse HTML documentation page."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else 'Documentation'
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract main content
            main_content = soup.find('main') or soup.find('body') or soup
            text = main_content.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            content = f"# {title_text}\n\n{text}"
            
            # Create metadata
            metadata = {
                'title': title_text,
                'source_url': url,
                'type': 'html_documentation'
            }
            
            doc = Document.from_text(
                content=content,
                metadata=metadata
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Error parsing HTML documentation: {e}")
            return None
    
    async def _scrape_documentation_page(self, url: str, selectors: Optional[Dict[str, str]] = None) -> Optional[Document]:
        """Scrape a documentation page with optional CSS selectors."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_html_documentation(content, url)
                        
        except Exception as e:
            logger.error(f"Error scraping documentation page: {e}")
        
        return None
    
    def _parse_github_url(self, url: str) -> Optional[tuple]:
        """Parse GitHub URL to extract owner and repo."""
        parsed = urlparse(url)
        
        if parsed.hostname != 'github.com':
            return None
        
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2:
            return path_parts[0], path_parts[1]
        
        return None
    
    async def _get_github_readme(self, owner: str, repo_name: str, github_token: Optional[str] = None) -> Optional[Document]:
        """Get README file from a GitHub repository."""
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/vnd.github.v3+json'
            }
            
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.github_api}/repos/{owner}/{repo_name}/readme"
                async with session.get(url, headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        readme_data = await response.json()
                        
                        # Decode content (it's base64 encoded)
                        import base64
                        content = base64.b64decode(readme_data['content']).decode('utf-8')
                        
                        # Create metadata
                        metadata = {
                            'repository': f"{owner}/{repo_name}",
                            'file_name': readme_data['name'],
                            'file_path': readme_data['path'],
                            'sha': readme_data['sha'],
                            'download_url': readme_data['download_url'],
                            'type': 'github_readme'
                        }
                        
                        doc = Document.from_text(
                            content=content,
                            metadata=metadata
                        )
                        
                        return doc
                        
        except Exception as e:
            logger.error(f"Error getting GitHub README: {e}")
        
        return None
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content to plain text."""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return html_content 