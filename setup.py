"""Setup script for the Codebase RAG system."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="codebase-rag",
    version="0.1.0",
    description="A Retrieval-Augmented Generation system for codebase search and developer assistance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Developer",
    author_email="developer@example.com",
    url="https://github.com/your-username/codebase-rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "streamlit>=1.25.0",
        "chromadb>=0.4.0",
        "faiss-cpu>=1.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "tiktoken>=0.4.0",
        "tree-sitter>=0.20.0",
        "pygments>=2.15.0",
        "python-magic>=0.4.27",
        "gitpython>=3.1.30",
        "pyyaml>=6.0.0",
        "beautifulsoup4>=4.12.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "all": [
            "accelerate>=0.20.0",
            "gradio>=3.40.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "codebase-rag=codebase_rag.cli:main",
            "codebase-rag-web=codebase_rag.web.streamlit_app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Documentation",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="rag retrieval-augmented-generation codebase search nlp ai machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/your-username/codebase-rag/issues",
        "Source": "https://github.com/your-username/codebase-rag",
        "Documentation": "https://github.com/your-username/codebase-rag/blob/main/README.md",
    },
) 