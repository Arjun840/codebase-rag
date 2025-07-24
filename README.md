# RAG Tool for Codebase Help (CLI Only)

A command-line tool for codebase search and code-aware Q&A using Retrieval-Augmented Generation (RAG).

---

## Quick Start

1. **Activate your virtual environment:**
   ```bash
   cd codebase-rag
   source ../.venv/bin/activate
   export PYTHONPATH=src
   ```

2. **Index your codebase:**
   ```bash
   python3 -m codebase_rag.cli index ./src
   ```

3. **Ask questions via CLI:**
   ```bash
   python3 -m codebase_rag.cli search "How do I use the index_codebase API?"
   python3 -m codebase_rag.cli search "What does this code do? def add(a, b): return a + b"
   python3 -m codebase_rag.cli search "What does function get_embedding do?"
   python3 -m codebase_rag.cli search "What does ModuleNotFoundError: No module named 'foo' mean?"
   ```

4. **Interactive mode:**
   ```bash
   python3 -m codebase_rag.cli interactive
   ```

---

## Features
- Search and ask questions about your codebase from the terminal
- Supports API usage, error explanations, code snippet analysis, and more
- No web frontend, no browser required

---

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'codebase_rag'`, make sure you are in the `codebase-rag` directory and have set `export PYTHONPATH=src`.
- For more options, run:
  ```bash
  python3 -m codebase_rag.cli --help
  ```

---

## Technical Description

This project is a Python CLI tool for codebase analysis using Retrieval-Augmented Generation (RAG). It indexes your codebase, embeds code/documents, and allows you to query or chat with your codebase using natural language. The backend uses FastAPI and supports pluggable embedding and generation models. All web UI has been removed for a streamlined, developer-focused workflow.
