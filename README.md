# DevLens RAG System (CLI Only)

## Overview
This project provides a Retrieval-Augmented Generation (RAG) system for codebase analysis, now with a **command-line interface (CLI)** only. The web frontend has been removed for a streamlined, developer-focused experience.

---

## Usage

### 1. **Activate your virtual environment**
```bash
cd codebase-rag
source ../.venv/bin/activate
```

### 2. **Index your codebase**
```bash
python3 -m codebase_rag.cli --index
```

### 3. **Ask questions via CLI**
```bash
python3 -m codebase_rag.cli --query "How do I use the index_codebase API?"

python3 -m codebase_rag.cli --query "What does this code do?" --code "def add(a, b): return a + b"

python3 -m codebase_rag.cli --query "What does function get_embedding do?"

python3 -m codebase_rag.cli --query "What does ModuleNotFoundError: No module named 'foo' mean?"
```

### 4. **Supported Query Types**
- API usage questions
- Error message explanations
- Code snippet inquiries
- Function/variable lookups
- General codebase questions

---

## Example
```bash
python3 -m codebase_rag.cli --query "How does the RAG system work?"
```

---

## Troubleshooting
- If you get cut-off answers, increase `max_new_tokens` or `max_sequence_length` in `src/codebase_rag/config.py`.
- To check if the backend is healthy, run:
  ```bash
  curl http://localhost:8000/health
  ```
- For more advanced usage, see the CLI help:
  ```bash
  python3 -m codebase_rag.cli --help
  ```

---

## No Web Frontend
All HTML and web assets have been removed. Use the CLI for all interactions.
