# Technical Description: RAG Tool for Codebase Help (CLI Only)

## Overview
This project is a command-line Retrieval-Augmented Generation (RAG) system for codebase analysis, code search, and code-aware Q&A. It is designed for developers who want to query, explore, and understand large codebases using natural language, without a web frontend.

---

## Architecture

### 1. **Core Components**
- **CLI Interface (`cli.py`)**: Handles user input, command parsing, and output. Supports search, indexing, and interactive chat.
- **RAG System (`core/rag_system.py`)**: Orchestrates retrieval, context assembly, and answer generation.
- **Embedding Manager (`core/embeddings.py`)**: Generates vector embeddings for code and documentation chunks using [SentenceTransformers](https://www.sbert.net/).
- **Vector Store (`core/vector_store.py`)**: Stores and retrieves vector embeddings using [ChromaDB](https://www.trychroma.com/) (default, persistent, local) or [FAISS](https://github.com/facebookresearch/faiss) (optional, local, in-memory or file-based).
- **Retriever (`core/retriever.py`)**: Finds the most relevant code/doc chunks for a query.
- **Generator (`core/generator.py`)**: Uses a language model to generate answers based on retrieved context. Supports:
  - **Ollama** (local LLMs via Ollama API, e.g., `ollama/codellama:7b`)
  - **OpenAI API** (e.g., `openai/gpt-3.5-turbo`, `openai/gpt-4`) if API key is provided
  - **Local HuggingFace models** (e.g., `microsoft/DialoGPT-medium`) if weights are available
- **Processors (`processors/`)**: Specialized logic for code, document, and error message processing.
- **Config (`config.py`)**: Centralized configuration for models, chunking, and database settings.

### 2. **Data Flow**
1. **Indexing**: The CLI indexes a codebase by splitting files into chunks, generating embeddings (with SentenceTransformers), and storing them in a vector database (ChromaDB or FAISS).
2. **Querying**: User submits a natural language query via CLI.
3. **Retrieval**: The system embeds the query and retrieves the most relevant code/doc chunks.
4. **Context Assembly**: Retrieved chunks are assembled into a context window.
5. **Generation**: The context and query are sent to a language model (Ollama, OpenAI API, or local HuggingFace) to generate an answer.
6. **Output**: The answer and (optionally) source context are displayed in the terminal.

---

## Features
- **Codebase Indexing**: Supports Python and other languages. Ignores common noise (e.g., venv, .git).
- **Semantic Search**: Finds code by meaning, not just text.
- **Natural Language Q&A**: Ask about functions, errors, usage, or code snippets.
- **Error Explanation**: Recognizes error messages and provides explanations/fixes.
- **Code Snippet Analysis**: Explains what code does in plain English.
- **Interactive Mode**: Chat-like CLI session for iterative exploration.
- **Source Attribution**: Optionally shows which files/chunks were used for each answer.
- **Configurable Models**: Swap out embedding/generation models via config or CLI args.

---

## Extensibility
- **Pluggable Embedding Models**: Any [SentenceTransformers](https://www.sbert.net/) model.
- **Pluggable Generation Models**: Ollama (local), OpenAI API, or local HuggingFace models.
- **Custom Processors**: Add new processors for other file types or error patterns.
- **Vector DB Backends**: ChromaDB (default), FAISS (optional).
- **Chunking Strategies**: AST-based, text-based, or custom chunkers.

---

## Configuration
- **config.py**: Central place for model names, chunk size, overlap, vector DB path, etc.
- **Environment Variables**: Can override config for model selection, API keys, etc.
- **CLI Arguments**: Override config at runtime (e.g., `--embedding-model`, `--generation-model`).

---

## Example Workflow

1. **Index the codebase:**
   ```bash
   python3 -m codebase_rag.cli index ./src
   ```
2. **Ask a question:**
   ```bash
   python3 -m codebase_rag.cli search "How do I use the index_codebase API?"
   ```
3. **Interactive mode:**
   ```bash
   python3 -m codebase_rag.cli interactive
   ```

---

## Advanced Usage
- **Top-K Retrieval**: Control how many chunks are retrieved per query (`--top-k`).
- **Show Sources**: In interactive mode, use `--show-sources` to display file/chunk info.
- **Force Reindex**: Use `--force` to rebuild the vector DB from scratch.
- **Model Swapping**: Use `--embedding-model` or `--generation-model` to try different models.

---

## Limitations & Considerations
- **Token/Length Limits**: Answers may be cut off if the model’s max token limit is reached. Adjust `max_new_tokens` and `max_sequence_length` in config.
- **Model Quality**: The quality of answers depends on the underlying language model and embeddings.
- **Resource Usage**: Large models require significant RAM/CPU/GPU. Use smaller models for faster, lighter-weight operation.
- **No Web UI**: All interaction is via CLI for simplicity and portability.

---

## Example Use Cases
- Quickly find and understand code functions, classes, or patterns.
- Get explanations for error messages encountered during development.
- Ask for usage examples or best practices from your own codebase.
- Analyze code snippets or refactorings in plain English.

---

## Developer Notes
- **Code Structure**: Modular, with clear separation between retrieval, generation, and CLI logic.
- **Testing**: You can write unit tests for processors, retrievers, and generators. For code generation, use HumanEval-style tests.
- **Extending**: Add new subcommands to `cli.py` for batch queries, export, or integration with editors/IDEs.

---

## Summary
This CLI RAG tool brings LLM-powered codebase search and Q&A to your terminal. It is highly extensible, model-agnostic (within the supported backends), and designed for developer productivity without the overhead of a web UI. 