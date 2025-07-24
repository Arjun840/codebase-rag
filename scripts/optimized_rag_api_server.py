import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from codebase_rag.core.rag_system import RAGSystem
import asyncio
import logging
from typing import Dict, Any
import time
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for local development and remote use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = None

# Simple in-memory cache for responses
response_cache: Dict[str, Dict[str, Any]] = {}
CACHE_SIZE = 50
CACHE_TTL = 300  # 5 minutes

class QueryRequest(BaseModel):
    question: str
    context: str = None
    fast_mode: bool = False  # New parameter for faster responses

class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    cached: bool = False
    response_time: float = 0.0

def get_cache_key(question: str, fast_mode: bool) -> str:
    """Generate cache key for a query."""
    return f"{question.lower().strip()}:{fast_mode}"

def get_cached_response(cache_key: str) -> Dict[str, Any]:
    """Get cached response if available and not expired."""
    if cache_key in response_cache:
        cached_data = response_cache[cache_key]
        if time.time() - cached_data["timestamp"] < CACHE_TTL:
            return cached_data["response"]
        else:
            # Remove expired cache
            del response_cache[cache_key]
    return None

def cache_response(cache_key: str, response: Dict[str, Any]):
    """Cache a response with timestamp."""
    # Implement simple LRU by removing oldest if cache is full
    if len(response_cache) >= CACHE_SIZE:
        oldest_key = min(response_cache.keys(), 
                        key=lambda k: response_cache[k]["timestamp"])
        del response_cache[oldest_key]
    
    response_cache[cache_key] = {
        "response": response,
        "timestamp": time.time()
    }

@app.on_event("startup")
async def startup_event():
    global rag_system
    try:
        from codebase_rag.config import config
        
        # Optimize configuration for performance
        config.chunk_size = 800  # Smaller chunks for faster processing
        config.chunk_overlap = 100  # Reduced overlap
        
        logger.info(f"Generation model at startup: {config.generation_model}")
        
        # If using CodeLlama from HuggingFace, try to use Ollama version if available
        if config.generation_model == "codellama/CodeLlama-7b-hf":
            logger.info("Attempting to use Ollama CodeLlama instead of HuggingFace version...")
            config.generation_model = "ollama/codellama:7b"
            logger.info(f"Switched to Ollama model: {config.generation_model}")
        
        logger.info("Initializing optimized RAG system...")
        rag_system = RAGSystem()
        await rag_system.initialize()
        logger.info("Optimized RAG system initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    global rag_system
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = get_cache_key(request.question, request.fast_mode)
        cached_response = get_cached_response(cache_key)
        
        if cached_response:
            logger.info(f"Returning cached response for: {request.question[:50]}...")
            return QueryResponse(
                answer=cached_response["answer"],
                sources=cached_response.get("sources", []),
                cached=True,
                response_time=time.time() - start_time
            )
        
        # Choose top_k based on fast_mode
        top_k = 3 if request.fast_mode else 5  # Reduced from default 10
        
        # Use the ask method with optimized top_k
        result = await rag_system.ask(request.question, top_k=top_k)
        
        response_data = {
            "answer": result['answer'],
            "sources": []  # Simplified for performance
        }
        
        # Cache the response
        cache_response(cache_key, response_data)
        
        response_time = time.time() - start_time
        logger.info(f"Query processed in {response_time:.2f}s (top_k={top_k})")
        
        return QueryResponse(
            answer=response_data["answer"],
            sources=response_data["sources"],
            cached=False,
            response_time=response_time
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        
        # If no documents are indexed, provide a fallback response
        if "no documents" in str(e).lower() or "empty" in str(e).lower():
            return QueryResponse(
                answer="I apologize, but no documents have been indexed yet. Please index a codebase first before asking questions.",
                sources=[],
                cached=False,
                response_time=response_time
            )
        else:
            # Log the error and return a generic message
            logger.error(f"Error processing query: {e}")
            return QueryResponse(
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                sources=[],
                cached=False,
                response_time=response_time
            )

@app.post("/query/fast")
async def fast_query_endpoint(request: QueryRequest):
    """Ultra-fast query endpoint with minimal retrieval."""
    request.fast_mode = True
    return await query_endpoint(request)

@app.post("/index")
async def index_codebase():
    global rag_system
    try:
        from pathlib import Path
        logger.info("Starting optimized codebase indexing...")
        
        # Index only the source code directory, not the entire codebase with venv
        codebase_path = Path("./src")  # Relative to codebase-rag directory
        if not codebase_path.exists():
            codebase_path = Path("./codebase-rag/src")  # Fallback path
            
        logger.info(f"Indexing source code at: {codebase_path}")
        await rag_system.index_codebase(codebase_path, force_reindex=True)
        logger.info(f"Successfully indexed codebase at {codebase_path}")
        
        # Clear cache after reindexing
        global response_cache
        response_cache.clear()
        logger.info("Cleared response cache after reindexing")
        
        return {"status": "success", "message": f"Successfully indexed codebase at {codebase_path}"}
    except Exception as e:
        logger.error(f"Error indexing codebase: {e}")
        return {"status": "error", "message": f"Failed to index codebase: {str(e)}"}

@app.get("/health")
def health():
    cache_info = {
        "cache_size": len(response_cache),
        "max_cache_size": CACHE_SIZE,
        "cache_hit_ratio": "Not implemented"  # Could add metrics here
    }
    return {
        "status": "ok", 
        "cache_info": cache_info,
        "optimizations": "enabled"
    }

@app.get("/cache/clear")
def clear_cache():
    """Clear the response cache."""
    global response_cache
    cache_size = len(response_cache)
    response_cache.clear()
    return {"status": "success", "message": f"Cleared {cache_size} cached responses"}

@app.get("/cache/stats")
def cache_stats():
    """Get cache statistics."""
    return {
        "cache_size": len(response_cache),
        "max_cache_size": CACHE_SIZE,
        "cache_ttl": CACHE_TTL,
        "cached_queries": list(response_cache.keys())[:10]  # Show first 10
    }

if __name__ == "__main__":
    uvicorn.run("optimized_rag_api_server:app", host="0.0.0.0", port=8001, reload=True) 