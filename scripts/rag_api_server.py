import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from codebase_rag.core.rag_system import RAGSystem
import asyncio
import logging

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

class QueryRequest(BaseModel):
    question: str
    context: str = None

class QueryResponse(BaseModel):
    answer: str
    sources: list = []

@app.on_event("startup")
async def startup_event():
    global rag_system
    try:
        from codebase_rag.config import config
        logger.info(f"Generation model at startup: {config.generation_model}")
        
        # If using CodeLlama from HuggingFace, try to use Ollama version if available
        if config.generation_model == "codellama/CodeLlama-7b-hf":
            logger.info("Attempting to use Ollama CodeLlama instead of HuggingFace version...")
            config.generation_model = "ollama/codellama:7b"
            logger.info(f"Switched to Ollama model: {config.generation_model}")
        
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem()
        await rag_system.initialize()
        logger.info("RAG system initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    global rag_system
    try:
        # Use the ask method instead of query, and handle empty index gracefully
        result = await rag_system.ask(request.question)
        return QueryResponse(answer=result['answer'], sources=[])
    except Exception as e:
        # If no documents are indexed, provide a fallback response
        if "no documents" in str(e).lower() or "empty" in str(e).lower():
            return QueryResponse(
                answer="I apologize, but no documents have been indexed yet. Please index a codebase first before asking questions.",
                sources=[]
            )
        else:
            # Log the error and return a generic message
            import logging
            logging.error(f"Error processing query: {e}")
            return QueryResponse(
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                sources=[]
            )

@app.post("/index")
async def index_codebase():
    global rag_system
    try:
        from pathlib import Path
        logger.info("Starting codebase indexing...")
        # Index only the source code directory, not the entire codebase with venv
        codebase_path = Path("./codebase-rag/src")
        logger.info(f"Indexing source code at: {codebase_path}")
        await rag_system.index_codebase(codebase_path, force_reindex=True)
        logger.info(f"Successfully indexed codebase at {codebase_path}")
        return {"status": "success", "message": f"Successfully indexed codebase at {codebase_path}"}
    except Exception as e:
        logger.error(f"Error indexing codebase: {e}")
        return {"status": "error", "message": f"Failed to index codebase: {str(e)}"}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("rag_api_server:app", host="0.0.0.0", port=8000, reload=True) 