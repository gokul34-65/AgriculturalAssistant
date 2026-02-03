"""
FastAPI RAG Application with Web Search Integration
Enhanced version of app.py with web search capabilities
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime

from rag_pipeline_ollama import DocumentLoader, TextChunker, VectorStore, OllamaRAG
from web_search_module import (
    WebSearchService,
    AgriculturalSearchAssistant,
    EnhancedRAGWithWebSearch
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
KNOWLEDGE_BASE_DIR = "./knowledge_base"
INDEX_PATH = "./rag_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:1b"
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Optional: Get from https://serpapi.com/

# Ensure directories exist
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="RAG Knowledge Base API with Web Search",
    description="RAG system with chat, document management, and web search",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vector_store: Optional[VectorStore] = None
rag_system: Optional[OllamaRAG] = None
enhanced_rag: Optional[EnhancedRAGWithWebSearch] = None
web_search_service: Optional[WebSearchService] = None
is_initialized = False

# ==================== Pydantic Models ====================

class ChatRequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    k: int = Field(3, ge=1, le=10, description="Number of RAG chunks")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=50, le=2048)
    use_web_search: Optional[bool] = Field(None, description="Force web search on/off")
    user_context: Optional[dict] = Field(None, description="User context (district, crop, etc.)")

class EnhancedChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    rag_sources: List[str] = []
    web_sources: List[str] = []
    used_web_search: bool
    web_results_count: int
    chunks_retrieved: int
    model: str
    timestamp: str

class WebSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=20)
    user_context: Optional[dict] = None

class WebSearchResponse(BaseModel):
    query: str
    enhanced_query: str
    category: str
    results: List[dict]
    timestamp: str

# ==================== Helper Functions ====================

def initialize_rag_system():
    """Initialize RAG and Web Search systems"""
    global vector_store, rag_system, enhanced_rag, web_search_service, is_initialized
    
    try:
        logger.info("Initializing RAG system...")
        
        # Check if index exists
        if not os.path.exists(f"{INDEX_PATH}_index.faiss"):
            logger.warning("No index found. Please rebuild the knowledge base.")
            is_initialized = False
            return False
        
        # Load vector store
        vector_store = VectorStore(embedding_model_name=EMBEDDING_MODEL)
        vector_store.load(INDEX_PATH)
        logger.info(f"✓ Loaded vector store with {len(vector_store.chunks)} chunks")
        
        # Initialize RAG
        rag_system = OllamaRAG(
            model_name=OLLAMA_MODEL,
            vector_store=vector_store
        )
        logger.info(f"✓ RAG system initialized with model: {OLLAMA_MODEL}")
        
        # Initialize Web Search
        web_search_service = WebSearchService(cache_duration_minutes=60)
        logger.info("✓ Web search service initialized")
        
        # Initialize Enhanced RAG with Web Search
        enhanced_rag = EnhancedRAGWithWebSearch(
            rag_system=rag_system,
            web_search_service=web_search_service,
            serpapi_key=SERPAPI_KEY
        )
        logger.info("✓ Enhanced RAG with web search ready")
        
        is_initialized = True
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize systems: {e}")
        is_initialized = False
        return False

def rebuild_index_sync():
    """Rebuild the vector index"""
    global vector_store, rag_system, enhanced_rag
    
    try:
        logger.info("Starting index rebuild...")
        
        loader = DocumentLoader()
        documents = loader.load_documents(KNOWLEDGE_BASE_DIR)
        
        if not documents:
            logger.warning("No documents found")
            return {"documents_processed": 0, "chunks_created": 0}
        
        logger.info(f"Loaded {len(documents)} documents")
        
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        vector_store = VectorStore(embedding_model_name=EMBEDDING_MODEL)
        vector_store.build_index(chunks)
        vector_store.save(INDEX_PATH)
        logger.info("✓ Index saved")
        
        # Reinitialize systems
        initialize_rag_system()
        
        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise

# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    logger.info("="*50)
    logger.info("Starting RAG API Server with Web Search...")
    logger.info("="*50)
    
    initialize_rag_system()
    
    if not is_initialized:
        logger.warning("⚠️  RAG system not initialized. Upload documents and rebuild index.")
    else:
        logger.info("✓ RAG API Server with Web Search ready!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down RAG API Server...")

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    return {
        "message": "RAG Knowledge Base API with Web Search",
        "version": "2.0.0",
        "features": ["RAG", "Web Search", "Document Management"],
        "docs": "/docs",
        "endpoints": {
            "chat_enhanced": "/chat-enhanced",
            "web_search": "/search/web",
            "schemes": "/search/schemes",
            "news": "/search/news",
            "chat": "/chat",
            "upload": "/documents/upload",
            "rebuild": "/index/rebuild",
            "status": "/status"
        }
    }

@app.post("/chat-enhanced", response_model=EnhancedChatResponse)
async def chat_enhanced(request: ChatRequest):
    """
    Enhanced chat with automatic web search integration
    
    Features:
    - Automatically detects if web search is needed
    - Combines RAG knowledge base with real-time web data
    - Returns combined results with source attribution
    """
    if not is_initialized:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please rebuild the index."
        )
    
    try:
        logger.info(f"Enhanced chat request: {request.question[:100]}...")
        
        # Query enhanced RAG system
        result = enhanced_rag.query(
            question=request.question,
            user_context=request.user_context,
            k=request.k,
            use_web_search=request.use_web_search,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        logger.info(f"✓ Response generated (Web search: {result['used_web_search']})")
        
        return EnhancedChatResponse(
            question=result['question'],
            answer=result['answer'],
            sources=result['sources'],
            rag_sources=result.get('rag_sources', []),
            web_sources=result.get('web_sources', []),
            used_web_search=result['used_web_search'],
            web_results_count=result['web_results_count'],
            chunks_retrieved=len(result.get('retrieved_chunks', [])),
            model=OLLAMA_MODEL,
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/web", response_model=WebSearchResponse)
async def web_search(request: WebSearchRequest):
    """
    Direct web search endpoint
    
    Search the web for agricultural information:
    - Government schemes
    - Latest news
    - Crop advisories
    - Market information
    """
    if not web_search_service:
        raise HTTPException(
            status_code=503,
            detail="Web search service not initialized"
        )
    
    try:
        logger.info(f"Web search request: {request.query}")
        
        assistant = AgriculturalSearchAssistant(web_search_service)
        
        result = assistant.search_with_context(
            query=request.query,
            user_context=request.user_context
        )
        
        logger.info(f"✓ Found {len(result['results'])} results")
        
        return WebSearchResponse(**result)
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/schemes")
async def search_schemes(
    scheme_type: str = Query("agriculture subsidy", description="Type of scheme"),
    state: str = Query("Tamil Nadu", description="State name"),
    max_results: int = Query(10, ge=1, le=20)
):
    """
    Search for government schemes
    
    Examples:
    - /search/schemes?scheme_type=drip irrigation subsidy
    - /search/schemes?scheme_type=PM Kisan
    - /search/schemes?scheme_type=crop insurance
    """
    if not web_search_service:
        raise HTTPException(status_code=503, detail="Web search not available")
    
    try:
        logger.info(f"Scheme search: {scheme_type} in {state}")
        
        results = web_search_service.search_government_schemes(
            scheme_type=scheme_type,
            state=state
        )
        
        return {
            "scheme_type": scheme_type,
            "state": state,
            "results": results[:max_results],
            "total_found": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Scheme search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/news")
async def search_news(
    topic: str = Query("Tamil Nadu agriculture", description="News topic"),
    days: int = Query(7, ge=1, le=30, description="Days to look back"),
    max_results: int = Query(10, ge=1, le=20)
):
    """
    Search for agricultural news
    
    Examples:
    - /search/news?topic=paddy cultivation&days=7
    - /search/news?topic=pest outbreak&days=3
    - /search/news?topic=government schemes&days=14
    """
    if not web_search_service:
        raise HTTPException(status_code=503, detail="Web search not available")
    
    try:
        logger.info(f"News search: {topic} (last {days} days)")
        
        results = web_search_service.search_agricultural_news(
            topic=topic,
            days=days
        )
        
        return {
            "topic": topic,
            "days": days,
            "results": results[:max_results],
            "total_found": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"News search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/advisory")
async def search_advisory(
    crop: str = Query(..., description="Crop name"),
    issue: Optional[str] = Query(None, description="Specific issue (pest, disease, etc.)"),
    district: Optional[str] = Query(None, description="District name"),
    max_results: int = Query(5, ge=1, le=10)
):
    """
    Search for crop advisories
    
    Examples:
    - /search/advisory?crop=paddy&issue=stem borer&district=Thanjavur
    - /search/advisory?crop=cotton&issue=bollworm
    - /search/advisory?crop=sugarcane&district=Coimbatore
    """
    if not web_search_service:
        raise HTTPException(status_code=503, detail="Web search not available")
    
    try:
        logger.info(f"Advisory search: {crop} - {issue} - {district}")
        
        results = web_search_service.search_crop_advisory(
            crop=crop,
            issue=issue,
            district=district
        )
        
        return {
            "crop": crop,
            "issue": issue,
            "district": district,
            "results": results[:max_results],
            "total_found": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Advisory search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Original Endpoints (Kept for compatibility) ====================

@app.post("/chat")
async def chat(request: ChatRequest):
    """Original chat endpoint (without web search)"""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = rag_system.query(
            question=request.question,
            k=request.k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return {
            "question": result['question'],
            "answer": result['answer'],
            "sources": result['sources'],
            "chunks_retrieved": len(result['retrieved_chunks']),
            "model": OLLAMA_MODEL,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get system status"""
    from pathlib import Path
    documents_count = len(list(Path(KNOWLEDGE_BASE_DIR).rglob('*.[pt][xd][tf]')))
    
    return {
        "status": "ready" if is_initialized else "not_initialized",
        "is_initialized": is_initialized,
        "model": OLLAMA_MODEL,
        "documents_count": documents_count,
        "chunks_count": len(vector_store.chunks) if vector_store else 0,
        "web_search_enabled": web_search_service is not None,
        "serpapi_enabled": SERPAPI_KEY is not None,
        "embedding_model": EMBEDDING_MODEL,
        "knowledge_base_dir": KNOWLEDGE_BASE_DIR
    }

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    auto_rebuild: bool = False
):
    """Upload document"""
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files supported")
    
    try:
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, file.filename)
        
        if os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' already exists")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"✓ Uploaded: {file.filename} ({file_size} bytes)")
        
        response = {
            "status": "success",
            "message": f"File '{file.filename}' uploaded successfully",
            "filename": file.filename,
            "size_bytes": file_size,
            "filepath": file_path
        }
        
        if auto_rebuild:
            logger.info("Auto-rebuild requested...")
            result = rebuild_index_sync()
            response["index_rebuilt"] = True
            response["rebuild_stats"] = result
        else:
            response["index_rebuilt"] = False
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/rebuild")
async def rebuild_index(background: bool = False):
    """Rebuild index"""
    try:
        if background:
            # Note: For background tasks, you'd need BackgroundTasks
            logger.info("Background rebuild not implemented yet")
            return {"status": "queued", "message": "Rebuild queued"}
        else:
            result = rebuild_index_sync()
            return {
                "status": "completed",
                "documents_processed": result["documents_processed"],
                "chunks_created": result["chunks_created"],
                "message": "Index rebuilt successfully"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "initialized": is_initialized,
        "web_search": web_search_service is not None
    }

# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting Enhanced RAG API Server with Web Search")
    print("=" * 60)
    print(f"📂 Knowledge Base: {KNOWLEDGE_BASE_DIR}")
    print(f"🤖 Model: {OLLAMA_MODEL}")
    print(f"🌐 Web Search: Enabled")
    print(f"📊 Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
