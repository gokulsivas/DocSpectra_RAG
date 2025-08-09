# app/router.py - RAG with Vector DB using FastDocumentProcessor
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# ✅ LAZY INITIALIZATION: Create fast processor instance when needed
_processor = None

def get_processor():
    """Get or create FastDocumentProcessor instance"""
    global _processor
    if _processor is None:
        from .utils.fast_document_processor import FastDocumentProcessor
        _processor = FastDocumentProcessor()
        logger.info("✅ FastDocumentProcessor initialized successfully")
    return _processor

# ✅ Updated models to match HackRx API format
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@router.post(
    "/run",
    response_model=QueryResponse,
    summary="Process document URL and answer multiple questions using fast RAG"
)
async def process_query(req: QueryRequest) -> Dict[str, Any]:
    """
    Fast RAG document processing pipeline:
    1️⃣ Marker scans document and extracts text
    2️⃣ Chunk text and store in vectorDB (Pinecone) 
    3️⃣ For each question: Search Vector DB for relevant chunks
    4️⃣ Use Titan LLM with relevant context to generate answers
    5️⃣ Return answers wrapped in object
    """
    try:
        logger.info(f"Processing document: {req.documents}")
        logger.info(f"Questions: {len(req.questions)}")
        
        # ✅ Get fast processor instance (initialized on first use)
        processor = get_processor()
        
        # ✅ SINGLE ASYNC CALL to fast orchestrator
        result = await processor.process_document_qa_fast(
            document_url=req.documents,
            questions=req.questions
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Fast processing failed: {result.get('error', 'Unknown error')}"
            )
        
        # ✅ Return answers wrapped in object
        return {"answers": result['answers']}
        
    except Exception as e:
        logger.error(f"Error in fast process_query: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Fast processing pipeline error: {str(e)}"
        )

# ✅ Add health check endpoint
@router.get("/health")
async def health_check():
    """Check system health"""
    try:
        # Optional: Add quick health check for fast processor
        processor = get_processor()
        health_status = processor.health_check() if hasattr(processor, 'health_check') else {"status": "ready"}
        
        return {
            "status": "healthy",
            "service": "DocSpectra Fast RAG API",
            "processor": "FastDocumentProcessor",
            "endpoints": {
                "run": "POST /run - Fast RAG processing with Marker + VectorDB + Titan"
            },
            "fast_processor_health": health_status
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "healthy",
            "service": "DocSpectra Fast RAG API", 
            "note": "Basic health check (processor not initialized yet)"
        }