# app/router.py - RAG with Vector DB
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# ✅ LAZY INITIALIZATION: Create processor instance when needed
_processor = None

def get_processor():
    """Get or create DocumentProcessor instance"""
    global _processor
    if _processor is None:
        from .utils.document_processor import DocumentProcessor
        _processor = DocumentProcessor()
        logger.info("✅ DocumentProcessor initialized successfully")
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
    summary="Process document URL and answer multiple questions using RAG"
)
async def process_query(req: QueryRequest) -> Dict[str, Any]:
    """
    RAG document processing pipeline:
    1️⃣ Marker scans document and extracts text
    2️⃣ Chunk text and store in vectorDB (Pinecone)
    3️⃣ For each question: Search Vector DB for relevant chunks
    4️⃣ Use Titan LLM with relevant context to generate answers
    5️⃣ Return answers wrapped in object
    """
    try:
        logger.info(f"Processing document: {req.documents}")
        logger.info(f"Questions: {len(req.questions)}")
        
        # ✅ Get processor instance (initialized on first use)
        processor = get_processor()
        
        # ✅ SINGLE CALL to orchestrator
        result = processor.process_document_qa(
            document_url=req.documents,
            questions=req.questions
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result.get('error', 'Unknown error')}"
            )
        
        # ✅ Return answers wrapped in object
        return {"answers": result['answers']}
        
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Processing pipeline error: {str(e)}"
        )

# ✅ Add health check endpoint
@router.get("/health")
async def health_check():
    """Check system health"""
    return {
        "status": "healthy",
        "service": "DocSpectra RAG API",
        "endpoints": {
            "run": "POST /run - RAG processing with Marker + VectorDB + Titan"
        }
    }
