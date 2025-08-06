# app/router.py - Corrected version
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# ✅ LAZY INITIALIZATION: Create processor instance when needed
# This avoids immediate Pinecone connection attempts during import
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
    documents: str  # Changed from 'url' to match API spec
    questions: List[str]  # Changed from single 'query' to list of questions

class QueryResponse(BaseModel):
    answers: List[str]  # Direct list of answer strings

@router.post(
    "/run",
    response_model=QueryResponse,
    summary="Process document URL and answer multiple questions"
)
async def process_query(req: QueryRequest) -> Dict[str, Any]:
    """
    Enhanced end-to-end processing:
    1️⃣ Download + parse PDF using S3-cached Marker models
    2️⃣ Chunk with sentence boundary awareness  
    3️⃣ Generate embeddings with AWS Titan
    4️⃣ Store in Pinecone + retrieve relevant chunks
    5️⃣ Generate answers using AWS Titan LLM
    """
    try:
        logger.info(f"Processing document: {req.documents}")
        logger.info(f"Questions: {len(req.questions)}")
        
        # ✅ Get processor instance (initialized on first use)
        processor = get_processor()
        
        # ✅ SINGLE CALL to orchestrator (replaces your 5 separate function calls)
        result = processor.process_document_qa(
            document_url=req.documents,
            questions=req.questions
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result.get('error', 'Unknown error')}"
            )
        
        # ✅ Return answers in correct format
        return QueryResponse(answers=result['answers'])
        
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
    try:
        # Get processor instance
        processor = get_processor()
        
        # Check vector store health
        vector_health = processor.vector_store.health_check()
        
        # Check if processor is initialized
        processor_status = "initialized" if processor else "not_initialized"
        
        return {
            "status": "healthy" if vector_health["status"] in ["healthy", "index_missing"] else "unhealthy",
            "processor": processor_status,
            "vector_store": vector_health,
            "pinecone": "connected" if vector_health["pinecone_connected"] else "disconnected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "error": str(e)
        }