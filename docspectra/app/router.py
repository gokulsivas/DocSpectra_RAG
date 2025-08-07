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
_titan_processor = None

def get_processor():
    """Get or create DocumentProcessor instance"""
    global _processor
    if _processor is None:
        from .utils.document_processor import DocumentProcessor
        _processor = DocumentProcessor()
        logger.info("✅ DocumentProcessor initialized successfully")
    return _processor

def get_titan_processor():
    """Get or create DocumentProcessor instance with Titan QA enabled"""
    global _titan_processor
    if _titan_processor is None:
        from .utils.document_processor import DocumentProcessor
        _titan_processor = DocumentProcessor(use_bedrock_qa=False, use_titan_qa=True)
        logger.info("✅ Titan DocumentProcessor initialized successfully")
    return _titan_processor

# ✅ Updated models to match HackRx API format
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

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

@router.post(
    "/run/titan",
    response_model=QueryResponse,
    summary="Process document URL and answer multiple questions using Titan Q&A"
)
async def process_query_titan(req: QueryRequest) -> Dict[str, Any]:
    """
    Titan Q&A processing:
    1️⃣ Download + parse PDF using S3-cached Marker models
    2️⃣ Extract OCR text from document
    3️⃣ Use Titan directly on OCR text to answer questions
    4️⃣ Return answers in JSON format
    """
    try:
        logger.info(f"Processing document with Titan: {req.documents}")
        logger.info(f"Questions: {len(req.questions)}")
        
        # ✅ Get Titan processor instance
        processor = get_titan_processor()
        
        # ✅ Use Titan Q&A processing
        result = processor.process_document_qa_titan(
            document_url=req.documents,
            questions=req.questions
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Titan processing failed: {result.get('error', 'Unknown error')}"
            )
        
        # ✅ Return answers in correct format
        return QueryResponse(answers=result['answers'])
        
    except Exception as e:
        logger.error(f"Error in process_query_titan: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Titan processing pipeline error: {str(e)}"
        )

# ✅ Add health check endpoint
@router.get("/health")
async def health_check():
    """Check system health"""
    return {
        "status": "healthy",
        "service": "DocSpectra RAG API",
        "endpoints": {
            "run": "POST /run - Standard processing with vector search",
            "run_titan": "POST /run/titan - Direct Titan Q&A processing"
        }
    }