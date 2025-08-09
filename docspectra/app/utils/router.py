# app/router.py - RAG with Vector DB + Optional Authentication
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)
router = APIRouter()

# Security scheme
security = HTTPBearer(auto_error=False)

# Competition token from environment variable
HACKRX_TOKEN = os.getenv('HACKRX_TOKEN', '1a63ddf923df12069f9dbdf8b30e53d518c3c6684b1ce8d21cc37a3970001170')

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

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Verify Bearer token - optional for local development"""
    if credentials is None:
        # No token provided - allow for local development
        logger.info("No authentication token provided - allowing for local development")
        return None
    
    token = credentials.credentials
    if token == HACKRX_TOKEN:
        logger.info("✅ Valid HackRx token provided")
        return token
    else:
        logger.warning(f"❌ Invalid token provided: {token[:20]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )

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
async def process_query(
    req: QueryRequest, 
    token: Optional[str] = Depends(verify_token)
) -> Dict[str, Any]:
    """
    RAG document processing pipeline:
    1️⃣ PDF processor extracts text from PDF
    2️⃣ Chunk text and store in vectorDB (Pinecone)
    3️⃣ For each question: Search Vector DB for relevant chunks
    4️⃣ Use Titan LLM with relevant context to generate answers
    5️⃣ Return answers wrapped in object
    
    Authentication: Optional Bearer token for competition submission
    """
    try:
        # Log authentication status
        auth_status = "authenticated" if token else "local development"
        logger.info(f"Processing request ({auth_status})")
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
        
    except HTTPException:
        # Re-raise HTTP exceptions (like auth errors)
        raise
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Processing pipeline error: {str(e)}"
        )

# ✅ Add health check endpoint
@router.get("/health")
async def health_check(token: Optional[str] = Depends(verify_token)):
    """Check system health"""
    auth_status = "authenticated" if token else "local development"
    
    return {
        "status": "healthy",
        "service": "DocSpectra RAG API",
        "auth_mode": auth_status,
        "endpoints": {
            "run": "POST /run - RAG processing with PDF + VectorDB + Titan"
        }
    }

# ✅ Add token validation endpoint
@router.get("/validate-token")
async def validate_token_endpoint(token: str = Depends(verify_token)):
    """Validate authentication token"""
    return {
        "valid": True,
        "token_type": "HackRx Competition Token",
        "status": "Token is valid"
    }