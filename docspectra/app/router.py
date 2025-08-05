from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from .utils.marker_handler import parse_pdf_from_url
from .utils.chunker import chunk_text
from .utils.embedder import embed_chunks
from .utils.vector_store import store_and_search_chunks
from .utils.answer_generator import generate_answer

router = APIRouter()


class QueryRequest(BaseModel):
    url: str
    query: str


class AnswerItem(BaseModel):
    clause: str


class QueryResponse(BaseModel):
    answers: List[AnswerItem]


@router.post(
    "/run",
    response_model=QueryResponse,
    summary="Process a PDF URL and return top-matched clauses"
)
async def process_query(req: QueryRequest) -> Dict[str, Any]:
    """
    End-to-end semantic search:
    1️⃣ Download + parse PDF from URL
    2️⃣ Tokenize into chunks
    3️⃣ Embed chunks + query
    4️⃣ Store into Pinecone, retrieve top-k
    5️⃣ Use Titan to generate human-readable clauses
    """
    try:
        raw_text = parse_pdf_from_url(req.url)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"PDF parsing failed: {e}"
        )

    try:
        chunks = chunk_text(raw_text)
        embeddings = embed_chunks(chunks)
        top_chunks = store_and_search_chunks(embeddings, req.query)
        answers = generate_answer(req.query, top_chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing pipeline error: {e}"
        )

    # Ensure each answer is wrapped in AnswerItem
    results = [
        AnswerItem(clause=a["clause"]) if isinstance(a, dict) else AnswerItem(clause=str(a))
        for a in answers
    ]
    return {"answers": results}
