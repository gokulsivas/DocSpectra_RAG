# app/utils/document_processor.py
import logging
from typing import List, Dict, Any, Optional
from .marker_handler import MarkerS3Context
from .chunker import TextChunker
from .embedder import DocumentEmbedder
from .vector_store import VectorStore
from .answer_generator import AnswerGenerator
from ..config import aws_config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main orchestrator for document processing pipeline"""
    
    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = DocumentEmbedder()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
        self.s3_bucket = aws_config.s3_bucket
    
    def process_document_qa(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """Complete document processing and Q&A pipeline"""
        try:
            # Step 1: Process document with Marker
            with MarkerS3Context(self.s3_bucket) as marker_handler:
                doc_result = marker_handler.process_document_from_url(document_url)
                
                if not doc_result['success']:
                    return {
                        'success': False,
                        'error': f"Document processing failed: {doc_result.get('error', 'Unknown error')}"
                    }
                
                document_text = doc_result['text']
            
            # Step 2: Chunk the document
            chunks = self.chunker.chunk_text(document_text)
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Step 3: Generate embeddings
            embedded_chunks = self.embedder.embed_texts(chunks)
            
            # Step 4: Store in vector database
            self.vector_store.store_documents(embedded_chunks)
            
            # Step 5: Answer questions
            answers = []
            for question in questions:
                # Retrieve relevant chunks
                relevant_chunks = self.vector_store.search_similar(question)
                
                # Generate answer
                answer = self.answer_generator.generate_answer(question, relevant_chunks)
                answers.append(answer)
            
            return {
                'success': True,
                'answers': answers,
                'metadata': {
                    'chunks_created': len(chunks),
                    'chunks_embedded': len(embedded_chunks),
                    'questions_processed': len(questions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {e}")
            return {
                'success': False,
                'error': str(e)
            }
