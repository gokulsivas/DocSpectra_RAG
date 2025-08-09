# app/utils/document_processor.py - RAG with Vector DB using PDF processor
import logging
from typing import List, Dict, Any, Optional
from .pdf_processor import PDFProcessor
from .chunker import TextChunker
from .embedder import DocumentEmbedder
from .vector_store import VectorStore
from .answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main orchestrator for document processing pipeline with RAG"""
    
    def __init__(self):
        """Initialize document processor with required components"""
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker()
        self.embedder = DocumentEmbedder()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
        
        logger.info("DocumentProcessor initialized with PDF + OCR pipeline")
    
    def process_document_qa(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Complete RAG document processing pipeline:
        1. PyPDF/OCR extracts text from PDF
        2. Chunk text and store in vectorDB (Pinecone)
        3. For each question: Search Vector DB for relevant chunks
        4. Use Titan LLM with relevant chunks to generate answers
        5. Return JSON format answers
        """
        try:
            logger.info(f"Processing document: {document_url}")
            logger.info(f"Questions: {len(questions)}")
            
            # Step 1: Process document with PDF processor
            doc_result = self.pdf_processor.process_document_from_url(document_url)
            
            if not doc_result['success']:
                return {
                    'success': False,
                    'error': f"Document processing failed: {doc_result.get('error', 'Unknown error')}"
                }
            
            document_text = doc_result['text']
            extraction_method = doc_result.get('method', 'unknown')
            logger.info(f"Document processed successfully using {extraction_method}: {len(document_text)} characters")
            
            # Step 2: Chunk the document for vector search
            chunks = self.chunker.chunk_text(document_text)
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Step 3: Generate embeddings using AWS Titan
            embedded_chunks = self.embedder.embed_texts(chunks)
            logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
            
            # Step 4: Store in vector database (Pinecone)
            store_success = self.vector_store.store_documents(embedded_chunks)
            if not store_success:
                logger.warning("Failed to store embeddings in vector store")
                return {
                    'success': False,
                    'error': 'Failed to store document in vector database'
                }
            
            # Step 5: Process each question using RAG approach
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
                
                try:
                    # Search Vector DB for relevant chunks
                    relevant_chunks = self.vector_store.search_similar(question, top_k=5)
                    
                    if not relevant_chunks:
                        logger.warning(f"No relevant chunks found for question {i+1}")
                        answers.append("I couldn't find relevant information in the document to answer this question.")
                        continue
                    
                    # Combine relevant chunks into context
                    context = "\n\n---\n\n".join(relevant_chunks)
                    logger.info(f"Found {len(relevant_chunks)} relevant chunks for question {i+1}")
                    
                    # Use Titan LLM with relevant context
                    titan_result = self.answer_generator.generate_answer_with_context(question, context)
                    
                    if "error" in titan_result:
                        logger.error(f"Error generating answer for question {i+1}: {titan_result['error']}")
                        answers.append(f"I encountered an error while processing this question: {titan_result['error']}")
                    else:
                        answers.append(titan_result['answer'])
                        
                except Exception as e:
                    logger.error(f"Error processing question {i+1}: {e}")
                    answers.append("I encountered an error while processing this question.")
            
            return {
                'success': True,
                'answers': answers,
                'metadata': {
                    'document_length': len(document_text),
                    'extraction_method': extraction_method,
                    'pages_processed': doc_result.get('pages_processed', 0),
                    'chunks_created': len(chunks),
                    'chunks_stored': len(embedded_chunks),
                    'questions_processed': len(questions),
                    'vector_store': 'pinecone',
                    'model_used': 'amazon.titan-text-express-v1',
                    'rag_method': 'vector_search'
                }
            }
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {e}")
            return {
                'success': False,
                'error': f"Document processing failed: {str(e)}"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        try:
            # Check PDF processor health
            pdf_health = self.pdf_processor.health_check()
            
            # Check vector store health
            vector_health = self.vector_store.health_check()
            
            return {
                "status": "healthy" if (pdf_health["status"] == "healthy" and 
                                     vector_health["status"] in ["healthy", "index_missing"]) else "unhealthy",
                "pdf_processor": pdf_health,
                "vector_store": vector_health,
                "pinecone": "connected" if vector_health.get("pinecone_connected") else "disconnected"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy", 
                "error": str(e)
            }