# app/utils/document_processor.py - Updated for Bedrock integration
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
    """Main orchestrator for document processing pipeline with Bedrock integration"""
    
    def __init__(self, use_bedrock_qa: bool = True):
        """
        Initialize document processor with optional Bedrock Q&A integration.
        
        Args:
            use_bedrock_qa: Whether to use Bedrock Titan for Q&A (recommended)
        """
        self.chunker = TextChunker()
        self.embedder = DocumentEmbedder()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
        self.s3_bucket = aws_config.s3_bucket
        self.use_bedrock_qa = use_bedrock_qa
        
        logger.info(f"DocumentProcessor initialized with Bedrock Q&A: {use_bedrock_qa}")
    
    def process_document_qa(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Complete document processing and Q&A pipeline with enhanced Bedrock integration.
        
        This method now leverages both vector search AND Bedrock Titan for optimal results:
        1. Process document with Marker (S3 cached models)
        2. Chunk with sentence boundary awareness
        3. Generate embeddings with AWS Titan
        4. Store in Pinecone for vector search
        5. For each question:
           - Retrieve relevant chunks via vector search
           - Use Bedrock Titan to generate intelligent answers
        """
        try:
            logger.info(f"Starting document processing pipeline for: {document_url}")
            logger.info(f"Questions to answer: {len(questions)}")
            
            # Step 1: Process document with Marker using S3 models
            # The MarkerS3Context now includes Bedrock integration for Q&A
            with MarkerS3Context(self.s3_bucket, use_bedrock_qa=self.use_bedrock_qa) as marker_handler:
                doc_result = marker_handler.process_document_from_url(document_url)
                
                if not doc_result['success']:
                    return {
                        'success': False,
                        'error': f"Document processing failed: {doc_result.get('error', 'Unknown error')}"
                    }
                
                document_text = doc_result['text']
                logger.info(f"Document processed successfully: {len(document_text)} characters")
            
            # Step 2: Chunk the document for vector search
            chunks = self.chunker.chunk_text(document_text)
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Step 3: Generate embeddings using AWS Titan
            embedded_chunks = self.embedder.embed_texts(chunks)
            logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
            
            # Step 4: Store in vector database (Pinecone)
            store_success = self.vector_store.store_documents(embedded_chunks)
            if not store_success:
                logger.warning("Failed to store embeddings in vector store, continuing with direct processing")
            
            # Step 5: Answer questions using hybrid approach
            answers = []
            
            if self.use_bedrock_qa:
                # Enhanced approach: Vector search + Bedrock Titan
                answers = self._answer_questions_with_bedrock(questions, document_text, chunks)
            else:
                # Fallback approach: Vector search + traditional answer generation
                answers = self._answer_questions_traditional(questions)
            
            return {
                'success': True,
                'answers': answers,
                'metadata': {
                    'document_length': len(document_text),
                    'chunks_created': len(chunks),
                    'chunks_embedded': len(embedded_chunks),
                    'questions_processed': len(questions),
                    'qa_method': 'bedrock_hybrid' if self.use_bedrock_qa else 'traditional',
                    'vector_store_used': store_success
                }
            }
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _answer_questions_with_bedrock(self, questions: List[str], document_text: str, chunks: List[str]) -> List[str]:
        """
        Answer questions using hybrid approach: Vector search for context + Bedrock Titan for generation
        """
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Get relevant chunks via vector search
                relevant_chunks = self.vector_store.search_similar(question)
                
                if not relevant_chunks:
                    # Fallback: use all chunks or document sections
                    logger.warning(f"No relevant chunks found for question {i+1}, using document sections")
                    relevant_chunks = chunks[:5]  # Use first 5 chunks as fallback
                
                # Use Bedrock Titan to generate answer from relevant context
                context = "\n\n---\n\n".join(relevant_chunks)
                answer = self.answer_generator.generate_answer(question, relevant_chunks)
                
                answers.append(answer)
                logger.info(f"Generated answer {i+1} using Bedrock hybrid approach")
                
            except Exception as e:
                logger.error(f"Error answering question {i+1}: {e}")
                answers.append("I apologize, but I couldn't generate an answer for this question due to a technical issue.")
        
        return answers
    
    def _answer_questions_traditional(self, questions: List[str]) -> List[str]:
        """
        Answer questions using traditional approach: Vector search + rule-based generation
        """
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)} (traditional): {question[:50]}...")
            
            try:
                # Retrieve relevant chunks
                relevant_chunks = self.vector_store.search_similar(question)
                
                if not relevant_chunks:
                    answers.append("I couldn't find relevant information in the document to answer this question.")
                    continue
                
                # Generate answer using traditional method
                answer = self.answer_generator.generate_answer(question, relevant_chunks)
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error answering question {i+1}: {e}")
                answers.append("I apologize, but I couldn't generate an answer for this question.")
        
        return answers
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all pipeline components"""
        health_status = {
            'overall_status': 'checking...',
            'components': {}
        }
        
        # Check vector store
        try:
            vector_health = self.vector_store.health_check()
            health_status['components']['vector_store'] = vector_health
        except Exception as e:
            health_status['components']['vector_store'] = {'status': 'unhealthy', 'error': str(e)}
        
        # Check embedder (AWS Titan)
        try:
            from .aws_client import aws_client
            aws_health = aws_client.health_check()
            health_status['components']['aws_bedrock'] = aws_health
        except Exception as e:
            health_status['components']['aws_bedrock'] = {'status': 'unhealthy', 'error': str(e)}
        
        # Check Marker with S3 models
        try:
            with MarkerS3Context(self.s3_bucket, use_bedrock_qa=False) as marker_handler:
                marker_health = marker_handler.health_check()
                health_status['components']['marker_s3'] = marker_health
        except Exception as e:
            health_status['components']['marker_s3'] = {'status': 'unhealthy', 'error': str(e)}
        
        # Determine overall status
        component_statuses = []
        for component, status in health_status['components'].items():
            if isinstance(status, dict):
                if status.get('status') == 'healthy' or status.get('overall_status') == True:
                    component_statuses.append(True)
                else:
                    component_statuses.append(False)
            else:
                component_statuses.append(False)
        
        health_status['overall_status'] = 'healthy' if all(component_statuses) else 'degraded'
        health_status['bedrock_qa_enabled'] = self.use_bedrock_qa
        
        return health_status
    
    def process_batch_documents(self, document_urls: List[str], questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents with the same set of questions"""
        results = []
        
        for i, url in enumerate(document_urls):
            logger.info(f"Processing document {i+1}/{len(document_urls)}: {url}")
            
            try:
                result = self.process_document_qa(url, questions)
                results.append({
                    'document_url': url,
                    'document_index': i,
                    **result
                })
            except Exception as e:
                logger.error(f"Failed to process document {i+1}: {e}")
                results.append({
                    'document_url': url,
                    'document_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return results
