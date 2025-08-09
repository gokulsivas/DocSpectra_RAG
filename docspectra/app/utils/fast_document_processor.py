# app/utils/improved_fast_document_processor.py - Accuracy-focused fast processing
import logging
import asyncio
from typing import List, Dict, Any, Optional
from .marker_handler import MarkerS3Context
from .chunker import TextChunker
from .embedder import DocumentEmbedder
from .vector_store import VectorStore
from .answer_generator import AnswerGenerator
from ..config import aws_config, processing_config

logger = logging.getLogger(__name__)

class ImprovedFastDocumentProcessor:
    """Fast document processor with accuracy preservation"""
    
    def __init__(self):
        """Initialize with enhanced accuracy parameters"""
        # Enhanced chunking for better coverage
        self.chunker = TextChunker(
            max_words=120,  # Smaller chunks for better granularity
            overlap_words=30  # More overlap to avoid missing info
        )
        self.embedder = DocumentEmbedder()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
        self.s3_bucket = aws_config.s3_bucket
        
        logger.info("ImprovedFastDocumentProcessor initialized with accuracy focus")
    
    async def process_document_qa_fast(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Enhanced fast processing with accuracy preservation:
        1. Better chunking strategy
        2. Multi-level search (vector + keyword fallback)
        3. Enhanced context assembly
        4. Question-specific optimization
        """
        try:
            logger.info(f"Processing document with accuracy focus: {document_url}")
            logger.info(f"Questions: {len(questions)}")
            
            # Step 1: Process document with Marker
            with MarkerS3Context(self.s3_bucket, use_bedrock_qa=False) as marker_handler:
                doc_result = marker_handler.process_document_from_url(document_url)
                
                if not doc_result['success']:
                    return {
                        'success': False,
                        'error': f"Document processing failed: {doc_result.get('error', 'Unknown error')}"
                    }
                
                document_text = doc_result['text']
                logger.info(f"Document processed: {len(document_text)} characters")
            
            # Step 2: Enhanced chunking with better overlap
            chunks = self.chunker.chunk_text(document_text)
            logger.info(f"Created {len(chunks)} chunks with enhanced overlap")
            
            # Step 3: Generate embeddings
            embedded_chunks = self.embedder.embed_texts(chunks)
            logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
            
            # Step 4: Store in vector database
            store_success = self.vector_store.store_documents(embedded_chunks)
            if not store_success:
                return {
                    'success': False,
                    'error': 'Failed to store document in vector database'
                }
            
            # Step 5: Process questions with enhanced accuracy
            answers = await self._process_questions_enhanced(questions, document_text, chunks)
            
            return {
                'success': True,
                'answers': answers,
                'metadata': {
                    'document_length': len(document_text),
                    'chunks_created': len(chunks),
                    'processing_method': 'enhanced_accuracy',
                    'chunking_strategy': 'small_chunks_high_overlap'
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced fast processing: {e}")
            return {
                'success': False,
                'error': f"Enhanced processing failed: {str(e)}"
            }
    
    async def _process_questions_enhanced(self, questions: List[str], document_text: str, chunks: List[str]) -> List[str]:
        """Enhanced question processing with multi-level search"""
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Multi-level search approach
                answer = await self._get_enhanced_answer(question, document_text, chunks)
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers.append("I encountered an error while processing this question.")
        
        return answers
    
    async def _get_enhanced_answer(self, question: str, document_text: str, chunks: List[str]) -> str:
        """Enhanced answer generation with fallback strategies"""
        
        # Strategy 1: Vector search with higher retrieval count
        try:
            relevant_chunks = self.vector_store.search_similar(question, top_k=8)  # More chunks
            
            if relevant_chunks and len(relevant_chunks) >= 2:
                # Combine relevant chunks with better context
                context = self._build_enhanced_context(relevant_chunks, question)
                
                # Generate answer with Titan
                titan_result = self.answer_generator.generate_answer_with_context(question, context)
                
                if "error" not in titan_result and titan_result.get('answer'):
                    answer = titan_result['answer']
                    # Validate answer quality
                    if self._is_quality_answer(answer, question):
                        logger.info(f"Vector search successful for: {question[:30]}")
                        return answer
        
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
        
        # Strategy 2: Keyword-based fallback with full document
        try:
            logger.info(f"Using keyword fallback for: {question[:30]}")
            fallback_answer = await self._keyword_based_answer(question, document_text, chunks)
            
            if self._is_quality_answer(fallback_answer, question):
                return fallback_answer
                
        except Exception as e:
            logger.warning(f"Keyword fallback failed: {e}")
        
        # Strategy 3: Final fallback - direct document search
        return await self._direct_document_search(question, document_text)
    
    def _build_enhanced_context(self, relevant_chunks: List[str], question: str) -> str:
        """Build better context by organizing and filtering chunks"""
        if not relevant_chunks:
            return ""
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in relevant_chunks:
            chunk_key = chunk[:100]  # Use first 100 chars as key
            if chunk_key not in seen:
                seen.add(chunk_key)
                unique_chunks.append(chunk)
        
        # Prioritize chunks with question keywords
        question_words = set(question.lower().split())
        prioritized_chunks = []
        other_chunks = []
        
        for chunk in unique_chunks:
            chunk_words = set(chunk.lower().split())
            if len(question_words.intersection(chunk_words)) > 1:
                prioritized_chunks.append(chunk)
            else:
                other_chunks.append(chunk)
        
        # Combine prioritized first, then others
        final_chunks = prioritized_chunks + other_chunks[:3]  # Limit total chunks
        
        return "\n\n---\n\n".join(final_chunks)
    
    async def _keyword_based_answer(self, question: str, document_text: str, chunks: List[str]) -> str:
        """Enhanced keyword-based search as fallback"""
        question_lower = question.lower()
        
        # Extract key terms (enhanced)
        key_terms = self._extract_key_terms(question)
        logger.info(f"Extracted key terms: {key_terms}")
        
        # Score chunks based on keyword matches
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = 0
            
            # Exact phrase matching (higher weight)
            for term in key_terms:
                if len(term) > 3 and term in chunk_lower:
                    score += 3
            
            # Individual word matching
            question_words = question_lower.split()
            for word in question_words:
                if len(word) > 3 and word in chunk_lower:
                    score += 1
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        if not scored_chunks:
            return "I couldn't find relevant information in the document to answer this question."
        
        # Sort by score and take top chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        top_chunks = [chunk for score, chunk in scored_chunks[:5]]
        
        # Build context and generate answer
        context = "\n\n---\n\n".join(top_chunks)
        
        titan_result = self.answer_generator.generate_answer_with_context(question, context)
        
        return titan_result.get('answer', "I couldn't generate a proper answer from the available information.")
    
    async def _direct_document_search(self, question: str, document_text: str) -> str:
        """Final fallback: search entire document"""
        try:
            # Simple but effective full-text search
            sentences = [s.strip() for s in document_text.split('.') if s.strip()]
            key_terms = self._extract_key_terms(question)
            
            best_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for term in key_terms if term in sentence_lower)
                
                if score > 0:
                    best_sentences.append((score, sentence))
            
            if best_sentences:
                best_sentences.sort(reverse=True, key=lambda x: x[0])
                # Take top 3 sentences as context
                context = '. '.join([s for score, s in best_sentences[:3]])
                
                titan_result = self.answer_generator.generate_answer_with_context(question, context)
                return titan_result.get('answer', "Information may be available but couldn't be processed properly.")
            
            return "I couldn't find relevant information in the document to answer this question."
            
        except Exception as e:
            logger.error(f"Direct search failed: {e}")
            return "I couldn't find relevant information in the document to answer this question."
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from question for better matching"""
        # Common stop words to ignore
        stop_words = {
            'what', 'is', 'the', 'how', 'does', 'are', 'there', 'any', 'and', 
            'or', 'for', 'in', 'on', 'at', 'to', 'a', 'an', 'can', 'will',
            'would', 'should', 'could', 'when', 'where', 'why', 'who', 'which',
            'of', 'with', 'by', 'from', 'this', 'that', 'these', 'those'
        }
        
        # Domain-specific important terms (add more as needed)
        important_terms = {
            'grace period': ['grace period', 'grace'],
            'waiting period': ['waiting period', 'waiting'],
            'pre-existing': ['pre-existing', 'ped'],
            'maternity': ['maternity', 'pregnancy'],
            'coverage': ['coverage', 'covered'],
            'hospital': ['hospital'],
            'treatment': ['treatment'],
            'premium': ['premium'],
            'policy': ['policy'],
            'claim': ['claim']
        }
        
        question_lower = question.lower()
        key_terms = []
        
        # First, look for important multi-word terms
        for concept, terms in important_terms.items():
            for term in terms:
                if term in question_lower:
                    key_terms.append(term)
        
        # Then extract individual meaningful words
        words = [word.strip('?.,!()') for word in question_lower.split()]
        for word in words:
            if len(word) > 3 and word not in stop_words and word not in key_terms:
                key_terms.append(word)
        
        return key_terms
    
    def _is_quality_answer(self, answer: str, question: str) -> bool:
        """Check if the answer meets quality criteria"""
        if not answer or len(answer.strip()) < 10:
            return False
        
        # Check for common "no answer" phrases
        no_answer_phrases = [
            "couldn't find", "not available", "no information", 
            "unable to find", "not specified", "i don't have",
            "cannot determine", "insufficient information"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in no_answer_phrases):
            return False
        
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check"""
        try:
            vector_health = self.vector_store.health_check()
            
            return {
                "status": "healthy",
                "processor_type": "ImprovedFastDocumentProcessor",
                "accuracy_features": [
                    "Enhanced chunking with overlap",
                    "Multi-level search (vector + keyword)",
                    "Quality validation",
                    "Multiple fallback strategies"
                ],
                "vector_store": vector_health,
                "chunking_config": {
                    "max_words": 120,
                    "overlap_words": 30,
                    "top_k_retrieval": 8
                }
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}