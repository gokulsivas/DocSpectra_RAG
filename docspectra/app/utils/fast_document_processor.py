# 1. FAST Document Processing - app/utils/fast_document_processor.py
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from .pdf_processor import PDFProcessor
from .chunker import TextChunker
from .embedder import DocumentEmbedder
from .vector_store import VectorStore
from .answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)

class FastDocumentProcessor:
    """Optimized processor focusing on speed + accuracy"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker()
        self.embedder = DocumentEmbedder()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
        
        # Pre-compile question patterns for faster matching
        self.question_patterns = self._compile_question_patterns()
        
        logger.info("FastDocumentProcessor initialized with speed optimizations")
    
    def _compile_question_patterns(self) -> Dict[str, List[str]]:
        """Pre-compile common insurance question patterns"""
        return {
            "grace_period": ["grace period", "premium payment", "late payment"],
            "waiting_period": ["waiting period", "pre-existing", "PED"],
            "maternity": ["maternity", "pregnancy", "childbirth"],
            "cataract": ["cataract", "eye surgery", "vision"],
            "organ_donor": ["organ donor", "transplant", "donation"],
            "ncd": ["no claim discount", "NCD", "bonus"],
            "health_check": ["preventive", "health check", "screening"],
            "hospital": ["hospital", "definition", "establishment"],
            "ayush": ["ayurveda", "yoga", "naturopathy", "unani", "siddha", "homeopathy"],
            "sub_limits": ["sub-limits", "sub limits", "room rent", "ICU"]
        }
    
    async def process_document_qa_fast(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        OPTIMIZED pipeline with parallel processing and smart shortcuts
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Fast processing: {document_url} with {len(questions)} questions")
            
            # STEP 1: Fast PDF processing with size limits
            doc_result = await self._fast_pdf_processing(document_url)
            if not doc_result['success']:
                return {'success': False, 'error': doc_result.get('error')}
            
            document_text = doc_result['text']
            processing_time_1 = time.time() - start_time
            logger.info(f"⏱️ PDF processing: {processing_time_1:.2f}s")
            
            # STEP 2: Smart chunking with parallel embedding
            chunks_and_embeddings = await self._parallel_chunk_and_embed(document_text)
            processing_time_2 = time.time() - start_time
            logger.info(f"⏱️ Chunking + Embedding: {processing_time_2 - processing_time_1:.2f}s")
            
            # STEP 3: Fast vector store operations
            store_success = await self._fast_vector_store(chunks_and_embeddings)
            if not store_success:
                return {'success': False, 'error': 'Vector store failed'}
            
            processing_time_3 = time.time() - start_time
            logger.info(f"⏱️ Vector storage: {processing_time_3 - processing_time_2:.2f}s")
            
            # STEP 4: Parallel question processing with smart routing
            answers = await self._parallel_question_processing(questions)
            
            total_time = time.time() - start_time
            logger.info(f"🎉 Total processing time: {total_time:.2f}s")
            
            return {
                'success': True,
                'answers': answers,
                'metadata': {
                    'total_time': total_time,
                    'document_length': len(document_text),
                    'chunks_created': len(chunks_and_embeddings),
                    'processing_method': 'fast_parallel'
                }
            }
            
        except Exception as e:
            logger.error(f"Fast processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _fast_pdf_processing(self, document_url: str) -> Dict[str, Any]:
        """Fast PDF processing with early text detection"""
        loop = asyncio.get_event_loop()
        
        # Run PDF processing in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor, 
                self.pdf_processor.process_document_from_url, 
                document_url
            )
        
        # Early termination if text extraction failed
        if not result['success'] or len(result.get('text', '')) < 100:
            return {'success': False, 'error': 'Insufficient text extracted'}
        
        return result
    
    async def _parallel_chunk_and_embed(self, document_text: str) -> List[Dict[str, Any]]:
        """Chunk and embed in parallel for speed"""
        # Chunking
        chunks = self.chunker.chunk_text(document_text)
        
        # Parallel embedding with thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Split chunks into batches for parallel processing
            batch_size = max(1, len(chunks) // 3)
            batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            
            # Process batches in parallel
            futures = [
                loop.run_in_executor(executor, self._embed_batch, i, batch)
                for i, batch in enumerate(batches)
            ]
            
            batch_results = await asyncio.gather(*futures)
            
            # Combine results
            embedded_chunks = []
            for batch_result in batch_results:
                embedded_chunks.extend(batch_result)
            
            return embedded_chunks
    
    def _embed_batch(self, batch_id: int, chunk_batch: List[str]) -> List[Dict[str, Any]]:
        """Embed a batch of chunks"""
        logger.debug(f"Processing embedding batch {batch_id} with {len(chunk_batch)} chunks")
        return self.embedder.embed_texts(chunk_batch)
    
    async def _fast_vector_store(self, embedded_chunks: List[Dict[str, Any]]) -> bool:
        """Fast vector store with async operations"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            success = await loop.run_in_executor(
                executor,
                self.vector_store.store_documents,
                embedded_chunks
            )
        return success
    
    async def _parallel_question_processing(self, questions: List[str]) -> List[str]:
        """Process questions in parallel with smart routing"""
        if len(questions) <= 2:
            # Sequential for small batches (less overhead)
            return [self._process_question_fast(q) for q in questions]
        
        # Parallel for larger batches
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=min(4, len(questions))) as executor:
            futures = [
                loop.run_in_executor(executor, self._process_question_fast, question)
                for question in questions
            ]
            answers = await asyncio.gather(*futures)
            return answers
    
    def _process_question_fast(self, question: str) -> str:
        """Fast question processing with optimized search"""
        try:
            # Quick pattern matching for common questions
            enhanced_question = self._enhance_question_fast(question)
            
            # Optimized search with caching
            relevant_chunks = self.vector_store.search_similar(enhanced_question, top_k=5)
            
            if not relevant_chunks:
                return "I couldn't find relevant information in the document to answer this question."
            
            # Fast context assembly (limit context size for speed)
            context = "\n\n".join(relevant_chunks[:4])  # Limit to top 4 chunks
            if len(context) > 3000:  # Truncate long context for speed
                context = context[:3000] + "..."
            
            # Generate answer with timeout protection
            result = self.answer_generator.generate_answer_with_context(question, context)
            
            return result.get('answer', 'Error generating answer')
            
        except Exception as e:
            logger.error(f"Fast question processing error: {e}")
            return "Error processing this question."
    
    def _enhance_question_fast(self, question: str) -> str:
        """Fast question enhancement using pre-compiled patterns"""
        question_lower = question.lower()
        
        for pattern_key, keywords in self.question_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                # Add just the most relevant keywords (not all)
                primary_keyword = keywords[0]
                return f"{question} {primary_keyword}"
        
        return question
