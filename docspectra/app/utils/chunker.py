# app/utils/chunker.py
import re
import logging
from typing import List
from ..config import processing_config

logger = logging.getLogger(__name__)

class TextChunker:
    """Enhanced text chunking with overlap and sentence boundary awareness"""
    
    def __init__(self, max_words: int = None, overlap_words: int = 20):
        self.max_words = max_words or processing_config.chunk_max_words
        self.overlap_words = overlap_words
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text with sentence boundary awareness and robust error handling"""
        # Input validation and conversion
        if text is None:
            logger.warning("Received None as text input")
            return []
        
        if isinstance(text, list):
            logger.warning("Received list as text input, converting to string")
            text = '\n\n'.join(str(item) for item in text)
        elif not isinstance(text, str):
            logger.warning(f"Received unexpected type {type(text)} as text input, converting to string")
            text = str(text)
        
        # Clean and validate text
        text = text.strip()
        if not text:
            logger.warning("Received empty text after cleaning")
            return []
        
        logger.info(f"Chunking text of {len(text)} characters")
        
        try:
            # Split into sentences first
            sentences = self._split_into_sentences(text)
            if not sentences:
                logger.warning("No sentences found in text, returning full text as single chunk")
                return [text]
            
            logger.info(f"Found {len(sentences)} sentences in text")
            
            chunks = []
            current_chunk = []
            current_word_count = 0
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                sentence_words = len(sentence.split())
                
                # If adding this sentence exceeds max words, finalize current chunk
                if current_word_count + sentence_words > self.max_words and current_chunk:
                    chunk_text = ' '.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    overlap_sentences = self._get_overlap_sentences(current_chunk)
                    current_chunk = overlap_sentences
                    current_word_count = sum(len(s.split()) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_word_count += sentence_words
            
            # Add final chunk if it has content
            if current_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            
            # Fallback: if no chunks created, create one from the full text
            if not chunks and text.strip():
                logger.warning("No chunks created through normal process, using full text")
                chunks = [text[:self.max_words * 10]]  # Rough character limit fallback
            
            logger.info(f"Created {len(chunks)} chunks from text")
            return [chunk.strip() for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            logger.error(f"Error in chunk_text: {e}")
            logger.error(f"Text type: {type(text)}, Text length: {len(text) if text else 0}")
            logger.error(f"Text sample: {repr(text[:100]) if text else 'None'}")
            
            # Emergency fallback: split by characters
            if text and isinstance(text, str):
                logger.info("Using emergency fallback chunking")
                chunk_size = self.max_words * 6  # Rough estimate: 6 chars per word
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]
            else:
                return []
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with error handling"""
        try:
            # Handle edge cases
            if not text or not isinstance(text, str):
                return []
            
            # Simple sentence splitting - can be enhanced with spaCy/NLTK
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # If no sentences found with punctuation, split by newlines
            if not sentences:
                sentences = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Final fallback: split by length if still no sentences
            if not sentences and text.strip():
                # Split into 100-word chunks as "sentences"
                words = text.split()
                sentences = []
                for i in range(0, len(words), 100):
                    sentence = ' '.join(words[i:i+100])
                    if sentence.strip():
                        sentences.append(sentence)
            
            return sentences
            
        except Exception as e:
            logger.error(f"Error in _split_into_sentences: {e}")
            return [text] if text else []
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get last few sentences for overlap with error handling"""
        if not sentences:
            return []
        
        try:
            overlap_word_count = 0
            overlap_sentences = []
            
            for sentence in reversed(sentences):
                if not sentence:
                    continue
                word_count = len(sentence.split())
                if overlap_word_count + word_count <= self.overlap_words:
                    overlap_sentences.insert(0, sentence)
                    overlap_word_count += word_count
                else:
                    break
                    
            return overlap_sentences
            
        except Exception as e:
            logger.error(f"Error in _get_overlap_sentences: {e}")
            return sentences[-1:] if sentences else []