# app/utils/chunker.py
import re
from typing import List
from ..config import processing_config

class TextChunker:
    """Enhanced text chunking with overlap and sentence boundary awareness"""
    
    def __init__(self, max_words: int = None, overlap_words: int = 20):
        self.max_words = max_words or processing_config.chunk_max_words
        self.overlap_words = overlap_words
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text with sentence boundary awareness"""
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds max words, finalize current chunk
            if current_word_count + sentence_words > self.max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_word_count = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add final chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be enhanced with spaCy/NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get last few sentences for overlap"""
        overlap_word_count = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            word_count = len(sentence.split())
            if overlap_word_count + word_count <= self.overlap_words:
                overlap_sentences.insert(0, sentence)
                overlap_word_count += word_count
            else:
                break
                
        return overlap_sentences
