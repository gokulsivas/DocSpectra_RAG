# app/utils/chunker.py - Improved chunking strategy
import re
import logging
from typing import List
from ..config import processing_config

logger = logging.getLogger(__name__)

class TextChunker:
    """Enhanced text chunking with overlap and sentence boundary awareness"""
    
    def __init__(self, max_words: int = None, overlap_words: int = 50):
        # Increase chunk size for insurance documents
        self.max_words = max_words or 250  # Increased from 150
        self.overlap_words = overlap_words  # Increased from 20
        
        # Insurance-specific section patterns
        self.section_patterns = [
            r'^\s*\d+\.\s*',  # 1. Section
            r'^\s*[A-Z][a-z]+:',  # Title:
            r'^\s*COVERAGE',  # Coverage sections
            r'^\s*BENEFITS',  # Benefits
            r'^\s*EXCLUSIONS',  # Exclusions
            r'^\s*CONDITIONS',  # Conditions
            r'^\s*DEFINITIONS',  # Definitions
        ]
    
    def chunk_text(self, text: str) -> List[str]:
        """Enhanced chunking with insurance-specific logic"""
        if not text or not isinstance(text, str):
            return []
        
        text = text.strip()
        if not text:
            return []
        
        logger.info(f"Chunking text of {len(text)} characters")
        
        try:
            # First, try to identify major sections
            sections = self._identify_sections(text)
            
            if sections:
                # Chunk each section separately to maintain context
                all_chunks = []
                for section_title, section_content in sections:
                    section_chunks = self._chunk_section(section_title, section_content)
                    all_chunks.extend(section_chunks)
                return all_chunks
            else:
                # Fall back to sentence-based chunking
                return self._chunk_by_sentences(text)
            
        except Exception as e:
            logger.error(f"Error in chunk_text: {e}")
            # Emergency fallback
            return self._simple_chunk(text)
    
    def _identify_sections(self, text: str) -> List[tuple]:
        """Identify major sections in insurance documents"""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if line looks like a section header
            is_section_header = any(re.match(pattern, line.strip()) for pattern in self.section_patterns)
            
            if is_section_header:
                # Save previous section
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                
                # Start new section
                current_section = line.strip()
                current_content = []
            else:
                if line.strip():  # Skip empty lines
                    current_content.append(line)
        
        # Don't forget the last section
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        return sections if len(sections) > 2 else []  # Only use if we found meaningful sections
    
    def _chunk_section(self, section_title: str, section_content: str) -> List[str]:
        """Chunk a single section while preserving context"""
        if not section_content.strip():
            return []
        
        # Split into sentences
        sentences = self._split_into_sentences(section_content)
        if not sentences:
            return [f"{section_title}\n{section_content}"]
        
        chunks = []
        current_chunk = [section_title]  # Start each chunk with section title
        current_word_count = len(section_title.split())
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds max words, finalize current chunk
            if current_word_count + sentence_words > self.max_words and len(current_chunk) > 1:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with section title + overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk[1:])  # Exclude title
                current_chunk = [section_title] + overlap_sentences
                current_word_count = len(' '.join(current_chunk).split())
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add final chunk
        if len(current_chunk) > 1:  # More than just the title
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Original sentence-based chunking as fallback"""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text[:self.max_words * 6]]  # Rough fallback
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > self.max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Add overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_word_count = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting for insurance documents"""
        try:
            # Handle insurance-specific patterns
            text = re.sub(r'(\d+)\.\s*([A-Z])', r'\1. \2', text)  # Fix numbering
            text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)  # Add periods between sentences
            
            # Split on multiple sentence endings
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            
            # If no sentences found, split by newlines
            if not sentences:
                sentences = [line.strip() for line in text.split('\n') if line.strip()]
            
            return sentences
            
        except Exception as e:
            logger.error(f"Error in sentence splitting: {e}")
            return [text] if text else []
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap with increased size"""
        if not sentences:
            return []
        
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
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Simple character-based chunking as final fallback"""
        chunk_size = self.max_words * 6  # Rough estimate
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks if chunks else [text]