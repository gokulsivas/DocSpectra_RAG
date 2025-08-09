# app/config.py - Enhanced configuration management
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AWSConfig:
    """AWS configuration with Titan support"""
    region: str = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    titan_model_id: str = os.getenv('TITAN_MODEL_ID', 'amazon.titan-text-express-v1')
    titan_embed_model: str = os.getenv('TITAN_EMBED_MODEL', 'amazon.titan-embed-text-v1')
    temperature: float = float(os.getenv('TEMPERATURE', '0.3'))
    top_p: float = float(os.getenv('TOP_P', '0.9'))
    max_tokens: int = int(os.getenv('MAX_TOKEN_COUNT', '2048'))
    
    def get(self, key: str, default: str = '') -> str:
        """Get configuration value with fallback"""
        return getattr(self, key, default)

@dataclass
class PineconeConfig:
    """Pinecone configuration"""
    api_key: str = os.getenv('PINECONE_API_KEY', '')
    index_name: str = os.getenv('PINECONE_INDEX_NAME', 'docspectra-index')
    environment: str = os.getenv('PINECONE_ENV', 'us-east-1')
    dimension: int = int(os.getenv('PINECONE_DIMENSION', '1536'))

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    chunk_size: int = int(os.getenv('CHUNK_SIZE', '1000'))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', '200'))
    chunk_max_words: int = int(os.getenv('CHUNK_MAX_WORDS', '150'))
    top_k_retrieval: int = int(os.getenv('TOP_K_RETRIEVAL', '5'))

@dataclass
class OCRConfig:
    """OCR configuration"""
    tesseract_cmd: str = os.getenv('TESSERACT_CMD', 'tesseract')
    ocr_lang: str = os.getenv('OCR_LANG', 'eng')
    dpi: int = int(os.getenv('OCR_DPI', '300'))
    timeout: int = int(os.getenv('OCR_TIMEOUT', '60'))

# Global config instances
aws_config = AWSConfig()
pinecone_config = PineconeConfig()
processing_config = ProcessingConfig()
ocr_config = OCRConfig()

# Add validation function
def validate_config():
    """Validate that required environment variables are set"""
    if not pinecone_config.api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")