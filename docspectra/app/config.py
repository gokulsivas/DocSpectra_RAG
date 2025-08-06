# app/config.py - UPDATED to ensure processing_config is properly accessible
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AWSConfig:
    """Centralized AWS configuration"""
    region: str = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    titan_model_id: str = os.getenv('TITAN_MODEL_ID', 'amazon.titan-text-express-v1')
    titan_embed_model: str = os.getenv('TITAN_EMBED_MODEL', 'amazon.titan-embed-text-v1')
    s3_bucket: str = os.getenv('MARKER_S3_BUCKET', 'docspectra-models')

@dataclass
class PineconeConfig:
    """Pinecone configuration"""
    api_key: str = os.getenv('PINECONE_API_KEY')
    environment: str = os.getenv('PINECONE_ENV', 'us-east-1')
    index_name: str = 'docspectra-index'
    dimension: int = 1536

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    chunk_max_words: int = int(os.getenv('CHUNK_MAX_WORDS', '150'))
    max_tokens: int = int(os.getenv('MAX_TOKENS', '300'))
    temperature: float = float(os.getenv('TEMPERATURE', '0.7'))
    top_p: float = float(os.getenv('TOP_P', '0.9'))
    top_k_retrieval: int = int(os.getenv('TOP_K_RETRIEVAL', '5'))

# Global config instances - MAKE SURE these are created
aws_config = AWSConfig()
pinecone_config = PineconeConfig()
processing_config = ProcessingConfig()

# Add validation to ensure critical configs are set
if not pinecone_config.api_key:
    raise ValueError("PINECONE_API_KEY environment variable is required")