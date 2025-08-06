# app/utils/embedder.py
import logging
from typing import List, Dict, Any
from .aws_client import aws_client
from ..config import aws_config

logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """Simplified document embedding service"""
    
    def __init__(self):
        self.model_id = aws_config.titan_embed_model
    
    def embed_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Embed multiple texts and return with metadata"""
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self._embed_single_text(text)
                embeddings.append({
                    "id": f"chunk_{i}",
                    "text": text,
                    "embedding": embedding
                })
            except Exception as e:
                logger.warning(f"Failed to embed text chunk {i}: {e}")
                continue
                
        return embeddings
    
    def _embed_single_text(self, text: str) -> List[float]:
        """Embed single text"""
        body = {"inputText": text}
        result = aws_client.invoke_bedrock_model(self.model_id, body)
        return result["embedding"]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        return self._embed_single_text(query)
