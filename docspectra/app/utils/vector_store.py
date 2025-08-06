# app/utils/vector_store.py - FIXED VERSION
import logging
from typing import List, Dict, Any, Optional
import pinecone
from pinecone import ServerlessSpec
from ..config import pinecone_config, processing_config  # FIXED: Added processing_config
from .embedder import DocumentEmbedder

logger = logging.getLogger(__name__)

class VectorStore:
    """Enhanced vector store with better error handling"""
    
    def __init__(self):
        self.config = pinecone_config
        self.embedder = DocumentEmbedder()
        self._index = None
        
        # FIXED: Initialize Pinecone with v3 API using config values
        pinecone.init(
            api_key=self.config.api_key,        # This gets PINECONE_API_KEY from env
            environment=self.config.environment  # This gets PINECONE_ENV from env
        )
        
    @property 
    def index(self):
        """Lazy-loaded Pinecone index"""
        if self._index is None:
            self._ensure_index_exists()
            self._index = pinecone.Index(self.config.index_name)
        return self._index
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist - FIXED for v3 API"""
        existing_indexes = pinecone.list_indexes()
        
        if self.config.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.config.index_name}")
            pinecone.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.config.environment
                )
            )
    
    def store_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Store document embeddings in vector store"""
        try:
            vectors_to_upsert = []
            
            for i, doc in enumerate(documents):
                vectors_to_upsert.append({
                    "id": doc.get("id", f"doc_{i}"),
                    "values": doc["embedding"],
                    "metadata": {
                        "text": doc["text"],
                        "source": doc.get("source", "unknown")
                    }
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
                
            logger.info(f"Stored {len(vectors_to_upsert)} documents in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = None) -> List[str]:
        """Search for similar documents"""
        top_k = top_k or self.config.top_k_retrieval  # FIXED: Use self.config instead of processing_config
        
        try:
            # Embed the query
            query_embedding = self.embedder.embed_query(query)
            
            # Search in vector store
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract text from results
            similar_texts = [
                match["metadata"]["text"] 
                for match in results.get("matches", [])
            ]
            
            logger.info(f"Found {len(similar_texts)} similar documents for query")
            return similar_texts
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []