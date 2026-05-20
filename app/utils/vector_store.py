# app/utils/vector_store.py - FIXED for Pinecone SDK 3.x+
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from ..config import pinecone_config, processing_config  # Both imports added
from .embedder import DocumentEmbedder

logger = logging.getLogger(__name__)

class VectorStore:
    """Enhanced vector store with Pinecone SDK 3.x+ compatibility"""
    
    def __init__(self):
        self.config = pinecone_config
        self.embedder = DocumentEmbedder()
        
        # FIXED: Initialize Pinecone client with v3+ API
        self.pc = Pinecone(api_key=self.config.api_key)
        self._index = None
        
        logger.info("Pinecone client initialized successfully")
    
    @property 
    def index(self):
        """Lazy-loaded Pinecone index"""
        if self._index is None:
            self._ensure_index_exists()
            self._index = self.pc.Index(self.config.index_name)
        return self._index
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist - FIXED for Pinecone SDK 3.x+"""
        try:
            # Get list of existing indexes
            existing_indexes = self.pc.list_indexes().names()
            
            if self.config.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.index_name}")
                
                # Create index with v3+ API
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,  # 1536 from your config
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.config.environment  # us-east-1 from your config
                    )
                )
                logger.info(f"Successfully created index: {self.config.index_name}")
            else:
                logger.info(f"Index {self.config.index_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise
    
    def store_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Store document embeddings in vector store"""
        try:
            vectors_to_upsert = []
            
            for i, doc in enumerate(documents):
                vectors_to_upsert.append({
                    "id": doc.get("id", f"doc_{i}"),
                    "values": doc["embedding"],
                    "metadata": {
                        "text": doc["text"][:1000],  # Limit metadata text to avoid size issues
                        "source": doc.get("source", "unknown")
                    }
                })
            
            # Upsert in batches (Pinecone recommends batch size of 100)
            batch_size = 100
            total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
            
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} vectors)")
                
                # Use the v3+ upsert method
                self.index.upsert(vectors=batch)
                
            logger.info(f"Successfully stored {len(vectors_to_upsert)} documents in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error storing documents in vector store: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = None) -> List[str]:
        """Search for similar documents"""
        top_k = top_k or processing_config.top_k_retrieval
        
        try:
            # Embed the query
            query_embedding = self.embedder.embed_query(query)
            
            # Search in vector store using v3+ API
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract text from results
            similar_texts = []
            matches = results.get("matches", [])
            
            for match in matches:
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                if text:
                    similar_texts.append(text)
                    
                # Log similarity score for debugging
                score = match.get("score", 0)
                logger.debug(f"Found match with score {score:.3f}")
            
            logger.info(f"Found {len(similar_texts)} similar documents for query")
            return similar_texts
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check vector store health"""
        try:
            # Check if we can list indexes
            indexes = self.pc.list_indexes()
            index_names = indexes.names() if hasattr(indexes, 'names') else []
            
            # Check if our index exists
            index_exists = self.config.index_name in index_names
            
            # If index exists, get stats
            index_stats = None
            if index_exists:
                try:
                    index_stats = self.index.describe_index_stats()
                except Exception as e:
                    logger.warning(f"Could not get index stats: {e}")
            
            return {
                "status": "healthy" if index_exists else "index_missing",
                "pinecone_connected": True,
                "index_name": self.config.index_name,
                "index_exists": index_exists,
                "total_indexes": len(index_names),
                "index_stats": index_stats
            }
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "status": "unhealthy",
                "pinecone_connected": False,
                "error": str(e)
            }