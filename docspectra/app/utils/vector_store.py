# app/utils/vector_store.py - SAFE improvements to search
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from ..config import pinecone_config, processing_config
from .embedder import DocumentEmbedder

logger = logging.getLogger(__name__)

class VectorStore:
    """Enhanced vector store with better search strategies - SAFE VERSION"""
    
    def __init__(self):
        self.config = pinecone_config
        self.embedder = DocumentEmbedder()
        self.pc = Pinecone(api_key=self.config.api_key)
        self._index = None
        logger.info("Enhanced Pinecone client initialized")
    
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
            existing_indexes = self.pc.list_indexes().names()
            
            if self.config.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.index_name}")
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=self.config.environment)
                )
                logger.info(f"Successfully created index: {self.config.index_name}")
            else:
                logger.info(f"Index {self.config.index_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise
    
    def store_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Store document embeddings with better metadata"""
        try:
            vectors_to_upsert = []
            
            for i, doc in enumerate(documents):
                # Enhanced metadata but keeping it simple
                metadata = {
                    "text": doc["text"][:1500],  # Slightly increased metadata text
                    "source": doc.get("source", "policy_document"),
                    "chunk_id": doc.get("id", f"chunk_{i}"),
                    "word_count": len(doc["text"].split())
                }
                
                vectors_to_upsert.append({
                    "id": doc.get("id", f"doc_{i}"),
                    "values": doc["embedding"],
                    "metadata": metadata
                })
            
            # Upsert in batches (Pinecone recommends batch size of 100)
            batch_size = 100
            total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
            
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} vectors)")
                self.index.upsert(vectors=batch)
                
            logger.info(f"Successfully stored {len(vectors_to_upsert)} documents in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error storing documents in vector store: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = None) -> List[str]:
        """Enhanced search for similar documents"""
        # Increase search results for better coverage
        top_k = top_k or min(processing_config.top_k_retrieval * 2, 8)
        
        try:
            # Embed the query
            query_embedding = self.embedder.embed_query(query)
            
            # Search in vector store using v3+ API
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract text from results with better filtering
            similar_texts = []
            matches = results.get("matches", [])
            
            for match in matches:
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                score = match.get("score", 0)
                
                if text and score > 0.7:  # Keep good threshold first
                    similar_texts.append(text)
                    logger.debug(f"Found high-quality match with score {score:.3f}")
            
            # If we don't have enough results, lower the threshold
            if len(similar_texts) < 3:
                logger.info("Not enough high-quality matches, expanding search...")
                for match in matches:
                    metadata = match.get("metadata", {})
                    text = metadata.get("text", "")
                    score = match.get("score", 0)
                    
                    if text and score > 0.5 and text not in similar_texts:  # Lower threshold
                        similar_texts.append(text)
                        logger.debug(f"Found expanded match with score {score:.3f}")
                        
                        if len(similar_texts) >= 5:  # Cap at 5 total results
                            break
            
            logger.info(f"Found {len(similar_texts)} similar documents for query")
            return similar_texts
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check vector store health"""
        try:
            indexes = self.pc.list_indexes()
            index_names = indexes.names() if hasattr(indexes, 'names') else []
            index_exists = self.config.index_name in index_names
            
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