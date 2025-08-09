# 2. CACHING SYSTEM - app/utils/cache_manager.py
import hashlib
import json
import os
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Simple file-based cache for embeddings and results"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_ttl = 3600  # 1 hour TTL
    
    def _get_cache_key(self, data: str) -> str:
        """Generate cache key from data"""
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_cached_embeddings(self, text: str) -> Optional[list]:
        """Get cached embeddings for text"""
        try:
            cache_key = self._get_cache_key(text)
            cache_file = os.path.join(self.cache_dir, f"embed_{cache_key}.json")
            
            if os.path.exists(cache_file):
                # Check if cache is still valid
                if time.time() - os.path.getmtime(cache_file) < self.cache_ttl:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def cache_embeddings(self, text: str, embeddings: list):
        """Cache embeddings for text"""
        try:
            cache_key = self._get_cache_key(text)
            cache_file = os.path.join(self.cache_dir, f"embed_{cache_key}.json")
            
            with open(cache_file, 'w') as f:
                json.dump(embeddings, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def get_cached_document(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached document processing result"""
        try:
            cache_key = self._get_cache_key(url)
            cache_file = os.path.join(self.cache_dir, f"doc_{cache_key}.json")
            
            if os.path.exists(cache_file):
                if time.time() - os.path.getmtime(cache_file) < self.cache_ttl:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            return None
        except Exception:
            return None
    
    def cache_document(self, url: str, result: Dict[str, Any]):
        """Cache document processing result"""
        try:
            cache_key = self._get_cache_key(url)
            cache_file = os.path.join(self.cache_dir, f"doc_{cache_key}.json")
            
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Document cache error: {e}")