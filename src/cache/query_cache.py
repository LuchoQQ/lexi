# Nuevo archivo: src/cache/query_cache.py
import hashlib
import json
import time
from typing import Dict, Any, Optional

class QueryCache:
    """Sistema de caché para consultas legales frecuentes."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        
    def _generate_key(self, query: str, filters: Dict[str, Any]) -> str:
        """Generate a cache key from query and filters."""
        key_data = {"query": query, "filters": filters}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, filters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get a cached result if it exists and is not expired."""
        key = self._generate_key(query, filters or {})
        
        if key in self.cache:
            cache_entry = self.cache[key]
            
            # Check if entry is expired
            if time.time() - cache_entry["timestamp"] > self.ttl:
                del self.cache[key]
                return None
                
            return cache_entry["data"]
        
        return None
        
    def set(self, query: str, filters: Dict[str, Any], data: Dict[str, Any]):
        """Store a query result in cache."""
        key = self._generate_key(query, filters or {})
        
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "timestamp": time.time(),
            "data": data
        }