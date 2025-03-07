# embeddings/model.py
"""Embedding model handling with caching."""

import os
import hashlib
import pickle
import numpy as np
import torch
from typing import List, Dict, Optional, Union
import logging
from sentence_transformers import SentenceTransformer
from src.exceptions import EmbeddingError

class EmbeddingModel:
    """Embedding model with caching capabilities."""
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
            cache_dir: Directory for caching embeddings (None to disable)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Create cache directory if specified and doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Initialize model
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(model_name, device=self.device)
            logging.info(f"Loaded embedding model {model_name} on {self.device}")
            
            # Limit number of threads for CPU mode
            if self.device == "cpu":
                torch.set_num_threads(2)  # Cambia a un número bajo
                # También puedes establecer las variables de entorno
                os.environ["OMP_NUM_THREADS"] = "2"
                os.environ["MKL_NUM_THREADS"] = "2"
                self.model = SentenceTransformer(model_name, device=self.device)
                logging.info(f"Loaded embedding model {model_name} on {self.device} with {torch.get_num_threads()} threads")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model {model_name}: {e}")
            
    def _cache_key(self, text: str) -> str:
        """Generate a cache key for a text.
        
        Args:
            text: Input text
            
        Returns:
            Cache key
        """
        # Create a hash of the model name and text
        key = f"{self.model_name}_{text}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Get embeddings from cache if available.
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding or None if not found
        """
        if not self.cache_dir:
            return None
        
        cache_key = self._cache_key(text)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embeddings to cache.
        
        Args:
            text: Input text
            embedding: Embedding vector
        """
        if not self.cache_dir:
            return
        
        cache_key = self._cache_key(text)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logging.warning(f"Failed to save to cache: {e}")
    
    async def encode_async(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings asynchronously.
        
        Args:
            texts: Input text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Embeddings as numpy array
        """
        import asyncio
        
        # Handle single text
        if isinstance(texts, str):
            # Check cache first
            cached = self._get_from_cache(texts)
            if cached is not None:
                return cached
            
            # Encode and cache
            try:
                # Run in a thread to not block the event loop
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(None, self.model.encode, texts)
                self._save_to_cache(texts, embedding)
                return embedding
            except Exception as e:
                raise EmbeddingError(f"Failed to encode text: {e}")
        
        # Handle list of texts
        else:
            results = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_results = []
                
                # Check cache for each text
                for text in batch:
                    cached = self._get_from_cache(text)
                    if cached is not None:
                        batch_results.append(cached)
                    else:
                        batch_results.append(None)
                
                # Encode texts not found in cache
                texts_to_encode = [text for j, text in enumerate(batch) if batch_results[j] is None]
                if texts_to_encode:
                    try:
                        # Run in a thread to not block the event loop
                        loop = asyncio.get_event_loop()
                        embeddings = await loop.run_in_executor(None, self.model.encode, texts_to_encode)
                        
                        # Cache new embeddings
                        for text, embedding in zip(texts_to_encode, embeddings):
                            self._save_to_cache(text, embedding)
                        
                        # Update batch results
                        embed_idx = 0
                        for j in range(len(batch)):
                            if batch_results[j] is None:
                                batch_results[j] = embeddings[embed_idx]
                                embed_idx += 1
                                
                    except Exception as e:
                        raise EmbeddingError(f"Failed to encode batch: {e}")
                
                results.extend(batch_results)
            
            return np.array(results)
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings synchronously.
        
        Args:
            texts: Input text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Embeddings as numpy array
        """
        # Handle single text
        if isinstance(texts, str):
            # Check cache first
            cached = self._get_from_cache(texts)
            if cached is not None:
                return cached
            
            # Encode and cache
            try:
                embedding = self.model.encode(texts)
                self._save_to_cache(texts, embedding)
                return embedding
            except Exception as e:
                raise EmbeddingError(f"Failed to encode text: {e}")
        
        # Handle list of texts
        else:
            results = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_results = []
                
                # Check cache for each text
                for text in batch:
                    cached = self._get_from_cache(text)
                    if cached is not None:
                        batch_results.append(cached)
                    else:
                        batch_results.append(None)
                
                # Encode texts not found in cache
                texts_to_encode = [text for j, text in enumerate(batch) if batch_results[j] is None]
                if texts_to_encode:
                    try:
                        embeddings = self.model.encode(texts_to_encode)
                        
                        # Cache new embeddings
                        for text, embedding in zip(texts_to_encode, embeddings):
                            self._save_to_cache(text, embedding)
                        
                        # Update batch results
                        embed_idx = 0
                        for j in range(len(batch)):
                            if batch_results[j] is None:
                                batch_results[j] = embeddings[embed_idx]
                                embed_idx += 1
                                
                    except Exception as e:
                        raise EmbeddingError(f"Failed to encode batch: {e}")
                
                results.extend(batch_results)
            
            return np.array(results)