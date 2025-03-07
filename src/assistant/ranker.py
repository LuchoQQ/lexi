# assistant/ranker.py
"""Advanced ranking for legal information."""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from cross_encoder import CrossEncoder
from ..exceptions import RetrievalError

class LegalRanker:
    """Advanced ranking system for legal information."""
    
    def __init__(self, rerank_model: CrossEncoder):
        """Initialize the legal ranker.
        
        Args:
            rerank_model: Cross-encoder model for reranking
        """
        self.rerank_model = rerank_model
    
    def rank_chunks(self, 
                   query: str, 
                   chunks: List[Dict[str, Any]], 
                   context_aware: bool = True) -> List[Dict[str, Any]]:
        """Rank chunks based on relevance to query.
        
        Args:
            query: User query
            chunks: Retrieved chunks to rank
            context_aware: Whether to use contextual ranking
            
        Returns:
            Ranked chunks
        """
        if not chunks:
            return []
        
        try:
            # Create pairs for reranking
            pairs = [(query, chunk["content"]) for chunk in chunks if "content" in chunk]
            
            # Skip if no valid pairs
            if not pairs:
                return chunks
            
            # Predict relevance scores
            scores = self.rerank_model.predict(pairs)
            
            # Apply context-aware ranking if enabled
            if context_aware:
                # Identify chunks with article references
                has_articles = []
                for chunk in chunks:
                    content = chunk.get("content", "").lower()
                    has_article = "artículo" in content or "articulo" in content or "art." in content
                    has_articles.append(has_article)
                
                # Boost chunks with article references
                for i, has_article in enumerate(has_articles):
                    if has_article and i < len(scores):
                        scores[i] *= 1.1  # 10% boost
            
            # Update chunks with scores
            for chunk, score in zip(chunks, scores):
                chunk["relevance_score"] = float(score)
            
            # Sort by score
            ranked_chunks = sorted(chunks, key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            return ranked_chunks
            
        except Exception as e:
            logging.error(f"Error ranking chunks: {e}")
            raise RetrievalError(f"Failed to rank chunks: {e}")
    
    def diversify_results(self, 
                         chunks: List[Dict[str, Any]], 
                         diversity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Diversify results to avoid redundancy.
        
        Args:
            chunks: Ranked chunks
            diversity_threshold: Threshold for considering chunks similar
            
        Returns:
            Diversified chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        try:
            # Extract chunk content
            contents = [chunk.get("content", "") for chunk in chunks]
            
            # Initialize results with top chunk
            results = [chunks[0]]
            used_indices = {0}
            
            for i in range(1, len(chunks)):
                # Skip if already used
                if i in used_indices:
                    continue
                
                # Check similarity with already selected chunks
                is_similar = False
                for selected in results:
                    # Simple text overlap similarity as a heuristic
                    similarity = self._text_similarity(chunks[i].get("content", ""), selected.get("content", ""))
                    if similarity > diversity_threshold:
                        is_similar = True
                        break
                
                # Add if not too similar to existing results
                if not is_similar:
                    results.append(chunks[i])
                    used_indices.add(i)
                    
                # Break if we have enough results
                if len(results) >= min(5, len(chunks)):
                    break
            
            # If we don't have enough diverse results, add remaining by rank
            for i in range(len(chunks)):
                if i not in used_indices and len(results) < len(chunks):
                    results.append(chunks[i])
                    used_indices.add(i)
            
            return results
            
        except Exception as e:
            logging.error(f"Error diversifying results: {e}")
            # Return original results if diversification fails
            return chunks
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        # Convert to sets of words for Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0