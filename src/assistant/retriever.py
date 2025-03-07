# src/assistant/retriever.py
"""Enhanced retrieval system with hybrid search and query expansion."""

import asyncio
import logging
import traceback
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.cross_encoder import CrossEncoder
from src.exceptions import RetrievalError
from src.utils.text_processing import extract_articles, extract_legal_concepts, extract_penalties
from src.embeddings.model import EmbeddingModel
from src.embeddings.vector_store import VectorStore

class LegalRetriever:
    """Enhanced legal information retriever with hybrid search capabilities."""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 knowledge_graph,
                 embedding_model: EmbeddingModel,
                 rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 legal_synonyms: Optional[Dict[str, List[str]]] = None):
        """Initialize the legal retriever.
        
        Args:
            vector_store: Vector store for semantic search
            knowledge_graph: Knowledge graph for graph-based retrieval
            embedding_model: Embedding model for text encoding
            rerank_model_name: Name of the reranking model
            legal_synonyms: Dictionary of legal term synonyms (optional)
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model
        self.legal_synonyms = legal_synonyms or {}
        
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Loading reranking model {rerank_model_name} on {device}")
            self.rerank_model = CrossEncoder(rerank_model_name, device=device)
            logging.info(f"Loaded reranking model on {device}")
        except Exception as e:
            logging.error(f"Failed to load reranking model: {e}")
            logging.error(traceback.format_exc())
            raise RetrievalError(f"Failed to load reranking model: {e}")
    
    async def _vector_search(self, query: str, top_k: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform vector search in the database."""
        logging.info(f"Executing vector search for query: '{query}' with top_k={top_k}")
        
        try:
            # Fix empty where filter - ChromaDB requires specific format
            where_filter = None
            if where and len(where) > 0:
                logging.info(f"Applying metadata filter: {where}")
                if len(where) == 1:
                    # Si solo hay un filtro, usar directamente sin $and
                    key, value = next(iter(where.items()))
                    where_filter = {key: {"$eq": value}}
                else:
                    # Para múltiples filtros usar $and
                    where_filter = {"$and": [{k: {"$eq": v}} for k, v in where.items()]}
            
            # Query the vector store (usando query en lugar de query_async)
            results = self.vector_store.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )
            
            logging.info(f"Vector store returned {len(results['ids'][0])} results")
            
            # Process results into a standard format
            retrieved_chunks = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results["ids"][0], 
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )):
                # Convert distance to similarity score
                similarity = 1.0 - min(distance, 1.0)
                
                chunk = {
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": similarity,
                    "retrieval_method": "vector"
                }
                retrieved_chunks.append(chunk)
                
                if i < 3:  # Log a few top results for debugging
                    logging.debug(f"Vector result #{i+1} - ID: {doc_id}, Score: {similarity:.4f}")
            
            return retrieved_chunks
            
        except Exception as e:
            error_msg = f"Error in vector search: {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            # Return empty list instead of raising exception to allow fallback to other methods
            logging.warning("Returning empty result list after vector search error")
            return []
    
    
    async def _graph_search(self, 
                        articles: List[str], 
                        concepts: List[str],
                        penalties: List[str],
                        max_results: int = 10) -> List[Dict[str, Any]]:
        """Retrieve chunks using graph-based search.
        
        Args:
            articles: Article entities extracted from query
            concepts: Concept entities extracted from query
            penalties: Penalty entities extracted from query
            max_results: Maximum number of results to return
            
        Returns:
            List of retrieved chunks
        """
        logging.info(f"Executing graph search with {len(articles)} articles, {len(concepts)} concepts, and {len(penalties)} penalties")
        
        relevant_chunk_ids = set()
        entity_chunk_mapping = {}  # To track which entity led to which chunk
        
        # Process articles
        for article in articles:
            entity_type = self.knowledge_graph._infer_entity_type(article)
            normalized = self.knowledge_graph._normalize_entity(article, entity_type)
            
            logging.debug(f"Looking for chunks related to article: '{normalized}'")
            if self.knowledge_graph.graph.has_node(normalized):
                chunk_ids = self.knowledge_graph.get_related_chunks(normalized)
                logging.debug(f"Found {len(chunk_ids)} chunks related to article '{normalized}'")
                for chunk_id in chunk_ids:
                    relevant_chunk_ids.add(chunk_id)
                    entity_chunk_mapping[chunk_id] = entity_chunk_mapping.get(chunk_id, []) + [normalized]
            else:
                logging.debug(f"Article '{normalized}' not found in knowledge graph")
        
        # Process concepts
        for concept in concepts:
            entity_type = self.knowledge_graph._infer_entity_type(concept)
            normalized = self.knowledge_graph._normalize_entity(concept, entity_type)
            
            logging.debug(f"Looking for chunks related to concept: '{normalized}'")
            if self.knowledge_graph.graph.has_node(normalized):
                chunk_ids = self.knowledge_graph.get_related_chunks(normalized)
                logging.debug(f"Found {len(chunk_ids)} chunks related to concept '{normalized}'")
                for chunk_id in chunk_ids:
                    relevant_chunk_ids.add(chunk_id)
                    entity_chunk_mapping[chunk_id] = entity_chunk_mapping.get(chunk_id, []) + [normalized]
            else:
                logging.debug(f"Concept '{normalized}' not found in knowledge graph")
        
        # Process penalties
        for penalty in penalties:
            entity_type = self.knowledge_graph._infer_entity_type(penalty)
            normalized = self.knowledge_graph._normalize_entity(penalty, entity_type)
            
            logging.debug(f"Looking for chunks related to penalty: '{normalized}'")
            if self.knowledge_graph.graph.has_node(normalized):
                chunk_ids = self.knowledge_graph.get_related_chunks(normalized)
                logging.debug(f"Found {len(chunk_ids)} chunks related to penalty '{normalized}'")
                for chunk_id in chunk_ids:
                    relevant_chunk_ids.add(chunk_id)
                    entity_chunk_mapping[chunk_id] = entity_chunk_mapping.get(chunk_id, []) + [normalized]
            else:
                logging.debug(f"Penalty '{normalized}' not found in knowledge graph")
        
        # Find additional chunks through graph relationships
        # For each entity, get related entities and their chunks
        all_entities = [self.knowledge_graph._normalize_entity(e, self.knowledge_graph._infer_entity_type(e)) 
                    for e in articles + concepts + penalties]
        
        for entity in all_entities:
            if self.knowledge_graph.graph.has_node(entity):
                # Get subgraph of depth 1
                subgraph = self.knowledge_graph.get_entity_subgraph(entity, depth=1)
                
                # For each node in the subgraph, add its chunks
                for node in subgraph.nodes():
                    if node != entity:  # Skip the original entity
                        chunk_ids = self.knowledge_graph.get_related_chunks(node)
                        for chunk_id in chunk_ids:
                            relevant_chunk_ids.add(chunk_id)
                            entity_chunk_mapping[chunk_id] = entity_chunk_mapping.get(chunk_id, []) + [f"{entity}>{node}"]
        
        logging.info(f"Graph search identified {len(relevant_chunk_ids)} relevant chunks")
        
        # Retrieve content for chunks that have numeric IDs
        # These should be available in the vector store
        chunk_ids_list = []
        for chunk_id in relevant_chunk_ids:
            if chunk_id.startswith("chunk_"):
                try:
                    # Extract numeric index
                    chunk_idx = int(chunk_id.split("_")[1])
                    chunk_ids_list.append(chunk_id)
                except (IndexError, ValueError):
                    logging.warning(f"Could not parse chunk ID: {chunk_id}")
        
        logging.info(f"Need to fetch content for {len(chunk_ids_list)} chunks from vector store")
        
        # Fetch content from vector store for all chunks at once
        chunk_contents = {}
        if chunk_ids_list:
            try:
                # Query vector store by IDs
                results = await self.vector_store.collection.get(
                    ids=chunk_ids_list,
                    include=["documents", "metadatas"]
                )
                
                # Process results
                if results and "documents" in results:
                    for i, doc_id in enumerate(results.get("ids", [])):
                        if i < len(results["documents"]):
                            chunk_contents[doc_id] = {
                                "content": results["documents"][i],
                                "metadata": results.get("metadatas", [{}])[i] if i < len(results.get("metadatas", [])) else {}
                            }
                    logging.info(f"Successfully retrieved {len(chunk_contents)} chunks from vector store")
                else:
                    logging.warning("No documents returned from vector store query")
            except Exception as e:
                logging.error(f"Error fetching chunk contents: {e}")
                logging.error(traceback.format_exc())
        
        # Create the graph-based results
        graph_chunks = []
        for chunk_id in relevant_chunk_ids:
            related_entities = entity_chunk_mapping.get(chunk_id, [])
            
            # Create chunk with content if available
            if chunk_id in chunk_contents:
                graph_chunks.append({
                    "id": chunk_id,
                    "content": chunk_contents[chunk_id]["content"],
                    "metadata": chunk_contents[chunk_id].get("metadata", {}),
                    "retrieved_by": ", ".join(related_entities),
                    "retrieval_method": "graph",
                    "graph_score": 0.8  # Placeholder score - would be calculated based on relevance
                })
            else:
                # For unknown chunks, add with placeholder
                graph_chunks.append({
                    "id": chunk_id,
                    "content": f"No content available for chunk {chunk_id}",
                    "retrieved_by": ", ".join(related_entities),
                    "retrieval_method": "graph",
                    "graph_score": 0.5  # Lower score for chunks without content
                })
        
        # Limit number of results
        result_chunks = graph_chunks[:max_results]
        logging.info(f"Returning {len(result_chunks)} graph search results (limited from {len(graph_chunks)})")
        return result_chunks
    async def retrieve(self, 
                  query: str, 
                  jurisdiction: Optional[str] = None,
                  legal_domain: Optional[str] = None,
                  top_k: int = 15, 
                  use_hybrid: bool = True,
                  use_expansion: bool = True) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query using hybrid search."""
        
        try:
            logging.info(f"Starting retrieval for query: '{query}'")
            logging.info(f"Parameters: top_k={top_k}, use_hybrid={use_hybrid}, use_expansion={use_expansion}")
            
            # Verificar si es una búsqueda específica de artículo
            import re
            article_pattern = r'\barticulo\s+(\d+)\b|\bart\.?\s+(\d+)\b'
            article_match = re.search(article_pattern, query.lower())
            
            # Si es una búsqueda directa de artículo, primero intentar por metadatos
            if article_match:
                article_num = article_match.group(1) or article_match.group(2)
                logging.info(f"Detectada búsqueda de artículo específico: {article_num}")
                
                # Crear filtro para búsqueda exacta por metadatos
                where_filter = {"article": article_num}
                
                # Realizar búsqueda vectorial con filtro exacto
                exact_results = await self._vector_search(query, top_k, where=where_filter)
                
                if exact_results and len(exact_results) > 0:
                    logging.info(f"Encontrados {len(exact_results)} resultados exactos para el artículo {article_num}")
                    # Añadir método de recuperación especial
                    for result in exact_results:
                        result["retrieval_method"] = "exact_article_match"
                    return exact_results
                else:
                    logging.info(f"No se encontraron resultados exactos para el artículo {article_num}, continuando con búsqueda normal")
            
            
            logging.info(f"Starting retrieval for query: '{query}'")
            logging.info(f"Parameters: top_k={top_k}, use_hybrid={use_hybrid}, use_expansion={use_expansion}")
            
            # 1. Extract key entities from query
            articles = extract_articles(query)
            concepts = extract_legal_concepts(query)
            penalties = extract_penalties(query)
            
            logging.info(f"Extracted entities - Articles: {articles}, Concepts: {concepts}, Penalties: {penalties}")
            
            # 2. Expand query if enabled
            expanded_query = query
            if use_expansion:
                expanded_query = await self._expand_query(query, articles, concepts)
                logging.info(f"Expanded query: '{expanded_query}'")
            
            # Crear filtro de metadata para la consulta vectorial
            where_filter = {}
            if jurisdiction:
                where_filter["jurisdiction"] = jurisdiction
            if legal_domain:
                where_filter["legal_domain"] = legal_domain
            
            # 3. Perform vector search
            logging.info("Performing vector search")
            vector_results = await self._vector_search(expanded_query, top_k, where=where_filter)
            logging.info(f"Vector search returned {len(vector_results)} results")
            
            # 4. Add graph-based results if enabled
            if use_hybrid:
                logging.info("Performing graph search")
                graph_results = await self._graph_search(articles, concepts, penalties)
                logging.info(f"Graph search returned {len(graph_results)} results")
                
                # Combine results
                all_results = self._combine_results(vector_results, graph_results)
                logging.info(f"Combined {len(vector_results)} vector results with {len(graph_results)} graph results for a total of {len(all_results)} results")
            else:
                all_results = vector_results
            
            # 5. Rerank results
            logging.info("Reranking results")
            reranked_results = await self._rerank_results(query, all_results, top_k)
            logging.info(f"Reranked to {len(reranked_results)} results")
            
            # 6. Ensure diversity in the results
            logging.info("Ensuring diversity in results")
            diverse_results = self._ensure_diversity(reranked_results)
            logging.info(f"Final diverse result set contains {len(diverse_results)} chunks")
            
            # Log the first result to help diagnose issues
            if diverse_results:
                logging.info(f"Top result - ID: {diverse_results[0].get('id')}, Score: {diverse_results[0].get('rerank_score')}")
                logging.debug(f"Top result content preview: {diverse_results[0].get('content', '')[:100]}...")
            else:
                logging.warning("No results found after all retrieval steps")
            
            return diverse_results
            
        except Exception as e:
            error_msg = f"Error retrieving information: {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            raise RetrievalError(f"Failed to retrieve information: {e}")
    
    async def _expand_query(self, 
                           query: str, 
                           articles: List[str], 
                           concepts: List[str]) -> str:
        """Expand the query with synonyms and related terms.
        
        Args:
            query: Original query
            articles: Articles mentioned in the query
            concepts: Concepts mentioned in the query
            
        Returns:
            Expanded query
        """
        logging.info(f"Expanding query with {len(articles)} articles and {len(concepts)} concepts")
        expansion_terms = []
        
        # 1. Add synonyms for legal concepts
        for concept in concepts:
            if concept in self.legal_synonyms:
                expansion_terms.extend(self.legal_synonyms[concept])
                logging.debug(f"Added synonyms for concept '{concept}': {self.legal_synonyms[concept]}")
        
        # 2. Add related concepts from knowledge graph
        for concept in concepts:
            entity_type = self.knowledge_graph._infer_entity_type(concept)
            normalized = self.knowledge_graph._normalize_entity(concept, entity_type)
            
            if self.knowledge_graph.graph.has_node(normalized):
                # Get directly connected concepts
                for neighbor in self.knowledge_graph.graph.successors(normalized):
                    if self.knowledge_graph.graph.nodes[neighbor].get("type") == "concept":
                        expansion_terms.append(neighbor)
                        logging.debug(f"Added related concept '{neighbor}' from knowledge graph")
        
        # 3. Add normalized article references
        for article in articles:
            entity_type = self.knowledge_graph._infer_entity_type(article)
            normalized = self.knowledge_graph._normalize_entity(article, entity_type)
            
            # Convert normalized form back to human-readable
            if normalized.startswith("articulo_"):
                readable = "artículo " + normalized[9:]
                expansion_terms.append(readable)
                logging.debug(f"Added normalized article reference: '{readable}'")
        
        # 4. Construct expanded query
        if expansion_terms:
            # Remove duplicates and limit number of expansion terms
            unique_terms = list(set(expansion_terms))[:5]
            expanded_query = f"{query} {' '.join(unique_terms)}"
            logging.info(f"Query expanded with {len(unique_terms)} unique terms")
            return expanded_query
        else:
            logging.info("No expansion terms found, using original query")
            return query
        
    def _combine_results(self, 
                        vector_results: List[Dict[str, Any]], 
                        graph_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine vector and graph search results.
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            
        Returns:
            Combined results
        """
        logging.info(f"Combining {len(vector_results)} vector results with {len(graph_results)} graph results")
        
        # Identify unique IDs
        vector_ids = {chunk["id"] for chunk in vector_results}
        logging.debug(f"Vector result IDs: {vector_ids}")
        
        # Merge results, avoiding duplicates
        combined = vector_results.copy()
        
        added_from_graph = 0
        updated_from_graph = 0
        
        for graph_chunk in graph_results:
            if graph_chunk["id"] not in vector_ids:
                # Add new chunk from graph search
                combined.append(graph_chunk)
                added_from_graph += 1
            else:
                # Update existing chunk with graph information
                for i, chunk in enumerate(combined):
                    if chunk["id"] == graph_chunk["id"]:
                        # Update with graph info
                        combined[i]["retrieval_method"] = "vector+graph"
                        combined[i]["retrieved_by"] = graph_chunk.get("retrieved_by", "")
                        # Boost score for items found by both methods
                        combined[i]["similarity_score"] = max(
                            combined[i].get("similarity_score", 0) * 1.2,  # 20% boost
                            0.95  # Cap at 0.95
                        )
                        updated_from_graph += 1
        
        logging.info(f"Combined results: {len(combined)} total (added {added_from_graph} new, updated {updated_from_graph} existing)")
        return combined
    
    async def _rerank_results(self, 
                             query: str, 
                             results: List[Dict[str, Any]],
                             top_k: int) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder model.
        
        Args:
            query: Original query
            results: Combined retrieval results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        if not results:
            logging.warning("No results to rerank")
            return []
        
        logging.info(f"Reranking {len(results)} results with top_k={top_k}")
        
        # Prepare pairs for reranking
        pairs = []
        chunk_contents = []
        
        # Check content presence
        no_content_count = 0
        for chunk in results:
            if "content" not in chunk:
                no_content_count += 1
        
        if no_content_count > 0:
            logging.warning(f"{no_content_count} chunks missing content field")
        
        # Fix chunks that are missing content
        for chunk in results:
            # Skip chunks without content
            if "content" not in chunk:
                # Try to retrieve content if this is a graph result
                if "id" in chunk and chunk.get("retrieval_method") == "graph":
                    # In a real implementation, this would fetch from the vector store
                    # For now, add a placeholder content
                    logging.debug(f"Adding placeholder content for graph result {chunk['id']}")
                    chunk["content"] = f"Placeholder content for {chunk['id']}"
                else:
                    logging.debug(f"Skipping chunk without content: {chunk.get('id', 'unknown')}")
                    continue
                
            pairs.append((query, chunk["content"]))
            chunk_contents.append(chunk)
        
        if not pairs:
            logging.warning("No valid pairs for reranking")
            return []
        
        # Rerank
        try:
            # Use model to predict
            logging.info(f"Running reranking model on {len(pairs)} pairs")
            scores = self.rerank_model.predict(pairs)
            
            # Update scores
            for i, (chunk, score) in enumerate(zip(chunk_contents, scores)):
                chunk["rerank_score"] = float(score)
                if i < 3:  # Log top scores for debugging
                    logging.debug(f"Rerank score for {chunk['id']}: {score:.4f}")
            
            # Sort by rerank score
            reranked = sorted(chunk_contents, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            # Trim to top_k
            result = reranked[:top_k]
            logging.info(f"Reranking complete. Top score: {result[0]['rerank_score']:.4f} if results exist")
            return result
            
        except Exception as e:
            error_msg = f"Error in reranking: {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            logging.warning("Falling back to original results order")
            # Fall back to original results
            return results[:top_k]
    
    def _ensure_diversity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity in the results by selecting varied content types.
        
        Args:
            results: Ranked retrieval results
            
        Returns:
            Diverse results
        """
        if len(results) <= 3:
            logging.info("Too few results to ensure diversity, returning all")
            return results
        
        logging.info(f"Ensuring diversity among {len(results)} results")
        
        # Categorize chunks
        definitions = []
        penalties = []
        procedures = []
        general = []
        
        for chunk in results:
            content = chunk.get("content", "").lower()
            
            # Simple categorization based on content
            if any(term in content for term in ["se entiende", "se define", "definición", "concepto de"]):
                definitions.append(chunk)
            elif any(term in content for term in ["pena", "sanción", "castigo", "prisión", "multa", "años"]):
                penalties.append(chunk)
            elif any(term in content for term in ["procedimiento", "proceso", "trámite"]):
                procedures.append(chunk)
            else:
                general.append(chunk)
        
        logging.info(f"Categorized chunks - Definitions: {len(definitions)}, Penalties: {len(penalties)}, Procedures: {len(procedures)}, General: {len(general)}")
        
        # Build diverse set by selecting from each category
        diverse = []
        
        # Always include top result
        if results:
            diverse.append(results[0])
        
        # Add one from each category if available
        categories = [definitions, penalties, procedures]
        for category in categories:
            if category:
                # Avoid duplicating top result
                if category[0] != diverse[0]:
                    diverse.append(category[0])
        
        # Fill remaining slots from general results
        general_filtered = [g for g in general if g not in diverse]
        remaining_slots = len(results) - len(diverse)
        
        diverse.extend(general_filtered[:remaining_slots])
        
        logging.info(f"Selected {len(diverse)} diverse results")
        return diverse
async def retrieve(self, 
                  query: str, 
                  jurisdiction: Optional[str] = None,
                  legal_domain: Optional[str] = None,
                  top_k: int = 15, 
                  use_hybrid: bool = True,
                  use_expansion: bool = True) -> List[Dict[str, Any]]:
    """Retrieve with domain filtering."""
    try:
        logging.info(f"Starting retrieval for query: '{query}'")
        logging.info(f"Parameters: top_k={top_k}, use_hybrid={use_hybrid}, use_expansion={use_expansion}")
        
        # 1. Extract key entities from query
        articles = extract_articles(query)
        concepts = extract_legal_concepts(query)
        penalties = extract_penalties(query)
        
        logging.info(f"Extracted entities - Articles: {articles}, Concepts: {concepts}, Penalties: {penalties}")
        
        # 2. Expand query if enabled
        expanded_query = query
        if use_expansion:
            expanded_query = await self._expand_query(query, articles, concepts)
            logging.info(f"Expanded query: '{expanded_query}'")
        
        # Crear filtro de metadata para la consulta vectorial
        where_filter = {}
        if jurisdiction:
            where_filter["jurisdiction"] = jurisdiction
        if legal_domain:
            where_filter["legal_domain"] = legal_domain
        
        # 3. Perform vector search
        logging.info("Performing vector search")
        vector_results = await self._vector_search(expanded_query, top_k, where=where_filter)
        logging.info(f"Vector search returned {len(vector_results)} results")
        
        # 4. Add graph-based results if enabled
        if use_hybrid:
            logging.info("Performing graph search")
            graph_results = await self._graph_search(articles, concepts, penalties)
            logging.info(f"Graph search returned {len(graph_results)} results")
            
            # Combine results
            all_results = self._combine_results(vector_results, graph_results)
            logging.info(f"Combined {len(vector_results)} vector results with {len(graph_results)} graph results for a total of {len(all_results)} results")
        else:
            all_results = vector_results
        
        # 5. Rerank results
        logging.info("Reranking results")
        reranked_results = await self._rerank_results(query, all_results, top_k)
        logging.info(f"Reranked to {len(reranked_results)} results")
        
        # 6. Ensure diversity in the results
        logging.info("Ensuring diversity in results")
        diverse_results = self._ensure_diversity(reranked_results)
        logging.info(f"Final diverse result set contains {len(diverse_results)} chunks")
        
        # Log the first result to help diagnose issues
        if diverse_results:
            logging.info(f"Top result - ID: {diverse_results[0].get('id')}, Score: {diverse_results[0].get('rerank_score')}")
            logging.debug(f"Top result content preview: {diverse_results[0].get('content', '')[:100]}...")
        else:
            logging.warning("No results found after all retrieval steps")
        
        return diverse_results
        
    except Exception as e:
        error_msg = f"Error retrieving information: {e}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        raise RetrievalError(f"Failed to retrieve information: {e}")
    
    
    
