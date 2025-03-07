# knowledge_graph/queries.py
"""Advanced query operations for the knowledge graph."""

import networkx as nx
from typing import List, Dict, Any, Set, Optional
import logging
from src.exceptions import GraphError

class GraphReasoner:
    """Reasoning and querying operations for the legal knowledge graph."""
    
    def __init__(self, graph):
        """Initialize the graph reasoner.
        
        Args:
            graph: KnowledgeGraph instance
        """
        self.graph = graph
    
    def reason_over_graph(self, query: str, entities: List[str]) -> Dict[str, Any]:
        """Perform reasoning over the graph for a specific query.
        
        Args:
            query: User query text
            entities: List of entities extracted from the query
            
        Returns:
            Dictionary with reasoning results
        """
        # Initialize results structure
        graph_insights = {
            "key_entities": [],
            "definitions": [],
            "related_articles": [],
            "related_concepts": [],
            "hierarchical_relations": [],
            "contradictions": []
        }
        
        try:
            # Process each entity
            for entity in entities:
                entity_type = self.graph._infer_entity_type(entity)
                normalized_entity = self.graph._normalize_entity(entity, entity_type)
                
                # Skip if entity not in graph
                if not self.graph.graph.has_node(normalized_entity):
                    continue
                
                # Get entity information
                entity_info = self._get_entity_info(normalized_entity)
                if entity_info:
                    graph_insights["key_entities"].append(entity_info)
                
                # Get entity definitions
                if entity_type == "concept":
                    definitions = self._get_concept_definitions(normalized_entity)
                    graph_insights["definitions"].extend(definitions)
                
                # Get related articles for concepts
                if entity_type == "concept":
                    related_articles = self._get_concept_related_articles(normalized_entity)
                    graph_insights["related_articles"].extend(related_articles)
                
                # Get related concepts for articles
                if entity_type == "article":
                    related_concepts = self._get_article_related_concepts(normalized_entity)
                    graph_insights["related_concepts"].extend(related_concepts)
                
                # Get hierarchical relations for articles
                if entity_type == "article":
                    hierarchical = self._get_article_hierarchical_relations(normalized_entity)
                    graph_insights["hierarchical_relations"].extend(hierarchical)
            
            # Find potential contradictions between any mentioned articles
            article_entities = [e for e in entities if self.graph._infer_entity_type(e) == "article"]
            contradictions = self._find_article_contradictions(article_entities)
            graph_insights["contradictions"] = contradictions
            
            return graph_insights
            
        except Exception as e:
            logging.error(f"Error in graph reasoning: {e}")
            raise GraphError(f"Failed to perform reasoning over graph: {e}")
    
    def _get_entity_info(self, entity: str) -> Dict[str, Any]:
        """Get detailed information about an entity.
        
        Args:
            entity: Entity ID
            
        Returns:
            Entity information dictionary
        """
        if not self.graph.graph.has_node(entity):
            return {}
        
        node_data = self.graph.graph.nodes[entity]
        entity_type = node_data.get("type", "unknown")
        
        # Base information
        info = {
            "entity": entity,
            "type": entity_type,
            "mentioned_in": len(self.graph.get_related_chunks(entity))
        }
        
        # Type-specific information
        if entity_type == "article":
            # Get concepts defined/mentioned by this article
            concepts = []
            for succ in self.graph.graph.successors(entity):
                if self.graph.graph.nodes[succ].get("type") == "concept":
                    edge_data = self.graph.graph.get_edge_data(entity, succ)
                    concepts.append({
                        "concept": succ,
                        "relation": edge_data.get("type", "mentions")
                    })
            info["related_concepts"] = concepts
            
            # Get penalties imposed by this article
            penalties = []
            for succ in self.graph.graph.successors(entity):
                if self.graph.graph.nodes[succ].get("type") == "penalty":
                    penalties.append(succ)
            info["imposed_penalties"] = penalties
            
        elif entity_type == "concept":
            # Get articles that define this concept
            defining_articles = []
            for pred in self.graph.graph.predecessors(entity):
                if self.graph.graph.nodes[pred].get("type") == "article":
                    edge_data = self.graph.graph.get_edge_data(pred, entity)
                    if edge_data.get("type") == "define":
                        defining_articles.append(pred)
            info["defined_in"] = defining_articles
            
            # Get related concepts
            related_concepts = []
            for succ in self.graph.graph.successors(entity):
                if self.graph.graph.nodes[succ].get("type") == "concept":
                    related_concepts.append(succ)
            info["related_concepts"] = related_concepts
        
        return info
    
    def _get_concept_definitions(self, concept: str) -> List[Dict[str, Any]]:
        """Get definitions of a concept.
        
        Args:
            concept: Concept ID
            
        Returns:
            List of definition information
        """
        definitions = []
        
        if not self.graph.graph.has_node(concept):
            return definitions
        
        # Find articles that define this concept
        for pred in self.graph.graph.predecessors(concept):
            if self.graph.graph.nodes[pred].get("type") == "article":
                edge_data = self.graph.graph.get_edge_data(pred, concept)
                if edge_data.get("type") == "define":
                    # Get the chunk that contains this definition
                    chunk_id = edge_data.get("source")
                    if chunk_id:
                        chunks = list(self.graph.entity_to_chunks.get(concept, set()))
                        definitions.append({
                            "concept": concept,
                            "defined_in": pred,
                            "source_chunk": chunk_id
                        })
        
        return definitions
    
    def _get_concept_related_articles(self, concept: str) -> List[Dict[str, Any]]:
        """Get articles related to a concept.
        
        Args:
            concept: Concept ID
            
        Returns:
            List of related article information
        """
        related = []
        
        if not self.graph.graph.has_node(concept):
            return related
        
        # Find articles that mention or define this concept
        for pred in self.graph.graph.predecessors(concept):
            if self.graph.graph.nodes[pred].get("type") == "article":
                edge_data = self.graph.graph.get_edge_data(pred, concept)
                related.append({
                    "article": pred,
                    "concept": concept,
                    "relation": edge_data.get("type", "mentions")
                })
        
        return related
    
    def _get_article_related_concepts(self, article: str) -> List[Dict[str, Any]]:
        """Get concepts related to an article.
        
        Args:
            article: Article ID
            
        Returns:
            List of related concept information
        """
        related = []
        
        if not self.graph.graph.has_node(article):
            return related
        
        # Find concepts mentioned or defined by this article
        for succ in self.graph.graph.successors(article):
            if self.graph.graph.nodes[succ].get("type") == "concept":
                edge_data = self.graph.graph.get_edge_data(article, succ)
                related.append({
                    "article": article,
                    "concept": succ,
                    "relation": edge_data.get("type", "mentions")
                })
        
        return related
    
    def _get_article_hierarchical_relations(self, article: str) -> List[Dict[str, Any]]:
        """Get hierarchical relations for an article.
        
        Args:
            article: Article ID
            
        Returns:
            List of hierarchical relation information
        """
        relations = []
        
        if not self.graph.graph.has_node(article):
            return relations
        
        # Find articles that modify this article
        for pred in self.graph.graph.predecessors(article):
            if self.graph.graph.nodes[pred].get("type") == "article":
                edge_data = self.graph.graph.get_edge_data(pred, article)
                relations.append({
                    "from": pred,
                    "to": article,
                    "type": edge_data.get("type", "related")
                })
        
        # Find articles modified by this article
        for succ in self.graph.graph.successors(article):
            if self.graph.graph.nodes[succ].get("type") == "article":
                edge_data = self.graph.graph.get_edge_data(article, succ)
                relations.append({
                    "from": article,
                    "to": succ,
                    "type": edge_data.get("type", "related")
                })
        
        return relations
    
    def _find_article_contradictions(self, articles: List[str]) -> List[Dict[str, Any]]:
        """Find contradictions between articles.
        
        Args:
            articles: List of article entities
            
        Returns:
            List of contradiction information
        """
        contradictions = []
        
        # Normalize articles
        normalized_articles = []
        for article in articles:
            entity_type = self.graph._infer_entity_type(article)
            if entity_type == "article":
                normalized = self.graph._normalize_entity(article, "article")
                if self.graph.graph.has_node(normalized):
                    normalized_articles.append(normalized)
        
        # Check for direct modifications
        for i, article1 in enumerate(normalized_articles):
            for article2 in normalized_articles[i+1:]:
                # Check for direct relationship
                if self.graph.graph.has_edge(article1, article2):
                    edge_data = self.graph.graph.get_edge_data(article1, article2)
                    if edge_data.get("type") in ["modifies", "contradicts"]:
                        contradictions.append({
                            "article1": article1,
                            "article2": article2,
                            "relation": edge_data.get("type"),
                            "source": edge_data.get("source")
                        })
                
                # Check reverse direction
                if self.graph.graph.has_edge(article2, article1):
                    edge_data = self.graph.graph.get_edge_data(article2, article1)
                    if edge_data.get("type") in ["modifies", "contradicts"]:
                        contradictions.append({
                            "article1": article2,
                            "article2": article1,
                            "relation": edge_data.get("type"),
                            "source": edge_data.get("source")
                        })
        
        return contradictions
