# knowledge_graph/builder.py
"""Knowledge graph construction and management."""

import re
import os
import json
import logging
import pickle
import asyncio
import networkx as nx
from typing import List, Dict, Any, Set, Tuple, Optional
from ..utils.text_processing import (
    extract_articles, 
    extract_legal_concepts, 
    extract_penalties, 
    split_into_sentences
)
from src.exceptions import GraphError

class KnowledgeGraph:
    """Legal knowledge graph with enhanced entity and relation extraction."""
    
    def __init__(self, graph_path: Optional[str] = None):
        """Initialize the knowledge graph.
        
        Args:
            graph_path: Path to saved graph file (optional)
        """
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Index mappings
        self.entity_to_chunks = {}  # Entity -> Set of chunk IDs
        self.chunk_to_entities = {}  # Chunk ID -> Set of entity IDs
        
        # Legal entity lists (for normalization)
        self.article_aliases = {}  # Normalized article ID -> List of aliases
        self.concept_aliases = {}  # Normalized concept ID -> List of aliases
        
        # Load existing graph if provided
        if graph_path and os.path.exists(graph_path):
            self.load(graph_path)
    
    def save(self, path: str):
        """Save the knowledge graph to a file.
        
        Args:
            path: Output file path
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save graph and mapping data
            data = {
                "graph": nx.node_link_data(self.graph),
                "entity_to_chunks": self.entity_to_chunks,
                "chunk_to_entities": self.chunk_to_entities,
                "article_aliases": self.article_aliases,
                "concept_aliases": self.concept_aliases
            }
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            
            logging.info(f"Knowledge graph saved to {path}")
        except Exception as e:
            raise GraphError(f"Failed to save knowledge graph: {e}")
    
    def load(self, path: str):
        """Load the knowledge graph from a file.
        
        Args:
            path: Input file path
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Restore graph
            self.graph = nx.node_link_graph(data["graph"])
            
            # Restore mappings
            self.entity_to_chunks = data["entity_to_chunks"]
            self.chunk_to_entities = data["chunk_to_entities"]
            self.article_aliases = data.get("article_aliases", {})
            self.concept_aliases = data.get("concept_aliases", {})
            
            logging.info(f"Knowledge graph loaded from {path}")
        except Exception as e:
            raise GraphError(f"Failed to load knowledge graph: {e}")
    
    async def build_from_chunks_async(self, chunks: List[Dict[str, Any]], batch_size: int = 50, max_workers: int = 4):
        """Build the knowledge graph from chunks asynchronously.
        
        Args:
            chunks: List of text chunks with metadata
            batch_size: Number of chunks to process in each batch
        """
        logging.info(f"Building knowledge graph from {len(chunks)} chunks")
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            batch = chunks[i:end_idx]
            
            # Process batch synchronously to avoid concurrency issues
            await self._process_chunk_batch(batch)
                
        # Build relationships between entities
        await self._build_entity_relationships()
        
        logging.info(f"Knowledge graph built with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
    
    async def _process_chunk_batch(self, chunks: List[Dict[str, Any]]):
        """Process a batch of chunks to extract entities and basic relationships.
        
        Args:
            chunks: Batch of text chunks
        """
        for chunk in chunks:
            chunk_id = chunk.get("id", str(id(chunk)))
            content = chunk["content"]
            
            # Extract entities
            articles = extract_articles(content)
            concepts = extract_legal_concepts(content)
            penalties = extract_penalties(content)
            
            # Add nodes and create mappings
            for article in articles:
                self._add_node(article, "article", chunk_id)
                
            for concept in concepts:
                self._add_node(concept, "concept", chunk_id)
                
            for penalty in penalties:
                self._add_node(penalty, "penalty", chunk_id)
            
            # Extract and add basic relations within the chunk
            self._extract_chunk_relations(content, articles, concepts, penalties, chunk_id)
    
    def _add_node(self, entity: str, entity_type: str, chunk_id: str):
        """Add a node to the graph and update mappings.
        
        Args:
            entity: Entity identifier
            entity_type: Type of entity (article, concept, penalty)
            chunk_id: ID of the chunk containing this entity
        """
        # Normalize entity
        entity = self._normalize_entity(entity, entity_type)
        
        # Add or update node
        if self.graph.has_node(entity):
            self.graph.nodes[entity]["mentions"] = self.graph.nodes[entity].get("mentions", 0) + 1
        else:
            self.graph.add_node(entity, type=entity_type, mentions=1)
        
        # Update mappings
        self._add_mutual_index(chunk_id, entity)
    
    def _normalize_entity(self, entity: str, entity_type: str) -> str:
        """Normalize entity identifiers for consistent referencing.
        
        Args:
            entity: Raw entity identifier
            entity_type: Type of entity
            
        Returns:
            Normalized entity identifier
        """
        if entity_type == "article":
            # Handle article normalization
            match = re.search(r'(\d+(\s*[a-z]*)?)', entity)
            if match:
                normalized = f"articulo_{match.group(1).strip().lower()}"
                
                # Store original form as alias
                if normalized not in self.article_aliases:
                    self.article_aliases[normalized] = []
                if entity not in self.article_aliases[normalized]:
                    self.article_aliases[normalized].append(entity)
                
                return normalized
        
        elif entity_type == "concept":
            # Normalize legal concepts
            normalized = entity.lower().strip()
            
            # Store original form as alias
            if normalized not in self.concept_aliases:
                self.concept_aliases[normalized] = []
            if entity not in self.concept_aliases[normalized]:
                self.concept_aliases[normalized].append(entity)
            
            return normalized
        
        # Default normalization
        return entity.lower().strip()
    
    def _add_mutual_index(self, chunk_id: str, entity: str):
        """Add mutual indices between chunks and entities.
        
        Args:
            chunk_id: Chunk identifier
            entity: Entity identifier
        """
        # Chunk -> Entities mapping
        if chunk_id not in self.chunk_to_entities:
            self.chunk_to_entities[chunk_id] = set()
        self.chunk_to_entities[chunk_id].add(entity)
        
        # Entity -> Chunks mapping
        if entity not in self.entity_to_chunks:
            self.entity_to_chunks[entity] = set()
        self.entity_to_chunks[entity].add(chunk_id)
    
    def _extract_chunk_relations(self, text: str, articles: List[str], concepts: List[str], 
                                penalties: List[str], chunk_id: str):
        """Extract relationships between entities within a chunk.
        
        Args:
            text: Chunk text
            articles: List of articles in the chunk
            concepts: List of concepts in the chunk
            penalties: List of penalties in the chunk
            chunk_id: Chunk ID
        """
        text_lower = text.lower()
        
        # Normalize entities
        normalized_articles = [self._normalize_entity(article, "article") for article in articles]
        normalized_concepts = [self._normalize_entity(concept, "concept") for concept in concepts]
        normalized_penalties = [self._normalize_entity(penalty, "penalty") for penalty in penalties]
        
        # Relation patterns to search for
        define_patterns = ["define", "establece", "entiende por", "constituye"]
        modify_patterns = ["modifica", "deroga", "sustituye", "complementa"]
        impose_patterns = ["impone", "establece", "sanciona", "castiga", "pena"]
        
        # Article -> Concept relations
        for article in normalized_articles:
            for concept in normalized_concepts:
                # Determine type of relation
                relation_type = "mentions"
                
                # Check for more specific relations in the text
                for pattern in define_patterns:
                    if pattern in text_lower and self._are_in_same_sentence(text_lower, article, concept):
                        relation_type = "define"
                        break
                
                self.graph.add_edge(article, concept, type=relation_type, source=chunk_id)
        
        # Article -> Article relations
        for i, article1 in enumerate(normalized_articles):
            for article2 in normalized_articles[i+1:]:
                # Check for article references
                for pattern in modify_patterns:
                    if pattern in text_lower and self._are_in_same_sentence(text_lower, article1, article2):
                        self.graph.add_edge(article1, article2, type="modifies", source=chunk_id)
                        break
        
        # Article -> Penalty relations
        for article in normalized_articles:
            for penalty in normalized_penalties:
                # Check for penalty imposition
                for pattern in impose_patterns:
                    if pattern in text_lower:
                        self.graph.add_edge(article, penalty, type="imposes", source=chunk_id)
                        break
        
        # Concept -> Concept relations
        for i, concept1 in enumerate(normalized_concepts):
            for concept2 in normalized_concepts[i+1:]:
                if self._are_in_same_sentence(text_lower, concept1, concept2):
                    self.graph.add_edge(concept1, concept2, type="related", source=chunk_id)
    
    def _are_in_same_sentence(self, text: str, entity1: str, entity2: str) -> bool:
        """Check if two entities appear in the same sentence.
        
        Args:
            text: Text to check
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities are in the same sentence
        """
        sentences = split_into_sentences(text)
        
        # Remove the 'articulo_' prefix from article entities for checking
        search_entity1 = entity1
        search_entity2 = entity2
        
        if entity1.startswith('articulo_'):
            search_entity1 = entity1[9:]  # Remove 'articulo_' prefix
        if entity2.startswith('articulo_'):
            search_entity2 = entity2[9:]  # Remove 'articulo_' prefix
            
        for sentence in sentences:
            if search_entity1 in sentence.lower() and search_entity2 in sentence.lower():
                return True
        
        return False
    
    async def _build_entity_relationships(self):
        """Build relationships between entities based on co-occurrence and context."""
        # Build a list of entities
        entity_list = list(self.entity_to_chunks.keys())
        
        # Process entity pairs
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                # Skip if already have a direct relationship
                if self.graph.has_edge(entity1, entity2) or self.graph.has_edge(entity2, entity1):
                    continue
                
                # Check for co-occurrence in chunks
                common_chunks = self.entity_to_chunks[entity1].intersection(self.entity_to_chunks[entity2])
                if common_chunks:
                    # Add a co-occurrence relationship
                    self.graph.add_edge(entity1, entity2, 
                                       type="co_occurs", 
                                       weight=len(common_chunks),
                                       sources=list(common_chunks))
    
    def get_entity_subgraph(self, entity: str, depth: int = 2) -> nx.DiGraph:
        """Get a subgraph centered on a specific entity.
        
        Args:
            entity: Central entity
            depth: Traversal depth
            
        Returns:
            Subgraph as a new DiGraph
        """
        # Normalize entity
        entity_type = self._infer_entity_type(entity)
        entity = self._normalize_entity(entity, entity_type)
        
        if not self.graph.has_node(entity):
            return nx.DiGraph()
        
        # Implement BFS to get nodes up to specified depth
        nodes = {entity}
        current_nodes = {entity}
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                # Add neighbors (both predecessors and successors)
                next_nodes.update(self.graph.predecessors(node))
                next_nodes.update(self.graph.successors(node))
            
            nodes.update(next_nodes)
            current_nodes = next_nodes
        
        return self.graph.subgraph(nodes)
    
    def _infer_entity_type(self, entity: str) -> str:
        """Infer the type of an entity based on its format.
        
        Args:
            entity: Entity to check
            
        Returns:
            Inferred entity type
        """
        if re.search(r'art[íi]culo|articulo|\bart\.?\s*\d+', entity, re.IGNORECASE):
            return "article"
        elif entity.startswith("pena_"):
            return "penalty"
        else:
            return "concept"
    
    def get_related_chunks(self, entity: str) -> Set[str]:
        """Get chunks related to an entity.
        
        Args:
            entity: Entity to check
            
        Returns:
            Set of chunk IDs
        """
        # Normalize entity
        entity_type = self._infer_entity_type(entity)
        entity = self._normalize_entity(entity, entity_type)
        
        return self.entity_to_chunks.get(entity, set())
    
    def get_related_entities(self, chunk_id: str) -> Set[str]:
        """Get entities related to a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Set of entity IDs
        """
        return self.chunk_to_entities.get(chunk_id, set())
    
    def get_entity_by_alias(self, alias: str) -> Optional[str]:
        """Find an entity by one of its aliases.
        
        Args:
            alias: Entity alias to look for
            
        Returns:
            Normalized entity ID or None if not found
        """
        # Check article aliases
        for normalized, aliases in self.article_aliases.items():
            if alias in aliases or alias.lower() in [a.lower() for a in aliases]:
                return normalized
        
        # Check concept aliases
        for normalized, aliases in self.concept_aliases.items():
            if alias in aliases or alias.lower() in [a.lower() for a in aliases]:
                return normalized
        
        return None
    
    def find_path(self, entity1: str, entity2: str) -> List[str]:
        """Find the shortest path between two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            List of entities in the path or empty list if no path exists
        """
        # Normalize entities
        entity1_type = self._infer_entity_type(entity1)
        entity2_type = self._infer_entity_type(entity2)
        
        entity1 = self._normalize_entity(entity1, entity1_type)
        entity2 = self._normalize_entity(entity2, entity2_type)
        
        if not self.graph.has_node(entity1) or not self.graph.has_node(entity2):
            return []
        
        try:
            path = nx.shortest_path(self.graph, entity1, entity2)
            return path
        except nx.NetworkXNoPath:
            # Try reverse direction
            try:
                path = nx.shortest_path(self.graph, entity2, entity1)
                return list(reversed(path))
            except nx.NetworkXNoPath:
                return []
    
    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Find potential contradictions in the legal knowledge.
        
        Returns:
            List of contradiction information
        """
        contradictions = []
        
        # Find articles that modify or contradict each other
        for node in self.graph.nodes():
            if self.graph.nodes[node].get("type") == "article":
                # Check for articles that modify this one
                modifiers = []
                for pred in self.graph.predecessors(node):
                    if self.graph.nodes[pred].get("type") == "article":
                        edge_data = self.graph.get_edge_data(pred, node)
                        if edge_data.get("type") in ["modifies", "contradicts"]:
                            modifiers.append({
                                "article": pred,
                                "relation": edge_data.get("type"),
                                "source": edge_data.get("source")
                            })
                
                if len(modifiers) > 0:
                    contradictions.append({
                        "article": node,
                        "modifiers": modifiers
                    })
        
        return contradictions
    
    def partition_graph(self, partition_criteria: str = "article_prefix"):
        """Particiona el grafo según criterios como jurisdicción o prefijo de artículo.
        
        Args:
            partition_criteria: Criterio de particionamiento ('article_prefix', 'type', etc.)
            
        Returns:
            Diccionario de subgrafos particionados
        """
        logging.info(f"Particionando grafo con criterio: {partition_criteria}")
        subgraphs = {}
        
        for node in self.graph.nodes():
            # Extraer criterio de partición del nodo
            node_partition = self._extract_partition_key(node, partition_criteria)
            
            if node_partition not in subgraphs:
                subgraphs[node_partition] = nx.DiGraph()
            
            # Copiar nodo y sus atributos
            subgraphs[node_partition].add_node(node, **self.graph.nodes[node])
        
        # Copiar las aristas relevantes
        for partition, subgraph in subgraphs.items():
            for node in subgraph.nodes():
                for succ in self.graph.successors(node):
                    if subgraph.has_node(succ):
                        subgraph.add_edge(node, succ, **self.graph.edges[node, succ])
        
        logging.info(f"Grafo particionado en {len(subgraphs)} subgrafos")
        return subgraphs

    def _extract_partition_key(self, node: str, partition_criteria: str) -> str:
        """Extrae la clave de partición de un nodo según el criterio.
        
        Args:
            node: Identificador del nodo
            partition_criteria: Criterio de particionamiento
            
        Returns:
            Clave de partición
        """
        if partition_criteria == "type":
            # Particionar por tipo de nodo (artículo, concepto, pena)
            return self.graph.nodes[node].get("type", "unknown")
        
        elif partition_criteria == "article_prefix":
            # Particionar artículos por prefijo numérico (ej: artículos 100-199 juntos)
            if node.startswith("articulo_"):
                # Extraer número del artículo
                import re
                match = re.search(r'(\d+)', node)
                if match:
                    article_num = int(match.group(1))
                    # Agrupar por centenar
                    return f"art_{article_num // 100}xx"
            
            # Para nodos que no son artículos o no tienen número
            return "other"
        
        elif partition_criteria == "chapter":
            # Si tuviéramos metadatos de capítulo en los nodos
            return self.graph.nodes[node].get("chapter", "unknown")
        
        # Criterio no reconocido, usar partición por defecto
        return "default"