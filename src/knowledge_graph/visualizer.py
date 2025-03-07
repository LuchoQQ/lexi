# src/knowledge_graph/visualizer.py
"""Knowledge graph visualization with improved styling and interactivity."""

import os
import logging
import networkx as nx
from typing import Dict, Any, Optional, List
from src.exceptions import GraphError

class GraphVisualizer:
    """Visualizer for the legal knowledge graph."""
    
    def __init__(self, graph):
        """Initialize the graph visualizer.
        
        Args:
            graph: KnowledgeGraph instance
        """
        self.graph = graph
    
    def visualize(self, 
                 output_file: str = "legal_knowledge_graph.html", 
                 entities: Optional[List[str]] = None,
                 highlight_entities: Optional[List[str]] = None,
                 max_nodes: int = 200) -> str:
        """Visualize the knowledge graph using Pyvis.
        
        Args:
            output_file: Output HTML file path
            entities: Optional list of specific entities to visualize
            highlight_entities: Optional list of entities to highlight
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Path to the generated HTML file
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logging.error("Pyvis is not installed. Install it with 'pip install pyvis'")
            raise GraphError("Pyvis is required for visualization")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            
            # Create network
            net = Network(height="800px", width="100%", notebook=False, directed=True)
            
            # Configure physics
            net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
            
            # Select subgraph if entities are specified
            if entities:
                logging.info(f"Creating subgraph for entities: {entities}")
                subgraph = nx.DiGraph()
                
                for entity in entities:
                    # Try to find the entity both as-is and as normalized
                    entity_type = self.graph._infer_entity_type(entity)
                    normalized = self.graph._normalize_entity(entity, entity_type)
                    logging.info(f"Looking for entity: {normalized}")
                    
                    # Try also with 'articulo_' prefix for article numbers
                    if entity.isdigit() or (len(entity) > 1 and entity[0].isdigit()):
                        alt_normalized = f"articulo_{entity}"
                        if self.graph.graph.has_node(alt_normalized):
                            normalized = alt_normalized
                    
                    if self.graph.graph.has_node(normalized):
                        logging.info(f"Found entity in graph: {normalized}")
                        entity_subgraph = self.graph.get_entity_subgraph(normalized, depth=2)
                        subgraph = nx.compose(subgraph, entity_subgraph)
                    else:
                        logging.warning(f"Entity not found in graph: {normalized}")
                
                # Limit number of nodes if necessary
                if subgraph.number_of_nodes() > max_nodes:
                    # Keep important nodes (high degree centrality)
                    centrality = nx.degree_centrality(subgraph)
                    important_nodes = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
                    subgraph = subgraph.subgraph(important_nodes)
                
                graph_to_viz = subgraph
                logging.info(f"Created subgraph with {graph_to_viz.number_of_nodes()} nodes and {graph_to_viz.number_of_edges()} edges")
            else:
                # Use the whole graph
                graph_to_viz = self.graph.graph
                
                # Limit number of nodes if necessary
                if graph_to_viz.number_of_nodes() > max_nodes:
                    # Keep important nodes (high degree centrality)
                    centrality = nx.degree_centrality(graph_to_viz)
                    important_nodes = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
                    graph_to_viz = graph_to_viz.subgraph(important_nodes)
                
                logging.info(f"Using full graph (limited to {max_nodes} nodes): {graph_to_viz.number_of_nodes()} nodes")
            
            # Prepare highlight set
            highlight_set = set()
            if highlight_entities:
                for entity in highlight_entities:
                    entity_type = self.graph._infer_entity_type(entity)
                    normalized = self.graph._normalize_entity(entity, entity_type)
                    highlight_set.add(normalized)
                logging.info(f"Highlighted entities: {highlight_set}")
            
            # Add nodes with colors according to type
            for node in graph_to_viz.nodes():
                node_data = graph_to_viz.nodes[node]
                node_type = node_data.get("type", "unknown")
                mentions = node_data.get("mentions", 1)
                
                # Determine node color and size
                if node in highlight_set:
                    color = "#FF5733"  # Highlighted orange
                    size = 30
                    border_width = 3
                    font_size = 20
                else:
                    if node_type == "article":
                        color = "#4287f5"  # Blue
                    elif node_type == "concept":
                        color = "#42f5aa"  # Green
                    elif node_type == "penalty":
                        color = "#f54242"  # Red
                    else:
                        color = "#f5d442"  # Yellow
                    
                    # Scale size based on mentions
                    size = 15 + min(mentions, 10)
                    border_width = 1
                    font_size = 14
                
                # Create node label
                label = node
                if node.startswith("articulo_"):
                    label = "Art. " + node[9:]  # Remove 'articulo_' prefix
                elif node.startswith("pena_"):
                    label = "Pena: " + node[5:]  # Remove 'pena_' prefix
                
                # Create node title (tooltip)
                tooltip = f"{node_type.capitalize()}: {label}"
                if mentions > 1:
                    tooltip += f"\nMenciones: {mentions}"
                if node_data.get("source_chunks"):
                    tooltip += f"\nFragmentos: {len(node_data.get('source_chunks', []))}"
                
                # Add node to network
                net.add_node(node, 
                            label=label, 
                            color=color, 
                            size=size, 
                            title=tooltip,
                            borderWidth=border_width,
                            font={'size': font_size, 'face': 'Arial'})
            
            # Add edges with labels
            for source, target, data in graph_to_viz.edges(data=True):
                edge_type = data.get("type", "related")
                
                # Determine edge color and width
                if edge_type == "define":
                    color = "#00cc00"  # Green
                    width = 3
                    arrows = "to"
                    dash = False
                elif edge_type == "modifies":
                    color = "#cc0000"  # Red
                    width = 3
                    arrows = "to"
                    dash = False
                elif edge_type == "imposes":
                    color = "#9900cc"  # Purple
                    width = 2
                    arrows = "to"
                    dash = False
                elif edge_type == "mentions":
                    color = "#0066cc"  # Blue
                    width = 1
                    arrows = "to"
                    dash = False
                elif edge_type == "co_occurs":
                    color = "#999999"  # Gray
                    width = 1
                    arrows = "to"
                    dash = True
                else:
                    color = "#999999"  # Gray
                    width = 1
                    arrows = "to"
                    dash = False
                
                # Add edge to network
                net.add_edge(source, target, title=edge_type, color=color, width=width, arrows=arrows, dashes=dash)
            
            # Configure options for better visualization
            net.set_options("""
            {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                  "enabled": true,
                  "iterations": 1000
                }
              },
              "interaction": {
                "tooltipDelay": 200,
                "hideEdgesOnDrag": true,
                "multiselect": true,
                "navigationButtons": true,
                "hover": true
              },
              "edges": {
                "arrows": {
                  "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                  }
                },
                "smooth": {
                  "type": "dynamic",
                  "forceDirection": "none"
                }
              },
              "nodes": {
                "font": {
                  "strokeWidth": 0,
                  "strokeColor": "#ffffff"
                },
                "shadow": {
                  "enabled": true
                }
              },
              "layout": {
                "hierarchical": {
                  "enabled": false
                }
              }
            }
            """)
            
            # Add custom HTML controls for the visualization
            html_header = """
            <div style="padding: 10px; background-color: #f5f5f5; border-bottom: 1px solid #ddd; margin-bottom: 10px;">
              <h2 style="margin: 0; color: #333;">Grafo de Conocimiento Legal</h2>
              <p style="margin: 5px 0;">Leyenda: 
                <span style="color: #4287f5; font-weight: bold;">■</span> Artículos, 
                <span style="color: #42f5aa; font-weight: bold;">■</span> Conceptos,
                <span style="color: #f54242; font-weight: bold;">■</span> Penas,
                <span style="color: #FF5733; font-weight: bold;">■</span> Entidades destacadas
              </p>
              <p style="margin: 5px 0;">
                <span style="color: #00cc00; font-weight: bold;">——</span> Define, 
                <span style="color: #cc0000; font-weight: bold;">——</span> Modifica,
                <span style="color: #9900cc; font-weight: bold;">——</span> Impone,
                <span style="color: #999999; font-weight: bold;">- - -</span> Co-ocurre
              </p>
              <p style="font-size: 0.9em; margin-top: 5px;">Use la rueda del ratón para hacer zoom, arrastre para mover, haga doble clic en un nodo para enfocarlo.</p>
            </div>
            """
            
            # Save the visualization
            net.save_graph(output_file)
            
            # Add the custom header to the HTML file
            with open(output_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Insert header after the body tag
            html_content = html_content.replace('<body>', f'<body>\n{html_header}')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logging.info(f"Knowledge graph visualization saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            logging.error(f"Error visualizing graph: {e}")
            logging.error(traceback.format_exc())
            raise GraphError(f"Failed to visualize knowledge graph: {e}")