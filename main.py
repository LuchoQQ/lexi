# main.py
"""Main entry point for legal assistant application."""

import os
import json
import argparse
import asyncio
import logging
from typing import Dict, Any, Optional

from src.utils.logging import setup_logging
from src.assistant.generator import LegalAssistant
from src.exceptions import APIError, GenerationError, LegalAssistantError
from config import Config

async def init_assistant(config_path: Optional[str] = None) -> LegalAssistant:
    """Initialize the legal assistant.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Initialized LegalAssistant
    """
    # Load configuration
    logging.info(f"Initializing assistant with config path: {config_path}")
    config = Config(config_path)
    
    # Verify API key
    if not config.get("openai_api_key"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter your OpenAI API key: ")
        config.config["openai_api_key"] = api_key
        logging.info("API key acquired from environment or user input")
    else:
        logging.info("Using API key from configuration")
    
    # Initialize assistant
    logging.info("Creating LegalAssistant instance")
    assistant = LegalAssistant(config)
    logging.info("LegalAssistant instance created successfully")
    return assistant

async def process_file(assistant: LegalAssistant, file_path: str, rebuild: bool = False) -> Dict[str, Any]:
    """Process a JSON file with legal data.
    
    Args:
        assistant: LegalAssistant instance
        file_path: Path to JSON file
        rebuild: Whether to force rebuild embeddings and graph
        
    Returns:
        Processing results
    """
    logging.info(f"Processing file: {file_path} (rebuild={rebuild})")
    try:
        result = await assistant.load_data(file_path, rebuild=rebuild)
        logging.info(f"File processing completed successfully. Results: {result}")
        return result
    except Exception as e:
        logging.error(f"Error processing file: {e}", exc_info=True)
        raise

async def process_query(assistant: LegalAssistant, query: str) -> Dict[str, Any]:
    """Process a user query.
    
    Args:
        assistant: LegalAssistant instance
        query: User query
        
    Returns:
        Response with metadata
    """
    logging.info(f"Processing query: {query}")
    try:
        result = await assistant.process_query(query)
        logging.info(f"Query processed successfully. Response length: {len(result.get('response', ''))}")
        if not result.get("response"):
            logging.warning("Empty response returned from process_query")
        return result
    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        raise

async def visualize_graph(assistant: LegalAssistant, entities: Optional[list] = None, 
                         output_file: str = "legal_knowledge_graph.html"):
    """Visualize the knowledge graph.
    
    Args:
        assistant: LegalAssistant instance
        entities: Optional list of specific entities to visualize
        output_file: Output file path for the visualization
    """
    logging.info(f"Visualizing graph with entities: {entities}")
    try:
        from src.knowledge_graph.visualizer import GraphVisualizer
        visualizer = GraphVisualizer(assistant.knowledge_graph)
        highlight_entities = entities if entities else []
        
        html_path = visualizer.visualize(
            output_file=output_file,
            entities=entities,
            highlight_entities=highlight_entities
        )
        
        logging.info(f"Graph visualization saved to {html_path}")
        print(f"\nGraph visualization saved to {html_path}")
        
    except Exception as e:
        logging.error(f"Error visualizing graph: {e}", exc_info=True)
        print(f"\nError visualizing graph: {e}")

async def interactive_mode(assistant: LegalAssistant):
    """Run interactive mode.
    
    Args:
        assistant: LegalAssistant instance
    """
    logging.info("Starting interactive mode")
    print("\nModo interactivo iniciado. Escribe 'salir' para terminar.")
    print("Comandos especiales:")
    print("  visualizar <artículo> - Visualiza el grafo para un artículo específico")
    print("  info - Muestra estadísticas del sistema")
    
    while True:
        query = input("\nIngresa tu consulta legal: ")
        if query.lower() in ["salir", "exit", "quit"]:
            logging.info("Exiting interactive mode")
            break
        
        try:
            # Check for special commands
            if query.lower().startswith("visualizar"):
                parts = query.split(maxsplit=1)
                entities = []
                if len(parts) > 1:
                    from src.utils.text_processing import extract_articles
                    articles = extract_articles(parts[1])
                    entities = [article for article in articles]
                
                await visualize_graph(assistant, entities)
                continue
                
            elif query.lower() == "info":
                print("\nEstadísticas del sistema:")
                print(f"Nodos en el grafo: {len(assistant.knowledge_graph.graph.nodes())}")
                print(f"Conexiones en el grafo: {len(assistant.knowledge_graph.graph.edges())}")
                if assistant.vector_store:
                    collection_info = assistant.vector_store.collection.count()
                    print(f"Documentos en la base vectorial: {collection_info}")
                continue
            
            # Process regular query
            logging.info(f"Processing interactive query: {query}")
            result = await process_query(assistant, query)
            
            # Validación de confiabilidad de respuesta
            if result.get("validation") and result.get("validation").get("factual_accuracy", 1.0) < 0.7:
                print("\n⚠️ AVISO: Baja confiabilidad factual (precisión: {:.1f}%).".format(
                    result.get("validation").get("factual_accuracy", 0) * 100))
                print("Se recomienda verificar esta información con fuentes oficiales del Código Penal.")
            
            print("\nRespuesta:")
            print(result["response"])
            
            if result.get("articles_cited"):
                print("\nArtículos citados:")
                print(", ".join(result["articles_cited"]))
                
                # Mostrar información de validación si existe
                if result.get("validation") and result.get("validation").get("unvalidated_citations"):
                    unvalidated = [cite["article"] for cite in result["validation"]["unvalidated_citations"]]
                    if unvalidated:
                        print("\n⚠️ Artículos citados sin verificar:")
                        print(", ".join(unvalidated))
            
            if result.get("tokens_used"):
                print(f"\nTokens utilizados: {result['tokens_used']}")
            
        except Exception as e:
            logging.error(f"Error in interactive mode: {e}", exc_info=True)
            print(f"\nError: {e}")

async def main():
    """Main entry point."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Asistente Legal con Código Penal de Argentina")
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración")
    parser.add_argument("--json_path", type=str, help="Ruta al archivo JSON con los chunks del Código Penal")
    parser.add_argument("--json_dir", type=str, help="Ruta al directorio con múltiples archivos JSON")
    parser.add_argument("--query", type=str, help="Consulta legal (opcional)")
    parser.add_argument("--log_level", type=str, default="INFO", help="Nivel de logging (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--rebuild", action="store_true", help="Reconstruir embeddings y grafo de conocimiento")
    parser.add_argument("--visualize", action="store_true", help="Visualizar el grafo de conocimiento")
    parser.add_argument("--visualize_entities", type=str, help="Entidades a visualizar (separadas por comas)")
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level=log_level)
    
    logging.info("Legal Assistant application starting")
    
    try:
        # Initialize assistant
        logging.info("Initializing assistant")
        assistant = await init_assistant(args.config)
        
        # Process file or directory if provided
        if args.json_dir:
            logging.info(f"Processing directory: {args.json_dir}")
            result = await assistant.load_data_directory(args.json_dir, rebuild=args.rebuild)
            print(f"Procesamiento de directorio completado. {len(result)} archivos procesados.")
        elif args.json_path:
            logging.info(f"Processing file: {args.json_path}")
            result = await process_file(assistant, args.json_path, rebuild=args.rebuild)
            print(f"Procesamiento completado: {json.dumps(result, indent=2)}")
        
        # Visualize graph if requested
        if args.visualize:
            entities = None
            if args.visualize_entities:
                entities = [e.strip() for e in args.visualize_entities.split(",")]
            await visualize_graph(assistant, entities)
        
        # Process query if provided, otherwise run interactive mode
        if args.query:
            logging.info(f"Processing command-line query: {args.query}")
            result = await process_query(assistant, args.query)
            
            # Validación de confiabilidad de respuesta
            if result.get("validation") and result.get("validation").get("factual_accuracy", 1.0) < 0.7:
                print("\n⚠️ AVISO: Baja confiabilidad factual (precisión: {:.1f}%).".format(
                    result.get("validation").get("factual_accuracy", 0) * 100))
                print("Se recomienda verificar esta información con fuentes oficiales del Código Penal.")
            
            print("\nRespuesta:")
            if result.get("response"):
                print(result["response"])
            else:
                print("No se obtuvo una respuesta válida.")
            
            if result.get("articles_cited"):
                print("\nArtículos citados:")
                print(", ".join(result["articles_cited"]))
                
                # Mostrar información de validación si existe
                if result.get("validation") and result.get("validation").get("unvalidated_citations"):
                    unvalidated = [cite["article"] for cite in result["validation"]["unvalidated_citations"]]
                    if unvalidated:
                        print("\n⚠️ Artículos citados sin verificar:")
                        print(", ".join(unvalidated))
        elif not args.visualize:  # Only run interactive mode if not just visualizing
            logging.info("No query provided, running interactive mode")
            await interactive_mode(assistant)
            
    except LegalAssistantError as e:
        logging.error(f"Error en el asistente legal: {e}", exc_info=True)
        print(f"Error: {e}")
    except Exception as e:
        logging.exception("Error inesperado")
        print(f"Error inesperado: {e}")
    
    logging.info("Legal Assistant application terminating")

if __name__ == "__main__":
    asyncio.run(main())