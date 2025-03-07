# src/assistant/generator.py
"""Asynchronous response generator with enhanced error handling."""
from src.cache.query_cache import QueryCache  # Asegurarse de crear este archivo primero
from typing import List, Dict, Any, Optional, Set, Tuple

import os
import json
import logging
import asyncio
import traceback
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from src.exceptions import APIError, GenerationError
from src.assistant.retriever import LegalRetriever

class ResponseGenerator:
    """Generator for legal assistant responses."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        """Initialize the response generator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
        """
        self.model = model
        
        try:
            self.client = AsyncOpenAI(api_key=api_key)
            logging.info(f"Initialized response generator with model {model}")
        except Exception as e:
            logging.error(f"Failed to initialize response generator: {e}")
            raise APIError(f"Failed to initialize OpenAI client: {e}")
    
    async def generate(self, 
                     query: str, 
                     context: str, 
                     graph_knowledge: str, 
                     max_tokens: int = 1500,
                     temperature: float = 0.2) -> Dict[str, Any]:
        """Generate a response to a legal query.
        
        Args:
            query: User query
            context: Retrieved legal context
            graph_knowledge: Structured knowledge from graph
            max_tokens: Maximum token count for response
            temperature: Temperature for generation
            
        Returns:
            Response with metadata
        """
        # Create system prompt
        prompt = self._create_prompt(query, context, graph_knowledge)
        logging.info(f"Created prompt for query: '{query}' with length: {len(prompt)}")
        
        try:
            # Generate response
            logging.info(f"Sending request to OpenAI with model: {self.model}")
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            response_text = completion.choices[0].message.content.strip()
            logging.info(f"Received response with length: {len(response_text)}")
            
            # Extract cited articles using regex
            import re
            article_pattern = r"[Aa]rt[íi]culo\s+(\d+(\s*[a-z])?)|[Aa]rt\.\s*(\d+(\s*[a-z])?)"
            articles_cited = re.findall(article_pattern, response_text)
            
            # Clean and format cited articles
            clean_articles = []
            for article in articles_cited:
                article_num = article[0] if article[0] else article[2]
                clean_articles.append(article_num.strip())
            
            # Remove duplicates
            clean_articles = list(set(clean_articles))
            logging.info(f"Extracted {len(clean_articles)} cited articles")
            
            # Prepare response
            response = {
                "response": response_text,
                "articles_cited": clean_articles,
                "model": self.model,
                "tokens_used": completion.usage.total_tokens
            }
            
            logging.info(f"Response generated successfully. Tokens used: {completion.usage.total_tokens}")
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            raise GenerationError(f"Failed to generate response: {e}")
    
    def _create_prompt(self, query: str, context: str, graph_knowledge: str) -> str:
        """Create a prompt for the LLM."""
        # Check if context mostly contains placeholders
        if "Placeholder content for" in context and context.count("Placeholder content for") > context.count("Contenido:") / 2:
            logging.warning("Context contains mostly placeholder content, adding a warning")
            warning = "\nATENCIÓN: La información disponible es limitada. Indica claramente cuando no puedas proporcionar detalles específicos sobre el artículo consultado.\n"
        else:
            warning = ""
        
        prompt = f"""
        Eres un asesor legal experto especializado en el Código Penal de Argentina.
        
        INFORMACIÓN LEGAL:
        {context}
        
        ESTRUCTURA DE CONOCIMIENTO LEGAL:
        {graph_knowledge}{warning}
        
        Responde a la siguiente consulta con extrema precisión y detalle, utilizando tanto la información textual como la estructura de conocimiento proporcionada:
        {query}
        
        Instrucciones:
        1. Cita TEXTUALMENTE las partes relevantes de los artículos y secciones.
        2. Si el artículo solicitado no está en el contexto o no tienes suficiente información, indícalo claramente: "No dispongo de información suficiente sobre el artículo X."
        3. Nunca inventes el contenido de artículos o disposiciones legales. Es mejor admitir que no tienes suficiente información.
        4. Si solo tienes información sobre relaciones entre artículos pero no el contenido específico, acláralo.
        5. Razona paso a paso, estableciendo conexiones entre los conceptos legales según la estructura de conocimiento.
        6. Identifica claramente las relaciones jerárquicas entre artículos cuando existan Y SOLO si tienes evidencia de estas relaciones.
        7. Utiliza únicamente las definiciones oficiales proporcionadas en el grafo de conocimiento.
        8. Estructura tu respuesta de manera clara y comprensible.
        9. Verifica si las afirmaciones que haces están respaldadas por el contexto proporcionado. Si no lo están, indícalo claramente.
        10. Cuidado con los artículos "espurios" o mal formateados en el grafo de conocimiento. Si encuentras artículos como "articulo_14 d" que parecen extraños, analiza si podría ser un error de extracción.
        
        Finaliza siempre con un resumen conciso de los puntos clave y con las recomendaciones principales o, si corresponde, con una aclaración sobre la falta de información suficiente.
        """
        
        logging.debug(f"Created prompt with context size: {len(context)} and graph knowledge size: {len(graph_knowledge)}")
        return prompt
    
    async def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks as context for the LLM.
        
        Args:
            chunks: Retrieved and ranked chunks
            
        Returns:
            Formatted context string
        """
        logging.info(f"Formatting {len(chunks)} chunks as context")
        context = "CONTEXTO LEGAL:\n\n"
        
        for i, chunk in enumerate(chunks):
            context += f"--- FRAGMENTO {i+1}"
            
            if "rerank_score" in chunk:
                context += f" (Score: {chunk['rerank_score']:.4f})"
            
            context += " ---\n"
            context += f"Contenido: {chunk['content']}\n"
            
            if "metadata" in chunk:
                metadata = chunk["metadata"]
                for key, value in metadata.items():
                    # Skip certain metadata fields
                    if key not in ["header_level"] and value:
                        context += f"{key.capitalize()}: {value}\n"
            
            if "retrieval_method" in chunk:
                context += f"Método de recuperación: {chunk['retrieval_method']}\n"
            
            if "retrieved_by" in chunk:
                context += f"Recuperado por: {chunk['retrieved_by']}\n"
            
            context += "\n"
        
        logging.info(f"Context formatting complete, total length: {len(context)}")
        return context
    
    def format_graph_knowledge(self, graph_insights: Dict[str, Any]) -> str:
        """Format graph insights as structured knowledge for the LLM.
        
        Args:
            graph_insights: Insights extracted from knowledge graph
            
        Returns:
            Formatted knowledge string
        """
        logging.info(f"Formatting graph insights with {len(graph_insights)} sections")
        result = "CONOCIMIENTO ESTRUCTURADO DEL GRAFO LEGAL:\n\n"
        
        # 1. Entidades clave
        if graph_insights.get("key_entities"):
            result += "ENTIDADES CLAVE:\n"
            for entity in graph_insights["key_entities"]:
                result += f"- {entity['entity']} (Tipo: {entity['type']}, Mencionado en {entity.get('mentioned_in', 0)} fragmentos)\n"
                if entity['type'] == "concept" and entity.get('defined_in'):
                    result += f"  Definido en: {', '.join(entity['defined_in'])}\n"
                if entity['type'] == "article":
                    if entity.get('related_concepts'):
                        related_concepts = []
                        for concept in entity.get('related_concepts', []):
                            if isinstance(concept, str):
                                related_concepts.append(concept)
                            elif isinstance(concept, dict) and 'concept' in concept:
                                related_concepts.append(f"{concept['concept']} ({concept.get('relation', 'mentions')})")
                        result += f"  Conceptos relacionados: {', '.join(related_concepts)}\n"
                    if entity.get('imposed_penalties'):
                        result += f"  Penas impuestas: {', '.join(entity['imposed_penalties'])}\n"
            result += "\n"
        
        # 2. Definiciones
        if graph_insights.get("definitions"):
            result += "DEFINICIONES RELEVANTES:\n"
            for definition in graph_insights["definitions"]:
                result += f"- {definition['concept']} (definido en {definition['defined_in']}):\n"
                if "definition" in definition:
                    result += f"  \"{definition['definition']}\"\n\n"
                elif "source_chunk" in definition:
                    result += f"  [Ver fragmento relacionado con ID: {definition['source_chunk']}]\n\n"
        
        # 3. Relaciones jerárquicas
        if graph_insights.get("hierarchical_relations"):
            result += "RELACIONES JERÁRQUICAS ENTRE ARTÍCULOS:\n"
            for relation in graph_insights["hierarchical_relations"]:
                result += f"- {relation['from']} se refiere a {relation['to']} ({relation['type']})\n"
            result += "\n"
        
        # 4. Artículos relacionados
        if graph_insights.get("related_articles"):
            result += "ARTÍCULOS RELACIONADOS:\n"
            for relation in graph_insights["related_articles"]:
                if "shared_concept" in relation:
                    result += f"- {relation['article']} está relacionado con {relation['related_article']} a través del concepto: {relation['shared_concept']}\n"
                else:
                    result += f"- {relation['article']} está relacionado con {relation['concept']} ({relation.get('relation', 'mentions')})\n"
            result += "\n"
        
        # 5. Conceptos relacionados
        if graph_insights.get("related_concepts"):
            result += "CONCEPTOS RELACIONADOS:\n"
            for relation in graph_insights["related_concepts"]:
                if relation.get('relation') == "direct":
                    result += f"- {relation['concept1']} está directamente relacionado con {relation['concept2']}\n"
                elif relation.get('relation') == "indirect" and relation.get('path'):
                    path_str = " -> ".join(relation['path'])
                    result += f"- {relation['concept1']} está indirectamente relacionado con {relation['concept2']} a través de: {path_str}\n"
            result += "\n"
        
        # 6. Contradictions
        if graph_insights.get("contradictions"):
            result += "POSIBLES CONTRADICCIONES:\n"
            for contradiction in graph_insights["contradictions"]:
                if "article1" in contradiction and "article2" in contradiction:
                    result += f"- {contradiction['article1']} {contradiction.get('relation', 'contradice')} a {contradiction['article2']}\n"
                elif "article" in contradiction and "modifiers" in contradiction:
                    result += f"- {contradiction['article']} es modificado por: "
                    modifiers = [f"{m['article']} ({m['relation']})" for m in contradiction['modifiers']]
                    result += ", ".join(modifiers) + "\n"
            result += "\n"
        
        logging.info(f"Graph knowledge formatting complete, total length: {len(result)}")
        return result


class LegalAssistant:
    """Main legal assistant class that integrates all components."""
    
    def __init__(self, config):
        """Initialize the legal assistant.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.embedding_model = None
        self.vector_store = None
        self.knowledge_graph = None
        self.retriever = None
        self.generator = None
        
            # Add query cache
        from src.cache.query_cache import QueryCache
        self.query_cache = QueryCache(
            max_size=self.config.get("cache_max_size", 1000),
            ttl=self.config.get("cache_ttl", 3600)
        )
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all assistant components."""
        logging.info("Initializing LegalAssistant components")
        try:
            from src.embeddings.model import EmbeddingModel
            from src.embeddings.vector_store import VectorStore
            from src.knowledge_graph.builder import KnowledgeGraph
            from src.knowledge_graph.queries import GraphReasoner
            
            # Initialize embedding model
            model_name = self.config.get("embedding_model", "intfloat/multilingual-e5-large")
            cache_dir = self.config.get("embedding_cache_dir", "./embedding_cache")
            logging.info(f"Initializing embedding model: {model_name}")
            self.embedding_model = EmbeddingModel(model_name, cache_dir=cache_dir)
            
            # Initialize vector store
            persist_dir = self.config.get("chroma_db_dir", "./chroma_db")
            collection_name = self.config.get("collection_name", "codigo_penal_chunks")
            logging.info(f"Initializing vector store: {collection_name} in {persist_dir}")
            self.vector_store = VectorStore(persist_dir, collection_name, model_name)
            
            # Initialize knowledge graph
            graph_path = self.config.get("graph_path", "./knowledge_graph.pkl")
            logging.info(f"Initializing knowledge graph from: {graph_path}")
            self.knowledge_graph = KnowledgeGraph(graph_path)
            self.graph_reasoner = GraphReasoner(self.knowledge_graph)
            
            # Initialize retriever
            rerank_model = self.config.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            synonyms_path = self.config.get("synonyms_path")
            
            legal_synonyms = {}
            if synonyms_path and os.path.exists(synonyms_path):
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    legal_synonyms = json.load(f)
                logging.info(f"Loaded {len(legal_synonyms)} legal synonyms from {synonyms_path}")
            
            logging.info(f"Initializing legal retriever with rerank model: {rerank_model}")
            self.retriever = LegalRetriever(
                self.vector_store,
                self.knowledge_graph,
                self.embedding_model,
                rerank_model_name=rerank_model,
                legal_synonyms=legal_synonyms
            )
            
            # Initialize generator
            api_key = self.config["openai_api_key"]
            generation_model = self.config.get("generation_model", "gpt-4-turbo")
            logging.info(f"Initializing response generator with model: {generation_model}")
            self.generator = ResponseGenerator(api_key, model=generation_model)
            
            logging.info("All components initialized successfully")
            
        except Exception as e:
            error_msg = f"Error initializing components: {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            raise
    
    async def load_data(self, json_path: str, rebuild: bool = False):
        """Load legal data from JSON.
        
        Args:
            json_path: Path to JSON file with legal chunks
            rebuild: Whether to force rebuild embeddings and graph
        """
        try:
            logging.info(f"Loading data from {json_path} (rebuild={rebuild})")
            
            # Check if data is already loaded
            if not rebuild:
                vector_store_exists = self._check_vector_store()
                graph_exists = self._check_knowledge_graph()
                
                if vector_store_exists and graph_exists:
                    logging.info("Data already loaded, skipping loading process")
                    return {
                        "status": "skipped",
                        "message": "Data already loaded",
                        "graph_nodes": len(self.knowledge_graph.graph.nodes()),
                        "graph_edges": len(self.knowledge_graph.graph.edges()),
                        "vector_store_docs": self.vector_store.collection.count()
                    }
            
            # Load chunks
            with open(json_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            logging.info(f"Loaded {len(chunks)} chunks from {json_path}")
            
            # Create document IDs
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            # Extract documents and metadata
            documents = [chunk["content"] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {}
                
                # Include metadata if available
                if "metadata" in chunk:
                    metadata.update(chunk["metadata"])
                
                # Include header information
                if "header" in chunk:
                    metadata["header"] = chunk["header"]
                if "header_level" in chunk:
                    metadata["header_level"] = str(chunk["header_level"])
                
                metadatas.append(metadata)
            
            # Add documents to vector store
            batch_size = self.config.get("batch_size", 100)
            logging.info(f"Adding {len(documents)} documents to vector store with batch size {batch_size}")
            await self.vector_store.add_documents_async(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
                batch_size=batch_size
            )
            
            # Build knowledge graph
            logging.info(f"Building knowledge graph from {len(chunks)} chunks")
            await self.knowledge_graph.build_from_chunks_async(chunks, batch_size=batch_size)
            
            # Save graph if path configured
            graph_path = self.config.get("graph_path", "./knowledge_graph.pkl")
            if graph_path:
                logging.info(f"Saving knowledge graph to {graph_path}")
                self.knowledge_graph.save(graph_path)
                logging.info(f"Knowledge graph saved with {len(self.knowledge_graph.graph.nodes())} nodes and {len(self.knowledge_graph.graph.edges())} edges")
            
            return {
                "status": "success",
                "chunks_loaded": len(chunks),
                "graph_nodes": len(self.knowledge_graph.graph.nodes()),
                "graph_edges": len(self.knowledge_graph.graph.edges()),
                "vector_store_docs": self.vector_store.collection.count()
            }
            
        except Exception as e:
            error_msg = f"Error loading data: {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            raise
    async def load_data_directory(self, directory_path: str, rebuild: bool = False):
        """Carga todos los archivos JSON de un directorio.
        
        Args:
            directory_path: Ruta al directorio que contiene archivos JSON
            rebuild: Si se debe reconstruir embeddings y grafo
            
        Returns:
            Resultados del procesamiento de cada archivo
        """
        import os
        
        # Verificar que el directorio existe
        if not os.path.exists(directory_path):
            logging.error(f"El directorio {directory_path} no existe")
            raise FileNotFoundError(f"El directorio {directory_path} no existe")
        
        # Listar todos los archivos JSON en el directorio
        file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                    if f.endswith('.json')]
        
        logging.info(f"Encontrados {len(file_paths)} archivos JSON en {directory_path}")
        
        # Procesar cada archivo
        results = []
        for file_path in file_paths:
            logging.info(f"Procesando archivo: {file_path}")
            try:
                result = await self.load_data(file_path, rebuild=rebuild)
                results.append({"file": file_path, "status": "success", "result": result})
            except Exception as e:
                logging.error(f"Error procesando {file_path}: {e}", exc_info=True)
                results.append({"file": file_path, "status": "error", "error": str(e)})
        
        return results
    
    
    
    
    
    
    
    def _check_vector_store(self) -> bool:
        """Check if vector store has data.
        
        Returns:
            True if vector store has data
        """
        try:
            count = self.vector_store.collection.count()
            logging.info(f"Vector store has {count} documents")
            return count > 0
        except Exception as e:
            logging.warning(f"Error checking vector store: {e}")
            return False
    
    def _check_knowledge_graph(self) -> bool:
        """Check if knowledge graph has data.
        
        Returns:
            True if knowledge graph has data
        """
        try:
            node_count = len(self.knowledge_graph.graph.nodes())
            edge_count = len(self.knowledge_graph.graph.edges())
            logging.info(f"Knowledge graph has {node_count} nodes and {edge_count} edges")
            return node_count > 0
        except Exception as e:
            logging.warning(f"Error checking knowledge graph: {e}")
            return False
            
    async def validate_answer(self, 
                         query: str, 
                         response: str, 
                         context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Valida la precisión factual de la respuesta generada.
        
        Args:
            query: Consulta original
            response: Respuesta generada
            context: Chunks utilizados como contexto
            
        Returns:
            Resultados de la validación
        """
        logging.info(f"Validando precisión de respuesta para consulta: '{query}'")
        
        # Extraer referencias a artículos de la respuesta
        import re
        citation_pattern = r"([^.]*?[Aa]rt[íi]culo\s+(\d+(\s*[a-z])?)[^.]*\.)"
        citations = re.findall(citation_pattern, response)
        
        validation_results = {
            "validated_citations": [],
            "unvalidated_citations": [],
            "factual_accuracy": 1.0
        }
        
        # Para cada citación, verificar si está respaldada por el contexto
        for cite_text, article_num, _ in citations:
            article_normalized = f"articulo_{article_num.strip().lower()}"
            
            # Verificar si la citación está respaldada por el contexto
            supported = False
            supporting_context = None
            
            for chunk in context:
                chunk_content = chunk.get("content", "").lower()
                chunk_metadata = chunk.get("metadata", {})
                
                # Verificar en el contenido o en los metadatos
                if article_normalized in chunk_content or (
                    "article" in chunk_metadata and 
                    str(chunk_metadata["article"]).strip().lower() == article_num.strip().lower()
                ):
                    supported = True
                    supporting_context = chunk
                    break
            
            if supported:
                validation_results["validated_citations"].append({
                    "citation": cite_text.strip(),
                    "article": article_num.strip(),
                    "supporting_context": supporting_context.get("id", "unknown")
                })
            else:
                validation_results["unvalidated_citations"].append({
                    "citation": cite_text.strip(),
                    "article": article_num.strip()
                })
        
        # Calcular puntuación de precisión factual
        total_citations = len(citations)
        if total_citations > 0:
            validation_results["factual_accuracy"] = len(validation_results["validated_citations"]) / total_citations
        
        # Registrar resultados
        logging.info(f"Validación completada - Precisión: {validation_results['factual_accuracy']:.2f}")
        logging.info(f"Citas validadas: {len(validation_results['validated_citations'])}, No validadas: {len(validation_results['unvalidated_citations'])}")
        
        return validation_results            

    async def process_query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Process a user query.
        
        Args:
            query: User query
            use_cache: Whether to use query cache
            
        Returns:
            Response with metadata
        """
        try:
            logging.info(f"Processing query: '{query}'")
            
            # Check cache first if enabled
            if use_cache:
                cached_response = self.query_cache.get(query, {})
                if cached_response:
                    logging.info(f"Retrieved response from cache for query: '{query}'")
                    cached_response["from_cache"] = True
                    return cached_response
            
            # Check if data is loaded
            if not self._check_vector_store() or not self._check_knowledge_graph():
                logging.warning("No data loaded, cannot process query")
                return {
                    "response": "No hay datos cargados en el sistema. Por favor, cargue primero el Código Penal utilizando el parámetro --json_path.",
                    "articles_cited": [],
                    "chunks_used": []
                }
            
            # 1. Retrieve relevant information
            logging.info("Starting retrieval process")
            chunks = await self.retriever.retrieve(
                query, 
                top_k=self.config.get("retrieval_top_k", 10),
                use_hybrid=self.config.get("use_hybrid_search", True),
                use_expansion=self.config.get("use_query_expansion", True)
            )
            
            if not chunks:
                logging.warning("No relevant chunks found for query")
                return {
                    "response": "No se encontró información relevante para responder a su consulta.",
                    "articles_cited": [],
                    "chunks_used": []
                }
            
            logging.info(f"Retrieved {len(chunks)} relevant chunks")
            
            # 2. Extract entities for graph reasoning
            logging.info("Extracting entities from query")
            from src.utils.text_processing import extract_articles, extract_legal_concepts, extract_penalties
            articles = extract_articles(query)
            concepts = extract_legal_concepts(query)
            penalties = extract_penalties(query)
            
            all_entities = articles + concepts + penalties
            logging.info(f"Extracted entities: {len(all_entities)} - Articles: {len(articles)}, Concepts: {len(concepts)}, Penalties: {len(penalties)}")
            
            # 3. Perform graph reasoning
            logging.info("Performing graph reasoning")
            graph_insights = self.graph_reasoner.reason_over_graph(query, all_entities)
            logging.info(f"Graph reasoning complete with {sum(len(graph_insights.get(k, [])) for k in graph_insights)} insights")
            
            # 4. Format context information
            logging.info("Formatting context and graph knowledge")
            context = await self.generator.format_context(chunks)
            graph_knowledge = self.generator.format_graph_knowledge(graph_insights)
            
            # 5. Generate response
            logging.info("Generating response")
            response = await self.generator.generate(
                query, 
                context, 
                graph_knowledge,
                max_tokens=self.config.get("max_tokens", 1500),
                temperature=self.config.get("temperature", 0.2)
            )
            
            # 6. Validar precisión factual
            validation_results = await self.validate_answer(query, response["response"], chunks)
            response["validation"] = validation_results
            
            # Si hay citas no validadas, añadir advertencia
            if validation_results["unvalidated_citations"] and validation_results["factual_accuracy"] < 0.8:
                disclaimer = "\n\nADVERTENCIA: Esta respuesta contiene referencias a artículos que podrían no estar respaldados por la información disponible. Por favor, verifique con fuentes oficiales."
                response["response"] += disclaimer
            
            # 7. Add metadata
            response["chunks_used"] = [{"id": chunk["id"], "score": chunk.get("rerank_score", 0)} for chunk in chunks]
            response["graph_insights"] = {
                "entities": len(graph_insights.get("key_entities", [])),
                "definitions": len(graph_insights.get("definitions", [])),
                "related_articles": len(graph_insights.get("related_articles", [])),
                "related_concepts": len(graph_insights.get("related_concepts", []))
            }
            
            logging.info(f"Response generated successfully with {len(response.get('articles_cited', []))} cited articles")
            
            # Check if we have an actual response
            if not response.get("response"):
                logging.warning("Generated response is empty")
                response["response"] = "Lo siento, no se pudo generar una respuesta específica para tu consulta. Por favor, intenta reformular tu pregunta o proporciona más detalles."
            
            if use_cache and response.get("response"):
                self.query_cache.set(query, {}, response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            return {
                "response": f"Lo siento, ocurrió un error al procesar su consulta: {str(e)}",
                "error": str(e),
                "articles_cited": [],
                "chunks_used": []
            }
            
        
