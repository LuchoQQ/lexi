# embeddings/vector_store.py
"""Vector database interface with async support."""

import os
import logging
import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
from chromadb.utils import embedding_functions
import asyncio
from src.exceptions import VectorStoreError

class VectorStore:
    """Interface to ChromaDB for vector storage and retrieval."""
    
    def __init__(self, 
                 persist_directory: str, 
                 collection_name: str,
                 embedding_model_name: str):
        """Initialize the vector store.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection
            embedding_model_name: Name of the embedding model
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Create persistence directory if it doesn't exist
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Set up embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
            
            # Get or create collection
            self.get_or_create_collection()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector store: {e}")
    
    def get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            # Check if collection exists using the new API convention
            collection_names = self.client.list_collections()
            
            # In v0.6.0+, collection_names is just a list of strings
            collection_exists = self.collection_name in collection_names
            
            if collection_exists:
                logging.info(f"Using existing collection: {self.collection_name}")
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            else:
                logging.info(f"Creating new collection: {self.collection_name}")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
        except Exception as e:
            raise VectorStoreError(f"Failed to get or create collection: {e}")
    
    async def add_documents_async(self, 
                          documents: List[str], 
                          ids: List[str], 
                          metadatas: Optional[List[Dict[str, Any]]] = None,
                          batch_size: int = 100):
        """Add documents to the vector store asynchronously.
        
        Args:
            documents: List of document texts
            ids: List of document IDs
            metadatas: List of document metadata (optional)
            batch_size: Number of documents to process in each batch
        """
        if len(documents) != len(ids):
            raise ValueError("documents and ids must have the same length")
        
        if metadatas and len(metadatas) != len(documents):
            raise ValueError("metadatas must have the same length as documents")
        
        # Prepare metadata if not provided
        if not metadatas:
            metadatas = [{} for _ in range(len(documents))]
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            batch_docs = documents[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_meta = metadatas[i:end_idx]
            
            # Clean metadata (ChromaDB doesn't support lists or dicts as values)
            cleaned_meta = []
            for meta in batch_meta:
                cleaned = {}
                for key, value in meta.items():
                    if isinstance(value, list):
                        cleaned[key] = ", ".join(str(item) for item in value)
                    elif isinstance(value, dict):
                        # Flatten dict to string
                        cleaned[key] = ", ".join(f"{k}:{v}" for k, v in value.items())
                    elif value is not None:
                        cleaned[key] = str(value)
                cleaned_meta.append(cleaned)
            
            try:
                # Run add operation in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.collection.add(
                        documents=batch_docs,
                        ids=batch_ids,
                        metadatas=cleaned_meta
                    )
                )
                logging.info(f"Added documents {i+1} to {end_idx}")
            except Exception as e:
                raise VectorStoreError(f"Failed to add documents: {e}")
    
    def add_documents(self, 
                     documents: List[str], 
                     ids: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     batch_size: int = 100):
        """Add documents to the vector store synchronously.
        
        Args:
            documents: List of document texts
            ids: List of document IDs
            metadatas: List of document metadata (optional)
            batch_size: Number of documents to process in each batch
        """
        if len(documents) != len(ids):
            raise ValueError("documents and ids must have the same length")
        
        if metadatas and len(metadatas) != len(documents):
            raise ValueError("metadatas must have the same length as documents")
        
        # Prepare metadata if not provided
        if not metadatas:
            metadatas = [{} for _ in range(len(documents))]
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            batch_docs = documents[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_meta = metadatas[i:end_idx]
            
            # Clean metadata (ChromaDB doesn't support lists or dicts as values)
            cleaned_meta = []
            for meta in batch_meta:
                cleaned = {}
                for key, value in meta.items():
                    if isinstance(value, list):
                        cleaned[key] = ", ".join(str(item) for item in value)
                    elif isinstance(value, dict):
                        # Flatten dict to string
                        cleaned[key] = ", ".join(f"{k}:{v}" for k, v in value.items())
                    elif value is not None:
                        cleaned[key] = str(value)
                cleaned_meta.append(cleaned)
            
            try:
                self.collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=cleaned_meta
                )
                logging.info(f"Added documents {i+1} to {end_idx}")
            except Exception as e:
                raise VectorStoreError(f"Failed to add documents: {e}")
    
        async def query_async(self, 
                    query_texts: List[str], 
                    n_results: int = 10, 
                    where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Query the vector store asynchronously."""
            try:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: self.collection.query(
                        query_texts=query_texts,
                        n_results=n_results,
                        where=where
                    )
                )
                return results
            except Exception as e:
                raise VectorStoreError(f"Failed to query vector store: {e}")
    
    def query(self, 
             query_texts: List[str], 
             n_results: int = 10, 
             where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the vector store synchronously.
        
        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter (optional)
            
        Returns:
            Query results
        """
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            raise VectorStoreError(f"Failed to query vector store: {e}")