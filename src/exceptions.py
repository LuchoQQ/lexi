# src/exceptions.py
"""Custom exceptions for legal assistant."""

class LegalAssistantError(Exception):
    """Base exception for all legal assistant errors."""
    pass

class ConfigurationError(LegalAssistantError):
    """Exception raised for configuration errors."""
    pass

class APIError(LegalAssistantError):
    """Exception raised for API-related errors."""
    pass

class EmbeddingError(LegalAssistantError):
    """Exception raised for embedding-related errors."""
    pass

class VectorStoreError(LegalAssistantError):
    """Exception raised for vector store errors."""
    pass

class GraphError(LegalAssistantError):
    """Exception raised for knowledge graph errors."""
    pass

class RetrievalError(LegalAssistantError):
    """Exception raised for retrieval errors."""
    pass

class GenerationError(LegalAssistantError):
    """Exception raised for response generation errors."""
    pass