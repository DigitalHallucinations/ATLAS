"""Embedding providers for RAG vector generation.

This module provides abstraction for generating text embeddings using
various providers (OpenAI, Cohere, HuggingFace).

Basic Usage:
    >>> from modules.storage.embeddings import create_embedding_provider
    >>> provider = create_embedding_provider("huggingface", model="all-MiniLM-L6-v2")
    >>> result = await provider.embed_text("Hello, world!")
    >>> print(result.embedding[:5])  # First 5 dimensions
    [0.123, 0.456, ...]

Provider Selection:
    - "openai": OpenAI API (text-embedding-3-small, text-embedding-3-large, ada-002)
    - "cohere": Cohere API (embed-english-v3.0, embed-multilingual-v3.0)
    - "huggingface": HuggingFace/Sentence-transformers (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)

Configuration:
    Providers can be configured via ATLAS/config/rag.py settings or
    by passing kwargs directly to create_embedding_provider().
    
    All providers support config_manager parameter for API key resolution
    from the ATLAS provider system.
"""

from .base import (
    # Core classes
    EmbeddingProvider,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingInputType,
    # Registry functions
    register_embedding_provider,
    get_embedding_provider_class,
    create_embedding_provider,
    list_embedding_providers,
    # Model utilities
    MODEL_DIMENSIONS,
    get_model_dimension,
    # Exceptions
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingGenerationError,
    EmbeddingRateLimitError,
)

from .openai import OpenAIEmbeddingProvider
from .huggingface import HuggingFaceEmbeddingProvider
from .cohere import CohereEmbeddingProvider


__all__ = [
    # Core classes
    "EmbeddingProvider",
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "EmbeddingInputType",
    # Registry functions
    "register_embedding_provider",
    "get_embedding_provider_class",
    "create_embedding_provider",
    "list_embedding_providers",
    # Model utilities
    "MODEL_DIMENSIONS",
    "get_model_dimension",
    # Exceptions
    "EmbeddingError",
    "EmbeddingProviderError",
    "EmbeddingGenerationError",
    "EmbeddingRateLimitError",
    # Concrete providers
    "OpenAIEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "CohereEmbeddingProvider",
]
