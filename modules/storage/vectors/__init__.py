"""Vector store abstraction supporting multiple providers.

Supported providers:
- pgvector: PostgreSQL with pgvector extension
- pinecone: Pinecone managed vector database
- chroma: ChromaDB local/cloud vector store
"""

from .base import (
    VectorProvider,
    VectorStoreError,
    VectorDocument,
    VectorSearchResult,
    VectorStoreRegistry,
    get_vector_provider,
    register_vector_provider,
)
from .pgvector import PgVectorProvider
from .pinecone import PineconeProvider
from .chroma import ChromaProvider
from .adapter import (
    VectorProviderAdapter,
    create_storage_manager_adapter,
    register_storage_manager_adapter,
)

__all__ = [
    # Base
    "VectorProvider",
    "VectorStoreError",
    "VectorDocument",
    "VectorSearchResult",
    "VectorStoreRegistry",
    "get_vector_provider",
    "register_vector_provider",
    # Providers
    "PgVectorProvider",
    "PineconeProvider",
    "ChromaProvider",
    # Adapter bridge
    "VectorProviderAdapter",
    "create_storage_manager_adapter",
    "register_storage_manager_adapter",
]
