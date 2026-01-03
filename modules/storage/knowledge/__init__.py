"""Knowledge store for RAG document and chunk management.

This module provides abstraction for storing and searching documents
and their chunks with vector embeddings for RAG workflows.

Basic Usage:
    >>> from modules.storage.knowledge import create_knowledge_store
    >>> store = create_knowledge_store("postgres", engine=engine, session_factory=session)
    >>> await store.initialize()
    >>> 
    >>> # Create a knowledge base
    >>> kb = await store.create_knowledge_base("docs", embedding_model="all-MiniLM-L6-v2")
    >>> 
    >>> # Add a document
    >>> doc = await store.add_document(kb.id, "README", content="...")
    >>> 
    >>> # Search
    >>> results = await store.search_text("How do I configure X?", knowledge_base_ids=[kb.id])

Store Selection:
    - "postgres": PostgreSQL with pgvector (recommended for production)
    - Future: "sqlite", "memory" for development/testing

Configuration:
    Stores require database connections and optionally embedding providers
    and text splitters for automatic document processing.
"""

from .base import (
    # Exceptions
    KnowledgeStoreError,
    KnowledgeBaseNotFoundError,
    DocumentNotFoundError,
    ChunkNotFoundError,
    IngestionError,
    # Enums
    DocumentStatus,
    DocumentType,
    # Data classes
    ChunkMetadata,
    KnowledgeChunk,
    KnowledgeDocument,
    DocumentVersion,
    KnowledgeBase,
    SearchResult,
    SearchQuery,
    # Base class
    KnowledgeStore,
    # Registry
    register_knowledge_store,
    get_knowledge_store_class,
    list_knowledge_stores,
    create_knowledge_store,
)

from .postgres import PostgresKnowledgeStore


__all__ = [
    # Exceptions
    "KnowledgeStoreError",
    "KnowledgeBaseNotFoundError",
    "DocumentNotFoundError",
    "ChunkNotFoundError",
    "IngestionError",
    # Enums
    "DocumentStatus",
    "DocumentType",
    # Data classes
    "ChunkMetadata",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "DocumentVersion",
    "KnowledgeBase",
    "SearchResult",
    "SearchQuery",
    # Base class
    "KnowledgeStore",
    # Registry
    "register_knowledge_store",
    "get_knowledge_store_class",
    "list_knowledge_stores",
    "create_knowledge_store",
    # Concrete stores
    "PostgresKnowledgeStore",
]
