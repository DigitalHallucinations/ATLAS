"""RAG retrieval and context assembly.

This module provides the retrieval pipeline for RAG workflows:
- RAGRetriever: Semantic search with optional reranking
- Rerankers: Cross-encoder and Cohere reranking
- Context assembly for LLM prompts

Usage:
    >>> from modules.storage.retrieval import RAGRetriever, RerankerType
    >>> retriever = RAGRetriever(
    ...     knowledge_store=store,
    ...     embedding_provider=embedder,
    ...     reranker_type=RerankerType.CROSS_ENCODER,
    ... )
    >>> await retriever.initialize()
    >>> results = await retriever.retrieve("How do I configure?")
    >>> context = retriever.assemble_context(results)
"""

from modules.storage.retrieval.retriever import (
    # Exceptions
    RetrievalError,
    RerankError,
    # Enums
    RerankerType,
    ContextFormat,
    # Data classes
    RetrievalResult,
    ContextChunk,
    AssembledContext,
    # Rerankers
    Reranker,
    CrossEncoderReranker,
    CohereReranker,
    # Main class
    RAGRetriever,
)


__all__ = [
    # Exceptions
    "RetrievalError",
    "RerankError",
    # Enums
    "RerankerType",
    "ContextFormat",
    # Data classes
    "RetrievalResult",
    "ContextChunk",
    "AssembledContext",
    # Rerankers
    "Reranker",
    "CrossEncoderReranker",
    "CohereReranker",
    # Main class
    "RAGRetriever",
]
