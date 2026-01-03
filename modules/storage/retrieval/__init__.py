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
    # Utility functions
    rrf_fuse,
)

from modules.storage.retrieval.query_router import (
    # Enums
    QueryIntent,
    # Data classes
    ClassificationResult,
    # Routers
    QueryRouter,
    SimpleQueryRouter,
)

from modules.storage.retrieval.evidence import (
    # Data classes
    Citation,
    ClaimVerification,
    VerificationResult,
    SupportLevel,
    # Classes
    CitationExtractor,
    FaithfulnessScorer,
    EvidenceGate,
)

from modules.storage.retrieval.cache import (
    # Data classes
    CacheEntry,
    # Caches
    EmbeddingCache,
    QueryResultCache,
    # Provider wrapper
    CachedEmbeddingProvider,
)

from modules.storage.retrieval.compression import (
    # Data classes
    CompressionResult,
    # Compressors
    ContextCompressor,
    LLMLinguaCompressor,
    ExtractiveSummarizer,
    HybridCompressor,
    # Factory
    create_compressor,
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
    # Utility functions
    "rrf_fuse",
    # Query routing
    "QueryIntent",
    "ClassificationResult",
    "QueryRouter",
    "SimpleQueryRouter",
    # Evidence gating
    "Citation",
    "ClaimVerification",
    "VerificationResult",
    "SupportLevel",
    "CitationExtractor",
    "FaithfulnessScorer",
    "EvidenceGate",
    # Caching
    "CacheEntry",
    "EmbeddingCache",
    "QueryResultCache",
    "CachedEmbeddingProvider",
    # Compression
    "CompressionResult",
    "ContextCompressor",
    "LLMLinguaCompressor",
    "ExtractiveSummarizer",
    "HybridCompressor",
    "create_compressor",
]
