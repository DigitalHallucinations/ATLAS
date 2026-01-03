"""Embedding and RAG configuration settings.

Defines configuration dataclasses for embedding providers, text chunking,
retrieval, and reranking settings used throughout the RAG pipeline.

All major subsystems have an `enabled` flag for granular on/off control:
- RAGSettings.enabled: Master switch for entire RAG pipeline
- EmbeddingSettings.enabled: Enable/disable embedding generation
- ChunkingSettings.enabled: Enable/disable automatic chunking
- RetrievalSettings.enabled: Enable/disable context retrieval
- RerankingSettings.enabled: Enable/disable result reranking
- KnowledgeStoreSettings.enabled: Enable/disable knowledge store
- IngestionSettings.enabled: Enable/disable document ingestion
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class EmbeddingProviderType(str, Enum):
    """Available embedding provider types."""

    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"

    @classmethod
    def coerce(cls, value: Any, default: "EmbeddingProviderType") -> "EmbeddingProviderType":
        """Return the enum value matching ``value`` or ``default`` when invalid."""
        if isinstance(value, EmbeddingProviderType):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            for candidate in cls:
                if candidate.value == lowered:
                    return candidate
        return default


class TextSplitterType(str, Enum):
    """Available text splitter types."""

    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"

    @classmethod
    def coerce(cls, value: Any, default: "TextSplitterType") -> "TextSplitterType":
        """Return the enum value matching ``value`` or ``default`` when invalid."""
        if isinstance(value, TextSplitterType):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            for candidate in cls:
                if candidate.value == lowered:
                    return candidate
        return default


class RerankerType(str, Enum):
    """Available reranker types."""

    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"

    @classmethod
    def coerce(cls, value: Any, default: "RerankerType") -> "RerankerType":
        """Return the enum value matching ``value`` or ``default`` when invalid."""
        if isinstance(value, RerankerType):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            for candidate in cls:
                if candidate.value == lowered:
                    return candidate
        return default


@dataclass
class OpenAIEmbeddingSettings:
    """OpenAI embedding provider settings."""

    api_key: Optional[str] = None
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "base_url": self.base_url,
            "organization": self.organization,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "OpenAIEmbeddingSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        return cls(
            api_key=data.get("api_key") or os.getenv("OPENAI_API_KEY"),
            model=str(data.get("model", "text-embedding-3-small")),
            dimensions=int(data["dimensions"]) if data.get("dimensions") else None,
            base_url=data.get("base_url"),
            organization=data.get("organization"),
            batch_size=int(data.get("batch_size", 100)),
            max_retries=int(data.get("max_retries", 3)),
            timeout=float(data.get("timeout", 60.0)),
        )


@dataclass
class CohereEmbeddingSettings:
    """Cohere embedding provider settings."""

    api_key: Optional[str] = None
    model: str = "embed-english-v3.0"
    batch_size: int = 96
    max_retries: int = 3
    timeout: float = 60.0
    truncate: str = "END"

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "model": self.model,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "truncate": self.truncate,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CohereEmbeddingSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        return cls(
            api_key=data.get("api_key") or os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY"),
            model=str(data.get("model", "embed-english-v3.0")),
            batch_size=int(data.get("batch_size", 96)),
            max_retries=int(data.get("max_retries", 3)),
            timeout=float(data.get("timeout", 60.0)),
            truncate=str(data.get("truncate", "END")),
        )


@dataclass
class HuggingFaceEmbeddingSettings:
    """HuggingFace embedding provider settings (sentence-transformers)."""

    model: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None  # None = auto-detect
    normalize_embeddings: bool = True
    batch_size: int = 32
    cache_folder: Optional[str] = None
    trust_remote_code: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "model": self.model,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "cache_folder": self.cache_folder,
            "trust_remote_code": self.trust_remote_code,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "HuggingFaceEmbeddingSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        return cls(
            model=str(data.get("model", "all-MiniLM-L6-v2")),
            device=data.get("device"),
            normalize_embeddings=bool(data.get("normalize_embeddings", True)),
            batch_size=int(data.get("batch_size", 32)),
            cache_folder=data.get("cache_folder"),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
        )


@dataclass
class EmbeddingSettings:
    """Combined embedding settings.
    
    Attributes:
        enabled: Enable/disable embedding generation.
        default_provider: Which provider to use by default.
        openai: OpenAI provider configuration.
        cohere: Cohere provider configuration.
        huggingface: HuggingFace provider configuration.
    """

    enabled: bool = True
    default_provider: EmbeddingProviderType = EmbeddingProviderType.HUGGINGFACE
    openai: OpenAIEmbeddingSettings = field(default_factory=OpenAIEmbeddingSettings)
    cohere: CohereEmbeddingSettings = field(default_factory=CohereEmbeddingSettings)
    huggingface: HuggingFaceEmbeddingSettings = field(default_factory=HuggingFaceEmbeddingSettings)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "default_provider": self.default_provider.value,
            "openai": self.openai.to_dict(),
            "cohere": self.cohere.to_dict(),
            "huggingface": self.huggingface.to_dict(),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "EmbeddingSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        return cls(
            enabled=bool(data.get("enabled", True)),
            default_provider=EmbeddingProviderType.coerce(
                data.get("default_provider"), EmbeddingProviderType.HUGGINGFACE
            ),
            openai=OpenAIEmbeddingSettings.from_mapping(data.get("openai")),
            cohere=CohereEmbeddingSettings.from_mapping(data.get("cohere")),
            huggingface=HuggingFaceEmbeddingSettings.from_mapping(data.get("huggingface")),
        )


@dataclass
class ChunkingSettings:
    """Text chunking configuration.
    
    Attributes:
        enabled: Enable/disable automatic text chunking.
        default_splitter: Which splitter algorithm to use.
        chunk_size: Target size for each chunk in characters.
        chunk_overlap: Overlap between consecutive chunks.
        preserve_metadata: Whether to preserve document metadata in chunks.
    """

    enabled: bool = True
    default_splitter: TextSplitterType = TextSplitterType.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    preserve_metadata: bool = True

    # Splitter-specific settings
    recursive_separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )
    sentence_min_length: int = 20

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "default_splitter": self.default_splitter.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "preserve_metadata": self.preserve_metadata,
            "recursive_separators": self.recursive_separators,
            "sentence_min_length": self.sentence_min_length,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ChunkingSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        separators = data.get("recursive_separators")
        if not isinstance(separators, list):
            separators = ["\n\n", "\n", ". ", " ", ""]

        return cls(
            enabled=bool(data.get("enabled", True)),
            default_splitter=TextSplitterType.coerce(
                data.get("default_splitter"), TextSplitterType.RECURSIVE
            ),
            chunk_size=int(data.get("chunk_size", 512)),
            chunk_overlap=int(data.get("chunk_overlap", 50)),
            preserve_metadata=bool(data.get("preserve_metadata", True)),
            recursive_separators=separators,
            sentence_min_length=int(data.get("sentence_min_length", 20)),
        )


@dataclass
class RetrievalSettings:
    """RAG retrieval configuration."""

    enabled: bool = True
    top_k: int = 10
    similarity_threshold: float = 0.7
    max_context_chunks: int = 5
    include_metadata: bool = True
    default_knowledge_base_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "max_context_chunks": self.max_context_chunks,
            "include_metadata": self.include_metadata,
            "default_knowledge_base_ids": self.default_knowledge_base_ids,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RetrievalSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        kb_ids = data.get("default_knowledge_base_ids")
        if not isinstance(kb_ids, list):
            kb_ids = []

        return cls(
            enabled=bool(data.get("enabled", True)),
            top_k=int(data.get("top_k", 10)),
            similarity_threshold=float(data.get("similarity_threshold", 0.7)),
            max_context_chunks=int(data.get("max_context_chunks", 5)),
            include_metadata=bool(data.get("include_metadata", True)),
            default_knowledge_base_ids=kb_ids,
        )


@dataclass
class RerankingSettings:
    """Reranking configuration."""

    enabled: bool = False
    provider: RerankerType = RerankerType.CROSS_ENCODER
    top_n_rerank: int = 20
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Cohere-specific
    cohere_api_key: Optional[str] = None
    cohere_model: str = "rerank-english-v3.0"

    # Hybrid search
    hybrid_search_enabled: bool = False
    hybrid_search_weight: float = 0.5  # 0 = all keyword, 1 = all vector

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "provider": self.provider.value,
            "top_n_rerank": self.top_n_rerank,
            "model": self.model,
            "cohere_model": self.cohere_model,
            "hybrid_search_enabled": self.hybrid_search_enabled,
            "hybrid_search_weight": self.hybrid_search_weight,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RerankingSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        return cls(
            enabled=bool(data.get("enabled", False)),
            provider=RerankerType.coerce(
                data.get("provider"), RerankerType.CROSS_ENCODER
            ),
            top_n_rerank=int(data.get("top_n_rerank", 20)),
            model=str(data.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
            cohere_api_key=data.get("cohere_api_key") or os.getenv("COHERE_API_KEY"),
            cohere_model=str(data.get("cohere_model", "rerank-english-v3.0")),
            hybrid_search_enabled=bool(data.get("hybrid_search_enabled", False)),
            hybrid_search_weight=float(data.get("hybrid_search_weight", 0.5)),
        )


class KnowledgeStoreType(str, Enum):
    """Available knowledge store types."""

    POSTGRES = "postgres"

    @classmethod
    def coerce(cls, value: Any, default: "KnowledgeStoreType") -> "KnowledgeStoreType":
        """Return the enum value matching ``value`` or ``default`` when invalid."""
        if isinstance(value, KnowledgeStoreType):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            for candidate in cls:
                if candidate.value == lowered:
                    return candidate
        return default


@dataclass
class KnowledgeStoreSettings:
    """Knowledge store configuration."""

    enabled: bool = True
    store_type: KnowledgeStoreType = KnowledgeStoreType.POSTGRES
    schema: str = "public"
    index_type: str = "hnsw"  # hnsw or ivfflat
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    auto_create_tables: bool = True

    # Default knowledge base settings
    default_embedding_model: str = "all-MiniLM-L6-v2"
    default_embedding_dimension: int = 384
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "store_type": self.store_type.value,
            "schema": self.schema,
            "index_type": self.index_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "auto_create_tables": self.auto_create_tables,
            "default_embedding_model": self.default_embedding_model,
            "default_embedding_dimension": self.default_embedding_dimension,
            "default_chunk_size": self.default_chunk_size,
            "default_chunk_overlap": self.default_chunk_overlap,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KnowledgeStoreSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        return cls(
            enabled=bool(data.get("enabled", True)),
            store_type=KnowledgeStoreType.coerce(
                data.get("store_type"), KnowledgeStoreType.POSTGRES
            ),
            schema=str(data.get("schema", "public")),
            index_type=str(data.get("index_type", "hnsw")),
            hnsw_m=int(data.get("hnsw_m", 16)),
            hnsw_ef_construction=int(data.get("hnsw_ef_construction", 64)),
            auto_create_tables=bool(data.get("auto_create_tables", True)),
            default_embedding_model=str(
                data.get("default_embedding_model", "all-MiniLM-L6-v2")
            ),
            default_embedding_dimension=int(
                data.get("default_embedding_dimension", 384)
            ),
            default_chunk_size=int(data.get("default_chunk_size", 512)),
            default_chunk_overlap=int(data.get("default_chunk_overlap", 50)),
        )


@dataclass
class IngestionSettings:
    """Document ingestion configuration.
    
    Attributes:
        enabled: Enable/disable document ingestion pipeline.
        auto_detect_type: Automatically detect file types.
        extract_metadata: Extract metadata from documents.
        supported_extensions: List of supported file extensions.
        max_file_size_mb: Maximum file size for ingestion.
        url_timeout: Timeout for URL fetching in seconds.
    """

    enabled: bool = True
    auto_detect_type: bool = True
    extract_metadata: bool = True
    supported_extensions: List[str] = field(
        default_factory=lambda: [
            ".txt", ".md", ".html", ".htm", ".pdf",
            ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
            ".json", ".yaml", ".yml", ".xml", ".csv",
        ]
    )
    max_file_size_mb: float = 50.0
    url_timeout: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "auto_detect_type": self.auto_detect_type,
            "extract_metadata": self.extract_metadata,
            "supported_extensions": self.supported_extensions,
            "max_file_size_mb": self.max_file_size_mb,
            "url_timeout": self.url_timeout,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "IngestionSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        extensions = data.get("supported_extensions")
        if not isinstance(extensions, list):
            extensions = [
                ".txt", ".md", ".html", ".htm", ".pdf",
                ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h",
                ".json", ".yaml", ".yml", ".xml", ".csv",
            ]

        return cls(
            enabled=bool(data.get("enabled", True)),
            auto_detect_type=bool(data.get("auto_detect_type", True)),
            extract_metadata=bool(data.get("extract_metadata", True)),
            supported_extensions=extensions,
            max_file_size_mb=float(data.get("max_file_size_mb", 50.0)),
            url_timeout=float(data.get("url_timeout", 30.0)),
        )


@dataclass
class RAGSettings:
    """Complete RAG configuration combining all subsystems.
    
    Master control for the entire RAG pipeline with granular subsystem toggles.
    
    Attributes:
        enabled: Master switch - disables entire RAG pipeline when False.
        auto_retrieve: Automatically retrieve context for user queries.
        max_context_tokens: Maximum tokens for retrieved context.
        embeddings: Embedding generation settings.
        chunking: Text chunking settings.
        retrieval: Context retrieval settings.
        reranking: Result reranking settings.
        knowledge_store: Knowledge store settings.
        ingestion: Document ingestion settings.
    """

    enabled: bool = False  # Disabled by default - user must opt-in
    auto_retrieve: bool = True  # When enabled, auto-retrieve context
    max_context_tokens: int = 4000  # Max tokens for context injection
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    reranking: RerankingSettings = field(default_factory=RerankingSettings)
    knowledge_store: KnowledgeStoreSettings = field(default_factory=KnowledgeStoreSettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)

    @property
    def is_fully_enabled(self) -> bool:
        """Check if RAG pipeline is fully operational."""
        return (
            self.enabled
            and self.embeddings.enabled
            and self.knowledge_store.enabled
            and self.retrieval.enabled
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
        return {
            "enabled": self.enabled,
            "auto_retrieve": self.auto_retrieve,
            "max_context_tokens": self.max_context_tokens,
            "embeddings": self.embeddings.to_dict(),
            "chunking": self.chunking.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "reranking": self.reranking.to_dict(),
            "knowledge_store": self.knowledge_store.to_dict(),
            "ingestion": self.ingestion.to_dict(),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RAGSettings":
        """Create from configuration mapping."""
        if not isinstance(data, Mapping):
            return cls()

        return cls(
            enabled=bool(data.get("enabled", False)),
            auto_retrieve=bool(data.get("auto_retrieve", True)),
            max_context_tokens=int(data.get("max_context_tokens", 4000)),
            embeddings=EmbeddingSettings.from_mapping(data.get("embeddings")),
            chunking=ChunkingSettings.from_mapping(data.get("chunking")),
            retrieval=RetrievalSettings.from_mapping(data.get("retrieval")),
            reranking=RerankingSettings.from_mapping(data.get("reranking")),
            knowledge_store=KnowledgeStoreSettings.from_mapping(data.get("knowledge_store")),
            ingestion=IngestionSettings.from_mapping(data.get("ingestion")),
        )


# Default configuration instance
DEFAULT_RAG_SETTINGS = RAGSettings()


__all__ = [
    # Enums
    "EmbeddingProviderType",
    "TextSplitterType",
    "RerankerType",
    "KnowledgeStoreType",
    # Provider settings
    "OpenAIEmbeddingSettings",
    "CohereEmbeddingSettings",
    "HuggingFaceEmbeddingSettings",
    # Combined settings
    "EmbeddingSettings",
    "ChunkingSettings",
    "RetrievalSettings",
    "RerankingSettings",
    "KnowledgeStoreSettings",
    "IngestionSettings",
    "RAGSettings",
    # Defaults
    "DEFAULT_RAG_SETTINGS",
]
