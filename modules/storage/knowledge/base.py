"""Knowledge store base abstraction.

Defines the abstract interface for knowledge base storage backends,
including document and chunk management with vector embeddings.

The knowledge store builds on top of the vector store and embedding
providers to provide a higher-level abstraction for RAG workflows:

    Document → Chunks → Embeddings → Vector Store
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class KnowledgeStoreError(Exception):
    """Base exception for knowledge store operations."""

    pass


class KnowledgeBaseNotFoundError(KnowledgeStoreError):
    """Raised when a knowledge base is not found."""

    pass


class DocumentNotFoundError(KnowledgeStoreError):
    """Raised when a document is not found."""

    pass


class ChunkNotFoundError(KnowledgeStoreError):
    """Raised when a chunk is not found."""

    pass


class IngestionError(KnowledgeStoreError):
    """Raised when document ingestion fails."""

    pass


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class DocumentStatus(str, Enum):
    """Status of a document in the knowledge base."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentType(str, Enum):
    """Type of document content."""

    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    CODE = "code"
    JSON = "json"
    CSV = "csv"
    UNKNOWN = "unknown"


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class ChunkMetadata:
    """Metadata for a document chunk.

    Attributes:
        start_char: Starting character position in source document.
        end_char: Ending character position in source document.
        start_line: Starting line number (if applicable).
        end_line: Ending line number (if applicable).
        section: Section or heading the chunk belongs to.
        language: Programming language (for code chunks).
        extra: Additional metadata.
    """

    start_char: int = 0
    end_char: int = 0
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    section: Optional[str] = None
    language: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeChunk:
    """A chunk of a document with its embedding.

    Represents a portion of a document that has been split for
    embedding and retrieval.

    Attributes:
        id: Unique chunk identifier.
        document_id: Parent document identifier.
        knowledge_base_id: Knowledge base identifier.
        content: The chunk text content.
        embedding: Vector embedding (may be None if not yet computed).
        chunk_index: Position of this chunk in the document.
        token_count: Approximate token count.
        content_length: Character length of content (for BM25 normalization).
        section_path: Hierarchical section path (e.g., "Doc > Heading > Sub").
        parent_chunk_id: ID of parent chunk for hierarchical retrieval.
        is_parent: Whether this is a parent chunk (larger context window).
        metadata: Chunk-specific metadata.
        created_at: Creation timestamp.
    """

    id: str
    document_id: str
    knowledge_base_id: str
    content: str
    embedding: Optional[List[float]] = None
    chunk_index: int = 0
    token_count: int = 0
    content_length: int = 0
    section_path: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    is_parent: bool = False
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.content_length == 0 and self.content:
            self.content_length = len(self.content)


@dataclass(slots=True)
class KnowledgeDocument:
    """A document in a knowledge base.

    Represents a source document that has been or will be processed
    into chunks for RAG retrieval.

    Attributes:
        id: Unique document identifier.
        knowledge_base_id: Parent knowledge base identifier.
        title: Document title or filename.
        content: Full document content (may be None if only chunks stored).
        content_hash: Hash of content for deduplication.
        source_uri: Original source location (file path, URL, etc.).
        document_type: Type of document content.
        status: Processing status.
        chunk_count: Number of chunks created from this document.
        token_count: Total token count of the document.
        metadata: Document metadata (author, tags, etc.).
        error_message: Error message if processing failed.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        indexed_at: When the document was fully indexed.
    """

    id: str
    knowledge_base_id: str
    title: str
    content: Optional[str] = None
    content_hash: Optional[str] = None
    source_uri: Optional[str] = None
    document_type: DocumentType = DocumentType.TEXT
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None
    current_version: int = 1
    version_count: int = 1

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        now = datetime.utcnow()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now


@dataclass(slots=True)
class DocumentVersion:
    """A version snapshot of a document.

    Stores a historical version of a document's content for
    version control and rollback capabilities.

    Attributes:
        id: Unique version identifier.
        document_id: Parent document identifier.
        version_number: Sequential version number (1, 2, 3, ...).
        title: Document title at this version.
        content: Full document content at this version.
        content_hash: Hash of content for comparison.
        change_summary: Optional description of changes.
        created_by: User who created this version.
        created_at: When this version was created.
        metadata: Version-specific metadata.
    """

    id: str
    document_id: str
    version_number: int
    title: str
    content: str
    content_hash: Optional[str] = None
    change_summary: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass(slots=True)
class KnowledgeBase:
    """A knowledge base containing documents and chunks.

    Represents a logical grouping of documents for a specific use case
    (e.g., project documentation, code repository, user uploads).

    Attributes:
        id: Unique knowledge base identifier.
        name: Human-readable name.
        description: Description of the knowledge base purpose.
        embedding_model: Name of the embedding model used.
        embedding_dimension: Dimension of embeddings.
        chunk_size: Default chunk size for this knowledge base.
        chunk_overlap: Default chunk overlap.
        document_count: Number of documents in the knowledge base.
        chunk_count: Total number of chunks.
        owner_id: Owner user or tenant ID.
        metadata: Additional metadata.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    id: str
    name: str
    description: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 50
    document_count: int = 0
    chunk_count: int = 0
    owner_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        now = datetime.utcnow()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now


@dataclass(slots=True)
class SearchResult:
    """Result from a knowledge base search.

    Attributes:
        chunk: The matching chunk.
        document: The parent document (may be partial).
        score: Similarity score (higher is more similar).
        distance: Raw distance from query vector.
        highlights: Optional highlighted text snippets.
    """

    chunk: KnowledgeChunk
    document: Optional[KnowledgeDocument] = None
    score: float = 0.0
    distance: Optional[float] = None
    highlights: Optional[List[str]] = None


@dataclass(slots=True)
class SearchQuery:
    """Query parameters for knowledge base search.

    Attributes:
        query_text: The search query text.
        query_embedding: Pre-computed query embedding (optional).
        knowledge_base_ids: Limit search to specific knowledge bases.
        document_ids: Limit search to specific documents.
        top_k: Number of results to return.
        min_score: Minimum similarity score threshold.
        metadata_filter: Filter by chunk/document metadata.
        document_types: Filter by document types.
        include_content: Whether to include full chunk content.
        include_document: Whether to include parent document info.
    """

    query_text: str = ""
    query_embedding: Optional[List[float]] = None
    knowledge_base_ids: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
    top_k: int = 10
    min_score: float = 0.0
    metadata_filter: Optional[Dict[str, Any]] = None
    document_types: Optional[List[DocumentType]] = None
    include_content: bool = True
    include_document: bool = True


# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------


class KnowledgeStore(ABC):
    """Abstract base class for knowledge store implementations.

    The knowledge store provides a high-level interface for managing
    documents, chunks, and embeddings for RAG workflows. It coordinates
    between the embedding provider and vector store.

    Implementations should handle:
    - Knowledge base lifecycle (create, update, delete)
    - Document ingestion and chunking
    - Embedding generation and storage
    - Semantic search across chunks
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the store implementation name."""
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the store has been initialized."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the knowledge store connection and resources."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources and close connections."""
        ...

    @abstractmethod
    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the store is healthy and accessible."""
        ...

    # --- Knowledge Base Management ---

    @abstractmethod
    async def create_knowledge_base(
        self,
        name: str,
        *,
        description: str = "",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        owner_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBase:
        """Create a new knowledge base.

        Args:
            name: Human-readable name for the knowledge base.
            description: Description of the knowledge base purpose.
            embedding_model: Name of embedding model to use.
            embedding_dimension: Dimension of embeddings.
            chunk_size: Default chunk size for documents.
            chunk_overlap: Default overlap between chunks.
            owner_id: Owner user or tenant ID.
            metadata: Additional metadata.

        Returns:
            The created knowledge base.

        Raises:
            KnowledgeStoreError: If creation fails.
        """
        ...

    @abstractmethod
    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by ID.

        Args:
            kb_id: Knowledge base identifier.

        Returns:
            The knowledge base or None if not found.
        """
        ...

    @abstractmethod
    async def list_knowledge_bases(
        self,
        *,
        owner_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[KnowledgeBase]:
        """List knowledge bases.

        Args:
            owner_id: Filter by owner ID.
            limit: Maximum number to return.
            offset: Number to skip for pagination.

        Returns:
            List of knowledge bases.
        """
        ...

    @abstractmethod
    async def update_knowledge_base(
        self,
        kb_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[KnowledgeBase]:
        """Update a knowledge base.

        Args:
            kb_id: Knowledge base identifier.
            name: New name (if provided).
            description: New description (if provided).
            metadata: New metadata (merged with existing).

        Returns:
            Updated knowledge base or None if not found.
        """
        ...

    @abstractmethod
    async def delete_knowledge_base(
        self,
        kb_id: str,
        *,
        delete_documents: bool = True,
    ) -> bool:
        """Delete a knowledge base.

        Args:
            kb_id: Knowledge base identifier.
            delete_documents: Whether to also delete all documents.

        Returns:
            True if deleted, False if not found.
        """
        ...

    # --- Document Management ---

    @abstractmethod
    async def add_document(
        self,
        kb_id: str,
        title: str,
        content: str,
        *,
        source_uri: Optional[str] = None,
        document_type: DocumentType = DocumentType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
        auto_chunk: bool = True,
        auto_embed: bool = True,
    ) -> KnowledgeDocument:
        """Add a document to a knowledge base.

        Args:
            kb_id: Knowledge base identifier.
            title: Document title.
            content: Document content.
            source_uri: Original source location.
            document_type: Type of document.
            metadata: Document metadata.
            auto_chunk: Whether to automatically chunk the document.
            auto_embed: Whether to automatically generate embeddings.

        Returns:
            The created document.

        Raises:
            KnowledgeBaseNotFoundError: If knowledge base not found.
            IngestionError: If document processing fails.
        """
        ...

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """Get a document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            The document or None if not found.
        """
        ...

    @abstractmethod
    async def list_documents(
        self,
        kb_id: str,
        *,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[KnowledgeDocument]:
        """List documents in a knowledge base.

        Args:
            kb_id: Knowledge base identifier.
            status: Filter by status.
            limit: Maximum number to return.
            offset: Number to skip for pagination.

        Returns:
            List of documents.
        """
        ...

    @abstractmethod
    async def find_duplicate(
        self,
        kb_id: str,
        content: str,
    ) -> Optional[KnowledgeDocument]:
        """Find a duplicate document by content hash.

        Args:
            kb_id: Knowledge base identifier.
            content: Content to check for duplicates.

        Returns:
            Existing document if duplicate found, None otherwise.
        """
        ...

    @abstractmethod
    async def update_document(
        self,
        doc_id: str,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rechunk: bool = False,
        reembed: bool = False,
        create_version: bool = True,
        change_summary: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Optional[KnowledgeDocument]:
        """Update a document.

        Args:
            doc_id: Document identifier.
            title: New title (if provided).
            content: New content (if provided).
            metadata: New metadata (merged with existing).
            rechunk: Whether to re-chunk the document.
            reembed: Whether to regenerate embeddings.
            create_version: Whether to create a version snapshot before updating.
            change_summary: Optional description of changes for version history.
            created_by: User ID who made the change.

        Returns:
            Updated document or None if not found.
        """
        ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks.

        Args:
            doc_id: Document identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...

    # --- Version Management ---

    @abstractmethod
    async def get_document_versions(
        self,
        doc_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List["DocumentVersion"]:
        """Get version history for a document.

        Args:
            doc_id: Document identifier.
            limit: Maximum number of versions to return.
            offset: Number to skip for pagination.

        Returns:
            List of versions ordered by version_number descending (newest first).
        """
        ...

    @abstractmethod
    async def get_document_version(
        self,
        doc_id: str,
        version_number: int,
    ) -> Optional["DocumentVersion"]:
        """Get a specific version of a document.

        Args:
            doc_id: Document identifier.
            version_number: Version number to retrieve.

        Returns:
            The version or None if not found.
        """
        ...

    @abstractmethod
    async def restore_document_version(
        self,
        doc_id: str,
        version_number: int,
        *,
        rechunk: bool = True,
        reembed: bool = True,
        created_by: Optional[str] = None,
    ) -> Optional[KnowledgeDocument]:
        """Restore a document to a previous version.

        Creates a new version with the restored content.

        Args:
            doc_id: Document identifier.
            version_number: Version number to restore to.
            rechunk: Whether to re-chunk after restoration.
            reembed: Whether to regenerate embeddings.
            created_by: User ID who performed the restore.

        Returns:
            Updated document or None if not found.
        """
        ...

    @abstractmethod
    async def compare_document_versions(
        self,
        doc_id: str,
        version_a: int,
        version_b: int,
    ) -> Optional[Dict[str, Any]]:
        """Compare two versions of a document.

        Args:
            doc_id: Document identifier.
            version_a: First version number.
            version_b: Second version number.

        Returns:
            Comparison result with differences or None if versions not found.
        """
        ...

    @abstractmethod
    async def delete_document_version(
        self,
        doc_id: str,
        version_number: int,
    ) -> bool:
        """Delete a specific version from history.

        Cannot delete the current version.

        Args:
            doc_id: Document identifier.
            version_number: Version to delete.

        Returns:
            True if deleted, False if not found or is current version.
        """
        ...

    # --- Chunk Management ---

    @abstractmethod
    async def get_chunks(
        self,
        doc_id: str,
        *,
        include_embeddings: bool = False,
    ) -> List[KnowledgeChunk]:
        """Get all chunks for a document.

        Args:
            doc_id: Document identifier.
            include_embeddings: Whether to include embedding vectors.

        Returns:
            List of chunks ordered by chunk_index.
        """
        ...

    @abstractmethod
    async def get_chunk(
        self,
        chunk_id: str,
        *,
        include_embedding: bool = False,
    ) -> Optional[KnowledgeChunk]:
        """Get a specific chunk by ID.

        Args:
            chunk_id: Chunk identifier.
            include_embedding: Whether to include embedding vector.

        Returns:
            The chunk or None if not found.
        """
        ...

    @abstractmethod
    async def update_chunk(
        self,
        chunk_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reembed: bool = True,
    ) -> Optional[KnowledgeChunk]:
        """Update a chunk's content or metadata.

        Args:
            chunk_id: Chunk identifier.
            content: New content (if provided).
            metadata: New metadata (merged with existing).
            reembed: Whether to regenerate embedding if content changed.

        Returns:
            Updated chunk or None if not found.
        """
        ...

    # --- Search ---

    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for relevant chunks across knowledge bases.

        Args:
            query: Search query parameters.

        Returns:
            List of search results sorted by relevance.
        """
        ...

    async def search_text(
        self,
        query_text: str,
        *,
        knowledge_base_ids: Optional[List[str]] = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Convenience method for simple text search.

        Args:
            query_text: The search query.
            knowledge_base_ids: Knowledge bases to search.
            top_k: Number of results.
            min_score: Minimum similarity threshold.

        Returns:
            List of search results.
        """
        query = SearchQuery(
            query_text=query_text,
            knowledge_base_ids=knowledge_base_ids,
            top_k=top_k,
            min_score=min_score,
        )
        return await self.search(query)

    # --- Batch Operations ---

    async def add_documents(
        self,
        kb_id: str,
        documents: Sequence[Dict[str, Any]],
        *,
        auto_chunk: bool = True,
        auto_embed: bool = True,
    ) -> List[KnowledgeDocument]:
        """Add multiple documents to a knowledge base.

        Default implementation calls add_document for each document.
        Implementations may override for batch optimization.

        Args:
            kb_id: Knowledge base identifier.
            documents: List of document dicts with title, content, etc.
            auto_chunk: Whether to automatically chunk.
            auto_embed: Whether to automatically embed.

        Returns:
            List of created documents.
        """
        results = []
        for doc in documents:
            result = await self.add_document(
                kb_id,
                title=doc.get("title", "Untitled"),
                content=doc.get("content", ""),
                source_uri=doc.get("source_uri"),
                document_type=doc.get("document_type", DocumentType.TEXT),
                metadata=doc.get("metadata"),
                auto_chunk=auto_chunk,
                auto_embed=auto_embed,
            )
            results.append(result)
        return results


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

_KNOWLEDGE_STORE_REGISTRY: Dict[str, Type[KnowledgeStore]] = {}

T = TypeVar("T", bound=KnowledgeStore)


def register_knowledge_store(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a knowledge store implementation.

    Usage:
        @register_knowledge_store("postgres")
        class PostgresKnowledgeStore(KnowledgeStore):
            ...
    """

    def decorator(cls: Type[T]) -> Type[T]:
        _KNOWLEDGE_STORE_REGISTRY[name.lower()] = cls
        logger.debug("Registered knowledge store: %s -> %s", name, cls.__name__)
        return cls

    return decorator


def get_knowledge_store_class(name: str) -> Type[KnowledgeStore]:
    """Get a knowledge store class by name.

    Args:
        name: Store name (e.g., "postgres", "sqlite").

    Returns:
        The knowledge store class.

    Raises:
        KnowledgeStoreError: If store not found.
    """
    name_lower = name.lower()
    if name_lower not in _KNOWLEDGE_STORE_REGISTRY:
        available = ", ".join(_KNOWLEDGE_STORE_REGISTRY.keys()) or "(none)"
        raise KnowledgeStoreError(
            f"Unknown knowledge store: '{name}'. Available: {available}"
        )
    return _KNOWLEDGE_STORE_REGISTRY[name_lower]


def list_knowledge_stores() -> List[str]:
    """List all registered knowledge store names."""
    return list(_KNOWLEDGE_STORE_REGISTRY.keys())


def create_knowledge_store(name: str, **kwargs: Any) -> KnowledgeStore:
    """Create a knowledge store instance by name.

    Args:
        name: Store name (e.g., "postgres").
        **kwargs: Arguments to pass to the store constructor.

    Returns:
        A knowledge store instance.
    """
    cls = get_knowledge_store_class(name)
    return cls(**kwargs)


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

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
]
