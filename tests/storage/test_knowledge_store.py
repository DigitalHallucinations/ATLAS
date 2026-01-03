"""Tests for the knowledge store module.

Tests cover the base classes, data models, PostgreSQL implementation,
and knowledge store registry.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from modules.storage.knowledge import (
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
    KnowledgeBase,
    SearchResult,
    SearchQuery,
    # Base class
    KnowledgeStore,
    # Registry
    get_knowledge_store_class,
    list_knowledge_stores,
    create_knowledge_store,
    # Implementations
    PostgresKnowledgeStore,
)


# --- Test Data Classes ---


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        meta = ChunkMetadata()
        assert meta.start_char == 0
        assert meta.end_char == 0
        assert meta.start_line is None
        assert meta.end_line is None
        assert meta.section is None
        assert meta.language is None
        assert meta.extra == {}

    def test_with_values(self):
        """Test creating with custom values."""
        meta = ChunkMetadata(
            start_char=100,
            end_char=200,
            start_line=10,
            end_line=20,
            section="Introduction",
            language="python",
            extra={"key": "value"},
        )
        assert meta.start_char == 100
        assert meta.end_char == 200
        assert meta.start_line == 10
        assert meta.end_line == 20
        assert meta.section == "Introduction"
        assert meta.language == "python"
        assert meta.extra == {"key": "value"}


class TestKnowledgeChunk:
    """Tests for KnowledgeChunk dataclass."""

    def test_auto_generated_id(self):
        """Test that ID is auto-generated if not provided."""
        chunk = KnowledgeChunk(
            id="",
            document_id="doc-123",
            knowledge_base_id="kb-456",
            content="Test content",
        )
        assert chunk.id != ""
        assert len(chunk.id) == 36  # UUID format

    def test_auto_generated_timestamp(self):
        """Test that created_at is auto-generated."""
        chunk = KnowledgeChunk(
            id="chunk-1",
            document_id="doc-123",
            knowledge_base_id="kb-456",
            content="Test content",
        )
        assert chunk.created_at is not None
        assert isinstance(chunk.created_at, datetime)

    def test_with_embedding(self):
        """Test chunk with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        chunk = KnowledgeChunk(
            id="chunk-1",
            document_id="doc-123",
            knowledge_base_id="kb-456",
            content="Test content",
            embedding=embedding,
            chunk_index=0,
            token_count=5,
        )
        assert chunk.embedding == embedding
        assert chunk.chunk_index == 0
        assert chunk.token_count == 5


class TestKnowledgeDocument:
    """Tests for KnowledgeDocument dataclass."""

    def test_auto_generated_fields(self):
        """Test auto-generated ID and timestamps."""
        doc = KnowledgeDocument(
            id="",
            knowledge_base_id="kb-123",
            title="Test Document",
        )
        assert doc.id != ""
        assert doc.created_at is not None
        assert doc.updated_at is not None

    def test_default_status(self):
        """Test default status is PENDING."""
        doc = KnowledgeDocument(
            id="doc-1",
            knowledge_base_id="kb-123",
            title="Test Document",
        )
        assert doc.status == DocumentStatus.PENDING

    def test_document_type(self):
        """Test document type handling."""
        doc = KnowledgeDocument(
            id="doc-1",
            knowledge_base_id="kb-123",
            title="Test Document",
            document_type=DocumentType.MARKDOWN,
        )
        assert doc.document_type == DocumentType.MARKDOWN

    def test_full_document(self):
        """Test document with all fields."""
        doc = KnowledgeDocument(
            id="doc-1",
            knowledge_base_id="kb-123",
            title="README.md",
            content="# Test\n\nContent here.",
            content_hash="abc123",
            source_uri="/path/to/file.md",
            document_type=DocumentType.MARKDOWN,
            status=DocumentStatus.INDEXED,
            chunk_count=5,
            token_count=100,
            metadata={"author": "test"},
        )
        assert doc.title == "README.md"
        assert doc.chunk_count == 5
        assert doc.metadata["author"] == "test"


class TestKnowledgeBase:
    """Tests for KnowledgeBase dataclass."""

    def test_auto_generated_fields(self):
        """Test auto-generated ID and timestamps."""
        kb = KnowledgeBase(
            id="",
            name="Test KB",
        )
        assert kb.id != ""
        assert kb.created_at is not None
        assert kb.updated_at is not None

    def test_default_values(self):
        """Test default values."""
        kb = KnowledgeBase(
            id="kb-1",
            name="Test KB",
        )
        assert kb.embedding_model == "all-MiniLM-L6-v2"
        assert kb.embedding_dimension == 384
        assert kb.chunk_size == 512
        assert kb.chunk_overlap == 50
        assert kb.document_count == 0
        assert kb.chunk_count == 0

    def test_full_knowledge_base(self):
        """Test knowledge base with all fields."""
        kb = KnowledgeBase(
            id="kb-1",
            name="Project Docs",
            description="Documentation for the project",
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536,
            chunk_size=1024,
            chunk_overlap=100,
            document_count=10,
            chunk_count=50,
            owner_id="user-123",
            metadata={"category": "technical"},
        )
        assert kb.name == "Project Docs"
        assert kb.embedding_model == "text-embedding-3-small"
        assert kb.owner_id == "user-123"


class TestSearchQuery:
    """Tests for SearchQuery dataclass."""

    def test_default_values(self):
        """Test default query values."""
        query = SearchQuery()
        assert query.query_text == ""
        assert query.query_embedding is None
        assert query.top_k == 10
        assert query.min_score == 0.0
        assert query.include_content is True
        assert query.include_document is True

    def test_query_with_text(self):
        """Test query with text."""
        query = SearchQuery(
            query_text="How do I configure logging?",
            knowledge_base_ids=["kb-1", "kb-2"],
            top_k=5,
            min_score=0.7,
        )
        assert query.query_text == "How do I configure logging?"
        assert len(query.knowledge_base_ids) == 2
        assert query.top_k == 5
        assert query.min_score == 0.7

    def test_query_with_embedding(self):
        """Test query with pre-computed embedding."""
        embedding = [0.1] * 384
        query = SearchQuery(
            query_embedding=embedding,
            top_k=20,
        )
        assert query.query_embedding == embedding
        assert query.query_text == ""


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_basic_result(self):
        """Test basic search result."""
        chunk = KnowledgeChunk(
            id="chunk-1",
            document_id="doc-1",
            knowledge_base_id="kb-1",
            content="Test content",
        )
        result = SearchResult(
            chunk=chunk,
            score=0.95,
            distance=0.05,
        )
        assert result.chunk == chunk
        assert result.score == 0.95
        assert result.distance == 0.05
        assert result.document is None
        assert result.highlights is None


# --- Test Document Enums ---


class TestDocumentStatus:
    """Tests for DocumentStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.INDEXED.value == "indexed"
        assert DocumentStatus.FAILED.value == "failed"
        assert DocumentStatus.DELETED.value == "deleted"


class TestDocumentType:
    """Tests for DocumentType enum."""

    def test_type_values(self):
        """Test all type values exist."""
        assert DocumentType.TEXT.value == "text"
        assert DocumentType.MARKDOWN.value == "markdown"
        assert DocumentType.HTML.value == "html"
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.CODE.value == "code"
        assert DocumentType.JSON.value == "json"
        assert DocumentType.CSV.value == "csv"
        assert DocumentType.UNKNOWN.value == "unknown"


# --- Test Registry ---


class TestKnowledgeStoreRegistry:
    """Tests for knowledge store registry."""

    def test_list_stores_includes_postgres(self):
        """Test that postgres store is registered."""
        stores = list_knowledge_stores()
        assert "postgres" in stores

    def test_get_postgres_store_class(self):
        """Test getting postgres store class."""
        cls = get_knowledge_store_class("postgres")
        assert cls is PostgresKnowledgeStore

    def test_get_unknown_store_raises(self):
        """Test that unknown store raises error."""
        with pytest.raises(KnowledgeStoreError) as exc_info:
            get_knowledge_store_class("unknown_store")
        assert "Unknown knowledge store" in str(exc_info.value)


# --- Test Exceptions ---


class TestKnowledgeStoreExceptions:
    """Tests for knowledge store exceptions."""

    def test_base_exception(self):
        """Test base exception."""
        exc = KnowledgeStoreError("Test error")
        assert str(exc) == "Test error"

    def test_knowledge_base_not_found(self):
        """Test knowledge base not found exception."""
        exc = KnowledgeBaseNotFoundError("KB not found")
        assert isinstance(exc, KnowledgeStoreError)

    def test_document_not_found(self):
        """Test document not found exception."""
        exc = DocumentNotFoundError("Document not found")
        assert isinstance(exc, KnowledgeStoreError)

    def test_chunk_not_found(self):
        """Test chunk not found exception."""
        exc = ChunkNotFoundError("Chunk not found")
        assert isinstance(exc, KnowledgeStoreError)

    def test_ingestion_error(self):
        """Test ingestion error exception."""
        exc = IngestionError("Ingestion failed")
        assert isinstance(exc, KnowledgeStoreError)


# --- Test PostgresKnowledgeStore (unit tests, no DB) ---


class TestPostgresKnowledgeStoreUnit:
    """Unit tests for PostgresKnowledgeStore (no database required)."""

    def test_provider_name(self):
        """Test that provider name is postgres."""
        # Create with mock engine/session
        from unittest.mock import MagicMock
        
        mock_engine = MagicMock()
        mock_session_factory = MagicMock()
        
        store = PostgresKnowledgeStore(
            engine=mock_engine,
            session_factory=mock_session_factory,
        )
        assert store.name == "postgres"

    def test_not_initialized_by_default(self):
        """Test store is not initialized by default."""
        from unittest.mock import MagicMock
        
        mock_engine = MagicMock()
        mock_session_factory = MagicMock()
        
        store = PostgresKnowledgeStore(
            engine=mock_engine,
            session_factory=mock_session_factory,
        )
        assert store.is_initialized is False

    def test_set_embedding_provider(self):
        """Test setting embedding provider."""
        from unittest.mock import MagicMock
        
        mock_engine = MagicMock()
        mock_session_factory = MagicMock()
        mock_provider = MagicMock()
        
        store = PostgresKnowledgeStore(
            engine=mock_engine,
            session_factory=mock_session_factory,
        )
        store.set_embedding_provider(mock_provider)
        assert store._embedding_provider is mock_provider

    def test_set_text_splitter(self):
        """Test setting text splitter."""
        from unittest.mock import MagicMock
        
        mock_engine = MagicMock()
        mock_session_factory = MagicMock()
        mock_splitter = MagicMock()
        
        store = PostgresKnowledgeStore(
            engine=mock_engine,
            session_factory=mock_session_factory,
        )
        store.set_text_splitter(mock_splitter)
        assert store._text_splitter is mock_splitter


# --- Integration Tests (require PostgreSQL) ---
# These tests are marked to skip by default


@pytest.mark.skip(reason="Requires PostgreSQL with pgvector")
class TestPostgresKnowledgeStoreIntegration:
    """Integration tests for PostgresKnowledgeStore.
    
    These tests require a running PostgreSQL instance with pgvector.
    """

    @pytest.fixture
    async def store(self):
        """Create and initialize a test knowledge store."""
        # This would require actual DB connection setup
        # Placeholder for actual integration test setup
        pass

    async def test_create_knowledge_base(self, store):
        """Test creating a knowledge base."""
        pass

    async def test_add_document(self, store):
        """Test adding a document."""
        pass

    async def test_search(self, store):
        """Test searching for chunks."""
        pass
