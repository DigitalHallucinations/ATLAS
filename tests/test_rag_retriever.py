"""Tests for RAG retrieval pipeline.

Tests the RAGRetriever, rerankers, and context assembly.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from modules.storage.retrieval import (
    RetrievalError,
    RerankError,
    RerankerType,
    ContextFormat,
    RetrievalResult,
    ContextChunk,
    AssembledContext,
    Reranker,
    CrossEncoderReranker,
    CohereReranker,
    RAGRetriever,
)


# -----------------------------------------------------------------------------
# Data Class Tests
# -----------------------------------------------------------------------------


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = RetrievalResult(
            query="test query",
            chunks=[],
        )
        assert result.query == "test query"
        assert result.chunks == []
        assert result.documents == {}
        assert result.retrieval_time_ms == 0.0
        assert result.reranked is False

    def test_with_timing(self):
        """Test with timing info."""
        result = RetrievalResult(
            query="test",
            chunks=[],
            retrieval_time_ms=15.5,
            rerank_time_ms=10.2,
            total_time_ms=25.7,
            reranked=True,
        )
        assert result.retrieval_time_ms == 15.5
        assert result.rerank_time_ms == 10.2
        assert result.total_time_ms == 25.7
        assert result.reranked is True


class TestContextChunk:
    """Tests for ContextChunk dataclass."""

    def test_creation(self):
        """Test basic creation."""
        chunk = ContextChunk(
            content="Some content here",
            source="document.pdf",
            score=0.85,
        )
        assert chunk.content == "Some content here"
        assert chunk.source == "document.pdf"
        assert chunk.score == 0.85
        assert chunk.metadata == {}

    def test_with_metadata(self):
        """Test with metadata."""
        chunk = ContextChunk(
            content="Content",
            source="doc.txt",
            score=0.9,
            metadata={"chunk_id": "c1", "page": 5},
        )
        assert chunk.metadata["chunk_id"] == "c1"
        assert chunk.metadata["page"] == 5


class TestAssembledContext:
    """Tests for AssembledContext dataclass."""

    def test_creation(self):
        """Test basic creation."""
        context = AssembledContext(
            text="Formatted context here",
            chunks=[],
            token_count=100,
            truncated=False,
        )
        assert context.text == "Formatted context here"
        assert context.chunks == []
        assert context.token_count == 100
        assert context.truncated is False


# -----------------------------------------------------------------------------
# Reranker Tests
# -----------------------------------------------------------------------------


class TestRerankerBase:
    """Tests for base Reranker class."""

    def test_name(self):
        """Test reranker name."""
        reranker = Reranker()
        assert reranker.name == "base"

    @pytest.mark.asyncio
    async def test_rerank_not_implemented(self):
        """Test rerank raises NotImplementedError."""
        reranker = Reranker()
        with pytest.raises(NotImplementedError):
            await reranker.rerank("query", [])


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_creation(self):
        """Test reranker creation."""
        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
        )
        assert reranker.name == "cross_encoder"

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        """Test reranking empty results."""
        reranker = CrossEncoderReranker()
        # Don't initialize - just test empty case
        reranker._initialized = True
        reranker._model = MagicMock()
        
        results = await reranker.rerank("query", [])
        assert results == []


class TestCohereReranker:
    """Tests for CohereReranker."""

    def test_creation(self):
        """Test reranker creation."""
        reranker = CohereReranker(
            api_key="test-key",
            model="rerank-english-v3.0",
        )
        assert reranker.name == "cohere"

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        """Test reranking empty results."""
        reranker = CohereReranker(api_key="test")
        reranker._initialized = True
        reranker._client = MagicMock()
        
        results = await reranker.rerank("query", [])
        assert results == []


# -----------------------------------------------------------------------------
# RAGRetriever Tests
# -----------------------------------------------------------------------------


class TestRAGRetriever:
    """Tests for RAGRetriever."""

    @pytest.fixture
    def mock_knowledge_store(self):
        """Create mock knowledge store."""
        store = MagicMock()
        store.is_initialized = True
        store.search = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create mock embedding provider."""
        provider = MagicMock()
        provider.is_initialized = True
        provider.embed = AsyncMock(return_value=MagicMock(embedding=[0.1] * 384))
        return provider

    def test_creation(self, mock_knowledge_store):
        """Test retriever creation."""
        retriever = RAGRetriever(
            knowledge_store=mock_knowledge_store,
            top_k=10,
            min_score=0.5,
        )
        assert retriever._top_k == 10
        assert retriever._min_score == 0.5
        assert not retriever.is_initialized

    def test_creation_with_reranker_type(self, mock_knowledge_store):
        """Test creation with reranker type."""
        retriever = RAGRetriever(
            knowledge_store=mock_knowledge_store,
            reranker_type=RerankerType.CROSS_ENCODER,
        )
        assert retriever._reranker is not None
        assert isinstance(retriever._reranker, CrossEncoderReranker)

    def test_creation_with_custom_reranker(self, mock_knowledge_store):
        """Test creation with custom reranker."""
        custom_reranker = MagicMock(spec=Reranker)
        retriever = RAGRetriever(
            knowledge_store=mock_knowledge_store,
            reranker=custom_reranker,
        )
        assert retriever._reranker is custom_reranker

    @pytest.mark.asyncio
    async def test_initialize(self, mock_knowledge_store, mock_embedding_provider):
        """Test initialization."""
        retriever = RAGRetriever(
            knowledge_store=mock_knowledge_store,
            embedding_provider=mock_embedding_provider,
        )
        
        mock_knowledge_store.is_initialized = False
        mock_knowledge_store.initialize = AsyncMock()
        mock_embedding_provider.is_initialized = False
        mock_embedding_provider.initialize = AsyncMock()
        
        await retriever.initialize()
        
        assert retriever.is_initialized
        mock_knowledge_store.initialize.assert_called_once()
        mock_embedding_provider.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_empty(self, mock_knowledge_store):
        """Test retrieval with no results."""
        retriever = RAGRetriever(knowledge_store=mock_knowledge_store)
        await retriever.initialize()
        
        result = await retriever.retrieve("test query")
        
        assert result.query == "test query"
        assert result.chunks == []
        assert result.reranked is False


# -----------------------------------------------------------------------------
# Context Assembly Tests
# -----------------------------------------------------------------------------


class TestContextAssembly:
    """Tests for context assembly methods."""

    @pytest.fixture
    def retriever(self):
        """Create retriever for testing."""
        store = MagicMock()
        store.is_initialized = True
        return RAGRetriever(knowledge_store=store)

    @pytest.fixture
    def sample_result(self):
        """Create sample retrieval result."""
        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.content = "First chunk content"
        mock_chunk1.id = "c1"
        mock_chunk1.document_id = "d1"
        mock_chunk1.chunk_index = 0
        
        mock_chunk2 = MagicMock()
        mock_chunk2.content = "Second chunk content"
        mock_chunk2.id = "c2"
        mock_chunk2.document_id = "d1"
        mock_chunk2.chunk_index = 1
        
        mock_doc = MagicMock()
        mock_doc.title = "Test Document"
        mock_doc.source_uri = "test.pdf"
        
        mock_result1 = MagicMock()
        mock_result1.chunk = mock_chunk1
        mock_result1.document = mock_doc
        mock_result1.score = 0.9
        
        mock_result2 = MagicMock()
        mock_result2.chunk = mock_chunk2
        mock_result2.document = mock_doc
        mock_result2.score = 0.8
        
        return RetrievalResult(
            query="test",
            chunks=[mock_result1, mock_result2],
        )

    def test_assemble_plain(self, retriever, sample_result):
        """Test plain format assembly."""
        context = retriever.assemble_context(
            sample_result,
            format=ContextFormat.PLAIN,
            include_sources=True,
            include_scores=True,
        )
        
        assert len(context.chunks) == 2
        assert "First chunk content" in context.text
        assert "Second chunk content" in context.text
        assert "Test Document" in context.text
        assert "0.9" in context.text

    def test_assemble_markdown(self, retriever, sample_result):
        """Test markdown format assembly."""
        context = retriever.assemble_context(
            sample_result,
            format=ContextFormat.MARKDOWN,
        )
        
        assert "## Retrieved Context" in context.text
        assert "###" in context.text

    def test_assemble_xml(self, retriever, sample_result):
        """Test XML format assembly."""
        context = retriever.assemble_context(
            sample_result,
            format=ContextFormat.XML,
        )
        
        assert "<context>" in context.text
        assert "<chunk" in context.text
        assert "</context>" in context.text

    def test_assemble_json(self, retriever, sample_result):
        """Test JSON format assembly."""
        import json
        
        context = retriever.assemble_context(
            sample_result,
            format=ContextFormat.JSON,
        )
        
        # Should be valid JSON
        data = json.loads(context.text)
        assert "context" in data
        assert len(data["context"]) == 2

    def test_token_truncation(self, retriever, sample_result):
        """Test context truncation by tokens."""
        context = retriever.assemble_context(
            sample_result,
            max_tokens=10,  # Very small
        )
        
        assert context.truncated is True
        assert "[Context truncated...]" in context.text


# -----------------------------------------------------------------------------
# Exception Tests
# -----------------------------------------------------------------------------


class TestExceptions:
    """Tests for exception classes."""

    def test_retrieval_error(self):
        """Test RetrievalError."""
        error = RetrievalError("Retrieval failed")
        assert str(error) == "Retrieval failed"
        assert isinstance(error, Exception)

    def test_rerank_error(self):
        """Test RerankError."""
        error = RerankError("Rerank failed")
        assert str(error) == "Rerank failed"
        assert isinstance(error, RetrievalError)


# -----------------------------------------------------------------------------
# Enum Tests
# -----------------------------------------------------------------------------


class TestEnums:
    """Tests for enum classes."""

    def test_reranker_type(self):
        """Test RerankerType values."""
        assert RerankerType.NONE == "none"
        assert RerankerType.CROSS_ENCODER == "cross_encoder"
        assert RerankerType.COHERE == "cohere"

    def test_context_format(self):
        """Test ContextFormat values."""
        assert ContextFormat.PLAIN == "plain"
        assert ContextFormat.MARKDOWN == "markdown"
        assert ContextFormat.XML == "xml"
        assert ContextFormat.JSON == "json"
