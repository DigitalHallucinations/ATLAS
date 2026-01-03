"""Unit tests for embedding providers.

Tests cover the embedding provider base class, registry functions,
and concrete provider implementations.
"""

from __future__ import annotations

import pytest
from typing import List, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

from modules.storage.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingInputType,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingGenerationError,
    EmbeddingRateLimitError,
    register_embedding_provider,
    get_embedding_provider_class,
    create_embedding_provider,
    list_embedding_providers,
    MODEL_DIMENSIONS,
    get_model_dimensions,
    OpenAIEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    CohereEmbeddingProvider,
)


# --- Test EmbeddingResult and BatchEmbeddingResult ---


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_create_embedding_result(self):
        """Test creating an embedding result."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = EmbeddingResult(
            embedding=embedding,
            text="Hello, world!",
            model="test-model",
            token_count=5,
        )

        assert result.embedding == embedding
        assert result.text == "Hello, world!"
        assert result.model == "test-model"
        assert result.token_count == 5
        assert result.dimension == 5

    def test_embedding_result_dimension(self):
        """Test dimension property."""
        result = EmbeddingResult(
            embedding=[0.0] * 384,
            text="test",
            model="model",
        )
        assert result.dimension == 384

    def test_embedding_result_metadata(self):
        """Test metadata field defaults to empty dict."""
        result = EmbeddingResult(
            embedding=[0.1],
            text="test",
            model="model",
        )
        assert result.metadata == {}


class TestBatchEmbeddingResult:
    """Tests for BatchEmbeddingResult dataclass."""

    def test_create_batch_result(self):
        """Test creating a batch embedding result."""
        embeddings = [
            EmbeddingResult([0.1, 0.2], "text1", "model"),
            EmbeddingResult([0.3, 0.4], "text2", "model"),
        ]
        batch = BatchEmbeddingResult(
            embeddings=embeddings,
            model="model",
            total_tokens=10,
        )

        assert len(batch) == 2
        assert batch.model == "model"
        assert batch.total_tokens == 10

    def test_batch_iteration(self):
        """Test iterating over batch result."""
        embeddings = [
            EmbeddingResult([0.1], "text1", "model"),
            EmbeddingResult([0.2], "text2", "model"),
        ]
        batch = BatchEmbeddingResult(embeddings=embeddings, model="model")

        texts = [r.text for r in batch]
        assert texts == ["text1", "text2"]

    def test_batch_indexing(self):
        """Test indexing batch result."""
        embeddings = [
            EmbeddingResult([0.1], "text1", "model"),
            EmbeddingResult([0.2], "text2", "model"),
        ]
        batch = BatchEmbeddingResult(embeddings=embeddings, model="model")

        assert batch[0].text == "text1"
        assert batch[1].text == "text2"

    def test_batch_vectors_property(self):
        """Test vectors property returns just embeddings."""
        embeddings = [
            EmbeddingResult([0.1, 0.2], "text1", "model"),
            EmbeddingResult([0.3, 0.4], "text2", "model"),
        ]
        batch = BatchEmbeddingResult(embeddings=embeddings, model="model")

        assert batch.vectors == [[0.1, 0.2], [0.3, 0.4]]


# --- Test Provider Registry ---


class TestProviderRegistry:
    """Tests for embedding provider registry."""

    def test_list_providers_includes_builtin(self):
        """Test that built-in providers are registered."""
        providers = list_embedding_providers()
        assert "openai" in providers
        assert "huggingface" in providers
        assert "cohere" in providers

    def test_get_openai_provider_class(self):
        """Test getting OpenAI provider class."""
        cls = get_embedding_provider_class("openai")
        assert cls is OpenAIEmbeddingProvider

    def test_get_huggingface_provider_class(self):
        """Test getting HuggingFace provider class."""
        cls = get_embedding_provider_class("huggingface")
        assert cls is HuggingFaceEmbeddingProvider

    def test_get_cohere_provider_class(self):
        """Test getting Cohere provider class."""
        cls = get_embedding_provider_class("cohere")
        assert cls is CohereEmbeddingProvider

    def test_get_unknown_provider_raises(self):
        """Test that unknown provider raises error."""
        with pytest.raises(EmbeddingProviderError) as exc_info:
            get_embedding_provider_class("unknown_provider")

        assert "Unknown embedding provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)


# --- Test Model Dimensions ---


class TestModelDimensions:
    """Tests for model dimension lookup."""

    def test_known_model_dimensions(self):
        """Test looking up known model dimensions."""
        assert get_model_dimensions("text-embedding-3-small") == 1536
        assert get_model_dimensions("text-embedding-3-large") == 3072
        assert get_model_dimensions("text-embedding-ada-002") == 1536
        assert get_model_dimensions("all-MiniLM-L6-v2") == 384
        assert get_model_dimensions("all-mpnet-base-v2") == 768
        assert get_model_dimensions("embed-english-v3.0") == 1024

    def test_unknown_model_returns_none(self):
        """Test that unknown model returns None."""
        assert get_model_dimensions("unknown-model-xyz") is None

    def test_model_dimensions_dict_populated(self):
        """Test that MODEL_DIMENSIONS dict is populated."""
        assert len(MODEL_DIMENSIONS) > 0
        assert "text-embedding-3-small" in MODEL_DIMENSIONS


# --- Test EmbeddingInputType ---


class TestEmbeddingInputType:
    """Tests for EmbeddingInputType enum."""

    def test_input_type_values(self):
        """Test input type enum values."""
        assert EmbeddingInputType.DOCUMENT.value == "search_document"
        assert EmbeddingInputType.QUERY.value == "search_query"
        assert EmbeddingInputType.CLUSTERING.value == "clustering"
        assert EmbeddingInputType.CLASSIFICATION.value == "classification"


# --- Mock Provider for Testing ---


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock provider for testing base class functionality."""

    def __init__(self, dimension: int = 384, model: str = "mock-model"):
        self._dimension = dimension
        self._model = model
        self._initialized = False

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def embed_text(
        self,
        text: str,
        *,
        input_type: EmbeddingInputType | None = None,
    ) -> EmbeddingResult:
        return EmbeddingResult(
            embedding=[0.1] * self._dimension,
            text=text,
            model=self._model,
            token_count=len(text.split()),
        )

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: EmbeddingInputType | None = None,
    ) -> BatchEmbeddingResult:
        embeddings = [
            EmbeddingResult(
                embedding=[0.1] * self._dimension,
                text=text,
                model=self._model,
            )
            for text in texts
        ]
        return BatchEmbeddingResult(
            embeddings=embeddings,
            model=self._model,
        )


class TestEmbeddingProviderBase:
    """Tests for EmbeddingProvider base class methods."""

    @pytest.fixture
    def provider(self):
        return MockEmbeddingProvider()

    @pytest.mark.asyncio
    async def test_embed_query(self, provider: MockEmbeddingProvider):
        """Test embed_query convenience method."""
        result = await provider.embed_query("What is Python?")
        assert result.text == "What is Python?"
        assert result.dimension == 384

    @pytest.mark.asyncio
    async def test_embed_document(self, provider: MockEmbeddingProvider):
        """Test embed_document convenience method."""
        result = await provider.embed_document("Python is a programming language.")
        assert result.text == "Python is a programming language."
        assert result.dimension == 384

    @pytest.mark.asyncio
    async def test_embed_documents(self, provider: MockEmbeddingProvider):
        """Test embed_documents convenience method."""
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        result = await provider.embed_documents(texts)
        assert len(result) == 3
        assert [r.text for r in result] == texts

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider: MockEmbeddingProvider):
        """Test health check when provider is healthy."""
        is_healthy = await provider.health_check()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check when provider fails."""

        class FailingProvider(MockEmbeddingProvider):
            async def embed_text(self, text, **kwargs):
                raise EmbeddingGenerationError("Test failure")

        provider = FailingProvider()
        is_healthy = await provider.health_check()
        assert is_healthy is False

    def test_default_properties(self, provider: MockEmbeddingProvider):
        """Test default property values."""
        assert provider.supports_batch is True
        assert provider.supports_input_type is False
        assert provider.max_batch_size == 100
        assert provider.max_input_tokens is None


# --- Test OpenAI Provider ---


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAI embedding provider."""

    def test_provider_properties(self):
        """Test provider property values."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
        )
        assert provider.name == "openai"
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimension == 1536
        assert provider.supports_batch is True
        assert provider.supports_input_type is False

    def test_custom_dimensions(self):
        """Test custom dimensions for embedding-3 models."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=512,
        )
        assert provider.dimension == 512

    def test_ada_002_dimensions_ignored(self):
        """Test that custom dimensions are ignored for ada-002."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-ada-002",
            dimensions=512,  # Should be ignored
        )
        assert provider.dimension == 1536

    @pytest.mark.asyncio
    async def test_embed_text_mocked(self):
        """Test embed_text with mocked OpenAI client."""
        provider = OpenAIEmbeddingProvider(api_key="test-key")

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536, index=0)]
        mock_response.usage.total_tokens = 5

        with patch.object(
            provider, "_client", create=True
        ) as mock_client:
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            await provider.initialize()

            # Replace the internal client
            provider._client = mock_client

            result = await provider.embed_text("Hello world")
            assert result.dimension == 1536


# --- Test HuggingFace Provider ---


class TestHuggingFaceEmbeddingProvider:
    """Tests for HuggingFace sentence-transformers provider."""

    def test_provider_properties(self):
        """Test provider property values."""
        provider = HuggingFaceEmbeddingProvider(model="all-MiniLM-L6-v2")
        assert provider.name == "huggingface"
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.supports_batch is True

    def test_asymmetric_model_detection(self):
        """Test that asymmetric models are detected."""
        # BGE models are asymmetric
        provider = HuggingFaceEmbeddingProvider(model="BAAI/bge-small-en")
        assert provider.supports_input_type is True

        # MiniLM is symmetric
        provider = HuggingFaceEmbeddingProvider(model="all-MiniLM-L6-v2")
        assert provider.supports_input_type is False

    def test_e5_model_detection(self):
        """Test E5 model asymmetric detection."""
        provider = HuggingFaceEmbeddingProvider(model="intfloat/e5-small-v2")
        assert provider.supports_input_type is True


# --- Test Cohere Provider ---


class TestCohereEmbeddingProvider:
    """Tests for Cohere embedding provider."""

    def test_provider_properties(self):
        """Test provider property values."""
        provider = CohereEmbeddingProvider(
            api_key="test-key",
            model="embed-english-v3.0",
        )
        assert provider.name == "cohere"
        assert provider.model_name == "embed-english-v3.0"
        assert provider.dimension == 1024
        assert provider.supports_batch is True
        assert provider.supports_input_type is True  # Cohere supports input types
        assert provider.max_batch_size == 96

    def test_multilingual_model(self):
        """Test multilingual model dimensions."""
        provider = CohereEmbeddingProvider(
            api_key="test-key",
            model="embed-multilingual-v3.0",
        )
        assert provider.dimension == 1024


# --- Test Exception Classes ---


class TestExceptions:
    """Tests for embedding exception classes."""

    def test_embedding_error_base(self):
        """Test base embedding error."""
        error = EmbeddingError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert isinstance(error, Exception)

    def test_provider_error(self):
        """Test provider error."""
        error = EmbeddingProviderError("Invalid configuration")
        assert isinstance(error, EmbeddingError)

    def test_generation_error(self):
        """Test generation error."""
        error = EmbeddingGenerationError("Failed to embed text")
        assert isinstance(error, EmbeddingError)

    def test_rate_limit_error(self):
        """Test rate limit error with retry_after."""
        error = EmbeddingRateLimitError(
            "Rate limit exceeded",
            retry_after=30.0,
        )
        assert isinstance(error, EmbeddingError)
        assert error.retry_after == 30.0

    def test_rate_limit_error_no_retry(self):
        """Test rate limit error without retry_after."""
        error = EmbeddingRateLimitError("Rate limit exceeded")
        assert error.retry_after is None


# --- Integration-style Tests (require markers to skip without dependencies) ---


@pytest.mark.skipif(
    True,  # Always skip in unit tests - enable in integration tests
    reason="Requires sentence-transformers installed",
)
class TestHuggingFaceProviderIntegration:
    """Integration tests for HuggingFace provider (skipped by default)."""

    @pytest.mark.asyncio
    async def test_embed_text_real(self):
        """Test actual embedding with HuggingFace model."""
        provider = HuggingFaceEmbeddingProvider(model="all-MiniLM-L6-v2")
        await provider.initialize()

        try:
            result = await provider.embed_text("Hello, world!")
            assert result.dimension == 384
            assert len(result.embedding) == 384
        finally:
            await provider.shutdown()
