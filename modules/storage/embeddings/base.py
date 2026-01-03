"""Base embedding provider abstraction.

Defines the common interface for all embedding backends (OpenAI, Cohere,
local sentence-transformers, etc.) allowing seamless switching between providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
)

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding operations."""

    pass


class EmbeddingProviderError(EmbeddingError):
    """Raised when provider initialization or configuration fails."""

    pass


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""

    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class EmbeddingInputType(str, Enum):
    """Input type hints for asymmetric embedding models.

    Some models (e.g., Cohere embed-v3) produce different embeddings
    for documents vs queries to optimize retrieval.
    """

    DOCUMENT = "search_document"
    QUERY = "search_query"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"


@dataclass(slots=True)
class EmbeddingResult:
    """Result from an embedding operation.

    Attributes:
        embedding: The embedding vector as a list of floats.
        text: The original text that was embedded.
        model: The model used to generate the embedding.
        token_count: Number of tokens in the input text (if available).
        metadata: Additional metadata from the provider.
    """

    embedding: List[float]
    text: str
    model: str
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return len(self.embedding)


@dataclass(slots=True)
class BatchEmbeddingResult:
    """Result from a batch embedding operation.

    Attributes:
        embeddings: List of embedding results.
        model: The model used to generate embeddings.
        total_tokens: Total tokens across all inputs (if available).
        metadata: Additional metadata from the provider.
    """

    embeddings: List[EmbeddingResult]
    model: str
    total_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __iter__(self):
        return iter(self.embeddings)

    def __getitem__(self, index: int) -> EmbeddingResult:
        return self.embeddings[index]

    @property
    def vectors(self) -> List[List[float]]:
        """Return just the embedding vectors."""
        return [result.embedding for result in self.embeddings]


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding implementations (OpenAI, Cohere, local models)
    must implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'openai', 'cohere', 'local')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier being used."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for the current model."""
        ...

    @property
    def supports_batch(self) -> bool:
        """Whether this provider supports batch embedding natively."""
        return True

    @property
    def supports_input_type(self) -> bool:
        """Whether this provider supports asymmetric embeddings."""
        return False

    @property
    def max_batch_size(self) -> int:
        """Maximum number of texts that can be embedded in one batch."""
        return 100

    @property
    def max_input_tokens(self) -> Optional[int]:
        """Maximum tokens per input text (None if unlimited)."""
        return None

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding provider (load model, verify API key, etc.)."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        ...

    @abstractmethod
    async def embed_text(
        self,
        text: str,
        *,
        input_type: Optional[EmbeddingInputType] = None,
    ) -> EmbeddingResult:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.
            input_type: Optional hint for asymmetric embedding models.

        Returns:
            EmbeddingResult containing the embedding vector.

        Raises:
            EmbeddingGenerationError: If embedding generation fails.
            EmbeddingRateLimitError: If rate limits are exceeded.
        """
        ...

    @abstractmethod
    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: Optional[EmbeddingInputType] = None,
    ) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: The texts to embed.
            input_type: Optional hint for asymmetric embedding models.

        Returns:
            BatchEmbeddingResult containing all embedding vectors.

        Raises:
            EmbeddingGenerationError: If embedding generation fails.
            EmbeddingRateLimitError: If rate limits are exceeded.
        """
        ...

    async def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a query for retrieval (uses QUERY input type if supported).

        Args:
            text: The query text to embed.

        Returns:
            EmbeddingResult containing the embedding vector.
        """
        input_type = EmbeddingInputType.QUERY if self.supports_input_type else None
        return await self.embed_text(text, input_type=input_type)

    async def embed_document(self, text: str) -> EmbeddingResult:
        """Embed a document for indexing (uses DOCUMENT input type if supported).

        Args:
            text: The document text to embed.

        Returns:
            EmbeddingResult containing the embedding vector.
        """
        input_type = EmbeddingInputType.DOCUMENT if self.supports_input_type else None
        return await self.embed_text(text, input_type=input_type)

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> BatchEmbeddingResult:
        """Embed multiple documents for indexing.

        Args:
            texts: The document texts to embed.

        Returns:
            BatchEmbeddingResult containing all embedding vectors.
        """
        input_type = EmbeddingInputType.DOCUMENT if self.supports_input_type else None
        return await self.embed_batch(texts, input_type=input_type)

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the embedding provider is healthy and accessible.

        Args:
            timeout: Maximum time to wait for health check.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            result = await self.embed_text("health check")
            return len(result.embedding) == self.dimension
        except Exception as exc:
            logger.warning("Embedding provider health check failed: %s", exc)
            return False


# --- Provider Registry ---

_EMBEDDING_PROVIDERS: Dict[str, Type[EmbeddingProvider]] = {}

ProviderT = TypeVar("ProviderT", bound=EmbeddingProvider)


def register_embedding_provider(
    name: str,
) -> Callable[[Type[ProviderT]], Type[ProviderT]]:
    """Decorator to register an embedding provider.

    Args:
        name: The provider name to register under.

    Returns:
        Decorator function.

    Example:
        @register_embedding_provider("openai")
        class OpenAIEmbeddingProvider(EmbeddingProvider):
            ...
    """

    def decorator(cls: Type[ProviderT]) -> Type[ProviderT]:
        if name in _EMBEDDING_PROVIDERS:
            logger.warning(
                "Overwriting embedding provider registration for '%s'", name
            )
        _EMBEDDING_PROVIDERS[name] = cls
        logger.debug("Registered embedding provider: %s", name)
        return cls

    return decorator


def get_embedding_provider_class(name: str) -> Type[EmbeddingProvider]:
    """Get an embedding provider class by name.

    Args:
        name: The provider name.

    Returns:
        The provider class.

    Raises:
        EmbeddingProviderError: If provider is not found.
    """
    if name not in _EMBEDDING_PROVIDERS:
        available = ", ".join(sorted(_EMBEDDING_PROVIDERS.keys())) or "(none)"
        raise EmbeddingProviderError(
            f"Unknown embedding provider '{name}'. Available: {available}"
        )
    return _EMBEDDING_PROVIDERS[name]


def list_embedding_providers() -> List[str]:
    """List all registered embedding provider names.

    Returns:
        Sorted list of provider names.
    """
    return sorted(_EMBEDDING_PROVIDERS.keys())


async def create_embedding_provider(
    name: str,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Create and initialize an embedding provider.

    Args:
        name: The provider name.
        **kwargs: Provider-specific configuration.

    Returns:
        Initialized embedding provider.

    Raises:
        EmbeddingProviderError: If creation or initialization fails.
    """
    provider_class = get_embedding_provider_class(name)
    try:
        provider = provider_class(**kwargs)
        await provider.initialize()
        return provider
    except Exception as exc:
        raise EmbeddingProviderError(
            f"Failed to create embedding provider '{name}': {exc}"
        ) from exc


# --- Model Dimension Registry ---

# Common model dimensions for reference
MODEL_DIMENSIONS: Dict[str, int] = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Cohere
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    # Local / Sentence Transformers
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "e5-small-v2": 384,
    "e5-base-v2": 768,
    "e5-large-v2": 1024,
    "gte-small": 384,
    "gte-base": 768,
    "gte-large": 1024,
    # Voyage
    "voyage-2": 1024,
    "voyage-large-2": 1536,
    "voyage-code-2": 1536,
}


def get_model_dimension(model_name: str) -> Optional[int]:
    """Look up the expected dimension for a known model.

    Args:
        model_name: The model identifier.

    Returns:
        Dimension if known, None otherwise.
    """
    # Try exact match first
    if model_name in MODEL_DIMENSIONS:
        return MODEL_DIMENSIONS[model_name]

    # Try partial match for versioned models
    for known_model, dimension in MODEL_DIMENSIONS.items():
        if known_model in model_name or model_name in known_model:
            return dimension

    return None


__all__ = [
    # Exceptions
    "EmbeddingError",
    "EmbeddingProviderError",
    "EmbeddingGenerationError",
    "EmbeddingRateLimitError",
    # Enums
    "EmbeddingInputType",
    # Data classes
    "EmbeddingResult",
    "BatchEmbeddingResult",
    # Base class
    "EmbeddingProvider",
    # Registry functions
    "register_embedding_provider",
    "get_embedding_provider_class",
    "list_embedding_providers",
    "create_embedding_provider",
    # Model dimensions
    "MODEL_DIMENSIONS",
    "get_model_dimension",
]
