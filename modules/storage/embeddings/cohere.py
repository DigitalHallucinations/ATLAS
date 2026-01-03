"""Cohere embedding provider.

Implements embedding generation using Cohere's embed models
(embed-english-v3.0, embed-multilingual-v3.0, etc.) with support
for asymmetric embeddings via input_type parameter.

This provider integrates with the ATLAS provider system for
API key management.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

from modules.logging.logger import setup_logger

from .base import (
    BatchEmbeddingResult,
    EmbeddingGenerationError,
    EmbeddingInputType,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingResult,
    register_embedding_provider,
)

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager

logger = setup_logger(__name__)

# Default model
DEFAULT_MODEL = "embed-english-v3.0"

# Model configurations
COHERE_MODELS: Dict[str, Dict[str, Any]] = {
    "embed-english-v3.0": {
        "dimension": 1024,
        "max_tokens": 512,
    },
    "embed-multilingual-v3.0": {
        "dimension": 1024,
        "max_tokens": 512,
    },
    "embed-english-light-v3.0": {
        "dimension": 384,
        "max_tokens": 512,
    },
    "embed-multilingual-light-v3.0": {
        "dimension": 384,
        "max_tokens": 512,
    },
    # Legacy v2 models
    "embed-english-v2.0": {
        "dimension": 4096,
        "max_tokens": 512,
    },
    "embed-multilingual-v2.0": {
        "dimension": 768,
        "max_tokens": 256,
    },
}

# Map our input types to Cohere's input_type values
INPUT_TYPE_MAP: Dict[EmbeddingInputType, str] = {
    EmbeddingInputType.DOCUMENT: "search_document",
    EmbeddingInputType.QUERY: "search_query",
    EmbeddingInputType.CLUSTERING: "clustering",
    EmbeddingInputType.CLASSIFICATION: "classification",
}


@register_embedding_provider("cohere")
class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider using the Cohere API.

    Supports embed-english-v3.0, embed-multilingual-v3.0, and other
    Cohere embedding models with native support for asymmetric embeddings
    via the input_type parameter.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 96,
        timeout: float = 60.0,
        truncate: str = "END",
        config_manager: Optional["ConfigManager"] = None,
    ) -> None:
        """Initialize the Cohere embedding provider.

        Args:
            api_key: Cohere API key (defaults to config manager or env var).
            model: Model to use for embeddings.
            base_url: Custom API base URL.
            max_retries: Maximum retries on transient failures.
            retry_delay: Initial delay between retries (exponential backoff).
            batch_size: Maximum texts per batch request (Cohere limit is 96).
            timeout: Request timeout in seconds.
            truncate: How to handle texts exceeding max tokens ('NONE', 'START', 'END').
            config_manager: Optional ConfigManager for Cohere API key.
        """
        self._config_manager = config_manager
        
        # Get API key from config manager, explicit parameter, or environment
        resolved_key = api_key
        if not resolved_key and config_manager is not None:
            get_key = getattr(config_manager, "get_cohere_api_key", None)
            if callable(get_key):
                resolved_key = get_key()
        if not resolved_key:
            resolved_key = os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
        
        self._api_key = resolved_key
        self._model = model
        self._base_url = base_url
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._batch_size = min(batch_size, 96)  # Cohere limit
        self._timeout = timeout
        self._truncate = truncate

        # Get model config
        if model in COHERE_MODELS:
            self._model_config = COHERE_MODELS[model].copy()
        else:
            logger.warning(
                "Unknown Cohere model '%s', using default dimensions", model
            )
            self._model_config = {
                "dimension": 1024,
                "max_tokens": 512,
            }

        self._dimension = self._model_config["dimension"]
        self._client: Optional[Any] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "cohere"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def supports_input_type(self) -> bool:
        # V3 models support input types
        return "v3" in self._model.lower()

    @property
    def max_batch_size(self) -> int:
        return self._batch_size

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self._model_config.get("max_tokens")

    async def initialize(self) -> None:
        """Initialize the Cohere client."""
        if self._initialized:
            return

        if not self._api_key:
            raise EmbeddingProviderError(
                "Cohere API key not provided. Set CO_API_KEY or COHERE_API_KEY "
                "environment variable or pass api_key parameter."
            )

        try:
            # Import here to avoid hard dependency
            import cohere

            client_kwargs: Dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": self._timeout,
            }

            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            # Use async client
            self._client = cohere.AsyncClient(**client_kwargs)
            self._initialized = True

            logger.info(
                "Cohere embedding provider initialized with model %s (dimension=%d)",
                self._model,
                self._dimension,
            )

        except ImportError as exc:
            raise EmbeddingProviderError(
                "Cohere package not installed. Install with: pip install cohere"
            ) from exc
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Failed to initialize Cohere client: {exc}"
            ) from exc

    async def shutdown(self) -> None:
        """Clean up the Cohere client."""
        if self._client is not None:
            # Cohere client doesn't require explicit cleanup
            self._client = None
        self._initialized = False
        logger.debug("Cohere embedding provider shut down")

    def _get_cohere_input_type(
        self,
        input_type: Optional[EmbeddingInputType],
    ) -> Optional[str]:
        """Convert our input type to Cohere's format."""
        if not self.supports_input_type:
            return None
        if input_type is None:
            # Default to document for v3 models
            return "search_document"
        return INPUT_TYPE_MAP.get(input_type, "search_document")

    async def embed_text(
        self,
        text: str,
        *,
        input_type: Optional[EmbeddingInputType] = None,
    ) -> EmbeddingResult:
        """Generate an embedding for a single text."""
        if not self._initialized:
            await self.initialize()

        result = await self._embed_with_retry([text], input_type)
        return result.embeddings[0]

    async def embed_batch(
        self,
        texts: Sequence[str],
        *,
        input_type: Optional[EmbeddingInputType] = None,
    ) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts."""
        if not self._initialized:
            await self.initialize()

        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model=self._model,
            )

        # Process in batches if needed
        if len(texts) <= self._batch_size:
            return await self._embed_with_retry(list(texts), input_type)

        # Split into batches and process
        all_embeddings: List[EmbeddingResult] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = await self._embed_with_retry(list(batch), input_type)
            all_embeddings.extend(result.embeddings)

        return BatchEmbeddingResult(
            embeddings=all_embeddings,
            model=self._model,
        )

    async def _embed_with_retry(
        self,
        texts: List[str],
        input_type: Optional[EmbeddingInputType],
    ) -> BatchEmbeddingResult:
        """Embed texts with exponential backoff retry."""
        last_error: Optional[Exception] = None
        delay = self._retry_delay

        for attempt in range(self._max_retries + 1):
            try:
                return await self._do_embed(texts, input_type)
            except EmbeddingRateLimitError as exc:
                last_error = exc
                wait_time = exc.retry_after if exc.retry_after else delay
                logger.warning(
                    "Rate limited on attempt %d/%d, waiting %.1fs",
                    attempt + 1,
                    self._max_retries + 1,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
                delay *= 2
            except EmbeddingGenerationError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "Embedding attempt %d/%d failed: %s, retrying in %.1fs",
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2

        raise EmbeddingGenerationError(
            f"Failed to generate embeddings after {self._max_retries + 1} attempts: {last_error}"
        ) from last_error

    async def _do_embed(
        self,
        texts: List[str],
        input_type: Optional[EmbeddingInputType],
    ) -> BatchEmbeddingResult:
        """Perform the actual embedding API call."""
        try:
            kwargs: Dict[str, Any] = {
                "texts": texts,
                "model": self._model,
                "truncate": self._truncate,
            }

            # Add input_type for v3 models
            cohere_input_type = self._get_cohere_input_type(input_type)
            if cohere_input_type:
                kwargs["input_type"] = cohere_input_type

            response = await self._client.embed(**kwargs)

            embeddings: List[EmbeddingResult] = []
            for i, embedding in enumerate(response.embeddings):
                embeddings.append(
                    EmbeddingResult(
                        embedding=list(embedding),
                        text=texts[i] if i < len(texts) else "",
                        model=self._model,
                    )
                )

            metadata: Dict[str, Any] = {}
            if hasattr(response, "meta") and response.meta:
                if hasattr(response.meta, "billed_units"):
                    metadata["billed_units"] = {
                        "input_tokens": getattr(
                            response.meta.billed_units, "input_tokens", None
                        ),
                    }

            return BatchEmbeddingResult(
                embeddings=embeddings,
                model=self._model,
                metadata=metadata,
            )

        except Exception as exc:
            error_message = str(exc).lower()

            # Check for rate limit errors
            if "rate" in error_message or "429" in error_message:
                raise EmbeddingRateLimitError(
                    f"Cohere rate limit exceeded: {exc}",
                ) from exc

            # Check for invalid request errors
            if "invalid" in error_message or "400" in error_message:
                raise EmbeddingGenerationError(
                    f"Invalid embedding request: {exc}"
                ) from exc

            # Re-raise for retry
            raise


__all__ = ["CohereEmbeddingProvider"]
