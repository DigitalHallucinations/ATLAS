"""OpenAI embedding provider.

Implements embedding generation using OpenAI's embedding models
(text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002).

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
    MODEL_DIMENSIONS,
    register_embedding_provider,
)

if TYPE_CHECKING:
    from core.config import ConfigManager

logger = setup_logger(__name__)

# Default models
DEFAULT_MODEL = "text-embedding-3-small"

# Model configurations
OPENAI_MODELS: Dict[str, Dict[str, Any]] = {
    "text-embedding-3-small": {
        "dimension": 1536,
        "max_tokens": 8191,
        "supports_dimensions": True,  # Can request lower dimensions
    },
    "text-embedding-3-large": {
        "dimension": 3072,
        "max_tokens": 8191,
        "supports_dimensions": True,
    },
    "text-embedding-ada-002": {
        "dimension": 1536,
        "max_tokens": 8191,
        "supports_dimensions": False,
    },
}

# Rate limit defaults (requests per minute)
DEFAULT_RPM = 3000
DEFAULT_TPM = 1000000


@register_embedding_provider("openai")
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using the OpenAI API.

    Supports text-embedding-3-small, text-embedding-3-large, and
    text-embedding-ada-002 models with async batch processing and
    automatic retry on rate limits.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 100,
        timeout: float = 60.0,
        config_manager: Optional["ConfigManager"] = None,
    ) -> None:
        """Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (defaults to config manager or env var).
            model: Model to use for embeddings.
            dimensions: Output dimension (only for text-embedding-3-* models).
            base_url: Custom API base URL (for Azure or proxies).
            organization: OpenAI organization ID.
            max_retries: Maximum retries on transient failures.
            retry_delay: Initial delay between retries (exponential backoff).
            batch_size: Maximum texts per batch request.
            timeout: Request timeout in seconds.
            config_manager: Optional ConfigManager for OpenAI API key.
        """
        self._config_manager = config_manager
        
        # Get API key from config manager, explicit parameter, or environment
        resolved_key = api_key
        if not resolved_key and config_manager is not None:
            get_key = getattr(config_manager, "get_openai_api_key", None)
            if callable(get_key):
                resolved_key = get_key()
        if not resolved_key:
            resolved_key = os.getenv("OPENAI_API_KEY")
        
        self._api_key = resolved_key
        self._model = model
        self._base_url = base_url
        self._organization = organization
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._batch_size = min(batch_size, 2048)  # OpenAI limit
        self._timeout = timeout

        # Validate model
        if model not in OPENAI_MODELS:
            logger.warning(
                "Unknown OpenAI model '%s', using default dimensions", model
            )
            self._model_config = {
                "dimension": dimensions or 1536,
                "max_tokens": 8191,
                "supports_dimensions": False,
            }
        else:
            self._model_config = OPENAI_MODELS[model].copy()

        # Handle custom dimensions for text-embedding-3-* models
        self._requested_dimensions = dimensions
        if dimensions is not None:
            if self._model_config.get("supports_dimensions"):
                self._dimension = dimensions
            else:
                logger.warning(
                    "Model %s does not support custom dimensions, using default %d",
                    model,
                    self._model_config["dimension"],
                )
                self._dimension = self._model_config["dimension"]
        else:
            self._dimension = self._model_config["dimension"]

        self._client: Optional[Any] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_batch_size(self) -> int:
        return self._batch_size

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self._model_config.get("max_tokens")

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return

        if not self._api_key:
            raise EmbeddingProviderError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        try:
            # Import here to avoid hard dependency
            from openai import AsyncOpenAI

            client_kwargs: Dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": self._timeout,
                "max_retries": 0,  # We handle retries ourselves
            }

            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            if self._organization:
                client_kwargs["organization"] = self._organization

            self._client = AsyncOpenAI(**client_kwargs)
            self._initialized = True
            logger.info(
                "OpenAI embedding provider initialized with model %s (dimension=%d)",
                self._model,
                self._dimension,
            )

        except ImportError as exc:
            raise EmbeddingProviderError(
                "OpenAI package not installed. Install with: pip install openai"
            ) from exc
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Failed to initialize OpenAI client: {exc}"
            ) from exc

    async def shutdown(self) -> None:
        """Clean up the OpenAI client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._initialized = False
        logger.debug("OpenAI embedding provider shut down")

    async def embed_text(
        self,
        text: str,
        *,
        input_type: Optional[EmbeddingInputType] = None,
    ) -> EmbeddingResult:
        """Generate an embedding for a single text."""
        if not self._initialized:
            await self.initialize()

        result = await self._embed_with_retry([text])
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
                total_tokens=0,
            )

        # Process in batches if needed
        if len(texts) <= self._batch_size:
            return await self._embed_with_retry(list(texts))

        # Split into batches and process
        all_embeddings: List[EmbeddingResult] = []
        total_tokens = 0

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = await self._embed_with_retry(list(batch))
            all_embeddings.extend(result.embeddings)
            if result.total_tokens:
                total_tokens += result.total_tokens

        return BatchEmbeddingResult(
            embeddings=all_embeddings,
            model=self._model,
            total_tokens=total_tokens if total_tokens else None,
        )

    async def _embed_with_retry(
        self,
        texts: List[str],
    ) -> BatchEmbeddingResult:
        """Embed texts with exponential backoff retry."""
        last_error: Optional[Exception] = None
        delay = self._retry_delay

        for attempt in range(self._max_retries + 1):
            try:
                return await self._do_embed(texts)
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
                delay *= 2  # Exponential backoff
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

    async def _do_embed(self, texts: List[str]) -> BatchEmbeddingResult:
        """Perform the actual embedding API call."""
        try:
            # Build request kwargs
            kwargs: Dict[str, Any] = {
                "input": texts,
                "model": self._model,
            }

            # Add dimensions parameter for text-embedding-3-* models
            if (
                self._requested_dimensions is not None
                and self._model_config.get("supports_dimensions")
            ):
                kwargs["dimensions"] = self._requested_dimensions

            response = await self._client.embeddings.create(**kwargs)

            embeddings: List[EmbeddingResult] = []
            for i, item in enumerate(response.data):
                embeddings.append(
                    EmbeddingResult(
                        embedding=list(item.embedding),
                        text=texts[i] if i < len(texts) else "",
                        model=response.model,
                        token_count=None,  # Not available per-item
                    )
                )

            return BatchEmbeddingResult(
                embeddings=embeddings,
                model=response.model,
                total_tokens=response.usage.total_tokens if response.usage else None,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                },
            )

        except Exception as exc:
            # Check for rate limit errors
            error_message = str(exc).lower()
            if "rate" in error_message and "limit" in error_message:
                # Try to extract retry-after from error
                retry_after = None
                if hasattr(exc, "response") and hasattr(exc.response, "headers"):
                    retry_str = exc.response.headers.get("retry-after")
                    if retry_str:
                        try:
                            retry_after = float(retry_str)
                        except ValueError:
                            pass
                raise EmbeddingRateLimitError(
                    f"OpenAI rate limit exceeded: {exc}",
                    retry_after=retry_after,
                ) from exc

            # Check for invalid request errors
            if "invalid" in error_message or "400" in error_message:
                raise EmbeddingGenerationError(
                    f"Invalid embedding request: {exc}"
                ) from exc

            # Re-raise for retry
            raise


__all__ = ["OpenAIEmbeddingProvider"]
