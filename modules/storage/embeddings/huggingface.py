"""HuggingFace embedding provider.

Implements embedding generation using HuggingFace models via the
sentence-transformers library (all-MiniLM-L6-v2, bge-small-en-v1.5, etc.).

This provider integrates with the ATLAS provider system and uses
HuggingFace infrastructure for local model management.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from modules.logging.logger import setup_logger

from .base import (
    BatchEmbeddingResult,
    EmbeddingGenerationError,
    EmbeddingInputType,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResult,
    MODEL_DIMENSIONS,
    get_model_dimension,
    register_embedding_provider,
)

if TYPE_CHECKING:
    from core.config import ConfigManager

logger = setup_logger(__name__)

# Default model
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Model configurations with prompt templates for asymmetric models
LOCAL_MODELS: Dict[str, Dict[str, Any]] = {
    # MiniLM models - fast and lightweight
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_tokens": 256,
        "symmetric": True,
    },
    "all-MiniLM-L12-v2": {
        "dimension": 384,
        "max_tokens": 256,
        "symmetric": True,
    },
    # MPNet - good quality
    "all-mpnet-base-v2": {
        "dimension": 768,
        "max_tokens": 384,
        "symmetric": True,
    },
    # BGE models - strong retrieval performance
    "bge-small-en-v1.5": {
        "dimension": 384,
        "max_tokens": 512,
        "symmetric": False,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "bge-base-en-v1.5": {
        "dimension": 768,
        "max_tokens": 512,
        "symmetric": False,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "bge-large-en-v1.5": {
        "dimension": 1024,
        "max_tokens": 512,
        "symmetric": False,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    # E5 models
    "e5-small-v2": {
        "dimension": 384,
        "max_tokens": 512,
        "symmetric": False,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "e5-base-v2": {
        "dimension": 768,
        "max_tokens": 512,
        "symmetric": False,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "e5-large-v2": {
        "dimension": 1024,
        "max_tokens": 512,
        "symmetric": False,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    # GTE models
    "gte-small": {
        "dimension": 384,
        "max_tokens": 512,
        "symmetric": True,
    },
    "gte-base": {
        "dimension": 768,
        "max_tokens": 512,
        "symmetric": True,
    },
    "gte-large": {
        "dimension": 1024,
        "max_tokens": 512,
        "symmetric": True,
    },
}


def _get_device() -> str:
    """Detect the best available device for inference."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@register_embedding_provider("huggingface")
class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider using sentence-transformers.

    Runs embedding models locally using CPU or GPU. Supports various
    models from HuggingFace including MiniLM, BGE, E5, and GTE families.
    
    This provider integrates with the ATLAS provider system and can
    use HuggingFace API keys from the config manager for private models.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        max_workers: int = 1,
        config_manager: Optional["ConfigManager"] = None,
    ) -> None:
        """Initialize the HuggingFace embedding provider.

        Args:
            model: Model name or path (HuggingFace model ID or local path).
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto).
            normalize_embeddings: Whether to L2-normalize embeddings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar during encoding.
            cache_folder: Custom folder for model cache.
            trust_remote_code: Whether to trust remote code in models.
            max_workers: Max threads for async encoding.
            config_manager: Optional ConfigManager for HuggingFace API key.
        """
        self._model_name = model
        self._device = device or _get_device()
        self._normalize = normalize_embeddings
        self._batch_size = batch_size
        self._show_progress = show_progress
        self._cache_folder = cache_folder
        self._trust_remote_code = trust_remote_code
        self._max_workers = max_workers
        self._config_manager = config_manager

        # Get model config
        self._model_config = LOCAL_MODELS.get(model, {})
        self._dimension = self._model_config.get("dimension") or get_model_dimension(model) or 384
        self._max_tokens = self._model_config.get("max_tokens", 512)
        self._symmetric = self._model_config.get("symmetric", True)
        self._query_prefix = self._model_config.get("query_prefix", "")
        self._document_prefix = self._model_config.get("document_prefix", "")

        self._model: Optional[Any] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def supports_input_type(self) -> bool:
        return not self._symmetric

    @property
    def max_batch_size(self) -> int:
        return self._batch_size

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self._max_tokens

    async def initialize(self) -> None:
        """Initialize the sentence-transformers model."""
        if self._initialized:
            return

        try:
            # Import here to avoid hard dependency
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading HuggingFace embedding model '%s' on device '%s'",
                self._model_name,
                self._device,
            )

            # Get HuggingFace token from config manager or environment
            hf_token = None
            if self._config_manager is not None:
                get_key = getattr(self._config_manager, "get_huggingface_api_key", None)
                if callable(get_key):
                    hf_token = get_key()
            if not hf_token:
                hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

            # Load model in thread pool to avoid blocking
            def load_model() -> SentenceTransformer:
                kwargs: Dict[str, Any] = {
                    "device": self._device,
                }
                if self._cache_folder:
                    kwargs["cache_folder"] = self._cache_folder
                if self._trust_remote_code:
                    kwargs["trust_remote_code"] = self._trust_remote_code
                if hf_token:
                    kwargs["token"] = hf_token

                return SentenceTransformer(self._model_name, **kwargs)

            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(None, load_model)

            # Update dimension from actual model
            self._dimension = self._model.get_sentence_embedding_dimension()

            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
            self._initialized = True

            logger.info(
                "HuggingFace embedding model loaded: %s (dimension=%d, device=%s)",
                self._model_name,
                self._dimension,
                self._device,
            )

        except ImportError as exc:
            raise EmbeddingProviderError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Failed to load HuggingFace embedding model '{self._model_name}': {exc}"
            ) from exc

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._model = None
        self._initialized = False
        logger.debug("HuggingFace embedding provider shut down")

    def _prepare_text(
        self,
        text: str,
        input_type: Optional[EmbeddingInputType],
    ) -> str:
        """Apply prefix based on input type for asymmetric models."""
        if self._symmetric or input_type is None:
            return text

        if input_type == EmbeddingInputType.QUERY:
            return self._query_prefix + text
        elif input_type == EmbeddingInputType.DOCUMENT:
            return self._document_prefix + text

        return text

    async def embed_text(
        self,
        text: str,
        *,
        input_type: Optional[EmbeddingInputType] = None,
    ) -> EmbeddingResult:
        """Generate an embedding for a single text."""
        if not self._initialized:
            await self.initialize()

        prepared_text = self._prepare_text(text, input_type)

        def encode() -> List[float]:
            embedding = self._model.encode(
                prepared_text,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
            return embedding.tolist()

        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(self._executor, encode)

            return EmbeddingResult(
                embedding=embedding,
                text=text,
                model=self._model_name,
            )

        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to generate embedding: {exc}"
            ) from exc

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
                model=self._model_name,
            )

        prepared_texts = [self._prepare_text(t, input_type) for t in texts]

        def encode_batch() -> List[List[float]]:
            embeddings = self._model.encode(
                prepared_texts,
                normalize_embeddings=self._normalize,
                show_progress_bar=self._show_progress,
                batch_size=self._batch_size,
            )
            return [emb.tolist() for emb in embeddings]

        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(self._executor, encode_batch)

            results = [
                EmbeddingResult(
                    embedding=emb,
                    text=texts[i],
                    model=self._model_name,
                )
                for i, emb in enumerate(embeddings)
            ]

            return BatchEmbeddingResult(
                embeddings=results,
                model=self._model_name,
            )

        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to generate batch embeddings: {exc}"
            ) from exc


__all__ = ["HuggingFaceEmbeddingProvider", "LOCAL_MODELS", "DEFAULT_MODEL"]
