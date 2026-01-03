"""Context compression for RAG responses.

Provides compression strategies to reduce context size while preserving
information critical to answering the query. This enables:
1. More efficient LLM token usage
2. Reduced latency for context-heavy queries
3. Improved focus on relevant information

Compression Strategies:
- LLMLinguaCompressor: Uses language model perplexity for token-level compression
- ExtractiveSummarizer: Sentence-level extraction based on query relevance
- HybridCompressor: Combines multiple strategies

Usage:
    >>> from modules.storage.retrieval.compression import LLMLinguaCompressor
    >>> compressor = LLMLinguaCompressor(target_ratio=0.5)
    >>> await compressor.initialize()
    >>> compressed = await compressor.compress(context, query)
    >>> print(f"Compressed from {len(context)} to {len(compressed)} chars")
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class CompressionResult:
    """Result of context compression.

    Attributes:
        compressed_text: The compressed context.
        original_length: Original character count.
        compressed_length: Compressed character count.
        compression_ratio: Compression ratio achieved.
        tokens_removed: Estimated tokens removed.
        strategy: Compression strategy used.
        metadata: Additional strategy-specific metadata.
    """

    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float = 0.0
    tokens_removed: int = 0
    strategy: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.original_length > 0:
            self.compression_ratio = 1 - (self.compressed_length / self.original_length)


# -----------------------------------------------------------------------------
# Abstract Base
# -----------------------------------------------------------------------------


class ContextCompressor(ABC):
    """Abstract base class for context compressors."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the compressor."""
        ...

    @abstractmethod
    async def compress(
        self,
        context: str,
        query: str,
        *,
        target_ratio: Optional[float] = None,
    ) -> CompressionResult:
        """Compress context while preserving query-relevant information.

        Args:
            context: The context text to compress.
            query: The query for relevance scoring.
            target_ratio: Target compression ratio (0.5 = keep 50%).

        Returns:
            CompressionResult with compressed text and metadata.
        """
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if compressor is initialized."""
        ...


# -----------------------------------------------------------------------------
# LLMLingua-style Compressor
# -----------------------------------------------------------------------------


class LLMLinguaCompressor(ContextCompressor):
    """Token-level compressor using language model perplexity scoring.

    Implements a simplified version of LLMLingua compression:
    1. Scores each token by its perplexity (how "surprising" it is)
    2. Removes low-information tokens that don't affect meaning
    3. Preserves query-relevant tokens regardless of perplexity

    This approach typically achieves 2-4x compression while maintaining
    semantic fidelity for downstream LLM tasks.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        target_ratio: float = 0.5,
        preserve_ratio: float = 0.3,
        device: Optional[str] = None,
    ) -> None:
        """Initialize LLMLingua compressor.

        Args:
            model_name: HuggingFace model for perplexity scoring.
            target_ratio: Default target compression ratio (keep this fraction).
            preserve_ratio: Fraction of query-relevant tokens to always keep.
            device: Device for inference (None = auto-detect).
        """
        self._model_name = model_name
        self._target_ratio = target_ratio
        self._preserve_ratio = preserve_ratio
        self._device = device
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if compressor is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the language model for perplexity scoring."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Auto-detect device
            if self._device is None:
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

            logger.info(f"Loading LLMLingua model {self._model_name} on {self._device}")

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
            ).to(self._device)
            model.eval()
            
            self._tokenizer = tokenizer
            self._model = model

            self._initialized = True
            logger.info("LLMLingua compressor initialized")

        except ImportError:
            logger.warning("transformers not available, using fallback compression")
            self._initialized = True  # Use fallback mode

    async def compress(
        self,
        context: str,
        query: str,
        *,
        target_ratio: Optional[float] = None,
    ) -> CompressionResult:
        """Compress context using token-level perplexity scoring.

        Args:
            context: The context text to compress.
            query: The query for relevance scoring.
            target_ratio: Target compression ratio (overrides default).

        Returns:
            CompressionResult with compressed text.
        """
        if not self._initialized:
            await self.initialize()

        original_length = len(context)
        ratio = target_ratio if target_ratio is not None else self._target_ratio

        # Use fallback if model not loaded
        if self._model is None or self._tokenizer is None:
            return await self._fallback_compress(context, query, ratio)

        try:
            compressed = await self._perplexity_compress(context, query, ratio)
            return CompressionResult(
                compressed_text=compressed,
                original_length=original_length,
                compressed_length=len(compressed),
                strategy="llmlingua",
                metadata={
                    "model": self._model_name,
                    "target_ratio": ratio,
                },
            )
        except Exception as exc:
            logger.warning(f"LLMLingua compression failed: {exc}, using fallback")
            return await self._fallback_compress(context, query, ratio)

    async def _perplexity_compress(
        self,
        context: str,
        query: str,
        target_ratio: float,
    ) -> str:
        """Compress using perplexity-based token scoring."""
        import torch
        import asyncio
        
        # Capture references for closure (type checker satisfaction)
        tokenizer = self._tokenizer
        model = self._model
        device = self._device
        
        if tokenizer is None or model is None:
            raise RuntimeError("Model not initialized")

        def _compute() -> str:
            # Tokenize context
            tokens = tokenizer.encode(context, return_tensors="pt").to(device)
            token_strings = [tokenizer.decode([t]) for t in tokens[0]]

            # Compute perplexity for each token
            with torch.no_grad():
                outputs = model(tokens, labels=tokens)
                # Get per-token loss (approximates perplexity)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = tokens[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            # Score tokens: higher loss = more important (surprising)
            scores = token_losses.cpu().numpy()

            # Identify query-relevant tokens (always preserve)
            query_tokens = set(query.lower().split())
            preserve_indices = set()
            for i, tok in enumerate(token_strings[1:]):  # Skip first token
                if tok.strip().lower() in query_tokens:
                    preserve_indices.add(i)

            # Calculate how many tokens to keep
            n_tokens = len(scores)
            n_keep = max(int(n_tokens * target_ratio), 1)
            n_preserve = len(preserve_indices)

            # Get indices of highest-importance tokens
            sorted_indices = scores.argsort()[::-1]  # Descending by importance

            # Build set of tokens to keep
            keep_indices = set(preserve_indices)
            for idx in sorted_indices:
                if len(keep_indices) >= n_keep:
                    break
                keep_indices.add(int(idx))

            # Reconstruct text with kept tokens
            kept_tokens = []
            for i, tok in enumerate(token_strings[1:]):  # Skip first token
                if i in keep_indices:
                    kept_tokens.append(tok)

            return "".join(kept_tokens)

        return await asyncio.to_thread(_compute)

    async def _fallback_compress(
        self,
        context: str,
        query: str,
        target_ratio: float,
    ) -> CompressionResult:
        """Fallback compression using simple heuristics."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)

        # Score sentences by query term overlap
        query_terms = set(query.lower().split())
        scored = []
        for sent in sentences:
            sent_terms = set(sent.lower().split())
            overlap = len(query_terms & sent_terms)
            scored.append((overlap, sent))

        # Sort by relevance
        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top sentences up to target ratio
        target_length = int(len(context) * target_ratio)
        kept = []
        current_length = 0
        for _, sent in scored:
            if current_length + len(sent) > target_length:
                break
            kept.append(sent)
            current_length += len(sent)

        # Reconstruct in original order
        kept_set = set(kept)
        result = [s for s in sentences if s in kept_set]
        compressed = " ".join(result)

        return CompressionResult(
            compressed_text=compressed,
            original_length=len(context),
            compressed_length=len(compressed),
            strategy="fallback_sentence",
            metadata={"target_ratio": target_ratio},
        )


# -----------------------------------------------------------------------------
# Extractive Summarizer
# -----------------------------------------------------------------------------


class ExtractiveSummarizer(ContextCompressor):
    """Sentence-level extraction based on query relevance and importance.

    Uses a combination of:
    1. Query term overlap scoring
    2. TF-IDF-like term importance
    3. Position-based scoring (earlier = more important)
    4. Sentence length normalization

    This is a lightweight alternative to LLMLingua that doesn't require
    a language model but still achieves reasonable compression.
    """

    def __init__(
        self,
        target_ratio: float = 0.5,
        min_sentences: int = 1,
        position_weight: float = 0.1,
    ) -> None:
        """Initialize extractive summarizer.

        Args:
            target_ratio: Default target compression ratio.
            min_sentences: Minimum sentences to keep.
            position_weight: Weight for position-based scoring.
        """
        self._target_ratio = target_ratio
        self._min_sentences = min_sentences
        self._position_weight = position_weight
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if compressor is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the summarizer (no-op for this implementation)."""
        self._initialized = True

    async def compress(
        self,
        context: str,
        query: str,
        *,
        target_ratio: Optional[float] = None,
    ) -> CompressionResult:
        """Compress context using extractive summarization.

        Args:
            context: The context text to compress.
            query: The query for relevance scoring.
            target_ratio: Target compression ratio.

        Returns:
            CompressionResult with extracted sentences.
        """
        if not self._initialized:
            await self.initialize()

        original_length = len(context)
        ratio = target_ratio if target_ratio is not None else self._target_ratio

        # Split into sentences
        sentences = self._split_sentences(context)
        if not sentences:
            return CompressionResult(
                compressed_text=context,
                original_length=original_length,
                compressed_length=original_length,
                strategy="extractive",
            )

        # Score each sentence
        scored = self._score_sentences(sentences, query)

        # Select top sentences
        target_length = int(original_length * ratio)
        selected = self._select_sentences(scored, target_length)

        # Reconstruct in original order
        result = self._reconstruct(sentences, selected)

        return CompressionResult(
            compressed_text=result,
            original_length=original_length,
            compressed_length=len(result),
            strategy="extractive",
            metadata={
                "sentences_original": len(sentences),
                "sentences_kept": len(selected),
                "target_ratio": ratio,
            },
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentences(
        self,
        sentences: List[str],
        query: str,
    ) -> List[Tuple[int, float, str]]:
        """Score sentences by relevance.

        Returns list of (original_index, score, sentence).
        """
        query_terms = set(query.lower().split())

        # Build document term frequencies
        doc_freq: Dict[str, int] = {}
        for sent in sentences:
            terms = set(sent.lower().split())
            for term in terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n_sentences = len(sentences)
        scored = []

        for i, sent in enumerate(sentences):
            sent_terms = sent.lower().split()
            term_set = set(sent_terms)

            # Query overlap score
            query_score = len(query_terms & term_set) / max(len(query_terms), 1)

            # IDF-weighted term importance
            idf_score = 0.0
            for term in term_set:
                df = doc_freq.get(term, 1)
                idf = 1.0 / df  # Simplified IDF
                idf_score += idf
            idf_score /= max(len(term_set), 1)

            # Position score (earlier sentences weighted higher)
            position_score = 1.0 - (i / n_sentences) * self._position_weight

            # Length normalization (prefer medium-length sentences)
            length_score = min(len(sent_terms) / 20, 1.0)

            # Combined score
            total_score = (
                query_score * 0.5 +
                idf_score * 0.2 +
                position_score * 0.2 +
                length_score * 0.1
            )

            scored.append((i, total_score, sent))

        return scored

    def _select_sentences(
        self,
        scored: List[Tuple[int, float, str]],
        target_length: int,
    ) -> List[int]:
        """Select sentences to keep based on scores and target length."""
        # Sort by score descending
        by_score = sorted(scored, key=lambda x: x[1], reverse=True)

        selected_indices = []
        current_length = 0

        for idx, score, sent in by_score:
            if current_length + len(sent) > target_length:
                if len(selected_indices) >= self._min_sentences:
                    break
            selected_indices.append(idx)
            current_length += len(sent) + 1  # +1 for space

        return selected_indices

    def _reconstruct(
        self,
        sentences: List[str],
        selected_indices: List[int],
    ) -> str:
        """Reconstruct text from selected sentences in original order."""
        selected_set = set(selected_indices)
        kept = [s for i, s in enumerate(sentences) if i in selected_set]
        return " ".join(kept)


# -----------------------------------------------------------------------------
# Hybrid Compressor
# -----------------------------------------------------------------------------


class HybridCompressor(ContextCompressor):
    """Combines multiple compression strategies.

    Applies compressors in sequence, with each stage further
    reducing the context size. Useful for aggressive compression
    while maintaining quality.
    """

    def __init__(
        self,
        compressors: Optional[List[ContextCompressor]] = None,
        target_ratio: float = 0.3,
    ) -> None:
        """Initialize hybrid compressor.

        Args:
            compressors: List of compressors to apply in order.
            target_ratio: Target final compression ratio.
        """
        self._compressors = compressors or []
        self._target_ratio = target_ratio
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if all compressors are initialized."""
        return self._initialized and all(c.is_initialized for c in self._compressors)

    async def initialize(self) -> None:
        """Initialize all compressors."""
        for compressor in self._compressors:
            await compressor.initialize()
        self._initialized = True

    def add_compressor(self, compressor: ContextCompressor) -> None:
        """Add a compressor to the pipeline."""
        self._compressors.append(compressor)

    async def compress(
        self,
        context: str,
        query: str,
        *,
        target_ratio: Optional[float] = None,
    ) -> CompressionResult:
        """Apply all compressors in sequence.

        Args:
            context: The context to compress.
            query: The query for relevance.
            target_ratio: Target final compression ratio.

        Returns:
            CompressionResult from the final stage.
        """
        if not self._initialized:
            await self.initialize()

        if not self._compressors:
            return CompressionResult(
                compressed_text=context,
                original_length=len(context),
                compressed_length=len(context),
                strategy="hybrid_empty",
            )

        ratio = target_ratio if target_ratio is not None else self._target_ratio

        # Calculate per-stage ratio for even distribution
        n_stages = len(self._compressors)
        stage_ratio = ratio ** (1 / n_stages)  # Geometric mean

        original_length = len(context)
        current_text = context
        stages: List[Dict[str, Any]] = []

        for i, compressor in enumerate(self._compressors):
            result = await compressor.compress(
                current_text,
                query,
                target_ratio=stage_ratio,
            )
            stages.append({
                "stage": i,
                "strategy": result.strategy,
                "ratio": result.compression_ratio,
            })
            current_text = result.compressed_text

        return CompressionResult(
            compressed_text=current_text,
            original_length=original_length,
            compressed_length=len(current_text),
            strategy="hybrid",
            metadata={
                "stages": stages,
                "n_stages": n_stages,
                "target_ratio": ratio,
            },
        )


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


def create_compressor(
    strategy: str = "extractive",
    **kwargs: Any,
) -> ContextCompressor:
    """Create a context compressor by strategy name.

    Args:
        strategy: Compression strategy ('llmlingua', 'extractive', 'hybrid').
        **kwargs: Strategy-specific configuration.

    Returns:
        Configured ContextCompressor instance.
    """
    if strategy == "llmlingua":
        return LLMLinguaCompressor(**kwargs)
    elif strategy == "extractive":
        return ExtractiveSummarizer(**kwargs)
    elif strategy == "hybrid":
        return HybridCompressor(**kwargs)
    else:
        logger.warning(f"Unknown compression strategy: {strategy}, using extractive")
        return ExtractiveSummarizer(**kwargs)
