"""Semantic text splitter implementation.

Splits text based on semantic similarity between segments,
grouping semantically related content together.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional, Sequence, TYPE_CHECKING

from .base import (
    TextSplitter,
    Chunk,
    ChunkMetadata,
    register_text_splitter,
    ChunkingError,
)
from .sentence import SentenceTextSplitter

if TYPE_CHECKING:
    from modules.storage.embeddings import EmbeddingProvider

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity (0 to 1).
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


@register_text_splitter("semantic")
class SemanticTextSplitter(TextSplitter):
    """Split text based on semantic similarity.

    This splitter first splits text into sentences, then groups
    sentences together based on their semantic similarity using
    embeddings. It creates natural topic-based chunks.

    This approach produces higher quality chunks for RAG as it
    keeps semantically related content together.

    Example:
        >>> from modules.storage.embeddings import create_embedding_provider
        >>> embedding_provider = await create_embedding_provider("local")
        >>> splitter = SemanticTextSplitter(
        ...     embedding_provider=embedding_provider,
        ...     similarity_threshold=0.7
        ... )
        >>> chunks = await splitter.split_text_async("Document text...")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.75,
        embedding_provider: Optional["EmbeddingProvider"] = None,
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 10,
        length_function: Callable[[str], int] | None = None,
        strip_whitespace: bool = True,
        buffer_size: int = 3,
    ):
        """Initialize the semantic text splitter.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            similarity_threshold: Minimum similarity to group sentences (0-1).
            embedding_provider: Provider for generating embeddings.
            min_sentences_per_chunk: Minimum sentences per chunk.
            max_sentences_per_chunk: Maximum sentences per chunk.
            length_function: Function to measure text length.
            strip_whitespace: Whether to strip whitespace from chunks.
            buffer_size: Number of sentences to consider for similarity.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            strip_whitespace=strip_whitespace,
            keep_separator=True,
        )
        self.similarity_threshold = similarity_threshold
        self.embedding_provider = embedding_provider
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.buffer_size = buffer_size

        # Internal sentence splitter
        self._sentence_splitter = SentenceTextSplitter(
            chunk_size=chunk_size * 2,  # Larger to avoid premature splitting
            chunk_overlap=0,
            min_sentence_length=10,
        )

    @property
    def name(self) -> str:
        return "semantic"

    def split_text(self, text: str) -> List[str]:
        """Synchronous fallback - uses sentence splitting.

        For semantic splitting, use split_text_async instead.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        # Without async embedding, fall back to sentence splitting
        logger.warning(
            "Semantic splitter called synchronously; "
            "use split_text_async for semantic chunking"
        )
        return self._sentence_splitter.split_text(text)

    async def split_text_async(self, text: str) -> List[str]:
        """Split text based on semantic similarity.

        Args:
            text: The text to split.

        Returns:
            List of semantically coherent chunks.
        """
        if not self.embedding_provider:
            raise ChunkingError(
                "SemanticTextSplitter requires an embedding_provider"
            )

        # First, split into sentences
        sentences = self._get_sentences(text)

        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        # Get embeddings for all sentences
        try:
            embeddings = await self._get_sentence_embeddings(sentences)
        except Exception as exc:
            logger.warning("Failed to get embeddings, falling back: %s", exc)
            return self._sentence_splitter.split_text(text)

        # Find semantic break points
        break_points = self._find_semantic_breaks(sentences, embeddings)

        # Create chunks from break points
        chunks = self._create_chunks_from_breaks(sentences, break_points)

        return chunks

    def _get_sentences(self, text: str) -> List[str]:
        """Extract sentences from text.

        Args:
            text: The text to split.

        Returns:
            List of sentences.
        """
        # Use internal sentence splitter to get individual sentences
        raw_chunks = self._sentence_splitter.split_text(text)

        # Further split if chunks contain multiple sentences
        sentences = []
        for chunk in raw_chunks:
            # Split on sentence boundaries
            import re
            parts = re.split(r'(?<=[.!?])\s+', chunk)
            sentences.extend(p.strip() for p in parts if p.strip())

        return sentences

    async def _get_sentence_embeddings(
        self,
        sentences: List[str],
    ) -> List[List[float]]:
        """Get embeddings for sentences.

        Args:
            sentences: List of sentences.

        Returns:
            List of embedding vectors.
        """
        assert self.embedding_provider is not None

        result = await self.embedding_provider.embed_batch(sentences)
        return result.vectors

    def _find_semantic_breaks(
        self,
        sentences: List[str],
        embeddings: List[List[float]],
    ) -> List[int]:
        """Find indices where semantic breaks should occur.

        Uses a sliding window approach to detect topic shifts.

        Args:
            sentences: List of sentences.
            embeddings: Corresponding embeddings.

        Returns:
            List of break point indices.
        """
        if len(sentences) <= 1:
            return []

        break_points: List[int] = []
        current_chunk_size = 0
        sentences_in_chunk = 0

        for i in range(1, len(sentences)):
            current_chunk_size += self.length_function(sentences[i - 1])
            sentences_in_chunk += 1

            # Calculate similarity between adjacent sentences
            similarity = cosine_similarity(embeddings[i - 1], embeddings[i])

            # Also consider similarity with a buffer of recent sentences
            avg_buffer_similarity = self._calculate_buffer_similarity(
                embeddings, i
            )

            # Determine if we should break here
            should_break = False

            # Break if similarity drops below threshold
            if similarity < self.similarity_threshold:
                should_break = True

            # Break if chunk is getting too large
            estimated_size = current_chunk_size + self.length_function(sentences[i])
            if estimated_size > self.chunk_size:
                should_break = True

            # Break if we've hit max sentences
            if sentences_in_chunk >= self.max_sentences_per_chunk:
                should_break = True

            # Don't break if we haven't hit minimum sentences
            if sentences_in_chunk < self.min_sentences_per_chunk:
                should_break = False

            if should_break:
                break_points.append(i)
                current_chunk_size = 0
                sentences_in_chunk = 0

        return break_points

    def _calculate_buffer_similarity(
        self,
        embeddings: List[List[float]],
        current_idx: int,
    ) -> float:
        """Calculate average similarity with recent sentences.

        Args:
            embeddings: All embeddings.
            current_idx: Current position.

        Returns:
            Average similarity with buffer.
        """
        if current_idx <= 0:
            return 1.0

        start_idx = max(0, current_idx - self.buffer_size)
        similarities = []

        for i in range(start_idx, current_idx):
            sim = cosine_similarity(embeddings[i], embeddings[current_idx])
            similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 1.0

    def _create_chunks_from_breaks(
        self,
        sentences: List[str],
        break_points: List[int],
    ) -> List[str]:
        """Create chunk strings from break points.

        Args:
            sentences: List of sentences.
            break_points: Indices where chunks should break.

        Returns:
            List of chunk strings.
        """
        chunks: List[str] = []
        start = 0

        for break_idx in break_points:
            chunk_sentences = sentences[start:break_idx]
            if chunk_sentences:
                chunk_text = " ".join(chunk_sentences)
                if self.strip_whitespace:
                    chunk_text = chunk_text.strip()
                chunks.append(chunk_text)
            start = break_idx

        # Don't forget the last chunk
        if start < len(sentences):
            chunk_sentences = sentences[start:]
            chunk_text = " ".join(chunk_sentences)
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()
            chunks.append(chunk_text)

        return chunks

    async def create_chunks_async(
        self,
        text: str,
        source_id: Optional[str] = None,
        custom_metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """Create Chunk objects with semantic splitting.

        Args:
            text: The text to split.
            source_id: Identifier for the source document.
            custom_metadata: Additional metadata for chunks.

        Returns:
            List of Chunk objects.
        """
        chunk_texts = await self.split_text_async(text)

        chunks = []
        total_chunks = len(chunk_texts)
        current_pos = 0

        for i, chunk_text in enumerate(chunk_texts):
            start_char = text.find(chunk_text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_text)

            metadata = ChunkMetadata(
                source_id=source_id,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=start_char,
                end_char=end_char,
                custom=custom_metadata or {},
            )

            chunks.append(Chunk(text=chunk_text, metadata=metadata))
            current_pos = end_char

        return chunks


__all__ = [
    "SemanticTextSplitter",
    "cosine_similarity",
]
