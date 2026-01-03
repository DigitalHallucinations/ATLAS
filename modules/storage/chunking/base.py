"""Base text splitter abstraction.

Defines the common interface for all text splitting/chunking strategies
used to break documents into smaller pieces for embedding and retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)
import hashlib
import re

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


class ChunkingError(Exception):
    """Base exception for chunking operations."""

    pass


class ChunkingConfigError(ChunkingError):
    """Raised when chunking configuration is invalid."""

    pass


@dataclass(slots=True)
class ChunkMetadata:
    """Metadata for a text chunk.

    Attributes:
        source_id: Identifier of the source document.
        chunk_index: Index of this chunk within the document.
        total_chunks: Total number of chunks from the document.
        start_char: Start character position in original text.
        end_char: End character position in original text.
        overlap_prev: Number of overlapping characters from previous chunk.
        overlap_next: Number of overlapping characters with next chunk.
        custom: Additional custom metadata.
    """

    source_id: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    overlap_prev: int = 0
    overlap_next: int = 0
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "source_id": self.source_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "overlap_prev": self.overlap_prev,
            "overlap_next": self.overlap_next,
        }
        if self.custom:
            result["custom"] = self.custom
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChunkMetadata":
        """Create from dictionary."""
        return cls(
            source_id=data.get("source_id"),
            chunk_index=int(data.get("chunk_index", 0)),
            total_chunks=int(data.get("total_chunks", 0)),
            start_char=int(data.get("start_char", 0)),
            end_char=int(data.get("end_char", 0)),
            overlap_prev=int(data.get("overlap_prev", 0)),
            overlap_next=int(data.get("overlap_next", 0)),
            custom=dict(data.get("custom", {})),
        )


@dataclass(slots=True)
class Chunk:
    """A chunk of text with metadata.

    Attributes:
        text: The chunk text content.
        metadata: Chunk position and source metadata.
        chunk_id: Unique identifier for this chunk (auto-generated if not provided).
    """

    text: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    chunk_id: Optional[str] = None

    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID based on content and position."""
        content = f"{self.metadata.source_id}:{self.metadata.chunk_index}:{self.text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def length(self) -> int:
        """Return the length of the chunk text."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Return the approximate word count."""
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Chunk":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            metadata=ChunkMetadata.from_dict(data.get("metadata", {})),
            chunk_id=data.get("chunk_id"),
        )


class TextSplitter(ABC):
    """Abstract base class for text splitting strategies.

    All splitter implementations must implement the split_text method
    and optionally override other methods for specific behavior.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function: Callable[[str], int] | None = None,
        strip_whitespace: bool = True,
        keep_separator: bool = True,
    ):
        """Initialize the text splitter.

        Args:
            chunk_size: Target size for each chunk (in characters or tokens).
            chunk_overlap: Number of overlapping characters between chunks.
            length_function: Function to measure text length (default: len).
            strip_whitespace: Whether to strip whitespace from chunks.
            keep_separator: Whether to keep separators in split chunks.
        """
        if chunk_size <= 0:
            raise ChunkingConfigError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ChunkingConfigError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ChunkingConfigError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        self.strip_whitespace = strip_whitespace
        self.keep_separator = keep_separator

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the splitter name."""
        ...

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        ...

    def create_chunks(
        self,
        text: str,
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split text and create Chunk objects with metadata.

        Args:
            text: The text to split.
            source_id: Identifier for the source document.
            custom_metadata: Additional metadata to include in each chunk.

        Returns:
            List of Chunk objects with metadata.
        """
        texts = self.split_text(text)
        total_chunks = len(texts)
        chunks = []

        current_pos = 0
        for i, chunk_text in enumerate(texts):
            # Find the actual position in the original text
            start_char = text.find(chunk_text, current_pos)
            if start_char == -1:
                # Fallback if exact match fails (e.g., due to stripping)
                start_char = current_pos
            end_char = start_char + len(chunk_text)

            # Calculate overlap
            overlap_prev = 0
            overlap_next = 0
            if i > 0 and self.chunk_overlap > 0:
                overlap_prev = min(self.chunk_overlap, len(chunk_text))
            if i < total_chunks - 1 and self.chunk_overlap > 0:
                overlap_next = min(self.chunk_overlap, len(chunk_text))

            metadata = ChunkMetadata(
                source_id=source_id,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=start_char,
                end_char=end_char,
                overlap_prev=overlap_prev,
                overlap_next=overlap_next,
                custom=custom_metadata or {},
            )

            chunks.append(Chunk(text=chunk_text, metadata=metadata))
            current_pos = end_char - self.chunk_overlap if self.chunk_overlap else end_char

        return chunks

    def split_documents(
        self,
        documents: Sequence[Dict[str, Any]],
        text_key: str = "text",
        id_key: str = "id",
    ) -> List[Chunk]:
        """Split multiple documents into chunks.

        Args:
            documents: Sequence of document dictionaries.
            text_key: Key for the text content in each document.
            id_key: Key for the document ID in each document.

        Returns:
            List of all chunks from all documents.
        """
        all_chunks = []
        for doc in documents:
            text = doc.get(text_key, "")
            source_id = doc.get(id_key)
            # Pass through all other metadata
            custom_metadata = {k: v for k, v in doc.items() if k not in (text_key, id_key)}
            chunks = self.create_chunks(text, source_id=source_id, custom_metadata=custom_metadata)
            all_chunks.extend(chunks)
        return all_chunks

    def _merge_splits(
        self,
        splits: List[str],
        separator: str,
    ) -> List[str]:
        """Merge small splits into larger chunks with overlap.

        Args:
            splits: List of text splits.
            separator: Separator to use when joining.

        Returns:
            List of merged chunks.
        """
        if not splits:
            return []

        merged: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            # If this split alone is too large, we need to further split it
            if split_length > self.chunk_size:
                # First, add current chunk if non-empty
                if current_chunk:
                    merged.append(self._join_with_separator(current_chunk, separator))
                    current_chunk = []
                    current_length = 0
                # Add the large split as-is (caller should handle)
                merged.append(split)
                continue

            # Check if adding this split would exceed chunk_size
            new_length = current_length + split_length
            if current_chunk:
                new_length += self.length_function(separator)

            if new_length > self.chunk_size:
                # Finish current chunk
                if current_chunk:
                    merged.append(self._join_with_separator(current_chunk, separator))

                # Start new chunk with overlap from previous
                overlap_splits = self._get_overlap_splits(current_chunk, separator)
                current_chunk = overlap_splits + [split]
                current_length = sum(self.length_function(s) for s in current_chunk)
                if len(current_chunk) > 1:
                    current_length += (len(current_chunk) - 1) * self.length_function(separator)
            else:
                current_chunk.append(split)
                current_length = new_length

        # Don't forget the last chunk
        if current_chunk:
            merged.append(self._join_with_separator(current_chunk, separator))

        return merged

    def _get_overlap_splits(
        self,
        splits: List[str],
        separator: str,
    ) -> List[str]:
        """Get splits to use for overlap with next chunk.

        Args:
            splits: Current chunk splits.
            separator: Separator string.

        Returns:
            Splits to include as overlap in next chunk.
        """
        if not splits or self.chunk_overlap == 0:
            return []

        overlap_splits: List[str] = []
        overlap_length = 0

        # Work backwards from the end
        for split in reversed(splits):
            split_length = self.length_function(split)
            if overlap_length + split_length > self.chunk_overlap:
                break
            overlap_splits.insert(0, split)
            overlap_length += split_length + self.length_function(separator)

        return overlap_splits

    def _join_with_separator(
        self,
        splits: List[str],
        separator: str,
    ) -> str:
        """Join splits with separator.

        Args:
            splits: List of text splits.
            separator: Separator string.

        Returns:
            Joined text.
        """
        text = separator.join(splits)
        if self.strip_whitespace:
            text = text.strip()
        return text


# --- Splitter Registry ---

_TEXT_SPLITTERS: Dict[str, Type[TextSplitter]] = {}

SplitterT = TypeVar("SplitterT", bound=TextSplitter)


def register_text_splitter(
    name: str,
) -> Callable[[Type[SplitterT]], Type[SplitterT]]:
    """Decorator to register a text splitter.

    Args:
        name: The splitter name to register under.

    Returns:
        Decorator function.

    Example:
        @register_text_splitter("recursive")
        class RecursiveTextSplitter(TextSplitter):
            ...
    """

    def decorator(cls: Type[SplitterT]) -> Type[SplitterT]:
        if name in _TEXT_SPLITTERS:
            logger.warning("Overwriting text splitter registration for '%s'", name)
        _TEXT_SPLITTERS[name] = cls
        logger.debug("Registered text splitter: %s", name)
        return cls

    return decorator


def get_text_splitter_class(name: str) -> Type[TextSplitter]:
    """Get a text splitter class by name.

    Args:
        name: The splitter name.

    Returns:
        The splitter class.

    Raises:
        ChunkingError: If splitter is not found.
    """
    if name not in _TEXT_SPLITTERS:
        available = ", ".join(sorted(_TEXT_SPLITTERS.keys())) or "(none)"
        raise ChunkingError(
            f"Unknown text splitter '{name}'. Available: {available}"
        )
    return _TEXT_SPLITTERS[name]


def list_text_splitters() -> List[str]:
    """List all registered text splitter names.

    Returns:
        Sorted list of splitter names.
    """
    return sorted(_TEXT_SPLITTERS.keys())


def create_text_splitter(
    name: str,
    **kwargs: Any,
) -> TextSplitter:
    """Create a text splitter instance by name.

    Args:
        name: The splitter name (e.g., 'recursive', 'sentence').
        **kwargs: Configuration options for the splitter.

    Returns:
        Configured TextSplitter instance.

    Raises:
        ChunkingError: If splitter creation fails.
    """
    cls = get_text_splitter_class(name)
    try:
        return cls(**kwargs)
    except Exception as exc:
        raise ChunkingError(
            f"Failed to create text splitter '{name}': {exc}"
        ) from exc


__all__ = [
    # Core classes
    "TextSplitter",
    "Chunk",
    "ChunkMetadata",
    # Exceptions
    "ChunkingError",
    "ChunkingConfigError",
    # Registry functions
    "register_text_splitter",
    "get_text_splitter_class",
    "list_text_splitters",
    "create_text_splitter",
]
