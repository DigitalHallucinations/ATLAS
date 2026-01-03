"""Text chunking/splitting for RAG document processing.

This module provides various text splitting strategies for breaking
documents into smaller chunks suitable for embedding and retrieval.

Basic Usage:
    >>> from modules.storage.chunking import create_text_splitter
    >>> splitter = create_text_splitter("recursive", chunk_size=500)
    >>> chunks = splitter.create_chunks("Long document text...")

Splitter Types:
    - "recursive": Hierarchical splitting using multiple separators
    - "character": Simple separator-based splitting
    - "sentence": Split at sentence boundaries
    - "paragraph": Split at paragraph boundaries
    - "semantic": Embedding-based semantic chunking

Configuration:
    Splitters can be configured via ATLAS/config/rag.py settings or
    by passing kwargs directly to create_text_splitter().
"""

from .base import (
    # Core classes
    TextSplitter,
    Chunk,
    ChunkMetadata,
    # Exceptions
    ChunkingError,
    ChunkingConfigError,
    # Registry functions
    register_text_splitter,
    get_text_splitter_class,
    list_text_splitters,
    create_text_splitter,
)

from .recursive import (
    RecursiveTextSplitter,
    CharacterTextSplitter,
    DEFAULT_SEPARATORS,
)

from .sentence import (
    SentenceTextSplitter,
    ParagraphTextSplitter,
    SENTENCE_ENDINGS,
    ABBREVIATIONS,
)

from .semantic import (
    SemanticTextSplitter,
    cosine_similarity,
)

from .hierarchical import (
    HierarchicalChunk,
    HierarchicalSplitResult,
    HierarchicalChunker,
    expand_to_parent,
    get_context_window,
)


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
    # Recursive splitters
    "RecursiveTextSplitter",
    "CharacterTextSplitter",
    "DEFAULT_SEPARATORS",
    # Sentence splitters
    "SentenceTextSplitter",
    "ParagraphTextSplitter",
    "SENTENCE_ENDINGS",
    "ABBREVIATIONS",
    # Semantic splitter
    "SemanticTextSplitter",
    "cosine_similarity",
    # Hierarchical splitter
    "HierarchicalChunk",
    "HierarchicalSplitResult",
    "HierarchicalChunker",
    "expand_to_parent",
    "get_context_window",
]
