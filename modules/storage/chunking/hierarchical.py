"""Hierarchical parent-child text chunking.

Implements a hierarchical chunking strategy where:
1. Parent chunks are larger (e.g., 2048 tokens) for full context
2. Child chunks are smaller (e.g., 512 tokens) for precise retrieval
3. Child chunks maintain references to their parent for context expansion

This enables "small-to-big" retrieval:
- Search on child chunks for precision
- Expand to parent chunks for complete context

Usage:
    >>> from modules.storage.chunking.hierarchical import HierarchicalChunker
    >>> chunker = HierarchicalChunker(
    ...     parent_chunk_size=2048,
    ...     child_chunk_size=512,
    ... )
    >>> result = chunker.split_hierarchical(text)
    >>> for parent in result.parents:
    ...     print(f"Parent {parent.metadata.chunk_index}: {len(parent.children)} children")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from modules.logging.logger import setup_logger
from modules.storage.chunking.base import (
    Chunk,
    ChunkMetadata,
    ChunkingConfigError,
    TextSplitter,
)
from modules.storage.chunking.recursive import RecursiveTextSplitter

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class HierarchicalChunk:
    """A chunk with hierarchical relationships.

    Attributes:
        chunk: The underlying chunk.
        chunk_id: Unique identifier.
        parent_id: ID of parent chunk (None for parents).
        is_parent: Whether this is a parent chunk.
        children: Child chunk IDs (for parents only).
        section_path: Hierarchical path (e.g., "section/subsection").
    """

    chunk: Chunk
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    is_parent: bool = False
    children: List[str] = field(default_factory=list)
    section_path: str = ""

    @property
    def text(self) -> str:
        """Get the chunk text."""
        return self.chunk.text

    @property
    def metadata(self) -> ChunkMetadata:
        """Get the chunk metadata."""
        return self.chunk.metadata


@dataclass
class HierarchicalSplitResult:
    """Result of hierarchical splitting.

    Attributes:
        parents: List of parent chunks.
        children: List of child chunks.
        all_chunks: All chunks (parents + children).
        parent_child_map: Mapping from parent ID to child IDs.
    """

    parents: List[HierarchicalChunk] = field(default_factory=list)
    children: List[HierarchicalChunk] = field(default_factory=list)
    all_chunks: List[HierarchicalChunk] = field(default_factory=list)
    parent_child_map: Dict[str, List[str]] = field(default_factory=dict)

    def get_children_for_parent(self, parent_id: str) -> List[HierarchicalChunk]:
        """Get all children for a parent chunk."""
        child_ids = self.parent_child_map.get(parent_id, [])
        return [c for c in self.children if c.chunk_id in child_ids]


# -----------------------------------------------------------------------------
# Hierarchical Chunker
# -----------------------------------------------------------------------------


class HierarchicalChunker(TextSplitter):
    """Hierarchical parent-child text splitter.

    Creates a two-level hierarchy of chunks:
    - Parent chunks: Larger, for context expansion
    - Child chunks: Smaller, for precise retrieval
    """

    def __init__(
        self,
        parent_chunk_size: int = 2048,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 512,
        child_chunk_overlap: int = 50,
        length_function: Optional[Callable[[str], int]] = None,
        strip_whitespace: bool = True,
        parent_separators: Optional[List[str]] = None,
        child_separators: Optional[List[str]] = None,
    ) -> None:
        """Initialize hierarchical chunker.

        Args:
            parent_chunk_size: Target size for parent chunks.
            parent_chunk_overlap: Overlap between parent chunks.
            child_chunk_size: Target size for child chunks.
            child_chunk_overlap: Overlap between child chunks.
            length_function: Function to measure text length.
            strip_whitespace: Whether to strip whitespace.
            parent_separators: Separators for parent splitting.
            child_separators: Separators for child splitting.
        """
        # Validate configuration
        if child_chunk_size >= parent_chunk_size:
            raise ChunkingConfigError(
                f"child_chunk_size ({child_chunk_size}) must be less than "
                f"parent_chunk_size ({parent_chunk_size})"
            )

        super().__init__(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            length_function=length_function,
            strip_whitespace=strip_whitespace,
        )

        self._parent_chunk_size = parent_chunk_size
        self._parent_chunk_overlap = parent_chunk_overlap
        self._child_chunk_size = child_chunk_size
        self._child_chunk_overlap = child_chunk_overlap

        # Default separators for document structure
        self._parent_separators = parent_separators or [
            "\n\n\n",  # Multiple blank lines
            "\n## ",  # Markdown H2
            "\n# ",  # Markdown H1
            "\n\n",  # Paragraphs
        ]
        self._child_separators = child_separators or [
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            " ",  # Words
        ]

        # Create internal splitters
        self._parent_splitter = RecursiveTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            length_function=length_function or len,
            separators=self._parent_separators,
            strip_whitespace=strip_whitespace,
        )

        self._child_splitter = RecursiveTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            length_function=length_function or len,
            separators=self._child_separators,
            strip_whitespace=strip_whitespace,
        )

    @property
    def name(self) -> str:
        """Return the splitter name."""
        return "hierarchical"

    @property
    def parent_chunk_size(self) -> int:
        """Get parent chunk size."""
        return self._parent_chunk_size

    @property
    def child_chunk_size(self) -> int:
        """Get child chunk size."""
        return self._child_chunk_size

    def split_text(self, text: str) -> List[str]:
        """Split text into flat list of child chunks.

        This is the TextSplitter interface method.
        Use split_hierarchical() for full hierarchy.

        Args:
            text: Text to split.

        Returns:
            List of child chunk texts.
        """
        result = self.split_hierarchical(text)
        return [c.text for c in result.children]

    def split_hierarchical(
        self,
        text: str,
        source_id: Optional[str] = None,
        base_section_path: str = "",
    ) -> HierarchicalSplitResult:
        """Split text into hierarchical parent-child chunks.

        Args:
            text: Text to split.
            source_id: Source document identifier.
            base_section_path: Base path for section hierarchy.

        Returns:
            HierarchicalSplitResult with parents and children.
        """
        if not text or not text.strip():
            return HierarchicalSplitResult()

        # First, split into parent chunks
        parent_texts = self._parent_splitter.split_text(text)

        parents: List[HierarchicalChunk] = []
        children: List[HierarchicalChunk] = []
        parent_child_map: Dict[str, List[str]] = {}

        # Track character positions
        current_pos = 0

        for parent_idx, parent_text in enumerate(parent_texts):
            # Find position in original text
            start_pos = text.find(parent_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(parent_text)
            current_pos = start_pos + 1  # Move past start for next search

            # Create parent chunk
            parent_id = str(uuid.uuid4())
            section_path = f"{base_section_path}/chunk_{parent_idx}" if base_section_path else f"chunk_{parent_idx}"

            parent_chunk = HierarchicalChunk(
                chunk=Chunk(
                    text=parent_text,
                    metadata=ChunkMetadata(
                        source_id=source_id,
                        chunk_index=parent_idx,
                        total_chunks=len(parent_texts),
                        start_char=start_pos,
                        end_char=end_pos,
                    ),
                ),
                chunk_id=parent_id,
                parent_id=None,
                is_parent=True,
                children=[],
                section_path=section_path,
            )

            # Split parent into child chunks
            child_texts = self._child_splitter.split_text(parent_text)
            child_ids: List[str] = []
            child_current_pos = 0

            for child_idx, child_text in enumerate(child_texts):
                # Find position within parent
                child_start = parent_text.find(child_text, child_current_pos)
                if child_start == -1:
                    child_start = child_current_pos
                child_end = child_start + len(child_text)
                child_current_pos = child_start + 1

                # Absolute position in original text
                abs_start = start_pos + child_start
                abs_end = start_pos + child_end

                child_id = str(uuid.uuid4())
                child_ids.append(child_id)

                child_chunk = HierarchicalChunk(
                    chunk=Chunk(
                        text=child_text,
                        metadata=ChunkMetadata(
                            source_id=source_id,
                            chunk_index=child_idx,
                            total_chunks=len(child_texts),
                            start_char=abs_start,
                            end_char=abs_end,
                            custom={"parent_chunk_index": parent_idx},
                        ),
                    ),
                    chunk_id=child_id,
                    parent_id=parent_id,
                    is_parent=False,
                    children=[],
                    section_path=f"{section_path}/child_{child_idx}",
                )
                children.append(child_chunk)

            # Update parent with child references
            parent_chunk.children = child_ids
            parents.append(parent_chunk)
            parent_child_map[parent_id] = child_ids

        # Build all_chunks list (parents first, then children)
        all_chunks = parents + children

        return HierarchicalSplitResult(
            parents=parents,
            children=children,
            all_chunks=all_chunks,
            parent_child_map=parent_child_map,
        )

    def create_chunks_with_metadata(
        self,
        text: str,
        source_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Create flat list of chunks with parent metadata.

        Useful for storage systems that don't support hierarchy natively.
        Parent info is stored in chunk metadata.

        Args:
            text: Text to split.
            source_id: Source document identifier.

        Returns:
            List of Chunk objects with parent info in metadata.
        """
        result = self.split_hierarchical(text, source_id=source_id)
        chunks: List[Chunk] = []

        # Add parent chunks
        for parent in result.parents:
            parent.chunk.metadata.custom["is_parent"] = True
            parent.chunk.metadata.custom["child_ids"] = parent.children
            parent.chunk.metadata.custom["section_path"] = parent.section_path
            chunks.append(parent.chunk)

        # Add child chunks
        for child in result.children:
            child.chunk.metadata.custom["is_parent"] = False
            child.chunk.metadata.custom["parent_id"] = child.parent_id
            child.chunk.metadata.custom["section_path"] = child.section_path
            chunks.append(child.chunk)

        return chunks


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def expand_to_parent(
    child_chunk: HierarchicalChunk,
    parent_map: Dict[str, HierarchicalChunk],
) -> Optional[HierarchicalChunk]:
    """Expand a child chunk to its parent.

    Args:
        child_chunk: The child chunk.
        parent_map: Mapping from parent ID to parent chunk.

    Returns:
        Parent chunk or None if not found.
    """
    if child_chunk.is_parent:
        return child_chunk
    if child_chunk.parent_id and child_chunk.parent_id in parent_map:
        return parent_map[child_chunk.parent_id]
    return None


def get_context_window(
    chunks: List[HierarchicalChunk],
    target_idx: int,
    window_size: int = 1,
) -> List[HierarchicalChunk]:
    """Get surrounding chunks for context.

    Args:
        chunks: List of chunks.
        target_idx: Index of target chunk.
        window_size: Number of chunks before/after.

    Returns:
        List of chunks in the window.
    """
    start = max(0, target_idx - window_size)
    end = min(len(chunks), target_idx + window_size + 1)
    return chunks[start:end]
