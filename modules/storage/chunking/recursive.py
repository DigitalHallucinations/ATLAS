"""Recursive text splitter implementation.

Splits text by recursively trying different separators (paragraphs,
newlines, sentences, words) until chunks are small enough.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from .base import (
    TextSplitter,
    register_text_splitter,
    ChunkingError,
)

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


# Default separators in order of priority
DEFAULT_SEPARATORS = [
    "\n\n",   # Paragraph breaks
    "\n",     # Line breaks
    ". ",     # Sentence endings
    "! ",     # Exclamation endings
    "? ",     # Question endings
    "; ",     # Semicolons
    ", ",     # Commas
    " ",      # Words
    "",       # Characters (last resort)
]


@register_text_splitter("recursive")
class RecursiveTextSplitter(TextSplitter):
    """Split text recursively using a hierarchy of separators.

    This splitter tries to split text using the first separator.
    If resulting chunks are still too large, it recursively splits
    those using the next separator in the list.

    This approach preserves semantic boundaries as much as possible
    by preferring paragraph and sentence breaks over arbitrary splits.

    Example:
        >>> splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
        >>> chunks = splitter.split_text("Long document text...")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Sequence[str] | None = None,
        length_function: Callable[[str], int] | None = None,
        strip_whitespace: bool = True,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
    ):
        """Initialize the recursive text splitter.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: List of separators to try in order.
            length_function: Function to measure text length.
            strip_whitespace: Whether to strip whitespace from chunks.
            keep_separator: Whether to keep separators in chunks.
            is_separator_regex: Whether separators are regex patterns.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            strip_whitespace=strip_whitespace,
            keep_separator=keep_separator,
        )
        self.separators = list(separators) if separators else DEFAULT_SEPARATORS.copy()
        self.is_separator_regex = is_separator_regex

    @property
    def name(self) -> str:
        return "recursive"

    def split_text(self, text: str) -> List[str]:
        """Split text recursively using separators.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        return self._split_text(text, self.separators)

    def _split_text(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """Recursively split text using given separators.

        Args:
            text: The text to split.
            separators: Remaining separators to try.

        Returns:
            List of text chunks.
        """
        final_chunks: List[str] = []

        # Find the best separator to use
        separator = separators[-1]  # Default to last (most granular)
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if self.is_separator_regex:
                import re
                if re.search(sep, text):
                    separator = sep
                    new_separators = separators[i + 1:]
                    break
            elif sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split by the chosen separator
        splits = self._split_by_separator(text, separator)

        # Process each split
        good_splits: List[str] = []
        current_separator = separator if self.keep_separator else ""

        for split in splits:
            if self.length_function(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                # This split is too large
                if good_splits:
                    # First, merge the good splits we have
                    merged = self._merge_splits(good_splits, current_separator)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    # No more separators, just add the large chunk
                    final_chunks.append(split)
                else:
                    # Recursively split with finer separators
                    sub_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(sub_chunks)

        # Don't forget remaining good splits
        if good_splits:
            merged = self._merge_splits(good_splits, current_separator)
            final_chunks.extend(merged)

        return final_chunks

    def _split_by_separator(
        self,
        text: str,
        separator: str,
    ) -> List[str]:
        """Split text by a separator, optionally keeping it.

        Args:
            text: The text to split.
            separator: The separator to split on.

        Returns:
            List of splits.
        """
        if separator == "":
            # Split into individual characters
            return list(text)

        if self.is_separator_regex:
            import re
            if self.keep_separator:
                # Use positive lookbehind to keep separator with preceding text
                splits = re.split(f"({separator})", text)
                result = []
                i = 0
                while i < len(splits):
                    if i + 1 < len(splits) and len(splits[i + 1]) > 0:
                        result.append(splits[i] + splits[i + 1])
                        i += 2
                    else:
                        result.append(splits[i])
                        i += 1
                return [s for s in result if s]
            else:
                return [s for s in re.split(separator, text) if s]

        # Plain string separator
        if self.keep_separator:
            # Split and keep separator at end of each piece
            parts = text.split(separator)
            result = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    result.append(part + separator)
                elif part:  # Last part, only add if non-empty
                    result.append(part)
            return result
        else:
            return [s for s in text.split(separator) if s]


@register_text_splitter("character")
class CharacterTextSplitter(TextSplitter):
    """Split text by a single separator with fixed chunk size.

    This is a simpler splitter that just splits on one separator
    (default: double newline) and merges until chunk_size.

    Example:
        >>> splitter = CharacterTextSplitter(separator="\\n", chunk_size=500)
        >>> chunks = splitter.split_text(document_text)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n",
        length_function: Callable[[str], int] | None = None,
        strip_whitespace: bool = True,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
    ):
        """Initialize the character text splitter.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            separator: Separator to split on.
            length_function: Function to measure text length.
            strip_whitespace: Whether to strip whitespace from chunks.
            keep_separator: Whether to keep separator in chunks.
            is_separator_regex: Whether separator is a regex pattern.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            strip_whitespace=strip_whitespace,
            keep_separator=keep_separator,
        )
        self.separator = separator
        self.is_separator_regex = is_separator_regex

    @property
    def name(self) -> str:
        return "character"

    def split_text(self, text: str) -> List[str]:
        """Split text by separator and merge into chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        # Split by separator
        if self.is_separator_regex:
            import re
            splits = re.split(self.separator, text)
        else:
            splits = text.split(self.separator)

        # Filter empty splits
        splits = [s for s in splits if s.strip()] if self.strip_whitespace else [s for s in splits if s]

        # Merge into chunks with overlap
        separator = "" if not self.keep_separator else self.separator
        return self._merge_splits(splits, separator)


__all__ = [
    "RecursiveTextSplitter",
    "CharacterTextSplitter",
    "DEFAULT_SEPARATORS",
]
