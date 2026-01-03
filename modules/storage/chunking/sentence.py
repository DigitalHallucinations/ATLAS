"""Sentence-based text splitter implementation.

Splits text at sentence boundaries using regex patterns or NLTK,
preserving semantic coherence within chunks.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional

from .base import (
    TextSplitter,
    register_text_splitter,
    ChunkingError,
)

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


# Regex pattern for sentence boundary detection
# Matches period, exclamation, or question mark followed by space or end
SENTENCE_ENDINGS = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence boundary
    r'(?<=[.!?])\s*$|'          # End of text
    r'(?<=[.!?])\s*\n',         # Newline after sentence
    re.MULTILINE
)

# More aggressive pattern that handles more edge cases
SENTENCE_PATTERN_DETAILED = re.compile(
    r'(?<='
    r'[.!?]'           # Sentence ending punctuation
    r'[\"\'\)\]]*'     # Optional closing quotes/parens
    r')'
    r'\s+'             # Whitespace
    r'(?='
    r'[\"\'\(\[]*'     # Optional opening quotes/parens
    r'[A-Z]'           # Capital letter (start of new sentence)
    r')',
    re.MULTILINE
)

# Abbreviations that shouldn't be treated as sentence endings
ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'esp',
    'inc', 'ltd', 'co', 'corp', 'st', 'ave', 'blvd', 'rd', 'dept',
    'fig', 'al', 'ed', 'vol', 'rev', 'no', 'pp', 'pg', 'approx',
    'est', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep',
    'oct', 'nov', 'dec', 'e.g', 'i.e', 'cf', 'viz',
}


@register_text_splitter("sentence")
class SentenceTextSplitter(TextSplitter):
    """Split text at sentence boundaries.

    This splitter uses regex patterns to identify sentence boundaries
    and groups sentences into chunks that fit within the size limit.

    It's more semantically aware than character-based splitting
    as it preserves complete sentences.

    Example:
        >>> splitter = SentenceTextSplitter(chunk_size=500)
        >>> chunks = splitter.split_text("First sentence. Second sentence.")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_sentence_length: int = 20,
        length_function: Callable[[str], int] | None = None,
        strip_whitespace: bool = True,
        use_nltk: bool = False,
    ):
        """Initialize the sentence text splitter.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            min_sentence_length: Minimum length for a valid sentence.
            length_function: Function to measure text length.
            strip_whitespace: Whether to strip whitespace from chunks.
            use_nltk: Whether to use NLTK for sentence tokenization.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            strip_whitespace=strip_whitespace,
            keep_separator=True,
        )
        self.min_sentence_length = min_sentence_length
        self.use_nltk = use_nltk
        self._nltk_available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "sentence"

    def split_text(self, text: str) -> List[str]:
        """Split text at sentence boundaries and merge into chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        # Get sentences
        if self.use_nltk and self._check_nltk():
            sentences = self._split_sentences_nltk(text)
        else:
            sentences = self._split_sentences_regex(text)

        # Filter out very short sentences (likely noise)
        sentences = [s for s in sentences if len(s.strip()) >= self.min_sentence_length]

        if not sentences:
            # Fall back to the original text if no sentences found
            return [text.strip()] if text.strip() else []

        # Merge sentences into chunks with overlap
        return self._merge_sentences(sentences)

    def _check_nltk(self) -> bool:
        """Check if NLTK is available."""
        if self._nltk_available is not None:
            return self._nltk_available

        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
            self._nltk_available = True
        except (ImportError, LookupError):
            logger.debug("NLTK not available, falling back to regex")
            self._nltk_available = False

        return self._nltk_available

    def _split_sentences_nltk(self, text: str) -> List[str]:
        """Split text into sentences using NLTK.

        Args:
            text: The text to split.

        Returns:
            List of sentences.
        """
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except Exception as exc:
            logger.warning("NLTK tokenization failed: %s", exc)
            return self._split_sentences_regex(text)

    def _split_sentences_regex(self, text: str) -> List[str]:
        """Split text into sentences using regex.

        Args:
            text: The text to split.

        Returns:
            List of sentences.
        """
        # First, handle paragraph breaks
        paragraphs = text.split('\n\n')
        sentences = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Split paragraph into sentences
            para_sentences = self._split_paragraph_sentences(paragraph)
            sentences.extend(para_sentences)

        return sentences

    def _split_paragraph_sentences(self, paragraph: str) -> List[str]:
        """Split a paragraph into sentences.

        Args:
            paragraph: A single paragraph of text.

        Returns:
            List of sentences.
        """
        # Use the detailed pattern for better accuracy
        parts = SENTENCE_PATTERN_DETAILED.split(paragraph)

        sentences = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            current += (" " if current else "") + part

            # Check if this looks like a complete sentence
            if self._is_sentence_end(current):
                sentences.append(current)
                current = ""

        # Don't forget remaining text
        if current.strip():
            sentences.append(current.strip())

        # If no sentences found, try simpler split
        if not sentences:
            # Simple fallback: split on . ! ?
            simple_sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sentences = [s.strip() for s in simple_sentences if s.strip()]

        return sentences if sentences else [paragraph]

    def _is_sentence_end(self, text: str) -> bool:
        """Check if text ends with a sentence boundary.

        Args:
            text: The text to check.

        Returns:
            True if text ends with a sentence.
        """
        text = text.strip()
        if not text:
            return False

        # Must end with sentence-ending punctuation
        if not text[-1] in '.!?':
            return False

        # Check for abbreviations
        words = text.lower().split()
        if words:
            last_word = words[-1].rstrip('.!?')
            if last_word in ABBREVIATIONS:
                return False

        return True

    def _merge_sentences(self, sentences: List[str]) -> List[str]:
        """Merge sentences into chunks respecting size limits.

        Args:
            sentences: List of sentences.

        Returns:
            List of merged chunks.
        """
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = self.length_function(sentence)

            # If sentence alone is too large, add it as-is
            if sentence_length > self.chunk_size:
                # First, save current chunk if any
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                # Add large sentence as-is
                chunks.append(sentence)
                continue

            # Check if adding this sentence would exceed limit
            new_length = current_length + sentence_length
            if current_chunk:
                new_length += 1  # Space between sentences

            if new_length > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(self.length_function(s) for s in current_chunk)
                current_length += len(current_chunk) - 1  # Spaces
            else:
                current_chunk.append(sentence)
                current_length = new_length

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences to include as overlap in the next chunk.

        Args:
            sentences: Current chunk sentences.

        Returns:
            Sentences to include in overlap.
        """
        if not sentences or self.chunk_overlap == 0:
            return []

        overlap: List[str] = []
        overlap_length = 0

        for sentence in reversed(sentences):
            sentence_length = self.length_function(sentence) + 1  # +1 for space
            if overlap_length + sentence_length > self.chunk_overlap:
                break
            overlap.insert(0, sentence)
            overlap_length += sentence_length

        return overlap


@register_text_splitter("paragraph")
class ParagraphTextSplitter(TextSplitter):
    """Split text at paragraph boundaries.

    This splitter treats double newlines as paragraph breaks
    and groups paragraphs into chunks.

    Example:
        >>> splitter = ParagraphTextSplitter(chunk_size=1000)
        >>> chunks = splitter.split_text(document)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        length_function: Callable[[str], int] | None = None,
        strip_whitespace: bool = True,
    ):
        """Initialize the paragraph text splitter.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            length_function: Function to measure text length.
            strip_whitespace: Whether to strip whitespace from chunks.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            strip_whitespace=strip_whitespace,
            keep_separator=False,
        )

    @property
    def name(self) -> str:
        return "paragraph"

    def split_text(self, text: str) -> List[str]:
        """Split text at paragraph boundaries.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text.strip()] if text.strip() else []

        # Merge paragraphs into chunks
        return self._merge_splits(paragraphs, "\n\n")


__all__ = [
    "SentenceTextSplitter",
    "ParagraphTextSplitter",
    "SENTENCE_ENDINGS",
    "ABBREVIATIONS",
]
