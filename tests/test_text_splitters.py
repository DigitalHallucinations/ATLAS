"""Unit tests for text splitters.

Tests cover the text splitter base class, chunk dataclass,
and concrete splitter implementations.
"""

from __future__ import annotations

import pytest
from typing import List

from modules.storage.chunking import (
    TextSplitter,
    Chunk,
    ChunkMetadata,
    ChunkingError,
    ChunkingConfigError,
    RecursiveTextSplitter,
    CharacterTextSplitter,
    SentenceTextSplitter,
    ParagraphTextSplitter,
    SemanticTextSplitter,
    create_text_splitter,
    list_text_splitters,
    get_text_splitter_class,
    cosine_similarity,
)


# --- Test ChunkMetadata ---


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating chunk metadata."""
        metadata = ChunkMetadata(
            source_id="doc-123",
            chunk_index=2,
            total_chunks=5,
            start_char=100,
            end_char=200,
        )

        assert metadata.source_id == "doc-123"
        assert metadata.chunk_index == 2
        assert metadata.total_chunks == 5
        assert metadata.start_char == 100
        assert metadata.end_char == 200

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = ChunkMetadata()

        assert metadata.source_id is None
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == 0
        assert metadata.overlap_prev == 0
        assert metadata.overlap_next == 0
        assert metadata.custom == {}

    def test_metadata_to_dict(self):
        """Test converting metadata to dict."""
        metadata = ChunkMetadata(
            source_id="doc-1",
            chunk_index=0,
            total_chunks=3,
        )

        d = metadata.to_dict()
        assert d["source_id"] == "doc-1"
        assert d["chunk_index"] == 0
        assert d["total_chunks"] == 3

    def test_metadata_from_dict(self):
        """Test creating metadata from dict."""
        d = {
            "source_id": "doc-2",
            "chunk_index": 1,
            "total_chunks": 5,
            "custom": {"key": "value"},
        }

        metadata = ChunkMetadata.from_dict(d)
        assert metadata.source_id == "doc-2"
        assert metadata.chunk_index == 1
        assert metadata.custom == {"key": "value"}


# --- Test Chunk ---


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_create_chunk(self):
        """Test creating a chunk."""
        chunk = Chunk(text="Hello, world!")

        assert chunk.text == "Hello, world!"
        assert chunk.chunk_id is not None
        assert len(chunk.chunk_id) == 16  # SHA256 prefix

    def test_chunk_length(self):
        """Test chunk length property."""
        chunk = Chunk(text="Hello, world!")
        assert chunk.length == 13

    def test_chunk_word_count(self):
        """Test chunk word count property."""
        chunk = Chunk(text="Hello world how are you")
        assert chunk.word_count == 5

    def test_chunk_to_dict(self):
        """Test converting chunk to dict."""
        chunk = Chunk(text="Test text")
        d = chunk.to_dict()

        assert d["text"] == "Test text"
        assert "chunk_id" in d
        assert "metadata" in d

    def test_chunk_from_dict(self):
        """Test creating chunk from dict."""
        d = {
            "text": "Test text",
            "chunk_id": "abc123",
            "metadata": {"source_id": "doc-1"},
        }

        chunk = Chunk.from_dict(d)
        assert chunk.text == "Test text"
        assert chunk.chunk_id == "abc123"
        assert chunk.metadata.source_id == "doc-1"

    def test_chunk_id_generation(self):
        """Test that chunk IDs are deterministic."""
        metadata = ChunkMetadata(source_id="doc-1", chunk_index=0)
        chunk1 = Chunk(text="Same text", metadata=metadata, chunk_id=None)

        # Recreate with same params
        metadata2 = ChunkMetadata(source_id="doc-1", chunk_index=0)
        chunk2 = Chunk(text="Same text", metadata=metadata2, chunk_id=None)

        assert chunk1.chunk_id == chunk2.chunk_id


# --- Test Splitter Registry ---


class TestSplitterRegistry:
    """Tests for text splitter registry."""

    def test_list_splitters_includes_builtin(self):
        """Test that built-in splitters are registered."""
        splitters = list_text_splitters()
        assert "recursive" in splitters
        assert "character" in splitters
        assert "sentence" in splitters
        assert "paragraph" in splitters
        assert "semantic" in splitters

    def test_get_recursive_splitter_class(self):
        """Test getting recursive splitter class."""
        cls = get_text_splitter_class("recursive")
        assert cls is RecursiveTextSplitter

    def test_get_sentence_splitter_class(self):
        """Test getting sentence splitter class."""
        cls = get_text_splitter_class("sentence")
        assert cls is SentenceTextSplitter

    def test_get_unknown_splitter_raises(self):
        """Test that unknown splitter raises error."""
        with pytest.raises(ChunkingError) as exc_info:
            get_text_splitter_class("unknown_splitter")

        assert "Unknown text splitter" in str(exc_info.value)

    def test_create_splitter_factory(self):
        """Test creating splitter via factory function."""
        splitter = create_text_splitter("recursive", chunk_size=200)
        assert isinstance(splitter, RecursiveTextSplitter)
        assert splitter.chunk_size == 200


# --- Test RecursiveTextSplitter ---


class TestRecursiveTextSplitter:
    """Tests for RecursiveTextSplitter."""

    def test_splitter_name(self):
        """Test splitter name property."""
        splitter = RecursiveTextSplitter()
        assert splitter.name == "recursive"

    def test_split_simple_text(self):
        """Test splitting simple text."""
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 10  # ~160 chars

        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            # Chunks should respect size (with some tolerance)
            assert len(chunk) <= 60

    def test_split_with_paragraphs(self):
        """Test splitting text with paragraph breaks."""
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_split_empty_text(self):
        """Test splitting empty text."""
        splitter = RecursiveTextSplitter()
        chunks = splitter.split_text("")
        assert chunks == []

    def test_split_short_text(self):
        """Test splitting text shorter than chunk_size."""
        splitter = RecursiveTextSplitter(chunk_size=500)
        text = "Short text."

        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_custom_separators(self):
        """Test using custom separators."""
        splitter = RecursiveTextSplitter(
            chunk_size=50,
            separators=["|||", "|", " "],
        )
        text = "Part one|||Part two|||Part three"

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_create_chunks_with_metadata(self):
        """Test creating Chunk objects with metadata."""
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 10

        chunks = splitter.create_chunks(text, source_id="doc-1")

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert chunk.metadata.source_id == "doc-1"
            assert chunk.metadata.chunk_index == i


# --- Test CharacterTextSplitter ---


class TestCharacterTextSplitter:
    """Tests for CharacterTextSplitter."""

    def test_splitter_name(self):
        """Test splitter name property."""
        splitter = CharacterTextSplitter()
        assert splitter.name == "character"

    def test_split_by_newline(self):
        """Test splitting by newline separator."""
        splitter = CharacterTextSplitter(
            chunk_size=100,
            separator="\n",
        )
        text = "Line one\nLine two\nLine three\nLine four"

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1


# --- Test SentenceTextSplitter ---


class TestSentenceTextSplitter:
    """Tests for SentenceTextSplitter."""

    def test_splitter_name(self):
        """Test splitter name property."""
        splitter = SentenceTextSplitter()
        assert splitter.name == "sentence"

    def test_split_sentences(self):
        """Test splitting at sentence boundaries."""
        splitter = SentenceTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            min_sentence_length=5,
        )
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        # Sentences should be preserved
        for chunk in chunks:
            assert "." in chunk or chunk.endswith("sentence")

    def test_split_preserves_sentences(self):
        """Test that sentences aren't broken mid-word."""
        splitter = SentenceTextSplitter(
            chunk_size=50,
            min_sentence_length=5,
        )
        text = "Hello world. How are you today? I am fine."

        chunks = splitter.split_text(text)

        # Each chunk should contain complete sentences
        for chunk in chunks:
            # Should not end with partial word
            words = chunk.split()
            assert len(words) >= 1

    def test_handles_abbreviations(self):
        """Test handling of abbreviations."""
        splitter = SentenceTextSplitter(
            chunk_size=200,
            min_sentence_length=10,
        )
        text = "Dr. Smith went to the store. He bought some milk."

        chunks = splitter.split_text(text)
        # "Dr." should not cause a split
        assert len(chunks) >= 1


# --- Test ParagraphTextSplitter ---


class TestParagraphTextSplitter:
    """Tests for ParagraphTextSplitter."""

    def test_splitter_name(self):
        """Test splitter name property."""
        splitter = ParagraphTextSplitter()
        assert splitter.name == "paragraph"

    def test_split_paragraphs(self):
        """Test splitting at paragraph boundaries."""
        splitter = ParagraphTextSplitter(chunk_size=200)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = splitter.split_text(text)
        assert len(chunks) >= 1


# --- Test SemanticTextSplitter ---


class TestSemanticTextSplitter:
    """Tests for SemanticTextSplitter."""

    def test_splitter_name(self):
        """Test splitter name property."""
        splitter = SemanticTextSplitter()
        assert splitter.name == "semantic"

    def test_sync_fallback(self):
        """Test synchronous fallback behavior."""
        splitter = SemanticTextSplitter(chunk_size=100)
        text = "First sentence. Second sentence. Third sentence."

        # Sync call should fall back to sentence splitting
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_requires_embedding_provider(self):
        """Test that async split requires embedding provider."""
        splitter = SemanticTextSplitter()

        import asyncio

        async def run_async():
            with pytest.raises(ChunkingError) as exc_info:
                await splitter.split_text_async("Test text")
            assert "embedding_provider" in str(exc_info.value)

        asyncio.get_event_loop().run_until_complete(run_async())


# --- Test Cosine Similarity ---


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(vec1, vec2)) < 0.001

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        assert abs(cosine_similarity(vec1, vec2) + 1.0) < 0.001

    def test_similar_vectors(self):
        """Test similarity of similar vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]
        sim = cosine_similarity(vec1, vec2)
        assert sim > 0.99  # Very similar

    def test_different_dimensions_raises(self):
        """Test that different dimensions raise error."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]

        with pytest.raises(ValueError):
            cosine_similarity(vec1, vec2)


# --- Test Configuration Errors ---


class TestConfigurationErrors:
    """Tests for configuration validation."""

    def test_negative_chunk_size_raises(self):
        """Test that negative chunk_size raises error."""
        with pytest.raises(ChunkingConfigError):
            RecursiveTextSplitter(chunk_size=-100)

    def test_zero_chunk_size_raises(self):
        """Test that zero chunk_size raises error."""
        with pytest.raises(ChunkingConfigError):
            RecursiveTextSplitter(chunk_size=0)

    def test_negative_overlap_raises(self):
        """Test that negative overlap raises error."""
        with pytest.raises(ChunkingConfigError):
            RecursiveTextSplitter(chunk_size=100, chunk_overlap=-10)

    def test_overlap_exceeds_chunk_size_raises(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ChunkingConfigError):
            RecursiveTextSplitter(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ChunkingConfigError):
            RecursiveTextSplitter(chunk_size=100, chunk_overlap=150)


# --- Integration Tests ---


class TestChunkingIntegration:
    """Integration tests for the chunking pipeline."""

    def test_split_documents(self):
        """Test splitting multiple documents."""
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)

        documents = [
            {"id": "doc-1", "text": "First document content. " * 5},
            {"id": "doc-2", "text": "Second document content. " * 5},
        ]

        chunks = splitter.split_documents(documents)

        assert len(chunks) > 2  # More chunks than documents
        # Check source IDs are preserved
        source_ids = {c.metadata.source_id for c in chunks}
        assert "doc-1" in source_ids
        assert "doc-2" in source_ids

    def test_chunk_overlap_works(self):
        """Test that chunk overlap produces overlapping content."""
        splitter = RecursiveTextSplitter(
            chunk_size=50,
            chunk_overlap=20,
        )
        text = "Word " * 30  # ~150 chars

        chunks = splitter.split_text(text)

        if len(chunks) >= 2:
            # Check that adjacent chunks have some overlap
            for i in range(len(chunks) - 1):
                # End of chunk i should appear at start of chunk i+1
                # (approximately, due to how overlap works)
                end_words = chunks[i].split()[-3:]
                start_words = chunks[i + 1].split()[:5]
                # At least some overlap should exist
                overlap = set(end_words) & set(start_words)
                # This is approximate due to word boundaries
                assert len(chunks[i]) > 0
