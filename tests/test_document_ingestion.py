"""Tests for document ingestion pipeline.

Tests the DocumentIngester, parsers, and batch ingestion.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from modules.storage.ingestion import (
    IngestionError,
    ParseError,
    FileType,
    IngestionResult,
    IngestionOptions,
    BatchIngestionResult,
    DocumentParser,
    TextParser,
    HTMLParser,
    PDFParser,
    DocumentIngester,
)


# -----------------------------------------------------------------------------
# Data Class Tests
# -----------------------------------------------------------------------------


class TestFileType:
    """Tests for FileType enum."""

    def test_values(self):
        """Test enum values."""
        assert FileType.TEXT == "text"
        assert FileType.MARKDOWN == "markdown"
        assert FileType.HTML == "html"
        assert FileType.PDF == "pdf"
        assert FileType.CODE == "code"

    def test_extension_map_exists(self):
        """Test EXTENSION_MAP class attribute."""
        assert hasattr(FileType, "EXTENSION_MAP")
        assert ".txt" in FileType.EXTENSION_MAP
        assert ".md" in FileType.EXTENSION_MAP
        assert ".py" in FileType.EXTENSION_MAP

    def test_from_extension(self):
        """Test from_extension class method."""
        assert FileType.from_extension(".txt") == FileType.TEXT
        assert FileType.from_extension(".md") == FileType.MARKDOWN
        assert FileType.from_extension(".html") == FileType.HTML
        assert FileType.from_extension(".pdf") == FileType.PDF
        assert FileType.from_extension(".py") == FileType.CODE
        assert FileType.from_extension(".unknown") == FileType.UNKNOWN


class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = IngestionResult(
            document_id="doc_123",
            chunk_count=10,
            success=True,
        )
        assert result.document_id == "doc_123"
        assert result.chunk_count == 10
        assert result.success is True
        assert result.error is None

    def test_with_error(self):
        """Test with error."""
        result = IngestionResult(
            document_id="",
            chunk_count=0,
            success=False,
            error="Parse failed",
        )
        assert result.success is False
        assert result.error == "Parse failed"


class TestIngestionOptions:
    """Tests for IngestionOptions dataclass."""

    def test_defaults(self):
        """Test default values."""
        options = IngestionOptions()
        assert options.chunk_size == 1000
        assert options.chunk_overlap == 200
        assert options.auto_detect_type is True
        assert options.extract_metadata is True

    def test_custom(self):
        """Test custom values."""
        options = IngestionOptions(
            chunk_size=500,
            chunk_overlap=100,
            file_type=FileType.MARKDOWN,
            auto_detect_type=False,
        )
        assert options.chunk_size == 500
        assert options.chunk_overlap == 100
        assert options.file_type == FileType.MARKDOWN
        assert options.auto_detect_type is False


class TestBatchIngestionResult:
    """Tests for BatchIngestionResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = BatchIngestionResult(
            total_files=10,
            successful=8,
            failed=2,
            results=[],
        )
        assert result.total_files == 10
        assert result.successful == 8
        assert result.failed == 2

    def test_with_results(self):
        """Test with individual results."""
        individual = [
            IngestionResult("d1", 5, True),
            IngestionResult("", 0, False, "Error"),
        ]
        result = BatchIngestionResult(
            total_files=2,
            successful=1,
            failed=1,
            results=individual,
        )
        assert len(result.results) == 2


# -----------------------------------------------------------------------------
# Parser Tests
# -----------------------------------------------------------------------------


class TestTextParser:
    """Tests for TextParser."""

    def test_creation(self):
        """Test parser creation."""
        parser = TextParser()
        assert parser.name == "text"

    def test_supported_types(self):
        """Test supported file types."""
        parser = TextParser()
        assert FileType.TEXT in parser.supported_types
        assert FileType.MARKDOWN in parser.supported_types
        assert FileType.CODE in parser.supported_types

    @pytest.mark.asyncio
    async def test_parse_text(self):
        """Test parsing text content."""
        parser = TextParser()
        content = "Hello, world!\nThis is a test."
        
        result = await parser.parse(content)
        
        assert result == content

    @pytest.mark.asyncio
    async def test_parse_file(self):
        """Test parsing text file."""
        parser = TextParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test file content")
            temp_path = f.name
        
        try:
            result = await parser.parse_file(Path(temp_path))
            assert "Test file content" in result
        finally:
            os.unlink(temp_path)


class TestHTMLParser:
    """Tests for HTMLParser."""

    def test_creation(self):
        """Test parser creation."""
        parser = HTMLParser()
        assert parser.name == "html"

    def test_supported_types(self):
        """Test supported file types."""
        parser = HTMLParser()
        assert FileType.HTML in parser.supported_types

    @pytest.mark.asyncio
    async def test_parse_simple_html(self):
        """Test parsing simple HTML."""
        parser = HTMLParser()
        html = "<html><body><p>Hello, world!</p></body></html>"
        
        result = await parser.parse(html)
        
        assert "Hello, world!" in result
        # Tags should be removed
        assert "<p>" not in result

    @pytest.mark.asyncio
    async def test_parse_html_with_scripts(self):
        """Test scripts are removed."""
        parser = HTMLParser()
        html = """
        <html>
        <head><script>alert('bad');</script></head>
        <body><p>Content</p></body>
        </html>
        """
        
        result = await parser.parse(html)
        
        assert "Content" in result
        assert "alert" not in result


class TestPDFParser:
    """Tests for PDFParser."""

    def test_creation(self):
        """Test parser creation."""
        parser = PDFParser()
        assert parser.name == "pdf"

    def test_supported_types(self):
        """Test supported file types."""
        parser = PDFParser()
        assert FileType.PDF in parser.supported_types


# -----------------------------------------------------------------------------
# DocumentIngester Tests
# -----------------------------------------------------------------------------


class TestDocumentIngester:
    """Tests for DocumentIngester."""

    @pytest.fixture
    def mock_knowledge_store(self):
        """Create mock knowledge store."""
        store = MagicMock()
        store.is_initialized = True
        store.add_document = AsyncMock(return_value=MagicMock(id="doc_123"))
        return store

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create mock embedding provider."""
        provider = MagicMock()
        provider.is_initialized = True
        return provider

    @pytest.fixture
    def mock_text_splitter(self):
        """Create mock text splitter."""
        splitter = MagicMock()
        splitter.split.return_value = ["chunk1", "chunk2", "chunk3"]
        return splitter

    def test_creation(self, mock_knowledge_store):
        """Test ingester creation."""
        ingester = DocumentIngester(
            knowledge_store=mock_knowledge_store,
        )
        assert ingester._knowledge_store is mock_knowledge_store
        assert not ingester.is_initialized

    @pytest.mark.asyncio
    async def test_initialize(self, mock_knowledge_store):
        """Test initialization."""
        mock_knowledge_store.is_initialized = False
        mock_knowledge_store.initialize = AsyncMock()
        
        ingester = DocumentIngester(knowledge_store=mock_knowledge_store)
        await ingester.initialize()
        
        assert ingester.is_initialized
        mock_knowledge_store.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_text(self, mock_knowledge_store, mock_text_splitter):
        """Test text ingestion."""
        ingester = DocumentIngester(
            knowledge_store=mock_knowledge_store,
            text_splitter=mock_text_splitter,
        )
        await ingester.initialize()
        
        result = await ingester.ingest_text(
            content="This is test content.",
            knowledge_base_id="kb_1",
            title="Test Document",
        )
        
        assert result.success is True
        assert result.document_id == "doc_123"

    @pytest.mark.asyncio
    async def test_ingest_file(self, mock_knowledge_store, mock_text_splitter):
        """Test file ingestion."""
        ingester = DocumentIngester(
            knowledge_store=mock_knowledge_store,
            text_splitter=mock_text_splitter,
        )
        await ingester.initialize()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("File content here")
            temp_path = f.name
        
        try:
            result = await ingester.ingest_file(
                file_path=Path(temp_path),
                knowledge_base_id="kb_1",
            )
            
            assert result.success is True
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_ingest_file_not_found(self, mock_knowledge_store):
        """Test ingestion of non-existent file."""
        ingester = DocumentIngester(knowledge_store=mock_knowledge_store)
        await ingester.initialize()
        
        result = await ingester.ingest_file(
            file_path=Path("/nonexistent/file.txt"),
            knowledge_base_id="kb_1",
        )
        
        assert result.success is False
        assert "not found" in result.error.lower() or "exist" in result.error.lower()


class TestBatchIngestion:
    """Tests for batch ingestion."""

    @pytest.fixture
    def mock_knowledge_store(self):
        """Create mock knowledge store."""
        store = MagicMock()
        store.is_initialized = True
        store.add_document = AsyncMock(return_value=MagicMock(id="doc_123"))
        return store

    @pytest.fixture
    def mock_text_splitter(self):
        """Create mock text splitter."""
        splitter = MagicMock()
        splitter.split.return_value = ["chunk1", "chunk2"]
        return splitter

    @pytest.mark.asyncio
    async def test_ingest_directory(self, mock_knowledge_store, mock_text_splitter):
        """Test directory ingestion."""
        ingester = DocumentIngester(
            knowledge_store=mock_knowledge_store,
            text_splitter=mock_text_splitter,
        )
        await ingester.initialize()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            Path(temp_dir, "file1.txt").write_text("Content 1")
            Path(temp_dir, "file2.txt").write_text("Content 2")
            Path(temp_dir, "file3.md").write_text("# Markdown")
            
            result = await ingester.ingest_directory(
                directory_path=Path(temp_dir),
                knowledge_base_id="kb_1",
            )
            
            assert result.total_files == 3
            assert result.successful == 3
            assert result.failed == 0

    @pytest.mark.asyncio
    async def test_ingest_directory_with_pattern(self, mock_knowledge_store, mock_text_splitter):
        """Test directory ingestion with file pattern."""
        ingester = DocumentIngester(
            knowledge_store=mock_knowledge_store,
            text_splitter=mock_text_splitter,
        )
        await ingester.initialize()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            Path(temp_dir, "file1.txt").write_text("Content 1")
            Path(temp_dir, "file2.md").write_text("Markdown")
            Path(temp_dir, "file3.py").write_text("# Python")
            
            result = await ingester.ingest_directory(
                directory_path=Path(temp_dir),
                knowledge_base_id="kb_1",
                file_patterns=["*.txt"],
            )
            
            assert result.total_files == 1
            assert result.successful == 1

    @pytest.mark.asyncio
    async def test_ingest_empty_directory(self, mock_knowledge_store):
        """Test ingestion of empty directory."""
        ingester = DocumentIngester(knowledge_store=mock_knowledge_store)
        await ingester.initialize()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await ingester.ingest_directory(
                directory_path=Path(temp_dir),
                knowledge_base_id="kb_1",
            )
            
            assert result.total_files == 0
            assert result.successful == 0
            assert result.failed == 0


# -----------------------------------------------------------------------------
# Exception Tests
# -----------------------------------------------------------------------------


class TestExceptions:
    """Tests for exception classes."""

    def test_ingestion_error(self):
        """Test IngestionError."""
        error = IngestionError("Ingestion failed")
        assert str(error) == "Ingestion failed"
        assert isinstance(error, Exception)

    def test_parse_error(self):
        """Test ParseError."""
        error = ParseError("Parse failed")
        assert str(error) == "Parse failed"
        assert isinstance(error, IngestionError)
