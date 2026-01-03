"""Document ingestion pipeline for RAG.

Orchestrates the full document ingestion workflow:
1. Parse documents from various formats (text, markdown, PDF, code)
2. Chunk documents using configured text splitters
3. Generate embeddings for chunks
4. Store in knowledge base

Usage:
    >>> from modules.storage.ingestion import DocumentIngester
    >>> ingester = DocumentIngester(
    ...     knowledge_store=store,
    ...     embedding_provider=embedder,
    ...     text_splitter=splitter,
    ... )
    >>> await ingester.initialize()
    >>> doc = await ingester.ingest_file("/path/to/document.md", kb_id="kb-123")
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from modules.storage.knowledge import KnowledgeStore, KnowledgeDocument
    from modules.storage.embeddings import EmbeddingProvider
    from modules.storage.chunking import TextSplitter

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class IngestionError(Exception):
    """Base exception for ingestion errors."""

    pass


class ParseError(IngestionError):
    """Error parsing a document."""

    pass


class UnsupportedFormatError(IngestionError):
    """Document format is not supported."""

    pass


# -----------------------------------------------------------------------------
# Enums and Types
# -----------------------------------------------------------------------------


class FileType(str, Enum):
    """Supported file types for ingestion."""

    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    CODE = "code"
    JSON = "json"
    CSV = "csv"
    UNKNOWN = "unknown"


# Mapping from file extensions to FileType
EXTENSION_MAP: Dict[str, FileType] = {
    ".txt": FileType.TEXT,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".pdf": FileType.PDF,
    ".json": FileType.JSON,
    ".csv": FileType.CSV,
    # Code files
    ".py": FileType.CODE,
    ".js": FileType.CODE,
    ".ts": FileType.CODE,
    ".jsx": FileType.CODE,
    ".tsx": FileType.CODE,
    ".java": FileType.CODE,
    ".c": FileType.CODE,
    ".cpp": FileType.CODE,
    ".h": FileType.CODE,
    ".hpp": FileType.CODE,
    ".cs": FileType.CODE,
    ".go": FileType.CODE,
    ".rs": FileType.CODE,
    ".rb": FileType.CODE,
    ".php": FileType.CODE,
    ".swift": FileType.CODE,
    ".kt": FileType.CODE,
    ".scala": FileType.CODE,
    ".sh": FileType.CODE,
    ".bash": FileType.CODE,
    ".zsh": FileType.CODE,
    ".sql": FileType.CODE,
    ".yaml": FileType.CODE,
    ".yml": FileType.CODE,
    ".toml": FileType.CODE,
    ".xml": FileType.CODE,
    ".css": FileType.CODE,
    ".scss": FileType.CODE,
    ".less": FileType.CODE,
}

# Mapping from extension to language name for code files
LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
}


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Result of document ingestion.

    Attributes:
        document_id: ID of the ingested document.
        knowledge_base_id: ID of the knowledge base.
        title: Document title.
        chunk_count: Number of chunks created.
        token_count: Total tokens in document.
        file_path: Original file path (if from file).
        file_type: Detected file type.
        duration_seconds: Time taken to ingest.
        success: Whether ingestion succeeded.
        error: Error message if failed.
    """

    document_id: str
    knowledge_base_id: str
    title: str
    chunk_count: int = 0
    token_count: int = 0
    file_path: Optional[str] = None
    file_type: FileType = FileType.UNKNOWN
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class IngestionOptions:
    """Options for document ingestion.

    Attributes:
        chunk_size: Override default chunk size.
        chunk_overlap: Override default chunk overlap.
        auto_embed: Whether to generate embeddings.
        metadata: Additional metadata to attach.
        deduplicate: Whether to skip duplicate content.
        language: Language hint for code files.
        use_hierarchical_chunking: Use parent-child hierarchical chunks.
        parent_chunk_size: Size of parent chunks (if hierarchical).
        child_chunk_size: Size of child chunks (if hierarchical).
    """

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    auto_embed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    deduplicate: bool = True
    language: Optional[str] = None
    use_hierarchical_chunking: bool = False
    parent_chunk_size: int = 2048
    child_chunk_size: int = 512


@dataclass
class BatchIngestionResult:
    """Result of batch document ingestion.

    Attributes:
        total: Total documents attempted.
        successful: Number of successful ingestions.
        failed: Number of failed ingestions.
        results: Individual results for each document.
        duration_seconds: Total time taken.
    """

    total: int = 0
    successful: int = 0
    failed: int = 0
    results: List[IngestionResult] = field(default_factory=list)
    duration_seconds: float = 0.0


# -----------------------------------------------------------------------------
# Document Parsers
# -----------------------------------------------------------------------------


class DocumentParser:
    """Base class for document parsers."""

    def can_parse(self, file_type: FileType) -> bool:
        """Check if this parser can handle the file type."""
        return False

    def parse(self, content: bytes, file_type: FileType, **kwargs: Any) -> str:
        """Parse document content to text."""
        raise NotImplementedError


class TextParser(DocumentParser):
    """Parser for plain text and text-like files."""

    SUPPORTED_TYPES = {
        FileType.TEXT,
        FileType.MARKDOWN,
        FileType.CODE,
        FileType.JSON,
        FileType.CSV,
    }

    def can_parse(self, file_type: FileType) -> bool:
        return file_type in self.SUPPORTED_TYPES

    def parse(
        self,
        content: bytes,
        file_type: FileType,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> str:
        """Parse text content with encoding detection."""
        # Try specified encoding first
        encodings = [encoding, "utf-8", "latin-1", "cp1252"]
        
        for enc in encodings:
            try:
                return content.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Last resort: decode with replacement
        return content.decode("utf-8", errors="replace")


class HTMLParser(DocumentParser):
    """Parser for HTML documents."""

    def can_parse(self, file_type: FileType) -> bool:
        return file_type == FileType.HTML

    def parse(self, content: bytes, file_type: FileType, **kwargs: Any) -> str:
        """Parse HTML to plain text."""
        try:
            from html.parser import HTMLParser as StdHTMLParser
        except ImportError:
            # Fallback: strip tags with regex
            import re
            text = content.decode("utf-8", errors="replace")
            return re.sub(r"<[^>]+>", "", text)

        class TextExtractor(StdHTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts: List[str] = []
                self._skip_tags = {"script", "style", "head"}
                self._in_skip = 0

            def handle_starttag(self, tag: str, attrs: list) -> None:
                if tag.lower() in self._skip_tags:
                    self._in_skip += 1

            def handle_endtag(self, tag: str) -> None:
                if tag.lower() in self._skip_tags:
                    self._in_skip = max(0, self._in_skip - 1)

            def handle_data(self, data: str) -> None:
                if self._in_skip == 0:
                    text = data.strip()
                    if text:
                        self.text_parts.append(text)

        html_text = content.decode("utf-8", errors="replace")
        extractor = TextExtractor()
        extractor.feed(html_text)
        return "\n".join(extractor.text_parts)


class PDFParser(DocumentParser):
    """Parser for PDF documents (requires pypdf or pdfplumber)."""

    def can_parse(self, file_type: FileType) -> bool:
        return file_type == FileType.PDF

    def parse(self, content: bytes, file_type: FileType, **kwargs: Any) -> str:
        """Parse PDF to plain text."""
        # Try pypdf first
        try:
            from pypdf import PdfReader
            import io

            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        except ImportError:
            pass

        # Try pdfplumber
        try:
            import pdfplumber
            import io

            text_parts = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n\n".join(text_parts)
        except ImportError:
            pass

        raise ParseError(
            "PDF parsing requires 'pypdf' or 'pdfplumber'. "
            "Install with: pip install pypdf"
        )


# -----------------------------------------------------------------------------
# Document Ingester
# -----------------------------------------------------------------------------


class DocumentIngester:
    """Orchestrates document ingestion into a knowledge store.

    Handles the full pipeline of parsing, chunking, embedding, and storing
    documents in a knowledge base.
    """

    def __init__(
        self,
        knowledge_store: "KnowledgeStore",
        embedding_provider: Optional["EmbeddingProvider"] = None,
        text_splitter: Optional["TextSplitter"] = None,
        *,
        parsers: Optional[List[DocumentParser]] = None,
        default_chunk_size: int = 512,
        default_chunk_overlap: int = 50,
        batch_size: int = 10,
        hierarchical_chunking_enabled: bool = False,
        parent_chunk_size: int = 2048,
        child_chunk_size: int = 512,
    ) -> None:
        """Initialize the document ingester.

        Args:
            knowledge_store: Store for persisting documents and chunks.
            embedding_provider: Provider for generating embeddings.
            text_splitter: Splitter for chunking documents.
            parsers: Custom document parsers (defaults to built-in parsers).
            default_chunk_size: Default chunk size if not specified.
            default_chunk_overlap: Default chunk overlap if not specified.
            batch_size: Number of documents to process in parallel.
            hierarchical_chunking_enabled: Use hierarchical parent-child chunking.
            parent_chunk_size: Size of parent chunks in tokens.
            child_chunk_size: Size of child chunks in tokens.
        """
        self._knowledge_store = knowledge_store
        self._embedding_provider = embedding_provider
        self._text_splitter = text_splitter
        self._default_chunk_size = default_chunk_size
        self._default_chunk_overlap = default_chunk_overlap
        self._batch_size = batch_size
        self._hierarchical_chunking_enabled = hierarchical_chunking_enabled
        self._parent_chunk_size = parent_chunk_size
        self._child_chunk_size = child_chunk_size
        self._hierarchical_chunker: Optional[Any] = None
        self._initialized = False

        # Set up parsers
        if parsers:
            self._parsers = parsers
        else:
            self._parsers = [
                TextParser(),
                HTMLParser(),
                PDFParser(),
            ]

    @property
    def is_initialized(self) -> bool:
        """Check if the ingester is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the ingester and its dependencies."""
        if self._initialized:
            return

        # Initialize knowledge store
        if not self._knowledge_store.is_initialized:
            await self._knowledge_store.initialize()

        # Initialize embedding provider
        if self._embedding_provider and not self._embedding_provider.is_initialized:
            await self._embedding_provider.initialize()

        # Initialize hierarchical chunker if enabled
        if self._hierarchical_chunking_enabled and not self._hierarchical_chunker:
            try:
                from modules.storage.chunking.hierarchical import HierarchicalChunker
                self._hierarchical_chunker = HierarchicalChunker(
                    parent_chunk_size=self._parent_chunk_size,
                    child_chunk_size=self._child_chunk_size,
                )
                logger.info(
                    "Hierarchical chunker enabled: parent=%d, child=%d tokens",
                    self._parent_chunk_size,
                    self._child_chunk_size,
                )
            except ImportError as exc:
                logger.warning(f"Failed to load hierarchical chunker: {exc}")

        self._initialized = True
        logger.info("DocumentIngester initialized")

    async def shutdown(self) -> None:
        """Shutdown the ingester."""
        self._initialized = False
        logger.info("DocumentIngester shutdown")

    def detect_file_type(self, file_path: Union[str, Path]) -> FileType:
        """Detect the file type from path extension.

        Args:
            file_path: Path to the file.

        Returns:
            Detected FileType.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        return EXTENSION_MAP.get(ext, FileType.UNKNOWN)

    def detect_language(self, file_path: Union[str, Path]) -> Optional[str]:
        """Detect programming language from file extension.

        Args:
            file_path: Path to the file.

        Returns:
            Language name or None.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        return LANGUAGE_MAP.get(ext)

    def _get_parser(self, file_type: FileType) -> Optional[DocumentParser]:
        """Get a parser that can handle the file type."""
        for parser in self._parsers:
            if parser.can_parse(file_type):
                return parser
        return None

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def ingest_text(
        self,
        kb_id: str,
        title: str,
        content: str,
        *,
        source_uri: Optional[str] = None,
        file_type: FileType = FileType.TEXT,
        options: Optional[IngestionOptions] = None,
    ) -> IngestionResult:
        """Ingest text content directly.

        Args:
            kb_id: Knowledge base ID.
            title: Document title.
            content: Text content to ingest.
            source_uri: Optional source URI.
            file_type: Type of content.
            options: Ingestion options.

        Returns:
            IngestionResult with status and details.
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()
        options = options or IngestionOptions()

        try:
            # Map FileType to DocumentType
            from modules.storage.knowledge import DocumentType

            doc_type_map = {
                FileType.TEXT: DocumentType.TEXT,
                FileType.MARKDOWN: DocumentType.MARKDOWN,
                FileType.HTML: DocumentType.HTML,
                FileType.PDF: DocumentType.PDF,
                FileType.CODE: DocumentType.CODE,
                FileType.JSON: DocumentType.JSON,
                FileType.CSV: DocumentType.CSV,
                FileType.UNKNOWN: DocumentType.UNKNOWN,
            }
            doc_type = doc_type_map.get(file_type, DocumentType.TEXT)

            # Merge metadata
            metadata = options.metadata.copy()
            if options.language:
                metadata["language"] = options.language

            # Determine if using hierarchical chunking
            use_hierarchical = (
                options.use_hierarchical_chunking or 
                self._hierarchical_chunking_enabled
            )
            hierarchical_chunker = self._hierarchical_chunker if use_hierarchical else None
            
            if use_hierarchical:
                metadata["chunking_strategy"] = "hierarchical"

            # Add document to knowledge store
            doc = await self._knowledge_store.add_document(
                kb_id,
                title=title,
                content=content,
                source_uri=source_uri,
                document_type=doc_type,
                metadata=metadata,
                auto_chunk=True,
                auto_embed=options.auto_embed,
                hierarchical_chunker=hierarchical_chunker,
            )

            duration = (datetime.utcnow() - start_time).total_seconds()

            return IngestionResult(
                document_id=doc.id,
                knowledge_base_id=kb_id,
                title=title,
                chunk_count=doc.chunk_count,
                token_count=doc.token_count,
                file_type=file_type,
                duration_seconds=duration,
                success=True,
            )

        except Exception as exc:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Failed to ingest text '{title}': {exc}")
            return IngestionResult(
                document_id="",
                knowledge_base_id=kb_id,
                title=title,
                file_type=file_type,
                duration_seconds=duration,
                success=False,
                error=str(exc),
            )

    async def ingest_file(
        self,
        kb_id: str,
        file_path: Union[str, Path],
        *,
        title: Optional[str] = None,
        options: Optional[IngestionOptions] = None,
    ) -> IngestionResult:
        """Ingest a file into a knowledge base.

        Args:
            kb_id: Knowledge base ID.
            file_path: Path to the file to ingest.
            title: Optional title (defaults to filename).
            options: Ingestion options.

        Returns:
            IngestionResult with status and details.
        """
        if not self._initialized:
            await self.initialize()

        path = Path(file_path)
        start_time = datetime.utcnow()
        options = options or IngestionOptions()

        # Use filename as title if not provided
        if title is None:
            title = path.name

        # Detect file type
        file_type = self.detect_file_type(path)
        if file_type == FileType.UNKNOWN:
            return IngestionResult(
                document_id="",
                knowledge_base_id=kb_id,
                title=title,
                file_path=str(path),
                file_type=file_type,
                success=False,
                error=f"Unsupported file type: {path.suffix}",
            )

        # Get parser
        parser = self._get_parser(file_type)
        if not parser:
            return IngestionResult(
                document_id="",
                knowledge_base_id=kb_id,
                title=title,
                file_path=str(path),
                file_type=file_type,
                success=False,
                error=f"No parser available for file type: {file_type.value}",
            )

        try:
            # Read file
            content_bytes = await asyncio.to_thread(path.read_bytes)

            # Parse content
            content = parser.parse(content_bytes, file_type)

            # Detect language for code files
            language = options.language or self.detect_language(path)
            if language:
                options = IngestionOptions(
                    chunk_size=options.chunk_size,
                    chunk_overlap=options.chunk_overlap,
                    auto_embed=options.auto_embed,
                    metadata={**options.metadata, "language": language},
                    deduplicate=options.deduplicate,
                    language=language,
                    use_hierarchical_chunking=options.use_hierarchical_chunking,
                    parent_chunk_size=options.parent_chunk_size,
                    child_chunk_size=options.child_chunk_size,
                )

            # Ingest the parsed content
            result = await self.ingest_text(
                kb_id,
                title=title,
                content=content,
                source_uri=str(path.absolute()),
                file_type=file_type,
                options=options,
            )
            result.file_path = str(path)
            return result

        except Exception as exc:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Failed to ingest file '{path}': {exc}")
            return IngestionResult(
                document_id="",
                knowledge_base_id=kb_id,
                title=title,
                file_path=str(path),
                file_type=file_type,
                duration_seconds=duration,
                success=False,
                error=str(exc),
            )

    async def ingest_directory(
        self,
        kb_id: str,
        directory: Union[str, Path],
        *,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        options: Optional[IngestionOptions] = None,
    ) -> BatchIngestionResult:
        """Ingest all files in a directory.

        Args:
            kb_id: Knowledge base ID.
            directory: Directory path.
            recursive: Whether to recurse into subdirectories.
            extensions: Filter to specific extensions (e.g., [".md", ".py"]).
            exclude_patterns: Patterns to exclude (e.g., ["__pycache__", ".git"]).
            options: Ingestion options.

        Returns:
            BatchIngestionResult with all individual results.
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()
        dir_path = Path(directory)
        exclude_patterns = exclude_patterns or ["__pycache__", ".git", "node_modules", ".venv", "venv"]

        # Collect files
        files: List[Path] = []
        
        def should_exclude(path: Path) -> bool:
            for pattern in exclude_patterns:
                if pattern in str(path):
                    return True
            return False

        if recursive:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and not should_exclude(file_path):
                    if extensions is None or file_path.suffix.lower() in extensions:
                        files.append(file_path)
        else:
            for file_path in dir_path.iterdir():
                if file_path.is_file() and not should_exclude(file_path):
                    if extensions is None or file_path.suffix.lower() in extensions:
                        files.append(file_path)

        # Ingest files
        results: List[IngestionResult] = []
        successful = 0
        failed = 0

        for file_path in files:
            result = await self.ingest_file(kb_id, file_path, options=options)
            results.append(result)
            if result.success:
                successful += 1
            else:
                failed += 1

        duration = (datetime.utcnow() - start_time).total_seconds()

        return BatchIngestionResult(
            total=len(files),
            successful=successful,
            failed=failed,
            results=results,
            duration_seconds=duration,
        )

    async def ingest_url(
        self,
        kb_id: str,
        url: str,
        *,
        title: Optional[str] = None,
        options: Optional[IngestionOptions] = None,
    ) -> IngestionResult:
        """Ingest content from a URL.

        Args:
            kb_id: Knowledge base ID.
            url: URL to fetch and ingest.
            title: Optional title.
            options: Ingestion options.

        Returns:
            IngestionResult with status and details.
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()
        options = options or IngestionOptions()

        try:
            import urllib.request
            import urllib.error

            # Fetch URL content
            with urllib.request.urlopen(url, timeout=30) as response:
                content_bytes = response.read()
                content_type = response.headers.get("Content-Type", "")

            # Determine file type from content type
            if "html" in content_type:
                file_type = FileType.HTML
            elif "json" in content_type:
                file_type = FileType.JSON
            elif "pdf" in content_type:
                file_type = FileType.PDF
            else:
                file_type = FileType.TEXT

            # Get parser and parse
            parser = self._get_parser(file_type)
            if not parser:
                raise UnsupportedFormatError(f"Cannot parse content type: {content_type}")

            content = parser.parse(content_bytes, file_type)

            # Use URL as title if not provided
            if title is None:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                title = parsed.path.split("/")[-1] or parsed.netloc

            # Ingest content
            result = await self.ingest_text(
                kb_id,
                title=title,
                content=content,
                source_uri=url,
                file_type=file_type,
                options=options,
            )
            return result

        except Exception as exc:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Failed to ingest URL '{url}': {exc}")
            return IngestionResult(
                document_id="",
                knowledge_base_id=kb_id,
                title=title or url,
                duration_seconds=duration,
                success=False,
                error=str(exc),
            )


__all__ = [
    # Exceptions
    "IngestionError",
    "ParseError",
    "UnsupportedFormatError",
    # Enums
    "FileType",
    # Data classes
    "IngestionResult",
    "IngestionOptions",
    "BatchIngestionResult",
    # Parsers
    "DocumentParser",
    "TextParser",
    "HTMLParser",
    "PDFParser",
    # Main class
    "DocumentIngester",
    # Utilities
    "EXTENSION_MAP",
    "LANGUAGE_MAP",
]
