"""Document ingestion pipeline for RAG.

This module provides document ingestion capabilities:
- DocumentIngester: Main ingestion orchestrator
- Parsers for various file formats (text, HTML, PDF)
- Batch ingestion for directories

Usage:
    >>> from modules.storage.ingestion import DocumentIngester, IngestionOptions
    >>> ingester = DocumentIngester(
    ...     knowledge_store=store,
    ...     embedding_provider=embedder,
    ... )
    >>> await ingester.initialize()
    >>> result = await ingester.ingest_file("document.pdf", knowledge_base_id="kb_123")
"""

from modules.storage.ingestion.ingester import (
    # Exceptions
    IngestionError,
    ParseError,
    # Enums
    FileType,
    # Data classes
    IngestionResult,
    IngestionOptions,
    BatchIngestionResult,
    # Parsers
    DocumentParser,
    TextParser,
    HTMLParser,
    PDFParser,
    # Main class
    DocumentIngester,
)


__all__ = [
    # Exceptions
    "IngestionError",
    "ParseError",
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
]
