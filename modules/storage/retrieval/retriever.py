"""RAG retrieval pipeline.

Provides semantic search and context assembly for RAG workflows:
1. Query embedding generation
2. Vector similarity search
3. Optional reranking
4. Context assembly for LLM prompts

Usage:
    >>> from modules.storage.retrieval import RAGRetriever
    >>> retriever = RAGRetriever(
    ...     knowledge_store=store,
    ...     embedding_provider=embedder,
    ... )
    >>> await retriever.initialize()
    >>> results = await retriever.retrieve("How do I configure logging?")
    >>> context = retriever.assemble_context(results)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
)

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from modules.storage.knowledge import (
        KnowledgeStore,
        SearchQuery,
        SearchResult,
        KnowledgeChunk,
        KnowledgeDocument,
    )
    from modules.storage.embeddings import EmbeddingProvider

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class RetrievalError(Exception):
    """Base exception for retrieval errors."""

    pass


class RerankError(RetrievalError):
    """Error during reranking."""

    pass


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class RerankerType(str, Enum):
    """Available reranker types."""

    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"


class ContextFormat(str, Enum):
    """Format for assembled context."""

    PLAIN = "plain"  # Simple numbered list
    MARKDOWN = "markdown"  # Markdown formatted
    XML = "xml"  # XML tags for structure
    JSON = "json"  # JSON format


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Result from RAG retrieval.

    Attributes:
        query: Original query text.
        chunks: Retrieved chunks with scores.
        documents: Associated documents (deduplicated).
        retrieval_time_ms: Time for initial retrieval.
        rerank_time_ms: Time for reranking (if applied).
        total_time_ms: Total retrieval time.
        reranked: Whether results were reranked.
    """

    query: str
    chunks: List["SearchResult"]
    documents: Dict[str, "KnowledgeDocument"] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    reranked: bool = False


@dataclass
class ContextChunk:
    """A chunk prepared for context assembly.

    Attributes:
        content: Chunk text content.
        source: Source document title or URI.
        score: Relevance score.
        metadata: Additional metadata.
    """

    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledContext:
    """Assembled context ready for LLM prompt.

    Attributes:
        text: Formatted context text.
        chunks: Source chunks used.
        token_count: Estimated token count.
        truncated: Whether context was truncated.
    """

    text: str
    chunks: List[ContextChunk]
    token_count: int = 0
    truncated: bool = False


# -----------------------------------------------------------------------------
# Rerankers
# -----------------------------------------------------------------------------


class Reranker:
    """Base class for rerankers."""

    @property
    def name(self) -> str:
        return "base"

    async def initialize(self) -> None:
        """Initialize the reranker."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the reranker."""
        pass

    async def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_n: Optional[int] = None,
    ) -> List["SearchResult"]:
        """Rerank search results.

        Args:
            query: Original query.
            results: Search results to rerank.
            top_n: Number of results to return.

        Returns:
            Reranked results.
        """
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Cross-encoder based reranker using sentence-transformers."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        """Initialize cross-encoder reranker.

        Args:
            model: HuggingFace model name.
            device: Device to use (cuda, cpu, mps).
        """
        self._model_name = model
        self._device = device
        self._model: Optional[Any] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "cross_encoder"

    async def initialize(self) -> None:
        """Load the cross-encoder model."""
        if self._initialized:
            return

        def _load() -> Any:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise RerankError(
                    "Cross-encoder reranking requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                )

            device = self._device
            if device is None:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            return CrossEncoder(self._model_name, device=device)

        self._model = await asyncio.to_thread(_load)
        self._initialized = True
        logger.info(f"CrossEncoder reranker initialized with {self._model_name}")

    async def shutdown(self) -> None:
        """Cleanup model resources."""
        self._model = None
        self._initialized = False

    async def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_n: Optional[int] = None,
    ) -> List["SearchResult"]:
        """Rerank using cross-encoder scores."""
        if not self._initialized or not self._model:
            await self.initialize()

        if not results:
            return []

        # Prepare pairs for scoring
        pairs = [(query, r.chunk.content) for r in results]

        def _score() -> List[float]:
            return self._model.predict(pairs).tolist()

        scores = await asyncio.to_thread(_score)

        # Update results with new scores
        reranked = []
        for result, score in zip(results, scores):
            # Create new result with updated score
            from modules.storage.knowledge import SearchResult as SR
            reranked.append(SR(
                chunk=result.chunk,
                document=result.document,
                score=float(score),
                distance=result.distance,
                highlights=result.highlights,
            ))

        # Sort by new score (descending)
        reranked.sort(key=lambda r: r.score, reverse=True)

        if top_n:
            reranked = reranked[:top_n]

        return reranked


class CohereReranker(Reranker):
    """Cohere API based reranker."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
        config_manager: Optional[Any] = None,
    ) -> None:
        """Initialize Cohere reranker.

        Args:
            api_key: Cohere API key.
            model: Rerank model name.
            config_manager: Optional config manager for API key.
        """
        self._api_key = api_key
        self._model = model
        self._config_manager = config_manager
        self._client: Optional[Any] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "cohere"

    async def initialize(self) -> None:
        """Initialize Cohere client."""
        if self._initialized:
            return

        import os

        # Resolve API key
        api_key = self._api_key
        if not api_key and self._config_manager:
            get_key = getattr(self._config_manager, "get_cohere_api_key", None)
            if callable(get_key):
                api_key = get_key()
        if not api_key:
            api_key = os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")

        if not api_key:
            raise RerankError("Cohere API key not found")

        try:
            import cohere
            self._client = cohere.Client(api_key)
        except ImportError:
            raise RerankError(
                "Cohere reranking requires the cohere package. "
                "Install with: pip install cohere"
            )

        self._initialized = True
        logger.info(f"Cohere reranker initialized with {self._model}")

    async def shutdown(self) -> None:
        """Cleanup client."""
        self._client = None
        self._initialized = False

    async def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_n: Optional[int] = None,
    ) -> List["SearchResult"]:
        """Rerank using Cohere API."""
        if not self._initialized or not self._client:
            await self.initialize()

        if not results:
            return []

        documents = [r.chunk.content for r in results]

        def _rerank() -> Any:
            return self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n or len(documents),
            )

        response = await asyncio.to_thread(_rerank)

        # Build reranked results
        from modules.storage.knowledge import SearchResult as SR
        reranked = []
        for item in response.results:
            original = results[item.index]
            reranked.append(SR(
                chunk=original.chunk,
                document=original.document,
                score=float(item.relevance_score),
                distance=original.distance,
                highlights=original.highlights,
            ))

        return reranked


# -----------------------------------------------------------------------------
# RAG Retriever
# -----------------------------------------------------------------------------


class RAGRetriever:
    """RAG retrieval pipeline with optional reranking.

    Provides semantic search over knowledge bases with support for
    reranking and context assembly.
    """

    def __init__(
        self,
        knowledge_store: "KnowledgeStore",
        embedding_provider: Optional["EmbeddingProvider"] = None,
        *,
        reranker: Optional[Reranker] = None,
        reranker_type: RerankerType = RerankerType.NONE,
        top_k: int = 10,
        top_n_rerank: int = 5,
        min_score: float = 0.0,
        config_manager: Optional[Any] = None,
    ) -> None:
        """Initialize the RAG retriever.

        Args:
            knowledge_store: Store for searching chunks.
            embedding_provider: Provider for query embeddings.
            reranker: Custom reranker instance.
            reranker_type: Type of reranker to use if not provided.
            top_k: Number of results from initial search.
            top_n_rerank: Number of results after reranking.
            min_score: Minimum similarity score threshold.
            config_manager: Config manager for API keys.
        """
        self._knowledge_store = knowledge_store
        self._embedding_provider = embedding_provider
        self._top_k = top_k
        self._top_n_rerank = top_n_rerank
        self._min_score = min_score
        self._config_manager = config_manager
        self._initialized = False

        # Set up reranker
        if reranker:
            self._reranker = reranker
        elif reranker_type == RerankerType.CROSS_ENCODER:
            self._reranker = CrossEncoderReranker()
        elif reranker_type == RerankerType.COHERE:
            self._reranker = CohereReranker(config_manager=config_manager)
        else:
            self._reranker = None

    @property
    def is_initialized(self) -> bool:
        """Check if retriever is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the retriever and dependencies."""
        if self._initialized:
            return

        # Initialize knowledge store
        if not self._knowledge_store.is_initialized:
            await self._knowledge_store.initialize()

        # Initialize embedding provider
        if self._embedding_provider and not self._embedding_provider.is_initialized:
            await self._embedding_provider.initialize()

        # Initialize reranker
        if self._reranker:
            await self._reranker.initialize()

        self._initialized = True
        logger.info("RAGRetriever initialized")

    async def shutdown(self) -> None:
        """Shutdown the retriever."""
        if self._reranker:
            await self._reranker.shutdown()
        self._initialized = False

    async def retrieve(
        self,
        query: str,
        *,
        knowledge_base_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        rerank: bool = True,
        include_documents: bool = True,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: Search query text.
            knowledge_base_ids: Knowledge bases to search.
            top_k: Override default top_k.
            min_score: Override default min_score.
            rerank: Whether to apply reranking.
            include_documents: Whether to include document info.

        Returns:
            RetrievalResult with chunks and metadata.
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()
        top_k = top_k or self._top_k
        min_score = min_score if min_score is not None else self._min_score

        # Build search query
        from modules.storage.knowledge import SearchQuery

        search_query = SearchQuery(
            query_text=query,
            knowledge_base_ids=knowledge_base_ids,
            top_k=top_k if not (rerank and self._reranker) else top_k * 2,
            min_score=min_score if not rerank else 0.0,  # Filter after rerank
            include_content=True,
            include_document=include_documents,
        )

        # Execute search
        retrieval_start = datetime.utcnow()
        results = await self._knowledge_store.search(search_query)
        retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds() * 1000

        # Apply reranking
        rerank_time = 0.0
        reranked = False
        if rerank and self._reranker and results:
            rerank_start = datetime.utcnow()
            results = await self._reranker.rerank(
                query, results, top_n=self._top_n_rerank
            )
            rerank_time = (datetime.utcnow() - rerank_start).total_seconds() * 1000
            reranked = True

        # Apply min_score filter
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]

        # Limit to top_k
        results = results[:top_k]

        # Collect unique documents
        documents: Dict[str, "KnowledgeDocument"] = {}
        if include_documents:
            for result in results:
                if result.document and result.document.id not in documents:
                    documents[result.document.id] = result.document

        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return RetrievalResult(
            query=query,
            chunks=results,
            documents=documents,
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time,
            reranked=reranked,
        )

    def assemble_context(
        self,
        results: RetrievalResult,
        *,
        format: ContextFormat = ContextFormat.PLAIN,
        max_tokens: Optional[int] = None,
        include_sources: bool = True,
        include_scores: bool = False,
    ) -> AssembledContext:
        """Assemble retrieved chunks into LLM context.

        Args:
            results: Retrieval results.
            format: Output format.
            max_tokens: Maximum tokens (approximate).
            include_sources: Whether to include source references.
            include_scores: Whether to include relevance scores.

        Returns:
            AssembledContext ready for prompt injection.
        """
        chunks: List[ContextChunk] = []

        for result in results.chunks:
            source = ""
            if result.document:
                source = result.document.title or result.document.source_uri or ""
            
            chunks.append(ContextChunk(
                content=result.chunk.content,
                source=source,
                score=result.score,
                metadata={
                    "chunk_id": result.chunk.id,
                    "document_id": result.chunk.document_id,
                    "chunk_index": result.chunk.chunk_index,
                },
            ))

        # Format context
        if format == ContextFormat.MARKDOWN:
            text = self._format_markdown(chunks, include_sources, include_scores)
        elif format == ContextFormat.XML:
            text = self._format_xml(chunks, include_sources, include_scores)
        elif format == ContextFormat.JSON:
            text = self._format_json(chunks, include_sources, include_scores)
        else:
            text = self._format_plain(chunks, include_sources, include_scores)

        # Estimate token count (rough: ~4 chars per token)
        token_count = len(text) // 4

        # Truncate if needed
        truncated = False
        if max_tokens and token_count > max_tokens:
            # Truncate to approximate token limit
            max_chars = max_tokens * 4
            text = text[:max_chars] + "\n[Context truncated...]"
            truncated = True
            token_count = max_tokens

        return AssembledContext(
            text=text,
            chunks=chunks,
            token_count=token_count,
            truncated=truncated,
        )

    def _format_plain(
        self,
        chunks: List[ContextChunk],
        include_sources: bool,
        include_scores: bool,
    ) -> str:
        """Format as plain numbered list."""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[{i}]"
            if include_sources and chunk.source:
                header += f" {chunk.source}"
            if include_scores:
                header += f" (score: {chunk.score:.3f})"
            lines.append(header)
            lines.append(chunk.content)
            lines.append("")
        return "\n".join(lines)

    def _format_markdown(
        self,
        chunks: List[ContextChunk],
        include_sources: bool,
        include_scores: bool,
    ) -> str:
        """Format as markdown."""
        lines = ["## Retrieved Context", ""]
        for i, chunk in enumerate(chunks, 1):
            header = f"### [{i}]"
            if include_sources and chunk.source:
                header += f" {chunk.source}"
            if include_scores:
                header += f" *(score: {chunk.score:.3f})*"
            lines.append(header)
            lines.append("")
            lines.append(chunk.content)
            lines.append("")
        return "\n".join(lines)

    def _format_xml(
        self,
        chunks: List[ContextChunk],
        include_sources: bool,
        include_scores: bool,
    ) -> str:
        """Format as XML."""
        lines = ["<context>"]
        for i, chunk in enumerate(chunks, 1):
            attrs = f'index="{i}"'
            if include_sources and chunk.source:
                # Escape XML special chars
                source = chunk.source.replace("&", "&amp;").replace('"', "&quot;")
                attrs += f' source="{source}"'
            if include_scores:
                attrs += f' score="{chunk.score:.3f}"'
            lines.append(f"  <chunk {attrs}>")
            # Escape content
            content = chunk.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            lines.append(f"    {content}")
            lines.append("  </chunk>")
        lines.append("</context>")
        return "\n".join(lines)

    def _format_json(
        self,
        chunks: List[ContextChunk],
        include_sources: bool,
        include_scores: bool,
    ) -> str:
        """Format as JSON."""
        import json
        
        data = []
        for i, chunk in enumerate(chunks, 1):
            item: Dict[str, Any] = {
                "index": i,
                "content": chunk.content,
            }
            if include_sources:
                item["source"] = chunk.source
            if include_scores:
                item["score"] = chunk.score
            data.append(item)
        
        return json.dumps({"context": data}, indent=2)


__all__ = [
    # Exceptions
    "RetrievalError",
    "RerankError",
    # Enums
    "RerankerType",
    "ContextFormat",
    # Data classes
    "RetrievalResult",
    "ContextChunk",
    "AssembledContext",
    # Rerankers
    "Reranker",
    "CrossEncoderReranker",
    "CohereReranker",
    # Main class
    "RAGRetriever",
]
