"""RAG (Retrieval-Augmented Generation) service facade.

Provides a unified interface for initializing and using RAG capabilities
across ATLAS. Coordinates:
- ConfigManager for RAG settings
- RAGRetriever for semantic search
- DocumentIngester for knowledge base population
- EmbeddingProvider for vector generation

Example:
    >>> from ATLAS.services.rag import RAGService
    >>> from ATLAS.config import ConfigManager
    >>> 
    >>> config = ConfigManager()
    >>> rag_service = await RAGService.create(config)
    >>> 
    >>> # Ingest a document
    >>> await rag_service.ingest_text("kb_123", "Doc Title", "Document content...")
    >>> 
    >>> # Retrieve context for a query
    >>> result = await rag_service.retrieve("How does X work?")
    >>> context = rag_service.assemble_context(result)
    >>> print(context.text)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager
    from ATLAS.config.rag import RAGSettings
    from modules.storage.embeddings import EmbeddingProvider
    from modules.storage.ingestion import DocumentIngester
    from modules.storage.knowledge import KnowledgeStore
    from modules.storage.retrieval import RAGRetriever
    from modules.storage.retrieval.retriever import (
        RetrievalResult,
        AssembledContext,
        ContextFormat,
    )

logger = setup_logger(__name__)


@dataclass
class RAGServiceStatus:
    """Status information for the RAG service."""
    
    enabled: bool
    """Whether RAG is enabled in configuration."""
    
    operational: bool
    """Whether RAG is fully operational (all components ready)."""
    
    embedding_provider: Optional[str] = None
    """Active embedding provider name."""
    
    embedding_model: Optional[str] = None
    """Active embedding model."""
    
    knowledge_store_connected: bool = False
    """Whether knowledge store connection is active."""
    
    has_retriever: bool = False
    """Whether retriever is available."""
    
    has_ingester: bool = False
    """Whether ingester is available."""
    
    error: Optional[str] = None
    """Error message if not operational."""


class RAGService:
    """High-level facade for RAG operations.
    
    Provides simplified access to RAG functionality including:
    - Document ingestion
    - Context retrieval
    - Status monitoring
    - Configuration management
    
    Use `await RAGService.create(config)` to initialize with auto-setup,
    or create directly with pre-configured components.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        *,
        knowledge_store: Optional[KnowledgeStore] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        retriever: Optional[RAGRetriever] = None,
        ingester: Optional[DocumentIngester] = None,
    ):
        """Initialize RAG service.
        
        For full auto-initialization, use `RAGService.create()` instead.
        
        Args:
            config_manager: ConfigManager instance for settings.
            knowledge_store: Pre-configured knowledge store.
            embedding_provider: Pre-configured embedding provider.
            retriever: Pre-configured retriever.
            ingester: Pre-configured ingester.
        """
        self._config_manager = config_manager
        self._knowledge_store = knowledge_store
        self._embedding_provider = embedding_provider
        self._retriever = retriever
        self._ingester = ingester
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    @classmethod
    async def create(
        cls,
        config_manager: ConfigManager,
        *,
        knowledge_store: Optional[KnowledgeStore] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> "RAGService":
        """Create a RAG service instance with component initialization.
        
        Args:
            config_manager: ConfigManager instance for settings.
            knowledge_store: Pre-configured knowledge store (required).
            embedding_provider: Pre-configured embedding provider (required).
            
        Returns:
            Configured RAGService instance.
            
        Note:
            Knowledge store and embedding provider must be provided
            as they require database connections and API keys that
            are best configured by the caller.
        """
        service = cls(
            config_manager,
            knowledge_store=knowledge_store,
            embedding_provider=embedding_provider,
        )
        await service.initialize()
        return service
    
    async def initialize(self) -> bool:
        """Initialize RAG components based on configuration.
        
        Creates retriever and ingester if knowledge_store and
        embedding_provider are available.
        
        Returns:
            True if initialization succeeded, False otherwise.
        """
        async with self._init_lock:
            if self._initialized:
                return True
            
            try:
                settings = self._config_manager.get_rag_settings()
                
                if not settings.enabled:
                    logger.info("RAG is disabled in configuration")
                    self._initialized = True
                    return True
                
                if not self._knowledge_store or not self._embedding_provider:
                    logger.debug(
                        "RAG service missing components: knowledge_store=%s, embedding_provider=%s",
                        bool(self._knowledge_store),
                        bool(self._embedding_provider),
                    )
                    self._initialized = True
                    return True
                
                # Create retriever if not provided
                if not self._retriever:
                    self._retriever = self._create_retriever(settings)
                
                # Create ingester if not provided
                if not self._ingester:
                    self._ingester = self._create_ingester(settings)
                
                self._initialized = True
                logger.info("RAG service initialized successfully")
                return True
                
            except Exception as exc:
                logger.error("Failed to initialize RAG service: %s", exc, exc_info=True)
                return False
    
    def _create_retriever(self, settings: RAGSettings) -> Optional[RAGRetriever]:
        """Create RAG retriever based on settings."""
        try:
            from modules.storage.retrieval import RAGRetriever
            from modules.storage.retrieval.retriever import RerankerType as RetrieverRerankerType
            
            if not self._knowledge_store or not self._embedding_provider:
                return None
            
            retr_settings = settings.retrieval
            rerank_settings = settings.reranking
            
            # Map config RerankerType to retriever RerankerType
            reranker_type = RetrieverRerankerType.NONE
            if rerank_settings.enabled:
                provider_name = rerank_settings.provider.value
                if provider_name == "cross_encoder":
                    reranker_type = RetrieverRerankerType.CROSS_ENCODER
                elif provider_name == "cohere":
                    reranker_type = RetrieverRerankerType.COHERE
            
            retriever = RAGRetriever(
                knowledge_store=self._knowledge_store,
                embedding_provider=self._embedding_provider,
                top_k=retr_settings.top_k,
                top_n_rerank=rerank_settings.top_n_rerank,
                min_score=retr_settings.similarity_threshold,
                reranker_type=reranker_type,
                config_manager=self._config_manager,
            )
            return retriever
            
        except Exception as exc:
            logger.error("Failed to create retriever: %s", exc)
            return None
    
    def _create_ingester(self, settings: RAGSettings) -> Optional[DocumentIngester]:
        """Create document ingester based on settings."""
        try:
            from modules.storage.ingestion import DocumentIngester
            
            if not self._knowledge_store or not self._embedding_provider:
                return None
            
            chunk_settings = settings.chunking
            
            ingester = DocumentIngester(
                knowledge_store=self._knowledge_store,
                embedding_provider=self._embedding_provider,
                default_chunk_size=chunk_settings.chunk_size,
                default_chunk_overlap=chunk_settings.chunk_overlap,
            )
            return ingester
            
        except Exception as exc:
            logger.error("Failed to create ingester: %s", exc)
            return None
    
    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    
    @property
    def is_enabled(self) -> bool:
        """Check if RAG is enabled in configuration."""
        return self._config_manager.is_rag_enabled()
    
    @property
    def is_operational(self) -> bool:
        """Check if RAG is fully operational."""
        if not self._initialized or not self.is_enabled:
            return False
        return bool(
            self._knowledge_store
            and self._embedding_provider
            and self._retriever
        )
    
    @property
    def retriever(self) -> Optional[RAGRetriever]:
        """Get the RAG retriever instance (for LLMContextManager)."""
        return self._retriever
    
    @property
    def ingester(self) -> Optional[DocumentIngester]:
        """Get the document ingester instance."""
        return self._ingester
    
    @property
    def knowledge_store(self) -> Optional[KnowledgeStore]:
        """Get the knowledge store instance."""
        return self._knowledge_store
    
    def get_status(self) -> RAGServiceStatus:
        """Get detailed status information."""
        settings = self._config_manager.get_rag_settings()
        
        status = RAGServiceStatus(
            enabled=settings.enabled,
            operational=self.is_operational,
        )
        
        if settings.embeddings.enabled:
            status.embedding_provider = settings.embeddings.default_provider.value
            provider_config = getattr(
                settings.embeddings,
                settings.embeddings.default_provider.value,
                None,
            )
            if provider_config:
                status.embedding_model = getattr(provider_config, "model", None)
        
        status.knowledge_store_connected = bool(self._knowledge_store)
        status.has_retriever = bool(self._retriever)
        status.has_ingester = bool(self._ingester)
        
        return status
    
    async def retrieve(
        self,
        query: str,
        *,
        knowledge_base_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        rerank: bool = True,
    ) -> Optional[RetrievalResult]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: The query string.
            knowledge_base_ids: Optional list of KB IDs to search.
            top_k: Override default top_k results.
            rerank: Whether to apply reranking.
            
        Returns:
            RetrievalResult with chunks and metadata, or None if unavailable.
        """
        if not self.is_operational:
            if not self._initialized:
                await self.initialize()
            if not self.is_operational:
                return None
        
        if not self._retriever:
            return None
        
        try:
            return await self._retriever.retrieve(
                query=query,
                knowledge_base_ids=knowledge_base_ids,
                top_k=top_k,
                rerank=rerank,
            )
        except Exception as exc:
            logger.error("Failed to retrieve: %s", exc)
            return None
    
    def assemble_context(
        self,
        results: RetrievalResult,
        *,
        format: Optional[ContextFormat] = None,
        max_tokens: Optional[int] = None,
        include_sources: bool = True,
    ) -> Optional[AssembledContext]:
        """Assemble retrieved chunks into LLM context.
        
        Args:
            results: Retrieval results from retrieve().
            format: Output format (plain, markdown, xml, json).
            max_tokens: Maximum tokens (approximate).
            include_sources: Whether to include source references.
            
        Returns:
            AssembledContext with formatted text, or None if no retriever.
        """
        if not self._retriever:
            return None
        
        try:
            from modules.storage.retrieval.retriever import ContextFormat as CF
            fmt = format or CF.PLAIN
            
            if max_tokens is None:
                max_tokens = self._config_manager.get_rag_settings().max_context_tokens
            
            return self._retriever.assemble_context(
                results,
                format=fmt,
                max_tokens=max_tokens,
                include_sources=include_sources,
            )
        except Exception as exc:
            logger.error("Failed to assemble context: %s", exc)
            return None
    
    async def retrieve_and_assemble(
        self,
        query: str,
        *,
        knowledge_base_ids: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Convenience method to retrieve and assemble context in one call.
        
        Args:
            query: The query string.
            knowledge_base_ids: Optional KB IDs to search.
            max_tokens: Maximum tokens for context.
            
        Returns:
            Assembled context text or None.
        """
        results = await self.retrieve(query, knowledge_base_ids=knowledge_base_ids)
        if not results or not results.chunks:
            return None
        
        assembled = self.assemble_context(results, max_tokens=max_tokens)
        return assembled.text if assembled else None
    
    async def ingest_text(
        self,
        kb_id: str,
        title: str,
        content: str,
        *,
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Ingest text content into a knowledge base.
        
        Args:
            kb_id: Knowledge base ID.
            title: Document title.
            content: Text content to ingest.
            source_uri: Optional source URI.
            metadata: Optional document metadata.
            
        Returns:
            True if ingestion succeeded.
        """
        if not self._ingester:
            return False
        
        try:
            from modules.storage.ingestion import IngestionOptions, FileType
            
            options = IngestionOptions(metadata=metadata or {})
            
            await self._ingester.ingest_text(
                kb_id=kb_id,
                title=title,
                content=content,
                source_uri=source_uri,
                options=options,
            )
            return True
        except Exception as exc:
            logger.error("Failed to ingest text: %s", exc)
            return False
    
    async def ingest_file(
        self,
        kb_id: str,
        file_path: Union[str, Path],
        *,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Ingest a file into a knowledge base.
        
        Args:
            kb_id: Knowledge base ID.
            file_path: Path to the file.
            title: Optional document title (defaults to filename).
            metadata: Optional document metadata.
            
        Returns:
            True if ingestion succeeded.
        """
        if not self._ingester:
            return False
        
        try:
            from modules.storage.ingestion import IngestionOptions
            
            options = IngestionOptions(metadata=metadata or {})
            
            await self._ingester.ingest_file(
                kb_id=kb_id,
                file_path=Path(file_path),
                title=title,
                options=options,
            )
            return True
        except Exception as exc:
            logger.error("Failed to ingest file: %s", exc)
            return False
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._retriever:
            try:
                await self._retriever.shutdown()
            except Exception as exc:
                logger.warning("Error shutting down retriever: %s", exc)
        
        if self._ingester:
            try:
                await self._ingester.shutdown()
            except Exception as exc:
                logger.warning("Error shutting down ingester: %s", exc)
        
        self._initialized = False


__all__ = ["RAGService", "RAGServiceStatus"]
