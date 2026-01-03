"""Caching layer for RAG embeddings and query results.

Provides LRU caching with TTL support for:
1. EmbeddingCache: Cache computed embeddings to avoid recomputation
2. QueryResultCache: Cache query results with semantic similarity matching

Usage:
    >>> from modules.storage.retrieval.cache import EmbeddingCache, QueryResultCache
    >>> embed_cache = EmbeddingCache(max_size=10000, ttl_seconds=3600)
    >>> cached = embed_cache.get("query text")
    >>> if cached is None:
    ...     embedding = await provider.embed("query text")
    ...     embed_cache.set("query text", embedding)
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# Cache Entry
# -----------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A cached item with metadata.

    Attributes:
        value: The cached value.
        created_at: Timestamp when entry was created.
        expires_at: Timestamp when entry expires (None = never).
        access_count: Number of times accessed.
        last_accessed: Timestamp of last access.
    """

    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = time.time()
        self.access_count += 1


# -----------------------------------------------------------------------------
# Embedding Cache
# -----------------------------------------------------------------------------


class EmbeddingCache:
    """LRU cache for computed embeddings.

    Thread-safe cache with TTL support for storing embeddings.
    Uses content hash as key for deduplication.
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = 3600,
        hash_algorithm: str = "sha256",
    ) -> None:
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of cached embeddings.
            ttl_seconds: Time-to-live in seconds (None = no expiration).
            hash_algorithm: Hash algorithm for content keys.
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hash_algorithm = hash_algorithm
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current cache size."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
                "ttl_seconds": self._ttl_seconds,
            }

    def _hash_key(self, text: str) -> str:
        """Generate cache key from text content."""
        hasher = hashlib.new(self._hash_algorithm)
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text.

        Args:
            text: Text to look up.

        Returns:
            Cached embedding or None if not found/expired.
        """
        key = self._hash_key(text)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding.

        Args:
            text: Text that was embedded.
            embedding: The computed embedding vector.
        """
        key = self._hash_key(text)
        expires_at = time.time() + self._ttl_seconds if self._ttl_seconds else None

        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=embedding,
                expires_at=expires_at,
            )

    def get_batch(self, texts: List[str]) -> Tuple[Dict[str, List[float]], List[str]]:
        """Get cached embeddings for multiple texts.

        Args:
            texts: List of texts to look up.

        Returns:
            Tuple of (cached_embeddings, missing_texts).
        """
        cached: Dict[str, List[float]] = {}
        missing: List[str] = []

        for text in texts:
            embedding = self.get(text)
            if embedding is not None:
                cached[text] = embedding
            else:
                missing.append(text)

        return cached, missing

    def set_batch(self, embeddings: Dict[str, List[float]]) -> None:
        """Cache multiple embeddings.

        Args:
            embeddings: Mapping of text -> embedding.
        """
        for text, embedding in embeddings.items():
            self.set(text, embedding)

    def invalidate(self, text: str) -> bool:
        """Remove a specific entry from cache.

        Args:
            text: Text to invalidate.

        Returns:
            True if entry was removed.
        """
        key = self._hash_key(text)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        removed = 0

        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.expires_at and entry.expires_at < now
            ]
            for key in expired_keys:
                del self._cache[key]
                removed += 1

        return removed


# -----------------------------------------------------------------------------
# Query Result Cache
# -----------------------------------------------------------------------------


class QueryResultCache:
    """Cache for query results with semantic similarity matching.

    Caches full retrieval results keyed by query text.
    Supports optional semantic similarity matching for near-duplicate queries.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = 300,
        similarity_threshold: float = 0.95,
        embedding_cache: Optional[EmbeddingCache] = None,
    ) -> None:
        """Initialize query result cache.

        Args:
            max_size: Maximum cached query results.
            ttl_seconds: TTL for cached results.
            similarity_threshold: Threshold for semantic matching (0-1).
            embedding_cache: Optional embedding cache for semantic matching.
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._similarity_threshold = similarity_threshold
        self._embedding_cache = embedding_cache
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._query_embeddings: Dict[str, List[float]] = {}  # For semantic matching
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "semantic_hits": self._semantic_hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "ttl_seconds": self._ttl_seconds,
            }

    def _hash_key(self, query: str, knowledge_base_ids: Optional[List[str]] = None) -> str:
        """Generate cache key from query and filters."""
        key_parts = [query.lower().strip()]
        if knowledge_base_ids:
            key_parts.append(",".join(sorted(knowledge_base_ids)))
        content = "|".join(key_parts)
        return hashlib.sha256(content.encode()).hexdigest()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(
        self,
        query: str,
        knowledge_base_ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[Any]:
        """Get cached result for query.

        Args:
            query: Query text.
            knowledge_base_ids: Knowledge base filter.
            query_embedding: Optional query embedding for semantic matching.

        Returns:
            Cached result or None.
        """
        key = self._hash_key(query, knowledge_base_ids)

        with self._lock:
            # Exact match
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired:
                    self._cache.move_to_end(key)
                    entry.touch()
                    self._hits += 1
                    return entry.value
                else:
                    del self._cache[key]

            # Try semantic matching if embedding provided
            if query_embedding and self._query_embeddings:
                best_match: Optional[str] = None
                best_similarity = 0.0

                for cached_key, cached_embedding in self._query_embeddings.items():
                    if cached_key not in self._cache:
                        continue
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    if similarity > best_similarity and similarity >= self._similarity_threshold:
                        entry = self._cache[cached_key]
                        if not entry.is_expired:
                            best_match = cached_key
                            best_similarity = similarity

                if best_match:
                    entry = self._cache[best_match]
                    entry.touch()
                    self._hits += 1
                    self._semantic_hits += 1
                    logger.debug(f"Semantic cache hit: {best_similarity:.3f} similarity")
                    return entry.value

            self._misses += 1
            return None

    def set(
        self,
        query: str,
        result: Any,
        knowledge_base_ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> None:
        """Cache a query result.

        Args:
            query: Query text.
            result: Result to cache.
            knowledge_base_ids: Knowledge base filter used.
            query_embedding: Optional query embedding for semantic matching.
        """
        key = self._hash_key(query, knowledge_base_ids)
        expires_at = time.time() + self._ttl_seconds if self._ttl_seconds else None

        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._query_embeddings.pop(oldest_key, None)

            self._cache[key] = CacheEntry(value=result, expires_at=expires_at)

            if query_embedding:
                self._query_embeddings[key] = query_embedding

    def invalidate_knowledge_base(self, kb_id: str) -> int:
        """Invalidate all entries for a knowledge base.

        Args:
            kb_id: Knowledge base ID.

        Returns:
            Number of entries invalidated.
        """
        # This requires tracking which entries used which KB
        # For now, we clear all (simple but safe approach)
        return self.clear()

    def clear(self) -> int:
        """Clear all cached entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._query_embeddings.clear()
            self._hits = 0
            self._misses = 0
            self._semantic_hits = 0
            return count


# -----------------------------------------------------------------------------
# Cached Embedding Provider Wrapper
# -----------------------------------------------------------------------------


class CachedEmbeddingProvider:
    """Wrapper that adds caching to any embedding provider.

    Usage:
        >>> from modules.storage.embeddings import HuggingFaceEmbeddingProvider
        >>> base_provider = HuggingFaceEmbeddingProvider()
        >>> cached_provider = CachedEmbeddingProvider(base_provider)
        >>> await cached_provider.initialize()
        >>> embedding = await cached_provider.embed("Hello world")
    """

    def __init__(
        self,
        provider: Any,
        cache: Optional[EmbeddingCache] = None,
        cache_max_size: int = 10000,
        cache_ttl: Optional[float] = 3600,
    ) -> None:
        """Initialize cached embedding provider.

        Args:
            provider: Base embedding provider.
            cache: Optional existing cache.
            cache_max_size: Max cache size if creating new cache.
            cache_ttl: Cache TTL if creating new cache.
        """
        self._provider = provider
        self._cache = cache or EmbeddingCache(
            max_size=cache_max_size,
            ttl_seconds=cache_ttl,
        )

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return getattr(self._provider, "is_initialized", True)

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats

    async def initialize(self) -> None:
        """Initialize the base provider."""
        if hasattr(self._provider, "initialize"):
            await self._provider.initialize()

    async def shutdown(self) -> None:
        """Shutdown the base provider."""
        if hasattr(self._provider, "shutdown"):
            await self._provider.shutdown()

    async def embed(self, text: str) -> List[float]:
        """Embed text with caching.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        # Check cache
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = await self._provider.embed(text)

        # Cache result
        self._cache.set(text, embedding)

        return embedding

    async def embed_batch(
        self,
        texts: List[str],
    ) -> Dict[str, List[float]]:
        """Embed multiple texts with caching.

        Args:
            texts: List of texts to embed.

        Returns:
            Mapping of text -> embedding.
        """
        # Check cache for all texts
        cached, missing = self._cache.get_batch(texts)

        if not missing:
            # All cached
            return cached

        # Compute missing embeddings
        if hasattr(self._provider, "embed_batch"):
            new_embeddings = await self._provider.embed_batch(missing)
        else:
            # Fallback to individual calls
            new_embeddings = {}
            for text in missing:
                new_embeddings[text] = await self._provider.embed(text)

        # Cache new embeddings
        self._cache.set_batch(new_embeddings)

        # Merge results
        cached.update(new_embeddings)
        return cached
