# 06 - Integration: Wiring neuralcore-runtime into ATLAS

**Phase:** 6  
**Duration:** 2-3 weeks  
**Priority:** Critical (enables all previous work)  
**Dependencies:** 01-05 (all previous phases)  

## Objective

Wire the neuralcore-runtime Rust crates into the ATLAS Python codebase, replacing the current Python implementations with the high-performance Rust equivalents while maintaining full backwards compatibility and graceful fallback.

## Deliverables

- [ ] Python package structure for neuralcore-runtime
- [ ] Feature detection and graceful fallback
- [ ] Drop-in replacement for embedding provider
- [ ] Drop-in replacement for NCB message bus
- [ ] Drop-in replacement for vector operations
- [ ] Configuration integration with ATLAS ConfigManager
- [ ] Migration scripts for existing deployments
- [ ] Performance validation benchmarks
- [ ] Documentation updates

---

## 1. Package Structure

```
neuralcore-runtime/
├── Cargo.toml                    # Workspace root
├── pyproject.toml                # Python package metadata
├── README.md
├── crates/
│   └── (sr-core, sr-vectors, etc.)
└── python/
    └── neuralcore/
        ├── __init__.py           # Public API
        ├── _native.pyi           # Type stubs for Rust bindings
        ├── vectors.py            # Vector operations wrapper
        ├── embedding.py          # Embedding provider wrapper
        ├── messaging.py          # Message bus wrapper
        ├── fallback/
        │   ├── __init__.py
        │   ├── vectors.py        # Pure Python fallback
        │   ├── embedding.py      # sentence-transformers fallback
        │   └── messaging.py      # Python asyncio fallback
        └── compat/
            ├── __init__.py
            ├── atlas_embedding.py    # ATLAS EmbeddingProvider adapter
            ├── atlas_ncb.py          # ATLAS NCB adapter
            └── atlas_vectors.py      # ATLAS vector ops adapter
```

---

## 2. Python Package Configuration

### pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "neuralcore-runtime"
version = "0.1.0"
description = "High-performance Rust runtime for neural computing workloads"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
    { name = "ATLAS Team", email = "atlas@example.com" }
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
keywords = ["neural", "embedding", "vectors", "messaging", "performance"]

dependencies = [
    "numpy>=1.24.0",
]

[project.optional-dependencies]
fallback = [
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
]
full = [
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    "redis>=4.5.0",
    "confluent-kafka>=2.0.0",
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-benchmark>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
]

[project.urls]
Homepage = "https://github.com/atlas/neuralcore-runtime"
Documentation = "https://atlas.github.io/neuralcore-runtime"
Repository = "https://github.com/atlas/neuralcore-runtime"

[tool.maturin]
# Build settings
python-source = "python"
module-name = "neuralcore._native"
features = ["pyo3/extension-module"]

# Platform-specific builds
[tool.maturin.target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-feature=+avx2,+fma"]

[tool.maturin.target.aarch64-apple-darwin]
rustflags = ["-C", "target-feature=+neon"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true
```

---

## 3. Main Package Entry Point

```python
# python/neuralcore/__init__.py
"""
neuralcore-runtime: High-performance Rust runtime for neural computing workloads.

This package provides accelerated implementations of:
- Vector distance computations (cosine, euclidean, dot product)
- HNSW approximate nearest neighbor search
- ONNX-based embedding inference
- Lock-free message bus with priority queues

When Rust extensions are not available, pure Python fallbacks are used.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import TYPE_CHECKING

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Feature flags
_NATIVE_AVAILABLE = False
_NATIVE_VECTORS = False
_NATIVE_EMBEDDING = False
_NATIVE_MESSAGING = False

# Try to import native extensions
try:
    from neuralcore import _native
    _NATIVE_AVAILABLE = True
    
    # Check which features are available
    _NATIVE_VECTORS = hasattr(_native, 'cosine_similarity')
    _NATIVE_EMBEDDING = hasattr(_native, 'EmbeddingEngine')
    _NATIVE_MESSAGING = hasattr(_native, 'MessageBus')
    
    logger.info(
        f"neuralcore native extensions loaded: "
        f"vectors={_NATIVE_VECTORS}, embedding={_NATIVE_EMBEDDING}, "
        f"messaging={_NATIVE_MESSAGING}"
    )
except ImportError as e:
    logger.warning(f"neuralcore native extensions not available: {e}")
    logger.info("Using pure Python fallbacks")


def is_native_available() -> bool:
    """Check if native Rust extensions are available."""
    return _NATIVE_AVAILABLE


def get_available_features() -> dict[str, bool]:
    """Get dictionary of available features."""
    return {
        "native": _NATIVE_AVAILABLE,
        "vectors": _NATIVE_VECTORS,
        "embedding": _NATIVE_EMBEDDING,
        "messaging": _NATIVE_MESSAGING,
    }


# Public API - automatically uses native or fallback
from neuralcore.vectors import (
    cosine_similarity,
    cosine_similarity_batch,
    euclidean_distance,
    dot_product,
    normalize,
    HNSWIndex,
)

from neuralcore.embedding import (
    EmbeddingEngine,
    embed_text,
    embed_batch,
)

from neuralcore.messaging import (
    MessageBus,
    Message,
    Priority,
)

__all__ = [
    # Version
    "__version__",
    
    # Feature detection
    "is_native_available",
    "get_available_features",
    
    # Vectors
    "cosine_similarity",
    "cosine_similarity_batch",
    "euclidean_distance",
    "dot_product",
    "normalize",
    "HNSWIndex",
    
    # Embedding
    "EmbeddingEngine",
    "embed_text",
    "embed_batch",
    
    # Messaging
    "MessageBus",
    "Message",
    "Priority",
]
```

---

## 4. Vector Operations Wrapper

```python
# python/neuralcore/vectors.py
"""Vector operations with automatic native/fallback selection."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Try native, fall back to Python
try:
    from neuralcore._native import (
        cosine_similarity as _native_cosine,
        cosine_similarity_batch as _native_cosine_batch,
        euclidean_distance as _native_euclidean,
        dot_product as _native_dot,
        normalize as _native_normalize,
        HNSWIndex as _NativeHNSWIndex,
    )
    _USE_NATIVE = True
except ImportError:
    from neuralcore.fallback.vectors import (
        cosine_similarity as _native_cosine,
        cosine_similarity_batch as _native_cosine_batch,
        euclidean_distance as _native_euclidean,
        dot_product as _native_dot,
        normalize as _native_normalize,
        HNSWIndex as _NativeHNSWIndex,
    )
    _USE_NATIVE = False
    logger.info("Using Python fallback for vector operations")


Vector = Union[List[float], NDArray[np.float32]]


def _ensure_numpy(v: Vector) -> NDArray[np.float32]:
    """Convert to numpy array if needed."""
    if isinstance(v, np.ndarray):
        return v.astype(np.float32, copy=False)
    return np.array(v, dtype=np.float32)


def cosine_similarity(a: Vector, b: Vector) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity in range [-1, 1]
        
    Example:
        >>> a = [1.0, 0.0, 0.0]
        >>> b = [0.0, 1.0, 0.0]
        >>> cosine_similarity(a, b)
        0.0
    """
    return _native_cosine(_ensure_numpy(a), _ensure_numpy(b))


def cosine_similarity_batch(
    query: Vector,
    vectors: Sequence[Vector],
) -> NDArray[np.float32]:
    """
    Compute cosine similarity between query and multiple vectors.
    
    Uses SIMD acceleration when available.
    
    Args:
        query: Query vector
        vectors: List of vectors to compare against
        
    Returns:
        Array of similarity scores
        
    Example:
        >>> query = [1.0, 0.0]
        >>> vectors = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
        >>> cosine_similarity_batch(query, vectors)
        array([1.0, 0.0, 0.707], dtype=float32)
    """
    query_np = _ensure_numpy(query)
    vectors_np = np.array([_ensure_numpy(v) for v in vectors], dtype=np.float32)
    return _native_cosine_batch(query_np, vectors_np)


def euclidean_distance(a: Vector, b: Vector) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Euclidean distance (>= 0)
    """
    return _native_euclidean(_ensure_numpy(a), _ensure_numpy(b))


def dot_product(a: Vector, b: Vector) -> float:
    """
    Compute dot product between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Dot product value
    """
    return _native_dot(_ensure_numpy(a), _ensure_numpy(b))


def normalize(v: Vector) -> NDArray[np.float32]:
    """
    Normalize a vector to unit length.
    
    Args:
        v: Input vector
        
    Returns:
        Normalized vector with L2 norm = 1
    """
    return _native_normalize(_ensure_numpy(v))


class HNSWIndex:
    """
    Hierarchical Navigable Small World graph for approximate nearest neighbor search.
    
    Provides fast similarity search with configurable accuracy/speed tradeoff.
    
    Example:
        >>> index = HNSWIndex(dimension=384, max_elements=10000)
        >>> index.add([1.0] * 384, id=0)
        >>> index.add([0.9] * 384, id=1)
        >>> results = index.search([1.0] * 384, k=5)
    """
    
    def __init__(
        self,
        dimension: int,
        max_elements: int = 100_000,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        """
        Initialize HNSW index.
        
        Args:
            dimension: Vector dimension
            max_elements: Maximum number of elements
            m: Number of connections per element (higher = more accurate, slower)
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of dynamic candidate list during search
        """
        self._index = _NativeHNSWIndex(
            dimension=dimension,
            max_elements=max_elements,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        self._dimension = dimension
    
    def add(
        self,
        vector: Vector,
        id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Add a vector to the index.
        
        Args:
            vector: Vector to add
            id: Optional ID (auto-generated if not provided)
            metadata: Optional metadata dictionary
            
        Returns:
            ID of added vector
        """
        v = _ensure_numpy(vector)
        if len(v) != self._dimension:
            raise ValueError(f"Vector dimension {len(v)} != index dimension {self._dimension}")
        return self._index.add(v, id, metadata)
    
    def add_batch(
        self,
        vectors: Sequence[Vector],
        ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Add multiple vectors to the index.
        
        Args:
            vectors: Vectors to add
            ids: Optional IDs
            
        Returns:
            List of IDs
        """
        vectors_np = np.array([_ensure_numpy(v) for v in vectors], dtype=np.float32)
        return self._index.add_batch(vectors_np, ids)
    
    def search(
        self,
        query: Vector,
        k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[tuple[int, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of neighbors to return
            filter_fn: Optional filter function (id) -> bool
            
        Returns:
            List of (id, distance) tuples, sorted by distance
        """
        q = _ensure_numpy(query)
        return self._index.search(q, k, filter_fn)
    
    def search_batch(
        self,
        queries: Sequence[Vector],
        k: int = 10,
    ) -> List[List[tuple[int, float]]]:
        """
        Batch search for nearest neighbors.
        
        Args:
            queries: Query vectors
            k: Number of neighbors per query
            
        Returns:
            List of result lists
        """
        queries_np = np.array([_ensure_numpy(q) for q in queries], dtype=np.float32)
        return self._index.search_batch(queries_np, k)
    
    def __len__(self) -> int:
        """Return number of vectors in index."""
        return len(self._index)
    
    def save(self, path: str) -> None:
        """Save index to file."""
        self._index.save(path)
    
    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        """Load index from file."""
        native = _NativeHNSWIndex.load(path)
        instance = cls.__new__(cls)
        instance._index = native
        instance._dimension = native.dimension
        return instance
```

---

## 5. ATLAS Embedding Provider Adapter

```python
# python/neuralcore/compat/atlas_embedding.py
"""
Drop-in replacement for ATLAS HuggingFaceEmbeddingProvider.

Usage:
    # In ATLAS configuration or code:
    from neuralcore.compat.atlas_embedding import NeuralCoreEmbeddingProvider
    
    provider = NeuralCoreEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    embeddings = await provider.embed(["Hello world"])
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Import the engine (native or fallback)
from neuralcore.embedding import EmbeddingEngine


class NeuralCoreEmbeddingProvider:
    """
    ATLAS-compatible embedding provider using neuralcore-runtime.
    
    This class provides the same interface as HuggingFaceEmbeddingProvider
    but uses the Rust ONNX runtime for inference.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_path: Optional[str] = None,
        max_batch_size: int = 32,
        max_sequence_length: int = 512,
        normalize: bool = True,
        device: str = "auto",  # Ignored in ONNX (uses available providers)
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the embedding provider.
        
        Args:
            model_name: HuggingFace model name or ONNX model name
            model_path: Path to local ONNX model (overrides model_name)
            max_batch_size: Maximum batch size for inference
            max_sequence_length: Maximum sequence length
            normalize: Whether to normalize embeddings
            device: Ignored (ONNX uses available execution providers)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.normalize = normalize
        self._cache_dir = cache_dir or str(Path.home() / ".cache" / "neuralcore")
        
        # Resolve model path
        if model_path:
            self._model_path = model_path
        else:
            self._model_path = self._resolve_model_path(model_name)
        
        # Initialize engine
        self._engine = EmbeddingEngine(
            model_path=self._model_path,
            max_batch_size=max_batch_size,
            normalize=normalize,
        )
        
        self._dimension: Optional[int] = None
        logger.info(f"Initialized NeuralCoreEmbeddingProvider with {model_name}")
    
    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model name to ONNX file path."""
        # Check common model mappings
        onnx_models = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "multi-qa-MiniLM-L6-cos-v1": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        }
        
        resolved = onnx_models.get(model_name, model_name)
        
        # Check if it's a local path
        if Path(resolved).exists():
            return resolved
        
        # Download from HuggingFace Hub
        cache_path = Path(self._cache_dir) / resolved.replace("/", "_")
        if cache_path.exists():
            # Find ONNX file
            onnx_files = list(cache_path.glob("*.onnx"))
            if onnx_files:
                return str(onnx_files[0])
        
        # Need to download - use huggingface_hub
        try:
            from huggingface_hub import snapshot_download
            
            cache_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                resolved,
                local_dir=str(cache_path),
                allow_patterns=["*.onnx", "*.json", "*.txt"],
            )
            
            onnx_files = list(cache_path.glob("*.onnx"))
            if onnx_files:
                return str(onnx_files[0])
            
            raise FileNotFoundError(f"No ONNX model found for {model_name}")
            
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models. "
                "Install with: pip install huggingface_hub"
            )
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Get dimension from a test embedding
            test_emb = self._engine.embed(["test"])
            self._dimension = len(test_emb[0])
        return self._dimension
    
    async def embed(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar (ignored)
            
        Returns:
            List of embedding vectors
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._embed_sync,
            texts,
        )
        return result
    
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            embeddings = self._engine.embed(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous embedding (for non-async contexts).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._embed_sync(texts)
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query (convenience method).
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed([query])
        return embeddings[0]
    
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed documents (alias for embed).
        
        Args:
            documents: List of document texts
            
        Returns:
            List of embedding vectors
        """
        return await self.embed(documents)
    
    def __repr__(self) -> str:
        return f"NeuralCoreEmbeddingProvider(model={self.model_name}, dim={self.dimension})"
```

---

## 6. ATLAS NCB Adapter

```python
# python/neuralcore/compat/atlas_ncb.py
"""
Drop-in replacement for ATLAS Neural Cognitive Bus (NCB).

Usage:
    # Replace NCB import:
    # from core.messaging.NCB import NCB
    from neuralcore.compat.atlas_ncb import NeuralCoreNCB as NCB
    
    ncb = NCB()
    await ncb.start()
    await ncb.publish("channel", {"key": "value"})
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Import messaging (native or fallback)
from neuralcore.messaging import MessageBus, Message, Priority


class NeuralCoreNCB:
    """
    ATLAS-compatible Neural Cognitive Bus using neuralcore-runtime.
    
    Provides the same interface as the Python NCB but uses
    Rust lock-free queues for message dispatch.
    """
    
    # Standard ATLAS channels
    CHANNELS = [
        "llm_request",
        "llm_response", 
        "tool_request",
        "tool_response",
        "user_input",
        "user_output",
        "memory_read",
        "memory_write",
        "system_event",
        "error",
        "debug",
        "metrics",
        # ... (add all 36 channels)
    ]
    
    def __init__(
        self,
        persistence_path: Optional[str] = None,
        worker_count: int = 4,
        enable_metrics: bool = True,
    ):
        """
        Initialize the NCB.
        
        Args:
            persistence_path: Path for SQLite persistence (optional)
            worker_count: Number of dispatch worker threads
            enable_metrics: Whether to collect metrics
        """
        self._bus = MessageBus(
            persistence_path=persistence_path,
            worker_count=worker_count,
        )
        self._enable_metrics = enable_metrics
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._started = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Create standard channels
        for channel in self.CHANNELS:
            self._bus.create_channel(channel)
    
    async def start(self) -> None:
        """Start the NCB."""
        if self._started:
            return
        
        self._loop = asyncio.get_event_loop()
        self._started = True
        logger.info("NeuralCoreNCB started")
    
    async def stop(self) -> None:
        """Stop the NCB."""
        if not self._started:
            return
        
        self._bus.stop()
        self._started = False
        logger.info("NeuralCoreNCB stopped")
    
    async def publish(
        self,
        channel: str,
        payload: Dict[str, Any],
        priority: int = 50,
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Publish a message to a channel.
        
        Args:
            channel: Target channel
            payload: Message payload (dict)
            priority: Priority (0=highest, 100=lowest)
            trace_id: Optional trace ID for distributed tracing
            correlation_id: Optional correlation ID for request/response
        """
        if not self._started:
            raise RuntimeError("NCB not started")
        
        self._bus.publish(
            channel=channel,
            payload=payload,
            priority=priority,
            trace_id=trace_id,
        )
    
    async def subscribe(
        self,
        channel: str,
        callback: Callable[[Dict[str, Any]], Any],
        module_name: str = "unknown",
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> str:
        """
        Subscribe to a channel.
        
        Args:
            channel: Channel to subscribe to
            callback: Callback function (can be sync or async)
            module_name: Name of subscribing module
            filter_fn: Optional filter function
            
        Returns:
            Subscription ID
        """
        if not self._started:
            raise RuntimeError("NCB not started")
        
        # Wrap callback to handle async
        async def async_callback(msg: Message) -> None:
            payload = msg.payload
            
            # Apply filter
            if filter_fn and not filter_fn(payload):
                return
            
            # Call callback
            if asyncio.iscoroutinefunction(callback):
                await callback(payload)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, callback, payload
                )
        
        # Register with native bus
        # Note: This needs bridge between sync Rust callbacks and async Python
        # Implementation depends on pyo3-asyncio setup
        
        subscription_id = f"{module_name}_{channel}_{id(callback)}"
        
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        self._subscriptions[channel].append((subscription_id, async_callback))
        
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Unsubscribe from a channel.
        
        Args:
            subscription_id: Subscription ID from subscribe()
        """
        for channel, subs in self._subscriptions.items():
            self._subscriptions[channel] = [
                (sid, cb) for sid, cb in subs if sid != subscription_id
            ]
    
    async def request(
        self,
        channel: str,
        payload: Dict[str, Any],
        response_channel: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Send request and wait for response.
        
        Args:
            channel: Request channel
            payload: Request payload
            response_channel: Response channel (auto-generated if not provided)
            timeout: Timeout in seconds
            
        Returns:
            Response payload
        """
        import uuid
        
        correlation_id = str(uuid.uuid4())
        response_channel = response_channel or f"{channel}_response"
        
        # Create future for response
        response_future: asyncio.Future = asyncio.Future()
        
        async def response_handler(msg: Dict[str, Any]) -> None:
            if msg.get("correlation_id") == correlation_id:
                response_future.set_result(msg)
        
        # Subscribe to response channel
        sub_id = await self.subscribe(
            response_channel,
            response_handler,
            module_name="request_waiter",
        )
        
        try:
            # Publish request
            await self.publish(
                channel,
                {**payload, "correlation_id": correlation_id},
            )
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        finally:
            await self.unsubscribe(sub_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bus metrics."""
        return self._bus.metrics()
    
    def get_queue_size(self, channel: str) -> int:
        """Get queue size for a channel."""
        return self._bus.queue_size(channel) or 0
    
    async def __aenter__(self) -> "NeuralCoreNCB":
        await self.start()
        return self
    
    async def __aexit__(self, *args) -> None:
        await self.stop()
```

---

## 7. ATLAS Configuration Integration

```python
# python/neuralcore/compat/config.py
"""
Configuration integration with ATLAS ConfigManager.

Usage:
    from neuralcore.compat.config import configure_from_atlas
    
    # Reads from ATLAS config.yaml
    configure_from_atlas("/path/to/atlas")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_atlas_config(atlas_root: str) -> Dict[str, Any]:
    """Load ATLAS configuration."""
    config_path = Path(atlas_root) / "config.yaml"
    
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def configure_from_atlas(
    atlas_root: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Configure neuralcore from ATLAS settings.
    
    Args:
        atlas_root: Path to ATLAS root directory
        config: Pre-loaded config dictionary (optional)
        
    Returns:
        Neuralcore configuration dictionary
    """
    if config is None:
        atlas_root = atlas_root or os.environ.get("ATLAS_ROOT", ".")
        config = load_atlas_config(atlas_root)
    
    neuralcore_config = {}
    
    # Extract embedding settings
    embedding_config = config.get("embedding", {})
    if embedding_config:
        neuralcore_config["embedding"] = {
            "model_name": embedding_config.get("model", "all-MiniLM-L6-v2"),
            "max_batch_size": embedding_config.get("batch_size", 32),
            "normalize": embedding_config.get("normalize", True),
            "cache_dir": embedding_config.get("cache_dir"),
        }
    
    # Extract messaging settings
    messaging_config = config.get("messaging", {}) or config.get("ncb", {})
    if messaging_config:
        neuralcore_config["messaging"] = {
            "persistence_path": messaging_config.get("persistence_path"),
            "worker_count": messaging_config.get("workers", 4),
            "redis_url": messaging_config.get("redis_url"),
            "kafka_bootstrap": messaging_config.get("kafka_bootstrap"),
        }
    
    # Extract vector settings
    vector_config = config.get("vectors", {})
    if vector_config:
        neuralcore_config["vectors"] = {
            "hnsw_m": vector_config.get("hnsw_m", 16),
            "hnsw_ef_construction": vector_config.get("hnsw_ef_construction", 200),
            "hnsw_ef_search": vector_config.get("hnsw_ef_search", 50),
        }
    
    return neuralcore_config


def apply_config(neuralcore_config: Dict[str, Any]) -> None:
    """Apply configuration globally."""
    import neuralcore
    
    # Store config for lazy initialization
    neuralcore._config = neuralcore_config
```

---

## 8. Migration Script

```python
#!/usr/bin/env python3
"""
Migration script to switch ATLAS from Python implementations to neuralcore-runtime.

Usage:
    python -m neuralcore.migrate --atlas-root /path/to/atlas --dry-run
    python -m neuralcore.migrate --atlas-root /path/to/atlas --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


# Import patterns to replace
REPLACEMENTS = [
    # Embedding provider
    (
        r"from modules\.storage\.embeddings\.huggingface import HuggingFaceEmbeddingProvider",
        "from neuralcore.compat.atlas_embedding import NeuralCoreEmbeddingProvider as HuggingFaceEmbeddingProvider",
    ),
    (
        r"from modules\.storage\.embeddings import HuggingFaceEmbeddingProvider",
        "from neuralcore.compat.atlas_embedding import NeuralCoreEmbeddingProvider as HuggingFaceEmbeddingProvider",
    ),
    # NCB
    (
        r"from core\.messaging\.NCB import NCB",
        "from neuralcore.compat.atlas_ncb import NeuralCoreNCB as NCB",
    ),
    (
        r"from core\.messaging import NCB",
        "from neuralcore.compat.atlas_ncb import NeuralCoreNCB as NCB",
    ),
    # Vector operations (if any direct imports)
    (
        r"from modules\.storage\.retrieval\.cache import cosine_similarity",
        "from neuralcore import cosine_similarity",
    ),
]


def find_python_files(root: Path) -> List[Path]:
    """Find all Python files in directory."""
    files = []
    for pattern in ["**/*.py"]:
        files.extend(root.glob(pattern))
    
    # Exclude tests and migrations
    files = [
        f for f in files 
        if "test" not in str(f).lower() 
        and "migration" not in str(f).lower()
        and "__pycache__" not in str(f)
    ]
    
    return files


def analyze_file(path: Path) -> List[Tuple[int, str, str]]:
    """
    Analyze a file for potential replacements.
    
    Returns:
        List of (line_number, old_text, new_text) tuples
    """
    changes = []
    
    with open(path) as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        for pattern, replacement in REPLACEMENTS:
            if re.search(pattern, line):
                new_line = re.sub(pattern, replacement, line)
                changes.append((i, line.strip(), new_line.strip()))
    
    return changes


def apply_changes(path: Path, changes: List[Tuple[int, str, str]]) -> None:
    """Apply changes to a file."""
    with open(path) as f:
        content = f.read()
    
    for _, old_text, new_text in changes:
        content = content.replace(old_text, new_text)
    
    with open(path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Migrate ATLAS to neuralcore-runtime")
    parser.add_argument("--atlas-root", required=True, help="Path to ATLAS root directory")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        print("Please specify --dry-run or --apply")
        sys.exit(1)
    
    atlas_root = Path(args.atlas_root)
    if not atlas_root.exists():
        print(f"ATLAS root not found: {atlas_root}")
        sys.exit(1)
    
    print(f"Scanning {atlas_root}...")
    
    files = find_python_files(atlas_root)
    print(f"Found {len(files)} Python files")
    
    total_changes = 0
    files_changed = 0
    
    for path in files:
        changes = analyze_file(path)
        
        if changes:
            files_changed += 1
            total_changes += len(changes)
            
            print(f"\n{path}:")
            for line_num, old_text, new_text in changes:
                print(f"  Line {line_num}:")
                print(f"    - {old_text}")
                print(f"    + {new_text}")
            
            if args.apply:
                apply_changes(path, changes)
                print("  ✓ Applied")
    
    print(f"\n{'Would change' if args.dry_run else 'Changed'} {total_changes} imports in {files_changed} files")
    
    if args.dry_run:
        print("\nRun with --apply to apply changes")


if __name__ == "__main__":
    main()
```

---

## 9. Benchmarks

```python
# benchmarks/compare_performance.py
"""
Performance comparison between Python and neuralcore implementations.
"""

import asyncio
import time
from typing import List

import numpy as np


def benchmark_vectors():
    """Benchmark vector operations."""
    print("\n=== Vector Operations ===")
    
    # Generate test data
    dim = 384
    n_vectors = 10000
    vectors = [np.random.randn(dim).astype(np.float32).tolist() for _ in range(n_vectors)]
    query = np.random.randn(dim).astype(np.float32).tolist()
    
    # Python baseline
    from modules.storage.retrieval.cache import cosine_similarity as py_cosine
    
    start = time.perf_counter()
    for v in vectors[:1000]:
        py_cosine(query, v)
    py_time = time.perf_counter() - start
    print(f"Python cosine (1000 calls): {py_time*1000:.2f}ms")
    
    # Neuralcore
    from neuralcore import cosine_similarity as nc_cosine, cosine_similarity_batch
    
    start = time.perf_counter()
    for v in vectors[:1000]:
        nc_cosine(query, v)
    nc_time = time.perf_counter() - start
    print(f"Neuralcore cosine (1000 calls): {nc_time*1000:.2f}ms")
    print(f"Speedup: {py_time/nc_time:.1f}x")
    
    # Batch comparison
    start = time.perf_counter()
    for v in vectors:
        py_cosine(query, v)
    py_batch_time = time.perf_counter() - start
    print(f"\nPython batch (10K vectors): {py_batch_time*1000:.2f}ms")
    
    start = time.perf_counter()
    cosine_similarity_batch(query, vectors)
    nc_batch_time = time.perf_counter() - start
    print(f"Neuralcore batch (10K vectors): {nc_batch_time*1000:.2f}ms")
    print(f"Speedup: {py_batch_time/nc_batch_time:.1f}x")


async def benchmark_messaging():
    """Benchmark message bus."""
    print("\n=== Message Bus ===")
    
    n_messages = 100000
    
    # Python NCB
    from core.messaging.NCB import NCB
    
    py_ncb = NCB()
    await py_ncb.start()
    
    received = 0
    
    async def handler(msg):
        nonlocal received
        received += 1
    
    await py_ncb.subscribe("test", handler)
    
    start = time.perf_counter()
    for i in range(n_messages):
        await py_ncb.publish("test", {"i": i})
    
    # Wait for all messages
    while received < n_messages:
        await asyncio.sleep(0.01)
    
    py_time = time.perf_counter() - start
    await py_ncb.stop()
    
    print(f"Python NCB ({n_messages} messages): {py_time:.2f}s ({n_messages/py_time:.0f} msg/s)")
    
    # Neuralcore
    from neuralcore.compat.atlas_ncb import NeuralCoreNCB
    
    nc_ncb = NeuralCoreNCB()
    await nc_ncb.start()
    
    received = 0
    await nc_ncb.subscribe("test", handler)
    
    start = time.perf_counter()
    for i in range(n_messages):
        await nc_ncb.publish("test", {"i": i})
    
    while received < n_messages:
        await asyncio.sleep(0.01)
    
    nc_time = time.perf_counter() - start
    await nc_ncb.stop()
    
    print(f"Neuralcore NCB ({n_messages} messages): {nc_time:.2f}s ({n_messages/nc_time:.0f} msg/s)")
    print(f"Speedup: {py_time/nc_time:.1f}x")


async def benchmark_embedding():
    """Benchmark embedding generation."""
    print("\n=== Embedding ===")
    
    texts = [f"This is test sentence number {i}" for i in range(1000)]
    
    # Python (sentence-transformers)
    from modules.storage.embeddings.huggingface import HuggingFaceEmbeddingProvider
    
    py_provider = HuggingFaceEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    
    start = time.perf_counter()
    await py_provider.embed(texts)
    py_time = time.perf_counter() - start
    print(f"Python sentence-transformers (1000 texts): {py_time:.2f}s")
    
    # Neuralcore
    from neuralcore.compat.atlas_embedding import NeuralCoreEmbeddingProvider
    
    nc_provider = NeuralCoreEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    
    start = time.perf_counter()
    await nc_provider.embed(texts)
    nc_time = time.perf_counter() - start
    print(f"Neuralcore ONNX (1000 texts): {nc_time:.2f}s")
    print(f"Speedup: {py_time/nc_time:.1f}x")


async def main():
    print("=" * 60)
    print("ATLAS / neuralcore-runtime Performance Comparison")
    print("=" * 60)
    
    benchmark_vectors()
    await benchmark_messaging()
    await benchmark_embedding()
    
    print("\n" + "=" * 60)
    print("Benchmark complete")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 10. Acceptance Criteria

- [ ] neuralcore package installs cleanly via pip
- [ ] All ATLAS tests pass with neuralcore replacements
- [ ] Pure Python fallback works when native not available
- [ ] Vector operations achieve ≥10x speedup
- [ ] Embedding generation achieves ≥3x speedup
- [ ] Message bus achieves ≥20x throughput improvement
- [ ] Migration script correctly updates imports
- [ ] No regression in functionality
- [ ] Documentation is complete and accurate

---

## 11. Rollout Plan

### Phase 1: Opt-in Testing (Week 1-2)
- Install neuralcore as optional dependency
- Add feature flag in config.yaml: `neuralcore.enabled: false`
- Run parallel validation in development

### Phase 2: Default with Fallback (Week 3-4)
- Enable by default: `neuralcore.enabled: true`
- Keep Python fallback active
- Monitor performance metrics

### Phase 3: Full Migration (Week 5+)
- Remove feature flag
- Update all imports permanently
- Deprecate Python implementations (keep for compatibility)

---

## 12. Troubleshooting

### Common Issues

**Import Error: `neuralcore._native` not found**
- Cause: Rust extension not compiled
- Fix: `pip install neuralcore-runtime[fallback]` or build from source with `maturin develop`

**Performance regression**
- Cause: Falling back to Python
- Fix: Check `neuralcore.is_native_available()` and rebuild extension

**ONNX model not found**
- Cause: Model not downloaded or cached
- Fix: Ensure HuggingFace Hub access or provide local model path

**Async callback issues**
- Cause: Mixing sync/async callbacks
- Fix: Use `pyo3-asyncio` bridge or wrap callbacks appropriately
