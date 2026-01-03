"""RAG metrics collection module.

Provides metrics instrumentation for monitoring RAG pipeline performance:
- Latency histograms for each pipeline stage
- Throughput counters
- Quality gauges (relevancy scores, compression ratios)
- Cache hit/miss rates

Supports Prometheus metrics via OpenTelemetry and a fallback in-memory collector.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar
import functools
import logging
import threading
import time

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class MetricType(Enum):
    """Types of metrics."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Container for a metric observation.
    
    Attributes:
        name: Metric name.
        value: Metric value.
        labels: Label key-value pairs.
        timestamp: Observation timestamp.
    """
    
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""
    
    @abstractmethod
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        ...
    
    @abstractmethod
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        ...
    
    @abstractmethod
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        ...
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        ...


class InMemoryMetricsBackend(MetricsBackend):
    """In-memory metrics backend for development and testing.
    
    Stores metrics in memory with thread-safe access.
    """
    
    def __init__(self):
        """Initialize in-memory metrics backend."""
        self._lock = threading.RLock()
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    def _labels_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Convert labels dict to a hashable key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        key = self._labels_key(labels)
        with self._lock:
            self._counters[name][key] += value
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        key = self._labels_key(labels)
        with self._lock:
            self._gauges[name][key] = value
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        key = self._labels_key(labels)
        with self._lock:
            self._histograms[name][key].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        with self._lock:
            result = {
                "counters": {},
                "gauges": {},
                "histograms": {},
            }
            
            for name, values in self._counters.items():
                result["counters"][name] = dict(values)
            
            for name, values in self._gauges.items():
                result["gauges"][name] = dict(values)
            
            for name, values in self._histograms.items():
                hist_stats = {}
                for key, observations in values.items():
                    if observations:
                        hist_stats[key] = {
                            "count": len(observations),
                            "sum": sum(observations),
                            "min": min(observations),
                            "max": max(observations),
                            "mean": sum(observations) / len(observations),
                        }
                result["histograms"][name] = hist_stats
            
            return result
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


class OpenTelemetryMetricsBackend(MetricsBackend):
    """OpenTelemetry metrics backend for production monitoring.
    
    Integrates with OpenTelemetry for metrics export to
    Prometheus, OTLP collectors, etc.
    """
    
    def __init__(self, service_name: str = "atlas-rag"):
        """Initialize OpenTelemetry metrics backend.
        
        Args:
            service_name: Service name for metrics.
        """
        self._service_name = service_name
        self._meter: Optional[Any] = None
        self._counters: Dict[str, Any] = {}
        self._gauges: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
        self._initialized = False
        
        # Fallback for when OTel is not available
        self._fallback = InMemoryMetricsBackend()
    
    def _initialize(self) -> bool:
        """Initialize OpenTelemetry metrics."""
        if self._initialized:
            return self._meter is not None
        
        self._initialized = True
        
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.metrics.export import (
                PeriodicExportingMetricReader,
                ConsoleMetricExporter,
            )
            
            # Create resource
            resource = Resource.create({
                "service.name": self._service_name,
            })
            
            # Create meter provider with console exporter
            reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=60000,
            )
            provider = MeterProvider(resource=resource, metric_readers=[reader])
            
            # Set global meter provider
            metrics.set_meter_provider(provider)
            
            # Get meter
            self._meter = metrics.get_meter(__name__)
            
            logger.info("OpenTelemetry metrics initialized")
            return True
            
        except ImportError:
            logger.warning("OpenTelemetry SDK not available for metrics")
            return False
        except Exception as exc:
            logger.warning("Failed to initialize OpenTelemetry metrics: %s", exc)
            return False
    
    def _get_counter(self, name: str) -> Any:
        """Get or create a counter instrument."""
        if name not in self._counters and self._meter:
            self._counters[name] = self._meter.create_counter(
                name,
                description=f"Counter for {name}",
            )
        return self._counters.get(name)
    
    def _get_gauge(self, name: str) -> Any:
        """Get or create a gauge instrument."""
        if name not in self._gauges and self._meter:
            # OpenTelemetry uses observable gauges
            # For simplicity, we'll use up-down counters
            self._gauges[name] = self._meter.create_up_down_counter(
                name,
                description=f"Gauge for {name}",
            )
        return self._gauges.get(name)
    
    def _get_histogram(self, name: str) -> Any:
        """Get or create a histogram instrument."""
        if name not in self._histograms and self._meter:
            self._histograms[name] = self._meter.create_histogram(
                name,
                description=f"Histogram for {name}",
            )
        return self._histograms.get(name)
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        if not self._initialize():
            self._fallback.increment_counter(name, value, labels)
            return
        
        counter = self._get_counter(name)
        if counter:
            counter.add(value, labels or {})
        else:
            self._fallback.increment_counter(name, value, labels)
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        if not self._initialize():
            self._fallback.set_gauge(name, value, labels)
            return
        
        # OTel doesn't have direct gauge support, use fallback
        self._fallback.set_gauge(name, value, labels)
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        if not self._initialize():
            self._fallback.observe_histogram(name, value, labels)
            return
        
        histogram = self._get_histogram(name)
        if histogram:
            histogram.record(value, labels or {})
        else:
            self._fallback.observe_histogram(name, value, labels)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        # Return fallback metrics (OTel exports asynchronously)
        return self._fallback.get_metrics()


class RAGMetrics:
    """RAG-specific metrics with semantic naming.
    
    Provides high-level metrics methods for common RAG operations
    with automatic labeling and timing.
    
    Metric Naming Convention:
    - rag.retrieval.duration_ms - Retrieval latency
    - rag.retrieval.count - Number of retrievals
    - rag.embedding.duration_ms - Embedding latency
    - rag.search.results - Number of search results
    - rag.rerank.duration_ms - Reranking latency
    - rag.compression.ratio - Compression ratio achieved
    - rag.cache.hits - Cache hit count
    - rag.cache.misses - Cache miss count
    
    Example:
        metrics = RAGMetrics()
        
        with metrics.time_operation("retrieval"):
            results = retriever.search(query)
        
        metrics.record_search_results(len(results), search_type="hybrid")
    """
    
    # Metric names
    RETRIEVAL_DURATION = "rag.retrieval.duration_ms"
    RETRIEVAL_COUNT = "rag.retrieval.count"
    EMBEDDING_DURATION = "rag.embedding.duration_ms"
    EMBEDDING_COUNT = "rag.embedding.count"
    SEARCH_DURATION = "rag.search.duration_ms"
    SEARCH_RESULTS = "rag.search.results"
    RERANK_DURATION = "rag.rerank.duration_ms"
    RERANK_COUNT = "rag.rerank.count"
    COMPRESSION_DURATION = "rag.compression.duration_ms"
    COMPRESSION_RATIO = "rag.compression.ratio"
    CACHE_HITS = "rag.cache.hits"
    CACHE_MISSES = "rag.cache.misses"
    CACHE_HIT_RATE = "rag.cache.hit_rate"
    GENERATION_DURATION = "rag.generation.duration_ms"
    GENERATION_TOKENS = "rag.generation.tokens"
    FAITHFULNESS_SCORE = "rag.quality.faithfulness"
    RELEVANCY_SCORE = "rag.quality.relevancy"
    
    def __init__(
        self,
        *,
        backend: Optional[MetricsBackend] = None,
        use_opentelemetry: bool = True,
    ):
        """Initialize RAG metrics.
        
        Args:
            backend: Custom metrics backend. Auto-selected if None.
            use_opentelemetry: Prefer OpenTelemetry if available.
        """
        if backend:
            self._backend = backend
        elif use_opentelemetry:
            self._backend = OpenTelemetryMetricsBackend()
        else:
            self._backend = InMemoryMetricsBackend()
    
    def time_operation(
        self,
        operation: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> "TimerContext":
        """Create a timer context for an operation.
        
        Args:
            operation: Operation name (retrieval, embedding, search, etc.).
            labels: Additional labels.
            
        Returns:
            Timer context manager.
        """
        metric_name = f"rag.{operation}.duration_ms"
        return TimerContext(self._backend, metric_name, labels)
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        self._backend.increment_counter(name, value, labels)
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        self._backend.set_gauge(name, value, labels)
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        self._backend.observe_histogram(name, value, labels)
    
    # High-level convenience methods
    
    def record_retrieval(
        self,
        duration_ms: float,
        *,
        success: bool = True,
        num_results: int = 0,
        knowledge_base: Optional[str] = None,
    ) -> None:
        """Record a retrieval operation.
        
        Args:
            duration_ms: Operation duration in milliseconds.
            success: Whether the operation succeeded.
            num_results: Number of results retrieved.
            knowledge_base: Knowledge base ID.
        """
        labels = {"success": str(success).lower()}
        if knowledge_base:
            labels["kb"] = knowledge_base
        
        self.observe_histogram(self.RETRIEVAL_DURATION, duration_ms, labels)
        self.increment_counter(self.RETRIEVAL_COUNT, labels=labels)
        
        if num_results > 0:
            self.set_gauge(self.SEARCH_RESULTS, num_results, labels)
    
    def record_embedding(
        self,
        duration_ms: float,
        *,
        batch_size: int = 1,
        model: Optional[str] = None,
    ) -> None:
        """Record an embedding operation.
        
        Args:
            duration_ms: Operation duration in milliseconds.
            batch_size: Number of texts embedded.
            model: Embedding model name.
        """
        labels = {"batch_size": str(batch_size)}
        if model:
            labels["model"] = model
        
        self.observe_histogram(self.EMBEDDING_DURATION, duration_ms, labels)
        self.increment_counter(self.EMBEDDING_COUNT, batch_size, labels)
    
    def record_search(
        self,
        duration_ms: float,
        num_results: int,
        *,
        search_type: str = "hybrid",
    ) -> None:
        """Record a search operation.
        
        Args:
            duration_ms: Operation duration in milliseconds.
            num_results: Number of results returned.
            search_type: Type of search (dense, lexical, hybrid).
        """
        labels = {"type": search_type}
        
        self.observe_histogram(self.SEARCH_DURATION, duration_ms, labels)
        self.set_gauge(self.SEARCH_RESULTS, num_results, labels)
    
    def record_rerank(
        self,
        duration_ms: float,
        num_candidates: int,
        *,
        model: Optional[str] = None,
    ) -> None:
        """Record a reranking operation.
        
        Args:
            duration_ms: Operation duration in milliseconds.
            num_candidates: Number of candidates reranked.
            model: Reranker model name.
        """
        labels = {"num_candidates": str(num_candidates)}
        if model:
            labels["model"] = model
        
        self.observe_histogram(self.RERANK_DURATION, duration_ms, labels)
        self.increment_counter(self.RERANK_COUNT, labels=labels)
    
    def record_compression(
        self,
        duration_ms: float,
        compression_ratio: float,
        *,
        strategy: str = "extractive",
    ) -> None:
        """Record a compression operation.
        
        Args:
            duration_ms: Operation duration in milliseconds.
            compression_ratio: Achieved compression ratio.
            strategy: Compression strategy used.
        """
        labels = {"strategy": strategy}
        
        self.observe_histogram(self.COMPRESSION_DURATION, duration_ms, labels)
        self.set_gauge(self.COMPRESSION_RATIO, compression_ratio, labels)
    
    def record_cache_access(
        self,
        hit: bool,
        *,
        cache_type: str = "embedding",
    ) -> None:
        """Record a cache access.
        
        Args:
            hit: Whether the cache was hit.
            cache_type: Type of cache (embedding, query).
        """
        labels = {"cache": cache_type}
        
        if hit:
            self.increment_counter(self.CACHE_HITS, labels=labels)
        else:
            self.increment_counter(self.CACHE_MISSES, labels=labels)
    
    def record_cache_hit_rate(
        self,
        hit_rate: float,
        *,
        cache_type: str = "embedding",
    ) -> None:
        """Record cache hit rate.
        
        Args:
            hit_rate: Hit rate (0.0 to 1.0).
            cache_type: Type of cache.
        """
        labels = {"cache": cache_type}
        self.set_gauge(self.CACHE_HIT_RATE, hit_rate, labels)
    
    def record_generation(
        self,
        duration_ms: float,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None,
    ) -> None:
        """Record a generation operation.
        
        Args:
            duration_ms: Operation duration in milliseconds.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: LLM model name.
        """
        labels = {}
        if model:
            labels["model"] = model
        
        self.observe_histogram(self.GENERATION_DURATION, duration_ms, labels)
        
        if output_tokens > 0:
            self.set_gauge(self.GENERATION_TOKENS, output_tokens, {**labels, "type": "output"})
        if input_tokens > 0:
            self.set_gauge(self.GENERATION_TOKENS, input_tokens, {**labels, "type": "input"})
    
    def record_quality_score(
        self,
        metric: str,
        score: float,
        *,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a quality metric score.
        
        Args:
            metric: Quality metric name (faithfulness, relevancy).
            score: Score value (0.0 to 1.0).
            labels: Additional labels.
        """
        metric_name = f"rag.quality.{metric}"
        self.set_gauge(metric_name, score, labels)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        return self._backend.get_metrics()


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        backend: MetricsBackend,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize timer context.
        
        Args:
            backend: Metrics backend.
            metric_name: Histogram metric name.
            labels: Labels for the metric.
        """
        self._backend = backend
        self._metric_name = metric_name
        self._labels = labels or {}
        self._start_time: Optional[float] = None
    
    def __enter__(self) -> "TimerContext":
        """Start the timer."""
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer and record the observation."""
        if self._start_time is not None:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            
            labels = self._labels.copy()
            labels["success"] = "true" if exc_type is None else "false"
            
            self._backend.observe_histogram(self._metric_name, duration_ms, labels)
    
    def add_label(self, key: str, value: str) -> None:
        """Add a label during the timed operation."""
        self._labels[key] = value


# Global metrics instance
_global_metrics: Optional[RAGMetrics] = None


def get_metrics(
    *,
    use_opentelemetry: bool = True,
) -> RAGMetrics:
    """Get or create the global RAG metrics collector.
    
    Args:
        use_opentelemetry: Prefer OpenTelemetry if available.
        
    Returns:
        The global RAGMetrics instance.
    """
    global _global_metrics
    
    if _global_metrics is None:
        _global_metrics = RAGMetrics(use_opentelemetry=use_opentelemetry)
    
    return _global_metrics


def timed(
    operation: str,
    labels: Optional[Dict[str, str]] = None,
) -> Callable[[F], F]:
    """Decorator to time a function.
    
    Args:
        operation: Operation name for the metric.
        labels: Static labels to add.
        
    Returns:
        Decorated function.
        
    Example:
        @timed("search", {"type": "hybrid"})
        def search(query: str):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with get_metrics().time_operation(operation, labels):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with get_metrics().time_operation(operation, labels):
                return await func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


__all__ = [
    "MetricType",
    "MetricValue",
    "MetricsBackend",
    "InMemoryMetricsBackend",
    "OpenTelemetryMetricsBackend",
    "RAGMetrics",
    "TimerContext",
    "get_metrics",
    "timed",
]
