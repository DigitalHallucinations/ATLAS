"""OpenTelemetry tracing instrumentation for RAG pipeline.

Provides distributed tracing capabilities to monitor and debug
the RAG retrieval pipeline across all stages:
- Query routing
- Embedding generation
- Vector search
- Reranking
- Context compression
- Response generation

Supports both OpenTelemetry SDK and a lightweight fallback tracer.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, Optional, TypeVar, Union
import functools
import logging
import time
import uuid

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(Enum):
    """Types of spans in the RAG pipeline."""
    
    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a span."""
    
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context information for a span.
    
    Attributes:
        trace_id: Unique identifier for the trace.
        span_id: Unique identifier for this span.
        parent_span_id: Optional parent span identifier.
    """
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    
    @classmethod
    def create(cls, parent: Optional["SpanContext"] = None) -> "SpanContext":
        """Create a new span context, optionally with a parent."""
        if parent:
            return cls(
                trace_id=parent.trace_id,
                span_id=uuid.uuid4().hex[:16],
                parent_span_id=parent.span_id,
            )
        return cls(
            trace_id=uuid.uuid4().hex[:32],
            span_id=uuid.uuid4().hex[:16],
        )


@dataclass
class Span:
    """A span representing a unit of work in the pipeline.
    
    Attributes:
        name: Name of the operation.
        context: Span context with trace/span IDs.
        kind: Type of span.
        start_time: Start timestamp in nanoseconds.
        end_time: End timestamp in nanoseconds.
        status: Span status.
        attributes: Key-value attributes.
        events: List of events during the span.
    """
    
    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    start_time: int = field(default_factory=lambda: time.time_ns())
    end_time: Optional[int] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: list = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        self.status_message = message
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time_ns(),
            "attributes": attributes or {},
        })
    
    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the span."""
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )
        self.set_status(SpanStatus.ERROR, str(exception))
    
    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = time.time_ns()
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) / 1_000_000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events,
        }


class TracerBackend(ABC):
    """Abstract base class for tracer backends."""
    
    @abstractmethod
    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[SpanContext] = None,
    ) -> Span:
        """Start a new span."""
        ...
    
    @abstractmethod
    def end_span(self, span: Span) -> None:
        """End a span and export it."""
        ...
    
    @abstractmethod
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        ...


class InMemoryTracerBackend(TracerBackend):
    """In-memory tracer backend for development and testing.
    
    Stores spans in memory with optional logging output.
    """
    
    def __init__(
        self,
        *,
        log_spans: bool = True,
        max_spans: int = 1000,
    ):
        """Initialize in-memory tracer.
        
        Args:
            log_spans: Whether to log span completion.
            max_spans: Maximum spans to keep in memory.
        """
        self.log_spans = log_spans
        self.max_spans = max_spans
        self._spans: list[Span] = []
        self._active_span: Optional[Span] = None
        self._span_stack: list[Span] = []
    
    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[SpanContext] = None,
    ) -> Span:
        """Start a new span."""
        # Use parent from argument or current active span
        parent_ctx = parent
        if parent_ctx is None and self._active_span:
            parent_ctx = self._active_span.context
        
        context = SpanContext.create(parent_ctx)
        
        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
        )
        
        # Push to stack
        if self._active_span:
            self._span_stack.append(self._active_span)
        self._active_span = span
        
        return span
    
    def end_span(self, span: Span) -> None:
        """End a span and store it."""
        span.end()
        
        # Store span
        self._spans.append(span)
        
        # Trim if needed
        if len(self._spans) > self.max_spans:
            self._spans = self._spans[-self.max_spans:]
        
        # Pop from stack
        if self._active_span is span:
            if self._span_stack:
                self._active_span = self._span_stack.pop()
            else:
                self._active_span = None
        
        # Log if enabled
        if self.log_spans:
            logger.debug(
                "Span completed: %s (%.2fms) trace=%s span=%s",
                span.name,
                span.duration_ms,
                span.context.trace_id[:8],
                span.context.span_id[:8],
            )
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self._active_span
    
    def get_spans(self) -> list[Span]:
        """Get all stored spans."""
        return self._spans.copy()
    
    def clear(self) -> None:
        """Clear all stored spans."""
        self._spans.clear()


class OpenTelemetryBackend(TracerBackend):
    """OpenTelemetry SDK backend for production tracing.
    
    Integrates with OpenTelemetry for distributed tracing,
    exporting spans to configured backends (Jaeger, Zipkin, etc.).
    """
    
    def __init__(self, service_name: str = "atlas-rag"):
        """Initialize OpenTelemetry backend.
        
        Args:
            service_name: Name of the service for tracing.
        """
        self._service_name = service_name
        self._tracer: Optional[Any] = None
        self._initialized = False
        self._span_map: Dict[str, Any] = {}  # Map our spans to OTel spans
    
    def _initialize(self) -> bool:
        """Initialize OpenTelemetry SDK."""
        if self._initialized:
            return self._tracer is not None
        
        self._initialized = True
        
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )
            
            # Create resource
            resource = Resource.create({
                "service.name": self._service_name,
                "service.version": "1.0.0",
            })
            
            # Create tracer provider
            provider = TracerProvider(resource=resource)
            
            # Add console exporter for development
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(provider)
            
            # Get tracer
            self._tracer = trace.get_tracer(__name__)
            
            logger.info("OpenTelemetry tracing initialized")
            return True
            
        except ImportError:
            logger.warning("OpenTelemetry SDK not available")
            return False
        except Exception as exc:
            logger.warning("Failed to initialize OpenTelemetry: %s", exc)
            return False
    
    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[SpanContext] = None,
    ) -> Span:
        """Start a new span."""
        # Create our span representation
        parent_ctx = parent
        context = SpanContext.create(parent_ctx)
        
        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
        )
        
        # Start OTel span if available
        if self._initialize() and self._tracer:
            try:
                from opentelemetry.trace import SpanKind as OTelSpanKind
                
                otel_kind_map = {
                    SpanKind.INTERNAL: OTelSpanKind.INTERNAL,
                    SpanKind.CLIENT: OTelSpanKind.CLIENT,
                    SpanKind.SERVER: OTelSpanKind.SERVER,
                    SpanKind.PRODUCER: OTelSpanKind.PRODUCER,
                    SpanKind.CONSUMER: OTelSpanKind.CONSUMER,
                }
                
                otel_span = self._tracer.start_span(
                    name,
                    kind=otel_kind_map.get(kind, OTelSpanKind.INTERNAL),
                    attributes=attributes,
                )
                
                self._span_map[span.context.span_id] = otel_span
                
            except Exception as exc:
                logger.debug("Failed to start OTel span: %s", exc)
        
        return span
    
    def end_span(self, span: Span) -> None:
        """End a span and export it."""
        span.end()
        
        # End OTel span if available
        otel_span = self._span_map.pop(span.context.span_id, None)
        if otel_span:
            try:
                # Set final attributes
                for key, value in span.attributes.items():
                    otel_span.set_attribute(key, value)
                
                # Set status
                if span.status == SpanStatus.ERROR:
                    from opentelemetry.trace import StatusCode
                    otel_span.set_status(
                        StatusCode.ERROR,
                        span.status_message or "Error",
                    )
                
                otel_span.end()
                
            except Exception as exc:
                logger.debug("Failed to end OTel span: %s", exc)
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        # OpenTelemetry manages its own context
        if self._tracer:
            try:
                from opentelemetry import trace
                otel_span = trace.get_current_span()
                if otel_span:
                    # Return a wrapped representation
                    ctx = otel_span.get_span_context()
                    return Span(
                        name="current",
                        context=SpanContext(
                            trace_id=format(ctx.trace_id, "032x"),
                            span_id=format(ctx.span_id, "016x"),
                        ),
                    )
            except Exception:
                pass
        return None


class RAGTracer:
    """RAG-specific tracer with pipeline-aware instrumentation.
    
    Provides high-level tracing methods for common RAG operations
    and automatic attribute collection.
    
    Example:
        tracer = RAGTracer()
        
        with tracer.trace_retrieval("user query") as span:
            span.set_attribute("top_k", 10)
            results = retriever.search(query)
            span.set_attribute("num_results", len(results))
    """
    
    def __init__(
        self,
        *,
        backend: Optional[TracerBackend] = None,
        service_name: str = "atlas-rag",
        use_opentelemetry: bool = True,
    ):
        """Initialize RAG tracer.
        
        Args:
            backend: Custom tracer backend. Auto-selected if None.
            service_name: Service name for tracing.
            use_opentelemetry: Prefer OpenTelemetry if available.
        """
        if backend:
            self._backend = backend
        elif use_opentelemetry:
            # Try OpenTelemetry, fall back to in-memory
            otel_backend = OpenTelemetryBackend(service_name)
            if otel_backend._initialize():
                self._backend = otel_backend
            else:
                self._backend = InMemoryTracerBackend()
        else:
            self._backend = InMemoryTracerBackend()
        
        self._service_name = service_name
    
    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Start a span as a context manager.
        
        Args:
            name: Span name.
            kind: Span kind.
            attributes: Initial attributes.
            
        Yields:
            The active span.
        """
        span = self._backend.start_span(name, kind=kind, attributes=attributes)
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            self._backend.end_span(span)
    
    @contextmanager
    def trace_embedding(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        batch_size: int = 1,
    ) -> Generator[Span, None, None]:
        """Trace an embedding generation operation.
        
        Args:
            text: Text being embedded (for length tracking).
            model: Embedding model name.
            batch_size: Number of texts in batch.
            
        Yields:
            The active span.
        """
        attributes = {
            "rag.operation": "embedding",
            "rag.embedding.text_length": len(text),
            "rag.embedding.batch_size": batch_size,
        }
        if model:
            attributes["rag.embedding.model"] = model
        
        with self.start_span("rag.embedding", attributes=attributes) as span:
            yield span
    
    @contextmanager
    def trace_search(
        self,
        query: str,
        *,
        search_type: str = "hybrid",
        top_k: int = 10,
        knowledge_base_ids: Optional[list] = None,
    ) -> Generator[Span, None, None]:
        """Trace a search operation.
        
        Args:
            query: Search query.
            search_type: Type of search (dense, lexical, hybrid).
            top_k: Number of results requested.
            knowledge_base_ids: Knowledge bases being searched.
            
        Yields:
            The active span.
        """
        attributes = {
            "rag.operation": "search",
            "rag.search.query_length": len(query),
            "rag.search.type": search_type,
            "rag.search.top_k": top_k,
        }
        if knowledge_base_ids:
            attributes["rag.search.kb_count"] = len(knowledge_base_ids)
        
        with self.start_span("rag.search", attributes=attributes) as span:
            yield span
    
    @contextmanager
    def trace_rerank(
        self,
        query: str,
        num_candidates: int,
        *,
        model: Optional[str] = None,
    ) -> Generator[Span, None, None]:
        """Trace a reranking operation.
        
        Args:
            query: Rerank query.
            num_candidates: Number of candidates being reranked.
            model: Reranker model name.
            
        Yields:
            The active span.
        """
        attributes = {
            "rag.operation": "rerank",
            "rag.rerank.query_length": len(query),
            "rag.rerank.num_candidates": num_candidates,
        }
        if model:
            attributes["rag.rerank.model"] = model
        
        with self.start_span("rag.rerank", attributes=attributes) as span:
            yield span
    
    @contextmanager
    def trace_compression(
        self,
        input_length: int,
        *,
        strategy: str = "extractive",
        target_ratio: float = 0.5,
    ) -> Generator[Span, None, None]:
        """Trace a context compression operation.
        
        Args:
            input_length: Length of input text.
            strategy: Compression strategy.
            target_ratio: Target compression ratio.
            
        Yields:
            The active span.
        """
        attributes = {
            "rag.operation": "compression",
            "rag.compression.input_length": input_length,
            "rag.compression.strategy": strategy,
            "rag.compression.target_ratio": target_ratio,
        }
        
        with self.start_span("rag.compression", attributes=attributes) as span:
            yield span
    
    @contextmanager
    def trace_retrieval(
        self,
        query: str,
        *,
        knowledge_base_ids: Optional[list] = None,
    ) -> Generator[Span, None, None]:
        """Trace a full retrieval pipeline operation.
        
        Args:
            query: User query.
            knowledge_base_ids: Knowledge bases being searched.
            
        Yields:
            The active span.
        """
        attributes = {
            "rag.operation": "retrieval",
            "rag.retrieval.query_length": len(query),
        }
        if knowledge_base_ids:
            attributes["rag.retrieval.kb_count"] = len(knowledge_base_ids)
        
        with self.start_span(
            "rag.retrieval",
            kind=SpanKind.SERVER,
            attributes=attributes,
        ) as span:
            yield span
    
    @contextmanager
    def trace_generation(
        self,
        context_length: int,
        *,
        model: Optional[str] = None,
    ) -> Generator[Span, None, None]:
        """Trace a response generation operation.
        
        Args:
            context_length: Length of context provided.
            model: LLM model name.
            
        Yields:
            The active span.
        """
        attributes = {
            "rag.operation": "generation",
            "rag.generation.context_length": context_length,
        }
        if model:
            attributes["rag.generation.model"] = model
        
        with self.start_span("rag.generation", attributes=attributes) as span:
            yield span
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self._backend.get_current_span()


# Global tracer instance
_global_tracer: Optional[RAGTracer] = None


def get_tracer(
    *,
    service_name: str = "atlas-rag",
    use_opentelemetry: bool = True,
) -> RAGTracer:
    """Get or create the global RAG tracer.
    
    Args:
        service_name: Service name for tracing.
        use_opentelemetry: Prefer OpenTelemetry if available.
        
    Returns:
        The global RAGTracer instance.
    """
    global _global_tracer
    
    if _global_tracer is None:
        _global_tracer = RAGTracer(
            service_name=service_name,
            use_opentelemetry=use_opentelemetry,
        )
    
    return _global_tracer


def trace_operation(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.
    
    Args:
        name: Span name.
        kind: Span kind.
        attributes: Static attributes to add.
        
    Returns:
        Decorated function.
        
    Example:
        @trace_operation("my_function")
        def my_function(arg1, arg2):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_span(name, kind=kind, attributes=attributes) as span:
                span.set_attribute("function", func.__name__)
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_span(name, kind=kind, attributes=attributes) as span:
                span.set_attribute("function", func.__name__)
                return await func(*args, **kwargs)
        
        if asyncio_iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


def asyncio_iscoroutinefunction(func: Any) -> bool:
    """Check if function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


__all__ = [
    "SpanKind",
    "SpanStatus",
    "SpanContext",
    "Span",
    "TracerBackend",
    "InMemoryTracerBackend",
    "OpenTelemetryBackend",
    "RAGTracer",
    "get_tracer",
    "trace_operation",
]
