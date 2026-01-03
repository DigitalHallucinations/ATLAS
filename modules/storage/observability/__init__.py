"""RAG Observability Module.

Provides OpenTelemetry instrumentation for the RAG pipeline including:
- Distributed tracing with spans for each pipeline stage
- Metrics collection for latency, throughput, and quality
- Structured logging with trace context
"""

from modules.storage.observability.tracing import (
    RAGTracer,
    get_tracer,
    trace_operation,
    SpanKind,
)
from modules.storage.observability.metrics import (
    RAGMetrics,
    get_metrics,
    MetricType,
)

__all__ = [
    # Tracing
    "RAGTracer",
    "get_tracer",
    "trace_operation",
    "SpanKind",
    # Metrics
    "RAGMetrics",
    "get_metrics",
    "MetricType",
]
