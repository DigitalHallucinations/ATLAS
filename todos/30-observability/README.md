# Observability Service

> **Status**: 📋 Planning  
> **Priority**: Medium  
> **Complexity**: Medium  
> **Effort**: 3-4 days  
> **Created**: 2026-01-07

---

## Overview

Consolidate and enhance observability from `modules/storage/observability/`:

- Distributed tracing
- Metrics collection
- Log aggregation
- Health monitoring
- Performance profiling

---

## Phases

### Phase 1: Service Creation

- [ ] **1.1** Create `core/services/observability/` package
- [ ] **1.2** Migrate from `modules/storage/observability/`
- [ ] **1.3** Implement `ObservabilityService`:
  - `start_trace(operation, metadata)` - Begin trace
  - `end_trace(trace_id, result)` - Complete trace
  - `record_metric(name, value, tags)` - Record metric
  - `get_metrics(name, period)` - Query metrics
  - `get_traces(filters)` - Query traces
  - `get_health_status()` - System health
- [ ] **1.4** Write unit tests

### Phase 2: Distributed Tracing

- [ ] **2.1** Trace propagation across services
- [ ] **2.2** Span management
- [ ] **2.3** Trace sampling strategies
- [ ] **2.4** Trace visualization data export

### Phase 3: Metrics & Monitoring

- [ ] **3.1** Standard metrics:
  - Request latency (p50, p95, p99)
  - Error rates
  - Throughput
  - Resource usage
- [ ] **3.2** Custom metric registration
- [ ] **3.3** Metric aggregation
- [ ] **3.4** Alert thresholds

### Phase 4: Health Checks

- [ ] **4.1** Component health registry
- [ ] **4.2** Dependency health checks
- [ ] **4.3** Health endpoint
- [ ] **4.4** Health history

---

## Service Methods

```python
class ObservabilityService:
    # Tracing
    def start_trace(
        self,
        operation: str,
        parent_trace_id: str | None = None,
        metadata: dict | None = None
    ) -> Trace: ...
    
    def end_trace(
        self,
        trace_id: str,
        result: TraceResult
    ) -> None: ...
    
    def add_span(
        self,
        trace_id: str,
        name: str,
        metadata: dict | None = None
    ) -> Span: ...
    
    # Metrics
    def record_metric(
        self,
        name: str,
        value: float,
        tags: dict | None = None
    ) -> None: ...
    
    def get_metrics(
        self,
        name: str,
        period: TimePeriod,
        aggregation: Aggregation = Aggregation.AVG
    ) -> list[MetricPoint]: ...
    
    # Health
    def register_health_check(
        self,
        component: str,
        checker: Callable[[], HealthStatus]
    ) -> None: ...
    
    def get_health_status(self) -> SystemHealth: ...
    
    def get_component_health(self, component: str) -> HealthStatus: ...
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/observability/__init__.py` | Package exports |
| `core/services/observability/types.py` | Dataclasses |
| `core/services/observability/service.py` | ObservabilityService |
| `core/services/observability/tracing.py` | Distributed tracing |
| `core/services/observability/metrics.py` | Metrics collection |
| `core/services/observability/health.py` | Health checks |
| `tests/services/observability/` | Service tests |

---

## Files to Modify/Consolidate

| File | Action |
|------|--------|
| `modules/storage/observability/` | Migrate to core/services |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- All other services (for instrumentation)

---

## Success Criteria

1. Traces capture request flow
2. Metrics provide visibility
3. Health status accurate
4. Performance overhead minimal (<5%)
5. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Trace storage backend? | In-memory / PostgreSQL / External (Jaeger) | TBD |
| Metric retention? | 24h / 7d / 30d / Configurable | TBD |
| Sampling rate for high-volume? | 1% / 10% / Adaptive | TBD |
| External export format? | OpenTelemetry / Prometheus / Both | TBD |
