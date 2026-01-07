# Analytics Service

> **Status**: 📋 Planning  
> **Priority**: Low  
> **Complexity**: Medium  
> **Effort**: 3-4 days  
> **Created**: 2026-01-07

---

## Overview

Consolidate `modules/analytics/` into `core/services/analytics/`:

- Event recording and aggregation
- Usage statistics
- Performance metrics
- Cost breakdown

---

## Phases

### Phase 1: Service Creation

- [ ] **1.1** Create `core/services/analytics/` package
- [ ] **1.2** Implement AnalyticsService:
  - `record_event(event_type, data)` - Record metric
  - `get_summary(metric, period)` - Aggregate data
  - `get_usage_stats(entity_type, entity_id, period)` - Usage
  - `get_performance_metrics(period)` - Latency, throughput
  - `get_cost_breakdown(period)` - Cost analytics
  - `export_data(format, period)` - Export for analysis
- [ ] **1.3** Add MessageBus subscriptions (collect from other events)
- [ ] **1.4** Background aggregation jobs
- [ ] **1.5** Write unit tests

---

## Service Methods

```python
class AnalyticsService:
    # Recording
    def record_event(self, event_type: str, data: dict) -> OperationResult[None]: ...
    
    # Queries
    def get_summary(self, metric: str, period: TimePeriod) -> OperationResult[MetricSummary]: ...
    def get_usage_stats(
        self,
        entity_type: str,
        entity_id: UUID | None,
        period: TimePeriod
    ) -> OperationResult[UsageStats]: ...
    def get_performance_metrics(self, period: TimePeriod) -> OperationResult[PerformanceMetrics]: ...
    def get_cost_breakdown(self, period: TimePeriod) -> OperationResult[CostBreakdown]: ...
    
    # Export
    def export_data(
        self,
        format: ExportFormat,
        period: TimePeriod,
        metrics: list[str] | None = None
    ) -> OperationResult[ExportResult]: ...
    
    # Aggregation
    def run_aggregation(self, period: TimePeriod) -> OperationResult[None]: ...
```

---

## MessageBus Subscriptions

The Analytics Service subscribes to events from other services to build metrics:

- `job.*` - Job execution metrics
- `task.*` - Task completion rates
- `tool.*` - Tool usage patterns
- `budget.*` - Cost tracking
- `provider.*` - Provider performance
- `conversation.*` - Conversation analytics

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/analytics/__init__.py` | Package exports |
| `core/services/analytics/types.py` | Dataclasses |
| `core/services/analytics/service.py` | AnalyticsService |
| `tests/services/analytics/` | Service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/analytics/` - Existing analytics (consolidate)
- `core/messaging/` - MessageBus subscriptions
- `modules/background_tasks/` - Aggregation scheduling

---

## Success Criteria

1. Analytics operations centralized
2. Real-time event collection working
3. Background aggregation functional
4. Export feature working
5. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Data retention period? | 30 days / 90 days / Configurable | TBD |
| Aggregation granularity? | Hourly / Daily / Both | TBD |
| Export formats? | CSV / JSON / Parquet | TBD |
| Privacy considerations for analytics? | Anonymize / Full / Configurable | TBD |
