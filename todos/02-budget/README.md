# Budget Services

> **Status**: 📋 Planning  
> **Priority**: High  
> **Complexity**: High  
> **Effort**: 1 week  
> **Created**: 2026-01-07

---

## Overview

Split `modules/budget/manager.py` (1338 lines) into three focused services:

1. **BudgetPolicyService** - Budget policy CRUD, enforcement, pre-flight checks
2. **BudgetTrackingService** - Usage recording, aggregation, reporting
3. **BudgetAlertService** - Alert configuration, threshold monitoring, notifications

---

## Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Service structure | Split into 3 services | Current manager is too large (1338 lines) |
| Facade pattern | Optional BudgetManager facade | Backwards compatibility if needed |
| Alert evaluation | Background task | Don't block operations on alert checks |
| Usage recording | Synchronous | Cost must be recorded before operation completes |

---

## Phases

### Phase 1: BudgetPolicyService

- [ ] **1.1** Create `core/services/budget/` package structure
- [ ] **1.2** Create `core/services/budget/policy_service.py`:
  - `create_policy(actor, policy)` - Create budget policy
  - `update_policy(actor, policy_id, updates)` - Modify policy
  - `delete_policy(actor, policy_id)` - Remove policy
  - `get_policy(policy_id)` - Get single policy
  - `list_policies(scope, filters)` - List policies
  - `check_budget(operation, estimated_cost)` - Pre-flight check
- [ ] **1.3** Extract policy validation logic from manager
- [ ] **1.4** Add MessageBus events:
  - `budget.policy_created`
  - `budget.policy_updated`
  - `budget.policy_deleted`
- [ ] **1.5** Write unit tests

### Phase 2: BudgetTrackingService

- [ ] **2.1** Create `core/services/budget/tracking_service.py`:
  - `record_usage(operation, cost, metadata)` - Record spend
  - `get_usage_summary(scope, period)` - Aggregate usage
  - `get_spend_by_category(scope, period)` - Category breakdown
  - `get_spend_by_provider(scope, period)` - Provider breakdown
  - `get_spend_trend(scope, periods)` - Historical trend
- [ ] **2.2** Extract tracking logic from manager
- [ ] **2.3** Add MessageBus events:
  - `budget.usage_recorded`
  - `budget.threshold_reached`
- [ ] **2.4** Write unit tests

### Phase 3: BudgetAlertService

- [ ] **3.1** Create `core/services/budget/alert_service.py`:
  - `configure_alert(actor, alert_config)` - Set up alert
  - `remove_alert(actor, alert_id)` - Remove alert
  - `list_alerts(scope)` - List configured alerts
  - `get_active_alerts()` - Get triggered alerts
  - `acknowledge_alert(actor, alert_id)` - Dismiss alert
  - `evaluate_alerts()` - Check all thresholds (background)
- [ ] **3.2** Extract alert logic from manager
- [ ] **3.3** Add MessageBus events:
  - `budget.alert_triggered`
  - `budget.alert_acknowledged`
  - `budget.limit_exceeded`
  - `budget.approaching_limit`
- [ ] **3.4** Register background task for alert evaluation
- [ ] **3.5** Write unit tests

### Phase 4: Integration & Migration

- [ ] **4.1** Optional: Keep thin `BudgetManager` as facade
- [ ] **4.2** Update all callers to use new services
- [ ] **4.3** Update GTKUI/Budget_manager to use services
- [ ] **4.4** Add deprecation warnings on old paths
- [ ] **4.5** Write integration tests

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `budget.policy_created` | `BudgetPolicyEvent` | BudgetPolicyService |
| `budget.policy_updated` | `BudgetPolicyEvent` | BudgetPolicyService |
| `budget.policy_deleted` | `BudgetPolicyEvent` | BudgetPolicyService |
| `budget.usage_recorded` | `BudgetUsageEvent` | BudgetTrackingService |
| `budget.threshold_reached` | `BudgetThresholdEvent` | BudgetTrackingService |
| `budget.alert_triggered` | `BudgetAlertEvent` | BudgetAlertService |
| `budget.alert_acknowledged` | `BudgetAlertEvent` | BudgetAlertService |
| `budget.limit_exceeded` | `BudgetLimitEvent` | BudgetAlertService |
| `budget.approaching_limit` | `BudgetLimitEvent` | BudgetAlertService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/budget/__init__.py` | Package exports |
| `core/services/budget/types.py` | Dataclasses, events |
| `core/services/budget/permissions.py` | BudgetPermissionChecker |
| `core/services/budget/exceptions.py` | Service exceptions |
| `core/services/budget/policy_service.py` | BudgetPolicyService |
| `core/services/budget/tracking_service.py` | BudgetTrackingService |
| `core/services/budget/alert_service.py` | BudgetAlertService |
| `tests/services/budget/` | Service tests |

---

## Files to Modify

| File | Changes |
|------|---------|
| `core/services/__init__.py` | Export budget services |
| `core/ATLAS.py` | Add budget service properties |
| `modules/budget/manager.py` | Deprecate, delegate to services |
| `GTKUI/Budget_manager/*.py` | Use new services |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/budget/` - Repository layer (exists)
- `core/messaging/` - MessageBus for events
- `modules/background_tasks/` - Alert evaluation scheduling

---

## Success Criteria

1. Manager split into 3 focused services
2. All services follow standard pattern
3. Background alert evaluation working
4. Usage recording doesn't block operations
5. Pre-flight budget checks functional
6. UI updated to use services
7. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Should budget checks block LLM calls? | Block / Warn only / Configurable | TBD |
| Alert evaluation frequency? | 1 min / 5 min / On usage only | TBD |
| Support for budget rollover? | Yes with config / No | TBD |
| Multi-tenant budget isolation? | Strict / Shared pools | TBD |
