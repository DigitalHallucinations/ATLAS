# Budget Services

> **Status**: ✅ Complete  
> **Priority**: High  
> **Complexity**: High  
> **Effort**: 1 week  
> **Created**: 2026-01-07  
> **Updated**: 2026-01-09

---

## Overview

Split `modules/budget/manager.py` (1338 lines) into three focused services:

1. **BudgetPolicyService** - Budget policy CRUD, enforcement, pre-flight checks ✅
2. **BudgetTrackingService** - Usage recording, aggregation, reporting ✅
3. **BudgetAlertService** - Alert configuration, threshold monitoring, notifications ✅

---

## Architectural Decisions

| Decision | Choice | Rationale |
| -------- | ------ | --------- |
| Service structure | Split into 3 services | Current manager is too large (1338 lines) |
| Facade pattern | No facade - direct service usage | Clean break, no deprecated code |
| Alert evaluation | Background task | Don't block operations on alert checks |
| Usage recording | Synchronous | Cost must be recorded before operation completes |

---

## Phases

### Phase 1: BudgetPolicyService ✅

- [x] **1.1** Create `core/services/budget/` package structure
- [x] **1.2** Create `core/services/budget/policy_service.py`:
  - `create_policy(actor, policy)` - Create budget policy
  - `update_policy(actor, policy_id, updates)` - Modify policy
  - `delete_policy(actor, policy_id)` - Remove policy
  - `get_policy(policy_id)` - Get single policy
  - `list_policies(scope, filters)` - List policies
  - `check_budget(operation, estimated_cost)` - Pre-flight check
- [x] **1.3** Extract policy validation logic from manager
- [x] **1.4** Add MessageBus events:
  - `budget.policy_created`
  - `budget.policy_updated`
  - `budget.policy_deleted`
- [x] **1.5** Write unit tests (36 tests passing)
- [x] **1.6** Global budget ceiling enforcement:
  - No scoped policy can exceed global limit
  - Combined scoped policies cannot exceed global limit
  - Validation on create and update

### Phase 2: BudgetTrackingService ✅

- [x] **2.1** Create `core/services/budget/tracking_service.py`:
  - `record_usage(operation, cost, metadata)` - Record spend
  - `get_usage_summary(scope, period)` - Aggregate usage
  - `get_spend_by_category(scope, period)` - Category breakdown
  - `get_spend_by_provider(scope, period)` - Provider breakdown
  - `get_spend_trend(scope, periods)` - Historical trend
- [x] **2.2** Extract tracking logic from manager
- [x] **2.3** Add MessageBus events:
  - `budget.usage_recorded`
  - `budget.threshold_reached`
- [x] **2.4** Write unit tests (24 tests passing)

### Phase 3: BudgetAlertService ✅

- [x] **3.1** Create `core/services/budget/alert_service.py`:
  - `configure_alert(actor, alert_config)` - Set up alert
  - `remove_alert(actor, alert_id)` - Remove alert
  - `list_alerts(scope)` - List configured alerts
  - `get_active_alerts()` - Get triggered alerts
  - `acknowledge_alert(actor, alert_id)` - Dismiss alert
  - `evaluate_alerts()` - Check all thresholds (background)
- [x] **3.2** Extract alert logic from manager
- [x] **3.3** Add MessageBus events:
  - `budget.alert_triggered`
  - `budget.alert_acknowledged`
  - `budget.limit_exceeded`
  - `budget.approaching_limit`
- [x] **3.4** Register background task for alert evaluation
- [x] **3.5** Write unit tests (23 tests passing)

### Phase 4: Integration & Migration ✅

- [x] **4.1** Factory pattern for service instantiation
  - Created `core/services/budget/factory.py`
  - `get_policy_service()`, `get_tracking_service()`, `get_alert_service()`
  - `initialize_services()`, `shutdown_services()`, `reset_services()`
- [x] **4.2** Global budget ceiling enforcement:
  - Added `_validate_global_ceiling` method to `BudgetPolicyService`
  - No scoped policy can exceed global limit
  - Combined scoped policies at same level cannot exceed global
  - Validation on both create and update operations
  - 2 new tests added (38 total policy service tests)
- [x] **4.3** Update `modules/budget/api.py` to use new services
  - API layer now uses `get_policy_service()`, `get_tracking_service()`, `get_alert_service()`
  - All API functions delegate to appropriate service
- [x] **4.4** Update GTKUI/Budget_manager to use services
  - Dashboard, policy editor, alerts panel, reports view, usage history updated
- [x] **4.5** Deprecation phase completed - all deprecated code removed
- [x] **4.6** Integration tests added (94 service tests total)

### Phase 5: BudgetManager Removal ✅

- [x] **5.1** Refactored persistence layer to be independent of BudgetManager
  - Factory now uses `BudgetStore.get_instance()` directly
  - No dependency on BudgetManager class
- [x] **5.2** Updated `modules/budget/api.py` to use new services
  - Lifecycle functions renamed: `initialize_budget_services()`, `shutdown_budget_services()`
  - All deprecated aliases removed
- [x] **5.3** Updated `modules/budget/tracking.py` to use new services
  - UsageTracker now uses `api` module functions
  - Removed `_budget_manager` field
- [x] **5.4** Updated `modules/budget/alerts.py` to use BudgetStore directly
- [x] **5.5** Updated `modules/budget/reports.py` - ReportGenerator no longer takes budget_manager param
- [x] **5.6** Updated `modules/budget/integration.py` to use new API
- [x] **5.7** Deleted `modules/budget/manager.py` - The deprecated BudgetManager class
- [x] **5.8** Deleted `tests/budget/test_manager.py` - Tests for the deprecated class
- [x] **5.9** Updated `tests/budget/test_reports.py` to use new API
- [x] **5.10** Final cleanup - 243 budget tests passing

---

## MessageBus Events

| Event Type | Payload | Emitted By |
| ---------- | ------- | ---------- |
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

## Files Created

| File | Purpose | Status |
| ---- | ------- | ------ |
| `core/services/budget/__init__.py` | Package exports | ✅ |
| `core/services/budget/types.py` | Dataclasses, events | ✅ |
| `core/services/budget/factory.py` | Service instantiation | ✅ |
| `core/services/budget/policy_service.py` | BudgetPolicyService | ✅ |
| `core/services/budget/tracking_service.py` | BudgetTrackingService | ✅ |
| `core/services/budget/alert_service.py` | BudgetAlertService | ✅ |
| `modules/budget/scope_hierarchy.py` | Scope resolution | ✅ |
| `modules/budget/policy_matcher.py` | Policy matching | ✅ |
| `tests/services/budget/test_policy_service.py` | Policy service tests | ✅ |
| `tests/services/budget/test_tracking_service.py` | Tracking service tests | ✅ |
| `tests/services/budget/test_alert_service.py` | Alert service tests | ✅ |
| `tests/services/budget/test_factory.py` | Factory tests | ✅ |
| `tests/services/budget/test_integration.py` | Integration tests | ✅ |
| `tests/budget/test_scope_hierarchy.py` | Scope hierarchy tests | ✅ |
| `tests/budget/test_policy_matcher.py` | Policy matcher tests | ✅ |

## Files Deleted

| File | Reason |
| ---- | ------ |
| `modules/budget/manager.py` | Replaced by services |
| `tests/budget/test_manager.py` | Tests for deprecated class |

---

## Files Modified

| File | Changes | Status |
| ---- | ------- | ------ |
| `core/services/__init__.py` | Export budget services | ✅ |
| `modules/budget/__init__.py` | Updated exports, removed BudgetManager | ✅ |
| `modules/budget/api.py` | Uses services, new lifecycle functions | ✅ |
| `modules/budget/tracking.py` | Uses api module instead of BudgetManager | ✅ |
| `modules/budget/alerts.py` | Uses BudgetStore directly | ✅ |
| `modules/budget/reports.py` | ReportGenerator no longer takes budget_manager | ✅ |
| `modules/budget/integration.py` | Uses new API functions | ✅ |
| `GTKUI/Budget_manager/*.py` | Use new services | ✅ |
| `tests/budget/test_reports.py` | Updated for new API | ✅ |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/budget/` - Repository layer (exists)
- `core/messaging/` - MessageBus for events
- `modules/background_tasks/` - Alert evaluation scheduling

---

## Success Criteria

1. ✅ Manager split into 3 focused services
2. ✅ All services follow standard pattern
3. ✅ Background alert evaluation working
4. ✅ Usage recording doesn't block operations
5. ✅ Pre-flight budget checks functional
6. ✅ UI updated to use services
7. ✅ 243 tests passing (149 budget + 94 service tests)

---

## Open Questions

| Question | Options | Decision |
| -------- | ------- | -------- |
| Should budget checks block LLM calls? | Block / Warn only / Configurable | **Configurable** - per-policy `hard_limit_action` with defaults and notifications |
| Alert evaluation frequency? | 1 min / 5 min / On usage only | **On usage only** for Phase 1; background evaluation in Phase 3 |
| Support for budget rollover? | Yes with config / No | **Yes with config** - preserve existing `rollover_enabled` / `rollover_max_percent` |
| Multi-tenant budget isolation? | Strict / Shared pools | **Strict by default** - `BudgetPermissionChecker` enforces `policy.scope_id == actor.tenant_id` |
