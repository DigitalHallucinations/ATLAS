# Budget Service Issues

> **Epic**: Split Budget Manager
> **Parent**: [README.md](./README.md)

## ✅ Completed

### BUD-001: Budget Service Scaffold ✅

**Description**: Create `core/services/budget`.
**Acceptance Criteria**:

- ✅ `policy_service.py`, `types.py`, `exceptions.py`, `permissions.py` modules created.
- ✅ `types.py` with DTOs and domain events.

**Completed**: 2026-01-08
**Files Created**:
- `core/services/budget/__init__.py`
- `core/services/budget/types.py`
- `core/services/budget/exceptions.py`
- `core/services/budget/permissions.py`
- `core/services/budget/policy_service.py`

### BUD-002: Extract Policy Logic ✅

**Description**: Move policy definition and validation logic to `BudgetPolicyService`.
**Acceptance Criteria**:

- ✅ CRUD for Budget Policies (`create_policy`, `get_policy`, `update_policy`, `delete_policy`, `list_policies`).
- ✅ Validation that policies don't conflict (scope uniqueness checks).
- ✅ Pre-flight budget check (`check_budget`).
- ✅ Tenant isolation via `BudgetPermissionChecker`.
- ✅ Event publishing for policy lifecycle.

**Completed**: 2026-01-08
**Tests**: 36 unit tests passing (`tests/services/budget/test_policy_service.py`)

---

## 📋 Ready for Development

### BUD-003: Extract Tracking Logic

**Description**: Move usage recording to `BudgetTrackingService`.
**Acceptance Criteria**:

- `record_usage(actor, token_count, model_cost)`.
- Method must be high-performance (critical path).
- `get_current_usage(period)` aggregation.

### BUD-004: Extract Alerting Logic

**Description**: Move alert checking to `BudgetAlertService`.
**Acceptance Criteria**:

- `check_thresholds(usage_data)`.
- `configure_alert(policy_id, threshold)`.
- Decoupled from the recording path (possibly async).
