# Budget Service Issues

> **Epic**: Split Budget Manager
> **Parent**: [README.md](./README.md)

## 📋 Ready for Development

### BUD-001: Budget Service Scaffold

**Description**: Create `core/services/budget`.
**Acceptance Criteria**:

- `policies.py`, `tracking.py`, `alerts.py` modules created.
- `types.py` extracted from `modules/budget/manager.py`.

### BUD-002: Extract Policy Logic

**Description**: Move policy definition and validation logic to `BudgetPolicyService`.
**Acceptance Criteria**:

- CRUD for Budget Policies.
- Validation that policies don't conflict.

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
