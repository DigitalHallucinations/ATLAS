# Foundation Issues

> **Epic**: Establish Common Service Patterns
> **Parent**: [README.md](./README.md)

## 📋 Ready for Development

### FND-001: Implement Service Result Types

**Description**: Create the standard `OperationResult` generic and related types to ensure consistent return values across all services.
**Acceptance Criteria**:

- `OperationResult[T]` generic class defined.
- `success`, `data`, `error`, `error_code` fields present.
- Helper methods `success(data)` and `failure(error)` implemented.
- Unit tests covering success/failure cases.
**File**: `core/services/common/types.py`

### FND-002: Implement Domain Event & Actor Types

**Description**: Define the standard `DomainEvent` and `Actor` structures for the event bus and permission system.
**Context**: Needed for decoupling services and enforcing security.
**Acceptance Criteria**:

- `DomainEvent` frozen dataclass with `event_type`, `entity_id`, `actor`, `timestamp`.
- `Actor` dataclass with `type` (user, system, etc.), `id`, and `permissions`.
- JSON serialization helpers if needed.
**File**: `core/services/common/types.py`

### FND-003: Implement Permission Checker

**Description**: Create the component responsible for validating if an Actor has required permissions.
**Acceptance Criteria**:

- `PermissionChecker` class initialized.
- `require(actor, permission)` raises `PermissionDeniedError` if missing.
- `has_permission(actor, permission)` returns bool.
- `require_any` and `require_all` helpers.
**File**: `core/services/common/permissions.py`

### FND-004: Standard Exception Hierarchy

**Description**: Define the base exceptions for the service layer to avoid leaking implementation details (e.g., SQL errors).
**Acceptance Criteria**:

- `ServiceError` (base)
- `ValidationError`, `NotFoundError`, `ConflictError`, `PermissionDeniedError`.
- All inherit from `ServiceError` (or appropriate stdlib mapping).
**File**: `core/services/common/exceptions.py`

### FND-005: Service & Repository Protocols

**Description**: Define `Protocol` or ABCs for what constitutes a "Service" and "Repository" to enforce structure.
**Acceptance Criteria**:

- `Repository[T]` protocol with `get`, `save`, `delete`.
- `Service` protocol (marker or with common lifecycle methods like `initialize`).
**File**: `core/services/common/protocols.py`

### FND-006: Global Export & Init

**Description**: expose the new clean types from `core.services`.
**Acceptance Criteria**:

- `core/services/common/__init__.py` exports all the above.
- `core/services/__init__.py` re-exports them for easy access (`from core.services import OperationResult`).

### FND-007: Validate Messaging Integration

**Description**: Ensure existing `core.messaging.AgentBus` is compatible with new `DomainEvent` types.
**Acceptance Criteria**:

- Verify if `AgentMessage` can wrap or extend `DomainEvent`.
- Define standard topic/channel naming convention for service events.
- Create adapter if necessary to publish `DomainEvents` to the Bus.
