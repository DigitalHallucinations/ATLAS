# Foundation & Common Patterns

> **Status**: 📋 Planning  
> **Priority**: Critical (Prerequisite for all other phases)  
> **Complexity**: Low  
> **Effort**: 1-2 days  
> **Created**: 2026-01-07

---

## Overview

Establish the common service patterns, types, and infrastructure that all ATLAS services will use. This phase MUST be completed before any other service work begins.

---

## Deliverables

### 1. Common Types Package

Create `core/services/common/` with shared types:

- [ ] **1.1** Create `types.py`:

  ```python
  @dataclass
  class OperationResult(Generic[T]):
      success: bool
      data: T | None = None
      error: str | None = None
      error_code: str | None = None
  
  @dataclass(frozen=True)
  class DomainEvent:
      event_type: str
      entity_id: UUID
      tenant_id: str
      timestamp: datetime
      actor: Literal["user", "system", "agent", "sync", "job", "task"]
      metadata: dict | None = None
  
  @dataclass
  class Actor:
      type: Literal["user", "system", "agent", "sync", "job", "task"]
      id: str
      permissions: set[str]
  ```

- [ ] **1.2** Create `permissions.py`:

  ```python
  class PermissionChecker:
      def require(self, actor: Actor, permission: str) -> None: ...
      def has_permission(self, actor: Actor, permission: str) -> bool: ...
      def require_any(self, actor: Actor, permissions: list[str]) -> None: ...
      def require_all(self, actor: Actor, permissions: list[str]) -> None: ...
  ```

- [ ] **1.3** Create `exceptions.py`:
  - `ServiceError` - Base exception
  - `PermissionDeniedError` - Actor lacks permission
  - `ValidationError` - Input validation failed
  - `NotFoundError` - Entity not found
  - `ConflictError` - Concurrent modification

- [ ] **1.4** Create `protocols.py`:
  - Common repository protocol patterns
  - Common service interface patterns

### 2. Package Exports

- [ ] **2.1** Create `core/services/common/__init__.py` with exports
- [ ] **2.2** Update `core/services/__init__.py` to export common types

### 3. Documentation

- [ ] **3.1** Create `docs/developer/service-pattern.md`:
  - Standard service structure
  - Required methods and signatures
  - MessageBus event patterns
  - Permission check patterns
  - Testing patterns
  - Example service implementation

### 4. Generator Script (Optional)

- [ ] **4.1** Create `scripts/generate_service.py`:
  - Scaffolds new service package
  - Creates types.py, permissions.py, service.py, tests
  - Ensures consistent structure

---

## Standard Service Pattern

All services should follow this structure:

```python
# core/services/{domain}/types.py
from core.services.common import OperationResult, DomainEvent, Actor

@dataclass(frozen=True)
class DomainSpecificEvent(DomainEvent):
    """Typed event for this domain."""
    ...

# core/services/{domain}/permissions.py
from core.services.common import PermissionChecker

class DomainPermissionChecker(PermissionChecker):
    PERMISSIONS = {
        "domain:read": "View domain entities",
        "domain:write": "Create/update domain entities",
        "domain:delete": "Delete domain entities",
        "domain:admin": "Full administrative access",
    }

# core/services/{domain}/service.py
class DomainService:
    def __init__(
        self,
        *,
        repository: DomainRepositoryProtocol,
        permission_checker: DomainPermissionChecker,
        message_bus: MessageBus,
        logger: Logger,
        tenant_id: str,
    ) -> None:
        self._repository = repository
        self._permissions = permission_checker
        self._bus = message_bus
        self._logger = logger
        self._tenant_id = tenant_id
    
    def operation(self, actor: Actor, ...) -> OperationResult[T]:
        self._permissions.require(actor, "domain:write")
        # ... implementation
        self._bus.publish(event)
        return OperationResult(success=True, data=result)
```

---

## Files to Create

| File | Purpose |
| ---- | ------- |
| `core/services/common/__init__.py` | Package exports |
| `core/services/common/types.py` | OperationResult, Actor, DomainEvent |
| `core/services/common/permissions.py` | PermissionChecker base |
| `core/services/common/exceptions.py` | Service exceptions |
| `core/services/common/protocols.py` | Common protocols |
| `docs/developer/service-pattern.md` | Pattern documentation |
| `docs/developer/services/` | Service-specific docs folder |
| `scripts/generate_service.py` | Service scaffolding (optional) |

---

## Files to Modify

| File | Changes |
| ---- | ------- |
| `core/services/__init__.py` | Export common types |

---

## Validation Checklist

- [ ] All common types are exported
- [ ] Pattern documentation is complete with examples
- [ ] Existing services (RAGService, ConversationService) can adopt pattern
- [ ] Tests exist for PermissionChecker
- [ ] Tests exist for OperationResult helpers

---

## Success Criteria

1. Any developer can create a new service by following the pattern doc
2. Common types are importable from `core.services.common`
3. Existing services can be incrementally migrated
4. Generator script (if created) produces valid service scaffolds

---

## Open Questions

| Question | Options | Decision |
| -------- | ------- | -------- |
| Should Actor include tenant_id? | Yes - explicit / No - from context | TBD |
| Should we use Pydantic instead of dataclasses? | Pydantic for validation / dataclasses for simplicity | TBD |
| How to handle async vs sync operations? | All async / Mixed based on need | TBD |
| Should PermissionChecker be async for DB-backed permissions? | Yes - future-proof / No - keep simple | TBD |

---

## Dependencies

- `core/messaging/` - MessageBus (exists)
- None - this is the foundation

---

## Downstream Dependents

ALL other service phases depend on this phase completing first.
