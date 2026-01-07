# User Account Service Issues

> **Epic**: Migrate User Accounts to Core Service
> **Parent**: [README.md](./README.md)
> **Dependency**: [FND-001](../00-foundation/ISSUES.md), [FND-002](../00-foundation/ISSUES.md)

## 📋 Ready for Development

### ACC-001: Scaffold Account Service Package

**Description**: Create the directory structure and base files for the new service.
**Acceptance Criteria**:

- `core/services/accounts/` created.
- `__init__.py`, `types.py`, `service.py`, `permissions.py` created (empty/stubbed).

### ACC-002: Define Account Types & Events

**Description**: Define `UserAccount` domain model and events using the new distinct types.
**Acceptance Criteria**:

- `events.py` defines `AccountCreated`, `AccountLocked`, etc.
- `types.py` defines the pure data model (decoupled from DB model if possible, or using Pydantic).

### ACC-003: Migrate Password Policy

**Description**: Extract password validation and hashing logic from `modules/user_accounts` into strict policies.
**Acceptance Criteria**:

- `PasswordPolicy` class implementing rules (length, complexity).
- `PasswordHasher` protocol/implementation.
- decoupled from the massive service class.

### ACC-004: Implement Core Account Operations

**Description**: Implement create, update, delete, and authenticate methods in `AccountService`.
**Acceptance Criteria**:

- `create_account` returns `OperationResult[UserAccount]`.
- `authenticate` returns `OperationResult[Session/Token]`.
- Proper event emission on success.
- Uses `PermissionChecker` (even if just `system` allowed for now).

### ACC-005: Backward Compatibility Adapter

**Description**: Create an adapter so existing code using `modules/user_accounts` can leverage the new service without full rewrite.
**Acceptance Criteria**:

- `modules/user_accounts/user_account_service.py` modified to wrap/call `core.services.accounts.AccountService`.
- Deprecation warning added to old module.
