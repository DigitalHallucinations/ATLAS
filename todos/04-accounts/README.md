# User Account Service

> **Status**: 📋 Planning  
> **Priority**: High  
> **Complexity**: Medium  
> **Effort**: 3-5 days  
> **Created**: 2026-01-07

---

## Overview

Migrate `modules/user_accounts/user_account_service.py` to `core/services/accounts/` with enhanced patterns:

- Standardized service structure
- Permission checks with Actor
- MessageBus events
- Password policy enforcement
- Credential storage abstraction

---

## Phases

### Phase 1: Service Migration

- [ ] **1.1** Create `core/services/accounts/` package:
  - `__init__.py`
  - `types.py` - Account types, events
  - `permissions.py` - Account permission checker
  - `service.py` - Migrate from modules/user_accounts/
  - `password_policy.py` - Extract password policy logic
  - `credential_store.py` - Extract credential storage
- [ ] **1.2** Add `OperationResult` return types
- [ ] **1.3** Add MessageBus events:
  - `account.created`
  - `account.updated`
  - `account.deleted`
  - `account.password_changed`
  - `account.locked`
  - `account.unlocked`
  - `account.login_succeeded`
  - `account.login_failed`
- [ ] **1.4** Add permission checks (actor parameter)
- [ ] **1.5** Write unit tests

### Phase 2: UI Integration → [40-ui-integration](../40-ui-integration/)

> UI integration tasks moved to consolidated UI sprint.
- [ ] **2.3** Update GTKUI/UserAccounts to use service
- [ ] **2.4** Deprecate old import path with warning

---

## Service Methods

```python
class UserAccountService:
    # Account CRUD
    def create_account(self, actor: Actor, account: AccountCreate) -> OperationResult[Account]: ...
    def get_account(self, actor: Actor, account_id: UUID) -> OperationResult[Account]: ...
    def update_account(self, actor: Actor, account_id: UUID, updates: AccountUpdate) -> OperationResult[Account]: ...
    def delete_account(self, actor: Actor, account_id: UUID) -> OperationResult[None]: ...
    def list_accounts(self, actor: Actor, filters: AccountFilters) -> OperationResult[list[Account]]: ...
    
    # Authentication
    def authenticate(self, username: str, password: str) -> OperationResult[AuthToken]: ...
    def change_password(self, actor: Actor, account_id: UUID, old_password: str, new_password: str) -> OperationResult[None]: ...
    def reset_password(self, actor: Actor, account_id: UUID) -> OperationResult[str]: ...
    
    # Account state
    def lock_account(self, actor: Actor, account_id: UUID, reason: str) -> OperationResult[None]: ...
    def unlock_account(self, actor: Actor, account_id: UUID) -> OperationResult[None]: ...
    
    # Password policy
    def validate_password(self, password: str) -> OperationResult[PasswordValidation]: ...
    def get_password_policy(self) -> PasswordPolicy: ...
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `account.created` | `AccountEvent` | UserAccountService |
| `account.updated` | `AccountEvent` | UserAccountService |
| `account.deleted` | `AccountEvent` | UserAccountService |
| `account.password_changed` | `AccountPasswordEvent` | UserAccountService |
| `account.locked` | `AccountLockEvent` | UserAccountService |
| `account.unlocked` | `AccountLockEvent` | UserAccountService |
| `account.login_succeeded` | `AccountLoginEvent` | UserAccountService |
| `account.login_failed` | `AccountLoginEvent` | UserAccountService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/accounts/__init__.py` | Package exports |
| `core/services/accounts/types.py` | Dataclasses, events |
| `core/services/accounts/permissions.py` | AccountPermissionChecker |
| `core/services/accounts/exceptions.py` | Service exceptions |
| `core/services/accounts/service.py` | UserAccountService |
| `core/services/accounts/password_policy.py` | Password validation |
| `core/services/accounts/credential_store.py` | Credential storage |
| `tests/services/accounts/` | Service tests |

---

## Files to Modify

| File | Changes |
|------|---------|
| `core/services/__init__.py` | Export account services |
| `modules/user_accounts/user_account_service.py` | Deprecate, redirect to core |
| `GTKUI/UserAccounts/*.py` | Use new service |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/user_accounts/` - Repository layer (exists)
- `core/messaging/` - MessageBus for events

---

## Success Criteria

1. Service migrated to `core/services/accounts/`
2. All operations use OperationResult
3. MessageBus events firing on all mutations
4. Password policy extracted and configurable
5. Old import path shows deprecation warning
6. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Should login events include IP/device info? | Yes for audit / No for privacy | TBD |
| Multi-factor authentication support? | Yes in this phase / Future phase | TBD |
| Account session management location? | In this service / Separate SessionService | TBD |
| Credential encryption approach? | libsecret / keyring / custom | TBD |
