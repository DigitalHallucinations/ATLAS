# Provider Services

> **Status**: � Service Layer Complete (UI deferred to [40-ui-integration](../40-ui-integration/))  
> **Priority**: Medium  
> **Complexity**: Medium  
> **Effort**: 3-4 days  
> **Created**: 2026-01-07  
> **Updated**: 2026-01-11

---

## Overview

Extract provider management from `GTKUI/Provider_manager/` and `modules/Providers/` into two focused services:

1. **ProviderConfigService** - Provider configuration and credentials ✅
2. **ProviderHealthService** - Provider health monitoring ✅

---

## Phases

### Phase 1: ProviderConfigService ✅

- [x] **1.1** Create `core/services/providers/` package
- [x] **1.2** Implement ProviderConfigService:
  - `list_providers()` - Get all providers
  - `get_provider(provider_id)` - Get provider config
  - `configure_provider(actor, provider_id, config)` - Update config
  - `enable_provider(actor, provider_id)` - Enable
  - `disable_provider(actor, provider_id)` - Disable
  - `set_credentials(actor, provider_id, credentials)` - Set API keys
  - ~~`validate_credentials(provider_id)`~~ - Deferred
  - ~~`get_default_provider(operation_type)`~~ - Deferred
  - ~~`set_default_provider(actor, operation_type, provider_id)`~~ - Deferred
- [x] **1.3** Add MessageBus events:
  - `ProviderConfigEvent`
  - `ProviderStateEvent` (enabled/disabled)
- [x] **1.4** Write unit tests (6 tests)

### Phase 2: ProviderHealthService ✅

- [x] **2.1** Implement ProviderHealthService:
  - `check_health(provider_id)` - Single health check
  - `check_all_health()` - Check all enabled providers
  - `get_status(provider_id)` - Get current status
  - `get_all_statuses()` - Get all statuses
- [x] **2.2** Add MessageBus events:
  - `ProviderHealthEvent`
  - `ProviderErrorEvent`
- [ ] **2.3** Background health check scheduling (stub only)
- [x] **2.4** Write unit tests (5 tests)

### Phase 3: UI Integration → Deferred to [40-ui-integration](../40-ui-integration/)

- [ ] **3.1** Update UI and provider_manager to use services
- [ ] **3.2** Remove direct config access from UI

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `provider.configured` | `ProviderConfigEvent` | ProviderConfigService |
| `provider.enabled` | `ProviderStateEvent` | ProviderConfigService |
| `provider.disabled` | `ProviderStateEvent` | ProviderConfigService |
| `provider.health_changed` | `ProviderHealthEvent` | ProviderHealthService |
| `provider.error` | `ProviderErrorEvent` | ProviderHealthService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/providers/__init__.py` | Package exports |
| `core/services/providers/types.py` | Dataclasses, events |
| `core/services/providers/permissions.py` | ProviderPermissionChecker |
| `core/services/providers/config_service.py` | ProviderConfigService |
| `core/services/providers/health_service.py` | ProviderHealthService |
| `tests/services/providers/` | Service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/Providers/` - Provider definitions
- `core/messaging/` - MessageBus for events
- `modules/background_tasks/` - Health check scheduling

---

## Success Criteria

1. Provider configuration centralized
2. Health checks automated
3. Credentials securely stored
4. UI updated to use services
5. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Health check frequency? | 1 min / 5 min / On-demand only | TBD |
| Credential storage backend? | libsecret / keyring / encrypted file | TBD |
| Should provider errors auto-disable? | Yes / No / After N failures | TBD |
| Support for provider fallback chains? | Yes / No | TBD |
