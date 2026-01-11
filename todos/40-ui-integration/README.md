# UI Integration Sprint

> **Status**: 📋 Planning  
> **Priority**: Medium  
> **Complexity**: Medium  
> **Effort**: 5-7 days  
> **Created**: 2026-01-11

---

## Overview

Consolidate all UI integration work deferred from service layer refactors. This sprint updates GTKUI components to consume the new `core/services/` layer instead of directly accessing modules/config.

---

## Scope

UI integration tasks collected from:

| Source          | Component                   | Status      |
| --------------- | --------------------------- | ----------- |
| 06-personas     | Persona_manager             | Deferred    |
| 07-providers    | Provider_manager            | Deferred    |
| 04-accounts     | UserAccounts                | Not started |
| 08-skills-tools | Skill_manager, Tool_manager | Not started |
| 09-knowledge    | KnowledgeBase               | Not started |

---

## Phases

### Phase 1: Personas UI (from 06-personas)

- [ ] **1.1** Update `GTKUI/Persona_manager/` to use `PersonaService`
- [ ] **1.2** Wire MessageBus events for real-time updates
- [ ] **1.3** Remove direct `modules/Personas/` access from UI
- [ ] **1.4** Update persona selection/switching logic

### Phase 2: Providers UI (from 07-providers)

- [ ] **2.1** Update `GTKUI/Provider_manager/` to use `ProviderConfigService`
- [ ] **2.2** Update `GTKUI/Provider_manager/` to use `ProviderHealthService`
- [ ] **2.3** Wire health status indicators to service events
- [ ] **2.4** Remove direct config access from UI

### Phase 3: Accounts UI (from 04-accounts)

- [ ] **3.1** Update `GTKUI/UserAccounts/` to use account services
- [ ] **3.2** Wire authentication flows through service layer
- [ ] **3.3** Remove direct `modules/user_accounts/` access

### Phase 4: Skills & Tools UI (from 08-skills-tools)

- [ ] **4.1** Update `GTKUI/Skill_manager/` to use `SkillService`
- [ ] **4.2** Update `GTKUI/Tool_manager/` to use `ToolService`
- [ ] **4.3** Wire enable/disable events through MessageBus

### Phase 5: Knowledge UI (from 09-knowledge)

- [ ] **5.1** Update `GTKUI/KnowledgeBase/` to use `KnowledgeService`
- [ ] **5.2** Wire document indexing events

---

## UI Components Affected

| GTKUI Component     | Service Layer                                    | Events                                |
| ------------------- | ------------------------------------------------ | ------------------------------------- |
| `Persona_manager/`  | `PersonaService`                                 | `persona.switched`, `persona.updated` |
| `Provider_manager/` | `ProviderConfigService`, `ProviderHealthService` | `provider.*`                          |
| `UserAccounts/`     | `UserAccountService`                             | `user.*`                              |
| `Skill_manager/`    | `SkillService`                                   | `skill.*`                             |
| `Tool_manager/`     | `ToolService`                                    | `tool.*`                              |
| `KnowledgeBase/`    | `KnowledgeService`                               | `knowledge.*`                         |

---

## Pattern

All UI updates follow this pattern:

1. **Inject service** via dependency injection or singleton
2. **Subscribe to MessageBus** for reactive updates
3. **Remove direct module imports** from UI code
4. **Use Actor pattern** for permission-aware operations

Example:

```python
# Before (direct access)
from modules.Personas import PersonaManager
personas = PersonaManager().list_personas()

# After (service layer)
from core.services.personas import PersonaService
personas = persona_service.list_personas(actor)
```

---

## Dependencies

- **Prerequisites**:
  - [completed/06-personas/](../completed/06-personas/) - PersonaService ✅
  - [completed/07-providers/](../completed/07-providers/) - ProviderConfigService, ProviderHealthService ✅
  - [04-accounts/](../04-accounts/) - UserAccountService (when complete)
  - [08-skills-tools/](../08-skills-tools/) - SkillService, ToolService (when complete)
  - [09-knowledge/](../09-knowledge/) - KnowledgeService (when complete)

---

## Success Criteria

1. All GTKUI managers use service layer (no direct module access)
2. Real-time UI updates via MessageBus subscriptions
3. Consistent Actor-based permission checks in UI
4. No regressions in UI functionality
5. UI tests updated to mock services
6. **No Pylance errors** in modified files

---

## Notes

- This sprint can be parallelized across UI components
- Each phase is independently deployable
- Consider GTK async patterns for service calls
