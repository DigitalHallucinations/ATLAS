# Persona Service

> **Status**: ✅ Phase 1 Complete  
> **Priority**: Medium  
> **Complexity**: Medium  
> **Effort**: 3-4 days  
> **Created**: 2026-01-07  
> **Updated**: 2026-01-10

---

## Overview

Extract persona management from `GTKUI/Persona_manager/` and `modules/Personas/` into `core/services/personas/`:

- Centralized persona CRUD
- Schema validation
- Active persona management
- Capability listing

---

## Phases

### Phase 1: Service Creation ✅

- [x] **1.1** Create `core/services/personas/` package:
  - `types.py` - PersonaConfig, PersonaEvent
  - `permissions.py` - Persona permission checker
  - `service.py` - PersonaService
  - `validation.py` - Schema validation (from persona_manager.py)
  - `exceptions.py` - Custom exception hierarchy
- [x] **1.2** Implement PersonaService:
  - `list_personas()` - Get all available personas
  - `get_persona(persona_id)` - Get single persona config
  - `create_persona(actor, config)` - Create custom persona
  - `update_persona(actor, persona_id, updates)` - Modify persona
  - `delete_persona(actor, persona_id)` - Remove persona
  - `validate_persona(config)` - Validate against schema
  - `get_active_persona()` - Get currently active
  - `set_active_persona(actor, persona_id)` - Switch persona
  - `get_persona_capabilities(persona_id)` - List tools/skills
- [x] **1.3** Add MessageBus events:
  - `persona.created`
  - `persona.updated`
  - `persona.deleted`
  - `persona.activated`
  - `persona.deactivated`
  - `persona.validated`
- [x] **1.4** Write unit tests - **42 tests passing**

### Phase 2: UI Integration

- [ ] **2.1** Update `GTKUI/Persona_manager/` to use service
- [ ] **2.2** Update `core/persona_manager.py` to delegate to service
- [ ] **2.3** Remove direct file access from UI

---

## Service Methods

```python
class PersonaService:
    # CRUD
    def list_personas(self, actor: Actor) -> OperationResult[list[PersonaSummary]]: ...
    def get_persona(self, actor: Actor, persona_id: str) -> OperationResult[PersonaConfig]: ...
    def create_persona(self, actor: Actor, config: PersonaCreate) -> OperationResult[PersonaConfig]: ...
    def update_persona(self, actor: Actor, persona_id: str, updates: PersonaUpdate) -> OperationResult[PersonaConfig]: ...
    def delete_persona(self, actor: Actor, persona_id: str) -> OperationResult[None]: ...
    
    # Validation
    def validate_persona(self, config: dict) -> OperationResult[ValidationResult]: ...
    
    # Active persona
    def get_active_persona(self) -> OperationResult[PersonaConfig | None]: ...
    def set_active_persona(self, actor: Actor, persona_id: str) -> OperationResult[PersonaConfig]: ...
    
    # Capabilities
    def get_persona_capabilities(self, persona_id: str) -> OperationResult[PersonaCapabilities]: ...
    def get_persona_tools(self, persona_id: str) -> OperationResult[list[str]]: ...
    def get_persona_skills(self, persona_id: str) -> OperationResult[list[str]]: ...
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `persona.created` | `PersonaEvent` | PersonaService |
| `persona.updated` | `PersonaEvent` | PersonaService |
| `persona.deleted` | `PersonaEvent` | PersonaService |
| `persona.activated` | `PersonaActivationEvent` | PersonaService |
| `persona.deactivated` | `PersonaActivationEvent` | PersonaService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/personas/__init__.py` | Package exports |
| `core/services/personas/types.py` | Dataclasses, events |
| `core/services/personas/permissions.py` | PersonaPermissionChecker |
| `core/services/personas/service.py` | PersonaService |
| `core/services/personas/validation.py` | Schema validation |
| `tests/services/personas/` | Service tests |

---

## Files to Modify

| File | Changes |
|------|---------|
| `core/services/__init__.py` | Export persona services |
| `core/persona_manager.py` | Delegate to PersonaService |
| `GTKUI/Persona_manager/*.py` | Use new service |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/Personas/` - Persona definitions (read-only)
- `modules/Personas/schema.json` - Validation schema
- `core/messaging/` - MessageBus for events

---

## Success Criteria

1. All persona operations go through PersonaService
2. Schema validation extracted and reusable
3. Active persona changes emit events
4. UI updated to use service
5. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Where should persona files be stored? | File system / Database | TBD |
| Should persona switching require permission? | Yes / No | TBD |
| Can users create system-wide personas? | Yes with admin / No, user-only | TBD |
| Persona inheritance support? | Yes / No | TBD |
