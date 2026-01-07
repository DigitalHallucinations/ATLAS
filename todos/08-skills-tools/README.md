# Skill & Tool Services

> **Status**: 📋 Planning  
> **Priority**: Medium  
> **Complexity**: Medium  
> **Effort**: 6-8 days  
> **Created**: 2026-01-07

---

## Overview

Extract skill and tool management from `GTKUI/Skill_manager/`, `GTKUI/Tool_manager/`, `modules/Skills/`, and `modules/Tools/` into two services:

1. **SkillService** - Skill registration, invocation, lifecycle
2. **ToolService** - Tool registration, invocation, permissions

---

## Phases

### Phase 1: SkillService

- [ ] **1.1** Create `core/services/skills/` package
- [ ] **1.2** Implement SkillService:
  - `list_skills()` - Get all skills
  - `get_skill(skill_id)` - Get skill details
  - `register_skill(actor, manifest)` - Register new skill
  - `unregister_skill(actor, skill_id)` - Remove skill
  - `enable_skill(actor, skill_id)` - Enable for use
  - `disable_skill(actor, skill_id)` - Disable
  - `get_skill_dependencies(skill_id)` - List dependencies
  - `check_skill_health(skill_id)` - Validate skill works
  - `invoke_skill(actor, skill_id, params)` - Execute skill
- [ ] **1.3** Add MessageBus events:
  - `skill.registered`, `skill.unregistered`
  - `skill.enabled`, `skill.disabled`
  - `skill.invoked`, `skill.completed`, `skill.failed`
- [ ] **1.4** Update UI to use service
- [ ] **1.5** Write unit tests

### Phase 2: ToolService

- [ ] **2.1** Create `core/services/tools/` package
- [ ] **2.2** Implement ToolService:
  - `list_tools()` - Get all tools
  - `get_tool(tool_id)` - Get tool manifest
  - `register_tool(actor, manifest)` - Register tool
  - `unregister_tool(actor, tool_id)` - Remove tool
  - `enable_tool(actor, tool_id)` - Enable
  - `disable_tool(actor, tool_id)` - Disable
  - `get_tool_permissions(tool_id)` - Required permissions
  - `check_tool_available(tool_id)` - Is tool ready?
  - `invoke_tool(actor, tool_id, params)` - Execute tool
  - `get_tools_for_persona(persona_id)` - Tools available to persona
- [ ] **2.3** Add MessageBus events:
  - `tool.registered`, `tool.unregistered`
  - `tool.enabled`, `tool.disabled`
  - `tool.invoked`, `tool.completed`, `tool.failed`
- [ ] **2.4** Integrate with permission system (tool-level permissions)
- [ ] **2.5** Update UI and ToolManager to use service
- [ ] **2.6** Write unit tests

---

## MessageBus Events

### Skill Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `skill.registered` | `SkillEvent` | SkillService |
| `skill.unregistered` | `SkillEvent` | SkillService |
| `skill.enabled` | `SkillStateEvent` | SkillService |
| `skill.disabled` | `SkillStateEvent` | SkillService |
| `skill.invoked` | `SkillInvocationEvent` | SkillService |
| `skill.completed` | `SkillResultEvent` | SkillService |
| `skill.failed` | `SkillErrorEvent` | SkillService |

### Tool Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `tool.registered` | `ToolEvent` | ToolService |
| `tool.unregistered` | `ToolEvent` | ToolService |
| `tool.enabled` | `ToolStateEvent` | ToolService |
| `tool.disabled` | `ToolStateEvent` | ToolService |
| `tool.invoked` | `ToolInvocationEvent` | ToolService |
| `tool.completed` | `ToolResultEvent` | ToolService |
| `tool.failed` | `ToolErrorEvent` | ToolService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/skills/__init__.py` | Package exports |
| `core/services/skills/types.py` | Dataclasses, events |
| `core/services/skills/permissions.py` | SkillPermissionChecker |
| `core/services/skills/service.py` | SkillService |
| `core/services/tools/__init__.py` | Package exports |
| `core/services/tools/types.py` | Dataclasses, events |
| `core/services/tools/permissions.py` | ToolPermissionChecker |
| `core/services/tools/service.py` | ToolService |
| `tests/services/skills/` | Skill service tests |
| `tests/services/tools/` | Tool service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/Skills/` - Skill definitions
- `modules/Tools/` - Tool definitions
- `core/messaging/` - MessageBus for events
- [06-personas](../06-personas/) - Persona-tool mapping

---

## Success Criteria

1. Skills and tools managed through services
2. Invocations tracked with events
3. Tool permissions integrated
4. UIs updated
5. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Tool sandboxing approach? | Container / Process isolation / None | TBD |
| Skill dependency resolution? | Automatic / Manual | TBD |
| Should tool invocations be logged? | Always / On error / Configurable | TBD |
| Tool timeout defaults? | 30s / 60s / Configurable per-tool | TBD |
