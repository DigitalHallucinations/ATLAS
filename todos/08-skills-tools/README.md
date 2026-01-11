# Skill & Tool Services

> **Status**: 📋 Planning  
> **Priority**: Medium  
> **Complexity**: High  
> **Effort**: 12-16 days  
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
- [ ] **1.4** Write unit tests

> **UI Integration**: See [40-ui-integration](../40-ui-integration/)

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
- [ ] **2.5** Write unit tests

> **UI Integration**: See [40-ui-integration](../40-ui-integration/)

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

## Phase 3: Base_Tools Reorganization

> **Goal**: Transform the flat `modules/Tools/Base_Tools/` directory (67+ tools) into a categorized subpackage structure optimized for marketplace discovery and context-based tool injection.

### 3.1 Category Taxonomy

Create the following category structure under `modules/Tools/Base_Tools/`:

```
Base_Tools/
├── _shared/                    # Internal utilities (not exposed as tools)
│   ├── __init__.py
│   └── normalization.py        # (moved from utils/)
│
├── web/                        # 🌐 Web & Network
│   ├── __init__.py
│   ├── browser/                # Unified browser package
│   │   ├── __init__.py
│   │   ├── virtual.py          # (from browser.py - session tracking, no network)
│   │   ├── lite.py             # (from browser_lite.py - real HTTP/Playwright)
│   │   └── errors.py
│   ├── search.py               # (from Google_search.py)
│   └── fetch.py                # (from webpage_fetch.py)
│
├── time_location/              # 🕐 Temporal & Spatial
│   ├── __init__.py
│   ├── time.py
│   └── location.py             # (from current_location.py)
│
├── calendar/                   # 📅 Calendar & Scheduling (existing subdir, expanded)
│   ├── __init__.py
│   ├── backends/
│   ├── service.py              # (from calendar_service.py)
│   ├── debian12.py             # (from debian12_calendar.py)
│   └── ... (existing files)
│
├── filesystem/                 # 📁 File System Operations
│   ├── __init__.py
│   ├── io.py                   # (from filesystem_io.py)
│   ├── ingest.py               # (from file_ingest.py)
│   └── publisher.py            # (from workspace_publisher.py)
│
├── storage/                    # 💾 Data Persistence & Storage
│   ├── __init__.py
│   ├── kv_store/               # Key-value store with provider pattern
│   │   ├── __init__.py
│   │   ├── service.py          # Core KV logic
│   │   ├── backends/
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py
│   │   │   └── mongo.py        # (from kv_store_mongo.py - now a backend)
│   │   └── errors.py
│   ├── vector_store.py
│   └── content_repository.py
│
├── queue/                      # 📋 Task Queues & Job Management
│   ├── __init__.py
│   ├── task_queue.py
│   ├── priority_queue.py
│   └── stream_monitor.py
│
├── memory/                     # 🧠 Agent Memory & Context
│   ├── __init__.py
│   ├── episodic.py             # (from memory_episodic.py)
│   ├── graph.py                # (from memory_graph.py)
│   ├── recall.py               # (from memory_recall.py)
│   └── context_tracker.py
│
├── communication/              # 📧 Messaging & Notifications
│   ├── __init__.py
│   ├── email.py                # (from email_service.py)
│   ├── notification.py         # (from notification_service.py)
│   └── ticketing.py            # (from ticketing_system.py)
│
├── business/                   # 💼 Business & Enterprise
│   ├── __init__.py
│   ├── crm.py                  # (from crm_service.py)
│   ├── roadmap.py              # (from roadmap_service.py)
│   └── labor_market.py         # (from labor_market_feed.py)
│
├── dashboards/                 # 📊 Analytics & Reporting
│   ├── __init__.py
│   ├── analytics.py            # (from analytics_dashboard.py)
│   ├── atlas.py                # (from atlas_dashboard.py)
│   ├── service.py              # (from dashboard_service.py)
│   └── audit.py                # (from audit_reporter.py)
│
├── creative/                   # 🎨 Creative & Media
│   ├── __init__.py
│   ├── image/
│   │   ├── __init__.py
│   │   ├── generate.py         # (from generate_image.py)
│   │   ├── edit.py             # (from edit_image.py)
│   │   └── visual_prompt.py
│   ├── writing/
│   │   ├── __init__.py
│   │   ├── lyricist.py
│   │   └── story_weaver.py
│   └── notebook.py
│
├── analysis/                   # 🔬 Text & Data Analysis
│   ├── __init__.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── tone.py             # (from tone_analyzer.py)
│   │   ├── emotive.py          # (from emotive_tagger.py)
│   │   └── mood.py             # (from mood_map.py)
│   ├── schema_infer.py
│   ├── structured_parser.py
│   └── spreadsheet.py
│
├── system/                     # ⚙️ System Operations
│   ├── __init__.py
│   ├── terminal.py             # (from terminal_command.py)
│   ├── snapshot.py             # (from sys_snapshot.py)
│   └── logging/
│       ├── __init__.py
│       ├── event.py            # (from log_event.py)
│       └── parser.py           # (from log_parser.py)
│
├── agents/                     # 🤖 Agent Orchestration & Reasoning
│   ├── __init__.py
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── decompose.py        # (from planner_decompose.py)
│   │   └── prompt_compiler.py
│   ├── consensus/
│   │   ├── __init__.py
│   │   ├── vote.py             # (from consensus_vote.py)
│   │   ├── conflict.py         # (from conflict_resolver.py)
│   │   └── hitl.py             # (from hitl_approval.py)
│   └── reflection/
│       ├── __init__.py
│       └── reflective_prompt.py
│
├── evaluation/                 # ✅ Testing & Evaluation
│   ├── __init__.py
│   ├── _utils.py               # (from _evaluation_utils.py)
│   ├── judge.py                # (from eval_judge.py)
│   ├── regression.py           # (from eval_regression.py)
│   └── trace.py                # (from trace_explain.py)
│
├── policy/                     # 🛡️ Policy & Governance
│   ├── __init__.py
│   ├── reference.py            # (from policy_reference.py)
│   └── budget.py               # (from budget_limiter.py)
│
├── security/                   # 🔐 Security & Secrets
│   ├── __init__.py
│   ├── vault.py                # (from vault_secrets.py)
│   ├── threat.py               # (from threat_scanner.py)
│   └── registry.py             # (from registry_capability.py)
│
├── integrations/               # 🔌 External Service Connectors
│   ├── __init__.py
│   ├── api_connector.py
│   └── data_bridge.py
│
├── compute/                    # 🧮 Computation & ML
│   ├── __init__.py
│   ├── calculator.py
│   └── embeddings/
│       ├── __init__.py
│       └── clip.py             # (from clip_embeddings.py)
│
└── incidents/                  # 🚨 Incident Response
    ├── __init__.py
    └── summarizer.py           # (from incident_summarizer.py)
```

### 3.2 Tasks

- [ ] **3.2.1** Create all category directories with `__init__.py` files
- [ ] **3.2.2** Consolidate browser tools:
  - Merge `browser.py` (virtual/mock) and `browser_lite.py` (real HTTP/Playwright)
  - Create unified `web/browser/` package with shared errors
  - Virtual mode for testing/reasoning, lite mode for actual automation
- [ ] **3.2.3** Convert `kv_store_mongo.py` to provider pattern:
  - Move to `storage/kv_store/backends/mongo.py`
  - Refactor as `MongoKVStoreAdapter` implementing `KeyValueStoreAdapter` protocol
  - Update `kv_store` service to load backends dynamically
- [ ] **3.2.4** Move all tools to their category subpackages (see mapping below)
- [ ] **3.2.5** Update `Base_Tools/__init__.py`:
  - Re-export all public symbols for backward compatibility during transition
  - Add deprecation warnings for direct imports
- [ ] **3.2.6** Update `modules/Tools/tool_maps/functions.json`:
  - Add `category` field to each tool manifest
  - Add `subcategory` field where applicable
  - Format: `"category": "web/browser"`, `"subcategory": "automation"`
- [ ] **3.2.7** Update `modules/Tools/tool_maps/schema.json`:
  - Add `category` and `subcategory` to schema validation
- [ ] **3.2.8** Update all import references across codebase
- [ ] **3.2.9** Write migration tests to verify all tools still accessible

### 3.3 Tool-to-Category Mapping

| Current File | New Location | Category |
|--------------|--------------|----------|
| `Google_search.py` | `web/search.py` | web/search |
| `webpage_fetch.py` | `web/fetch.py` | web/fetch |
| `browser.py` | `web/browser/virtual.py` | web/browser |
| `browser_lite.py` | `web/browser/lite.py` | web/browser |
| `time.py` | `time_location/time.py` | time_location |
| `current_location.py` | `time_location/location.py` | time_location |
| `calendar_service.py` | `calendar/service.py` | calendar |
| `debian12_calendar.py` | `calendar/debian12.py` | calendar |
| `filesystem_io.py` | `filesystem/io.py` | filesystem |
| `file_ingest.py` | `filesystem/ingest.py` | filesystem |
| `workspace_publisher.py` | `filesystem/publisher.py` | filesystem |
| `kv_store.py` | `storage/kv_store/service.py` | storage/kv_store |
| `kv_store_mongo.py` | `storage/kv_store/backends/mongo.py` | storage/kv_store |
| `vector_store.py` | `storage/vector_store.py` | storage |
| `content_repository.py` | `storage/content_repository.py` | storage |
| `task_queue.py` | `queue/task_queue.py` | queue |
| `priority_queue.py` | `queue/priority_queue.py` | queue |
| `stream_monitor.py` | `queue/stream_monitor.py` | queue |
| `memory_episodic.py` | `memory/episodic.py` | memory |
| `memory_graph.py` | `memory/graph.py` | memory |
| `memory_recall.py` | `memory/recall.py` | memory |
| `context_tracker.py` | `memory/context_tracker.py` | memory |
| `email_service.py` | `communication/email.py` | communication |
| `notification_service.py` | `communication/notification.py` | communication |
| `ticketing_system.py` | `communication/ticketing.py` | communication |
| `crm_service.py` | `business/crm.py` | business |
| `roadmap_service.py` | `business/roadmap.py` | business |
| `labor_market_feed.py` | `business/labor_market.py` | business |
| `analytics_dashboard.py` | `dashboards/analytics.py` | dashboards |
| `atlas_dashboard.py` | `dashboards/atlas.py` | dashboards |
| `dashboard_service.py` | `dashboards/service.py` | dashboards |
| `audit_reporter.py` | `dashboards/audit.py` | dashboards |
| `generate_image.py` | `creative/image/generate.py` | creative/image |
| `edit_image.py` | `creative/image/edit.py` | creative/image |
| `visual_prompt.py` | `creative/image/visual_prompt.py` | creative/image |
| `lyricist.py` | `creative/writing/lyricist.py` | creative/writing |
| `story_weaver.py` | `creative/writing/story_weaver.py` | creative/writing |
| `notebook.py` | `creative/notebook.py` | creative |
| `tone_analyzer.py` | `analysis/nlp/tone.py` | analysis/nlp |
| `emotive_tagger.py` | `analysis/nlp/emotive.py` | analysis/nlp |
| `mood_map.py` | `analysis/nlp/mood.py` | analysis/nlp |
| `schema_infer.py` | `analysis/schema_infer.py` | analysis |
| `structured_parser.py` | `analysis/structured_parser.py` | analysis |
| `spreadsheet.py` | `analysis/spreadsheet.py` | analysis |
| `terminal_command.py` | `system/terminal.py` | system |
| `sys_snapshot.py` | `system/snapshot.py` | system |
| `log_event.py` | `system/logging/event.py` | system/logging |
| `log_parser.py` | `system/logging/parser.py` | system/logging |
| `planner_decompose.py` | `agents/planning/decompose.py` | agents/planning |
| `prompt_compiler.py` | `agents/planning/prompt_compiler.py` | agents/planning |
| `consensus_vote.py` | `agents/consensus/vote.py` | agents/consensus |
| `conflict_resolver.py` | `agents/consensus/conflict.py` | agents/consensus |
| `hitl_approval.py` | `agents/consensus/hitl.py` | agents/consensus |
| `reflective_prompt.py` | `agents/reflection/reflective_prompt.py` | agents/reflection |
| `_evaluation_utils.py` | `evaluation/_utils.py` | evaluation |
| `eval_judge.py` | `evaluation/judge.py` | evaluation |
| `eval_regression.py` | `evaluation/regression.py` | evaluation |
| `trace_explain.py` | `evaluation/trace.py` | evaluation |
| `policy_reference.py` | `policy/reference.py` | policy |
| `budget_limiter.py` | `policy/budget.py` | policy |
| `vault_secrets.py` | `security/vault.py` | security |
| `threat_scanner.py` | `security/threat.py` | security |
| `registry_capability.py` | `security/registry.py` | security |
| `api_connector.py` | `integrations/api_connector.py` | integrations |
| `data_bridge.py` | `integrations/data_bridge.py` | integrations |
| `calculator.py` | `compute/calculator.py` | compute |
| `clip_embeddings.py` | `compute/embeddings/clip.py` | compute/embeddings |
| `incident_summarizer.py` | `incidents/summarizer.py` | incidents |

### 3.4 Manifest Schema Updates

Add to `modules/Tools/tool_maps/schema.json`:

```json
{
  "category": {
    "type": "string",
    "description": "Primary category path (e.g., 'web/browser', 'storage/kv_store')",
    "pattern": "^[a-z_]+(/[a-z_]+)*$"
  },
  "subcategory": {
    "type": "string",
    "description": "Optional refinement (e.g., 'search_engines', 'automation')"
  },
  "marketplace_tags": {
    "type": "array",
    "items": { "type": "string" },
    "description": "Additional discovery tags for marketplace search"
  }
}
```

### 3.5 Context Injection Integration

The category structure enables future context-based tool injection:

```python
# Example: ToolService discovers relevant tools by context
async def get_tools_for_context(context: ConversationContext) -> list[ToolManifest]:
    """Inject tools based on detected user intent and conversation context."""
    categories = []
    
    if context.mentions_scheduling:
        categories.append("calendar")
    if context.mentions_web_research:
        categories.extend(["web/search", "web/fetch"])
    if context.mentions_files:
        categories.append("filesystem")
    if context.mentions_coding:
        categories.extend(["system/terminal", "evaluation"])
    
    return await tool_service.get_tools_by_categories(categories)
```

---

## Phase 4: Skills Categorization (Future)

> Apply the same categorization pattern to `modules/Skills/` for consistency.

### 4.1 Proposed Skill Categories

| Category | Description |
|----------|-------------|
| `reasoning/` | Logic, deduction, problem decomposition |
| `research/` | Information gathering, synthesis |
| `coding/` | Code generation, review, debugging |
| `communication/` | Writing, summarization, translation |
| `creative/` | Ideation, storytelling, design |
| `analysis/` | Data analysis, pattern recognition |
| `domain/` | Domain-specific expertise (legal, medical, finance) |

### 4.2 Tasks (TBD)

- [ ] **4.2.1** Audit existing skills in `modules/Skills/`
- [ ] **4.2.2** Define skill category taxonomy
- [ ] **4.2.3** Create skill subpackages
- [ ] **4.2.4** Update skill manifests with category fields
- [ ] **4.2.5** Integrate with SkillService for context-based injection

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Tool sandboxing approach? | Container / Process isolation / None | TBD |
| Skill dependency resolution? | Automatic / Manual | TBD |
| Should tool invocations be logged? | Always / On error / Configurable | TBD |
| Tool timeout defaults? | 30s / 60s / Configurable per-tool | TBD |
| Deprecation period for old imports? | None / 1 release / 2 releases | None (clean break) |
| Marketplace UI for tool discovery? | In-app / Web portal / Both | TBD |
