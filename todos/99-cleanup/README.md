# Final Cleanup & Documentation

> **Status**: 📋 Planning  
> **Priority**: Low (execute last)  
> **Complexity**: Low  
> **Effort**: 2-3 days  
> **Created**: 2026-01-07

---

## Overview

Final cleanup phase after all other phases are complete:

- Remove deprecated code paths
- Update all imports
- Comprehensive documentation
- Architecture diagrams
- AGENTS.md updates

---

## Phases

### Phase 1: Deprecation Cleanup

- [ ] **1.1** Remove deprecated paths with warnings that have been active for sufficient time
- [ ] **1.2** Remove backward-compatibility shims
- [ ] **1.3** Clean up unused imports
- [ ] **1.4** Remove dead code

### Phase 2: Import Consolidation

- [ ] **2.1** Update all imports across codebase
- [ ] **2.2** Ensure `core/services/__init__.py` exports all services
- [ ] **2.3** Verify no circular imports
- [ ] **2.4** Run import linter

### Phase 3: Service Documentation

- [ ] **3.1** Document all services in `docs/developer/services/`:
  - Service overview
  - API reference
  - Usage examples
  - Configuration options
- [ ] **3.2** Create service-specific guides:
  - `docs/developer/services/calendar.md`
  - `docs/developer/services/budget.md`
  - `docs/developer/services/library.md`
  - (etc. for all services)

### Phase 4: Architecture Documentation

- [ ] **4.1** Update `docs/architecture-overview.md`:
  - Service layer diagram
  - Data flow diagrams
  - Component relationships
- [ ] **4.2** Create architecture decision records (ADRs)
- [ ] **4.3** Document MessageBus event catalog

### Phase 5: AGENTS.md Updates

- [ ] **5.1** Update root `AGENTS.md` with service ownership
- [ ] **5.2** Update domain-specific AGENTS.md files:
  - `core/AGENTS.md`
  - `modules/AGENTS.md`
  - `GTKUI/AGENTS.md`
- [ ] **5.3** Define service boundaries for agents

### Phase 6: Testing & Validation

- [ ] **6.1** Run full test suite
- [ ] **6.2** Verify all tests pass
- [ ] **6.3** Check test coverage (>90% on services)
- [ ] **6.4** Manual integration testing

### Phase 7: Release Preparation

- [ ] **7.1** Update `docs/release-notes.md`
- [ ] **7.2** Update README.md with new architecture
- [ ] **7.3** Create migration guide for existing users
- [ ] **7.4** Version bump

---

## Documentation Structure

```
docs/
├── architecture-overview.md          # Updated with service layer
├── developer/
│   ├── service-pattern.md            # Standard service pattern
│   ├── services/
│   │   ├── README.md                 # Service index
│   │   ├── accounts.md               # UserAccountService
│   │   ├── analytics.md              # AnalyticsService
│   │   ├── budget.md                 # Budget services
│   │   ├── calendar.md               # Calendar services
│   │   ├── guardrails.md             # GuardrailsService
│   │   ├── jobs.md                   # JobService
│   │   ├── knowledge.md              # KnowledgeBaseService
│   │   ├── library.md                # LibraryService
│   │   ├── mcp.md                    # MCP integration
│   │   ├── memory.md                 # Memory services
│   │   ├── observability.md          # ObservabilityService
│   │   ├── orchestration.md          # Multi-agent orchestration
│   │   ├── personas.md               # PersonaService
│   │   ├── planning.md               # Planning & reflection
│   │   ├── providers.md              # Provider services
│   │   ├── skills.md                 # SkillService
│   │   ├── speech.md                 # SpeechService
│   │   ├── storage.md                # StorageService
│   │   ├── tasks.md                  # TaskService
│   │   └── tools.md                  # ToolService
│   └── messagebus-events.md          # Event catalog
└── contributing/
    └── agent-owners.md               # Updated ownership
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `docs/developer/services/README.md` | Service index |
| `docs/developer/services/*.md` | Individual service docs |
| `docs/developer/messagebus-events.md` | Event catalog |

---

## Files to Update

| File | Changes |
|------|---------|
| `docs/architecture-overview.md` | Add service layer section |
| `README.md` | Update architecture description |
| `AGENTS.md` | Add service ownership |
| `core/AGENTS.md` | Update for new structure |
| `modules/AGENTS.md` | Update for repositories |
| `GTKUI/AGENTS.md` | Update for UI-only |
| `docs/release-notes.md` | Document changes |

---

## Dependencies

- **All other phases must be complete before this phase**

---

## Success Criteria

1. No deprecated code remains
2. All imports are clean
3. Documentation is comprehensive
4. Architecture diagrams are current
5. AGENTS.md files are accurate
6. All tests pass
7. Ready for release

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Deprecation period before removal? | 1 release / 2 releases | TBD |
| Documentation format? | Markdown only / + API docs | TBD |
| Architecture diagram tool? | Mermaid / Draw.io / ASCII | TBD |
