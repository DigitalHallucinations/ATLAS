# ATLAS Todo Index

> **Last Updated**: 2026-01-07

This directory contains all planning documents for ATLAS development, organized by domain.

---

## Quick Navigation

### 🏗️ Architecture & Foundation

- [00-foundation/](00-foundation/) - Core patterns, common types, base infrastructure

### 📅 Domain Services

- [01-calendar/](01-calendar/) - Calendar, sync, and reminder services
- [02-budget/](02-budget/) - Budget policies, tracking, and alerts
- [03-library/](03-library/) - Artifact storage, collections, versioning
- [04-accounts/](04-accounts/) - User accounts, authentication, credentials
- [05-jobs-tasks/](05-jobs-tasks/) - Job and task service migration
- [06-personas/](06-personas/) - Persona management service
- [07-providers/](07-providers/) - Provider configuration and health
- [08-skills-tools/](08-skills-tools/) - Skill and tool registry services
- [09-knowledge/](09-knowledge/) - Knowledge base management
- [10-speech/](10-speech/) - Speech services (TTS/STT)
- [11-analytics/](11-analytics/) - Analytics and metrics
- [12-storage/](12-storage/) - File storage service

### 🧠 SOTA Agent Capabilities

- [20-memory/](20-memory/) - Long-term memory, episodic recall, user preferences
- [21-multi-agent/](21-multi-agent/) - Orchestrator-worker, agent handoff, parallelization
- [22-guardrails/](22-guardrails/) - Safety, content filtering, human-in-the-loop
- [23-planning/](23-planning/) - Task decomposition, self-reflection, chain-of-thought

### 🔧 Infrastructure

- [30-observability/](30-observability/) - Telemetry, tracing, metrics
- [31-mcp/](31-mcp/) - Model Context Protocol integration

### 📋 Cleanup & Documentation

- [99-cleanup/](99-cleanup/) - Final cleanup, deprecation, documentation

---

## Status Legend

| Status | Meaning |
| ------ | ------- |
| 📋 Planning | Design phase, gathering requirements |
| 🚧 In Progress | Active development |
| ✅ Complete | Finished and merged |
| ⏸️ Blocked | Waiting on dependency |
| 🔄 Review | Ready for code review |

---

## Implementation Order

### Phase 1: Foundation (Week 1-2)

1. `00-foundation/` - Common patterns MUST come first

### Phase 2: Core Services (Week 2-5)

1. `05-jobs-tasks/` - Already partially implemented
2. `04-accounts/` - Already partially implemented  
3. `01-calendar/` - High complexity, start early
4. `02-budget/` - High complexity

### Phase 3: New Features (Week 5-8)

1. `03-library/` - New feature
2. `20-memory/` - SOTA capability
3. `06-personas/` through `12-storage/` - Medium complexity

### Phase 4: Advanced SOTA (Week 8-11)

1. `21-multi-agent/` - Advanced orchestration
2. `22-guardrails/` - Safety layer
3. `23-planning/` - Agent enhancement

### Phase 5: Finalization (Week 11-12)

1. `30-observability/` - Telemetry consolidation
2. `31-mcp/` - External tool protocol
3. `99-cleanup/` - Documentation, deprecation

---

## Effort Estimates

| Domain | Complexity | Effort |
| ------ | ---------- | ------ |
| 00-foundation | Low | 1-2 days |
| 01-calendar | High | 1-2 weeks |
| 02-budget | High | 1 week |
| 03-library | Medium | 1 week |
| 04-accounts | Medium | 3-4 days |
| 05-jobs-tasks | Medium | 3-5 days |
| 06-personas | Medium | 3-4 days |
| 07-providers | Medium | 3-4 days |
| 08-skills-tools | Medium | 4-5 days |
| 09-knowledge | Medium | 3-4 days |
| 10-speech | Low | 2-3 days |
| 11-analytics | Medium | 3-4 days |
| 12-storage | Low | 2-3 days |
| 20-memory | High | 1-2 weeks |
| 21-multi-agent | High | 1-2 weeks |
| 22-guardrails | Medium | 4-5 days |
| 23-planning | Medium | 4-5 days |
| 30-observability | Low | 2-3 days |
| 31-mcp | Medium | 3-4 days |
| 99-cleanup | Low | 2-3 days |

**Total Estimated**: 10-14 weeks

---

## Cross-Cutting Concerns

All service todos share these requirements:

1. **Pattern Compliance**: Use `OperationResult[T]`, Actor-based permissions, MessageBus events
2. **Testing**: >90% coverage on service layer
3. **Documentation**: Update `docs/developer/services/` for each service
4. **AGENTS.md**: Update ownership in relevant AGENTS.md files
5. **Migration**: Deprecation warnings on old import paths

---

## References

- Architecture overview: `docs/architecture-overview.md`
- Service pattern: `00-foundation/service-pattern.md`
- MessageBus: `core/messaging/`
