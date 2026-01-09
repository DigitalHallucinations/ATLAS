# ATLAS Todo Index

> **Last Updated**: 2026-01-09

This directory contains all planning documents for ATLAS development, organized by domain.

> **Note**: Folder numbers indicate **domain groupings**, not build order. See [Implementation Order](#implementation-order) for the recommended build sequence.

---

## Quick Navigation

### ✅ Completed Phases

- [completed/00-foundation/](completed/00-foundation/) - ✅ **COMPLETED** Core patterns, common types, base infrastructure
- [completed/01-calendar/](completed/01-calendar/) - ✅ **COMPLETED** Calendar, sync, and reminder services
- [completed/02-budget/](completed/02-budget/) - ✅ **COMPLETED** Budget policies, tracking, and alerts (243 tests)

### 📅 Domain Services

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
- [13-content-editing/](13-content-editing/) - **NEW** SOTA content editing platform with AI-powered tools

### 🧠 SOTA Agent Capabilities

- [20-memory/](20-memory/) - Long-term memory, episodic recall, user preferences
- [21-multi-agent/](21-multi-agent/) - Orchestrator-worker, agent handoff, parallelization
- [22-guardrails/](22-guardrails/) - Safety, content filtering, human-in-the-loop
- [23-planning/](23-planning/) - Task decomposition, self-reflection, chain-of-thought
- [24-recursive-llm/](24-recursive-llm/) - Recursive Language Models for long-context handling

### 🔧 Infrastructure

- [30-observability/](30-observability/) - Telemetry, tracing, metrics
- [31-mcp/](31-mcp/) - Model Context Protocol integration

### 🔐 Security

- [32-authentication/](32-authentication/) - **NEW** Argon2id, MFA (TOTP/WebAuthn), sessions, OAuth2/OIDC federation
- [33-security/](33-security/) - **NEW** Crypto services, secrets management, agentic AI security, RBAC/policy engine

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

### Phase 4.5: Recursive Processing (Week 11-12)

1. `24-recursive-llm/` - Recursive Language Models for arbitrarily long prompts

### Phase 5: Security Hardening (Week 12-14)

1. `32-authentication/` - MFA, session management, federation
2. `33-security/` - Crypto, secrets, agentic security, RBAC

### Phase 6: Finalization (Week 13-14)

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
| 24-recursive-llm | High | 1-2 weeks |
| 30-observability | Low | 2-3 days |
| 31-mcp | Medium | 3-4 days |
| 32-authentication | High | 1-2 weeks |
| 33-security | High | 2-3 weeks |
| 99-cleanup | Low | 2-3 days |

**Total Estimated**: 14-18 weeks

---

## Cross-Cutting Concerns

All service todos share these requirements:

1. **Pattern Compliance**: Use `OperationResult[T]`, Actor-based permissions, MessageBus events
2. **Testing**: >90% coverage on service layer
3. **Documentation**: Update `docs/developer/services/` for each service
4. **AGENTS.md**: Update ownership in relevant AGENTS.md files
5. **Delete Deprecated Code**: See policy below

---

## Deprecated Code Policy

> ⚠️ **IMPORTANT**: ATLAS is in production with no external user base. When completing refactoring tasks:

1. **No Backward Compatibility Shims**: Do not maintain deprecated APIs, facades, or compatibility layers
2. **Delete Old Code Immediately**: When replacing components with upgrades, delete the old implementation entirely
3. **No Deprecation Warnings**: Skip the deprecation warning phase - just remove the old code
4. **Update All Callers**: Migrate all callers to the new API in the same PR
5. **Clean Imports**: Remove old exports from `__init__.py` files

**Rationale**: Without external users depending on our APIs, maintaining deprecated code paths:
- Adds unnecessary complexity
- Increases maintenance burden
- Creates confusion about which code path to use
- Wastes test cycles on dead code

**Example** (from 02-budget completion):
- ❌ Wrong: Keep `BudgetManager` as a facade with deprecation warnings
- ✅ Correct: Delete `manager.py`, update all callers to use new services directly

---

## References

- Architecture overview: `docs/architecture-overview.md`
- Service pattern: `00-foundation/service-pattern.md`
- MessageBus: `core/messaging/`
