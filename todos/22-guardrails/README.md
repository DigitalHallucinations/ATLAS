# Guardrails & Safety Service (SOTA)

> **Status**: 📋 Planning  
> **Priority**: High  
> **Complexity**: Medium  
> **Effort**: 1 week  
> **Created**: 2026-01-07

---

## Overview

Implement comprehensive guardrails and safety mechanisms based on state-of-the-art research:

### Research References

- [OpenAI: Practices for Governing Agentic AI Systems](https://openai.com/index/practices-for-governing-agentic-ai-systems/)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - Human-in-the-loop patterns
- Action reversibility detection
- Content filtering and safety checks

---

## Core Concepts

### 1. Human-in-the-Loop (HITL)

Require human approval for high-risk operations:
- Financial transactions
- Data deletion
- External communications
- System modifications

### 2. Action Reversibility

Classify actions by reversibility:
- **Reversible**: Can be undone (file edits, soft deletes)
- **Irreversible**: Cannot be undone (external API calls, hard deletes)
- **Partially Reversible**: Can be partially restored

### 3. Content Safety

Filter and validate:
- Input content (user messages)
- Output content (agent responses)
- Tool outputs (command results)

---

## Phases

### Phase 1: Guardrails Service Core

- [ ] **1.1** Create `core/services/guardrails/` package
- [ ] **1.2** Implement `GuardrailsService`:
  - `check_action(action, context)` - Pre-flight check
  - `require_approval(action, reason)` - Request HITL
  - `log_action(action, result)` - Audit trail
  - `is_reversible(action)` - Reversibility check
- [ ] **1.3** Action classification system
- [ ] **1.4** Risk scoring

### Phase 2: Human-in-the-Loop

- [ ] **2.1** Implement approval workflow:
  - `request_approval(action, context)` - Submit for approval
  - `approve(request_id, actor)` - Approve action
  - `reject(request_id, actor, reason)` - Reject action
  - `get_pending_approvals()` - List pending
- [ ] **2.2** Approval timeout handling
- [ ] **2.3** Escalation paths
- [ ] **2.4** Bulk approval for similar actions

### Phase 3: Action Reversibility

- [ ] **3.1** Implement reversibility registry:
  - Register action patterns with reversibility
  - Automatic classification for tools
- [ ] **3.2** Undo operation generation
- [ ] **3.3** Rollback transaction support
- [ ] **3.4** Reversibility warnings in UI

### Phase 4: Content Safety

- [ ] **4.1** Implement `ContentFilter`:
  - `check_input(content)` - Validate user input
  - `check_output(content)` - Validate agent output
  - `check_tool_output(output)` - Validate tool results
- [ ] **4.2** Configurable filter rules
- [ ] **4.3** PII detection (optional)
- [ ] **4.4** Content policy enforcement

### Phase 5: Rate Limiting & Quotas

- [ ] **5.1** Implement rate limiting:
  - Per-user limits
  - Per-agent limits
  - Per-tool limits
- [ ] **5.2** Quota management
- [ ] **5.3** Circuit breaker for failing services

### Phase 6: Audit & Compliance

- [ ] **6.1** Comprehensive audit logging
- [ ] **6.2** Audit log search/export
- [ ] **6.3** Compliance reporting
- [ ] **6.4** Retention policies

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Guardrails Service                                 │
│                      (core/services/guardrails/)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Action         │  │  Human-in-      │  │  Content        │             │
│  │  Classifier     │  │  the-Loop       │  │  Filter         │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Reversibility  │  │  Rate           │  │  Audit          │             │
│  │  Registry       │  │  Limiter        │  │  Logger         │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Policy Engine                                        │
│  • Rule evaluation • Risk scoring • Decision logging                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Risk Levels

| Level | Description | Approval Required | Examples |
|-------|-------------|-------------------|----------|
| `low` | Reversible, minimal impact | No | Read operations, drafts |
| `medium` | Reversible, moderate impact | Optional | File edits, DB updates |
| `high` | Irreversible or significant impact | Yes | Delete, external API |
| `critical` | System-level or financial | Yes + 2FA | Admin operations |

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `guardrails.action_blocked` | `GuardrailEvent` | GuardrailsService |
| `guardrails.approval_requested` | `ApprovalRequestEvent` | HITL |
| `guardrails.approval_granted` | `ApprovalEvent` | HITL |
| `guardrails.approval_denied` | `ApprovalEvent` | HITL |
| `guardrails.content_filtered` | `ContentFilterEvent` | ContentFilter |
| `guardrails.rate_limited` | `RateLimitEvent` | RateLimiter |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/guardrails/__init__.py` | Package exports |
| `core/services/guardrails/types.py` | Types and events |
| `core/services/guardrails/service.py` | GuardrailsService |
| `core/services/guardrails/hitl.py` | Human-in-the-loop |
| `core/services/guardrails/reversibility.py` | Action reversibility |
| `core/services/guardrails/content_filter.py` | Content filtering |
| `core/services/guardrails/rate_limiter.py` | Rate limiting |
| `core/services/guardrails/audit.py` | Audit logging |
| `core/services/guardrails/policies.py` | Policy engine |
| `tests/services/guardrails/` | Service tests |

---

## Configuration

```yaml
guardrails:
  hitl:
    enabled: true
    timeout_minutes: 30
    escalation_after_minutes: 15
  
  content_filter:
    enabled: true
    pii_detection: true
    custom_rules: []
  
  rate_limits:
    per_user:
      requests_per_minute: 60
      tokens_per_hour: 100000
    per_tool:
      dangerous_commands:
        requests_per_hour: 10
  
  reversibility:
    require_confirmation:
      - irreversible
      - critical
```

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `core/messaging/` - MessageBus for events
- All other services (for action interception)

---

## Success Criteria

1. High-risk actions require approval
2. Irreversible actions clearly identified
3. Content filtering prevents harmful outputs
4. Rate limiting prevents abuse
5. Comprehensive audit trail
6. Configurable policies

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| HITL notification method? | In-app / Email / Push / All | TBD |
| Approval timeout behavior? | Auto-reject / Auto-approve / Escalate | TBD |
| Content filter models? | Rule-based / ML-based / Hybrid | TBD |
| Rate limit scope? | Per-tenant / Per-user / Both | TBD |
| Audit log storage? | Same DB / Separate / External | TBD |
| PII detection scope? | Input only / All / Configurable | TBD |
