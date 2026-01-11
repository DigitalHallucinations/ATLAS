# Persona SOTA Enhancements

> **Status**: ✅ Complete  
> **Priority**: High  
> **Complexity**: High  
> **Effort**: 2-3 weeks  
> **Created**: 2026-01-10  
> **Completed**: 2026-01-15  
> **Depends On**: 06-personas Phase 1 (✅ Complete)

---

## Overview

Enhance the PersonaService with state-of-the-art agent capabilities. This bridges the persona system with memory, multi-agent coordination, guardrails, and self-improvement systems.

### Selected Enhancements

1. ✅ **Agent Memory Integration** - Persona-scoped memory contexts
2. ❌ ~~Dynamic Persona Composition~~ - Deferred
3. ✅ **Persona Switching Strategies** - Handoff protocols, conditional switching
4. ✅ **Multi-Agent Coordination** - Orchestrator patterns, parallel execution
5. ✅ **Self-Improvement Capabilities** - Performance analytics, prompt refinement
6. ✅ **Guardrails per Persona** - Safety policies, output filtering, HITL triggers

---

## Phase 1: Persona Memory Context

> **Goal**: Each persona maintains its own memory context for continuity across sessions

### 1.1 Persona Working Memory

```python
# New types in core/services/personas/types.py
@dataclass
class PersonaMemoryContext:
    """Working memory scoped to a persona."""
    persona_id: str
    scratchpad: Dict[str, Any]  # Key-value working memory
    active_goals: List[str]     # Current objectives
    task_context: Optional[str] # Current task being worked on
    last_accessed: datetime

@dataclass  
class PersonaKnowledge:
    """Semantic knowledge specific to a persona."""
    persona_id: str
    learned_facts: List[LearnedFact]
    user_preferences: Dict[str, Any]
    domain_expertise: Dict[str, float]  # Topic -> confidence
```

### 1.2 Memory Service Integration

- [x] **1.2.1** Add `PersonaMemoryService` (extends MemoryService with persona scope)
- [x] **1.2.2** Implement persona-scoped storage:
  - `get_working_memory(persona_id)` - Get persona's scratchpad
  - `update_working_memory(persona_id, key, value)` - Update scratchpad
  - `clear_working_memory(persona_id)` - Reset scratchpad
  - `get_persona_knowledge(persona_id)` - Get learned facts
  - `learn_from_interaction(persona_id, interaction)` - Extract & store learnings
- [x] **1.2.3** Memory injection into persona context prompts
- [x] **1.2.4** Cross-session memory persistence

### 1.3 Episodic Memory per Persona

- [x] **1.3.1** Track interaction history per persona
- [x] **1.3.2** Implement `recall_similar_episodes(persona_id, context)`
- [x] **1.3.3** Episode summarization for long-term storage

---

## Phase 2: Persona Switching Strategies

> **Goal**: Intelligent switching between personas with context preservation

### 2.1 Handoff Protocol

```python
@dataclass
class PersonaHandoff:
    """Context transfer between personas."""
    handoff_id: str
    from_persona: str
    to_persona: str
    context: HandoffContext
    reason: str
    timestamp: datetime
    status: HandoffStatus  # pending, accepted, completed, failed

@dataclass
class HandoffContext:
    """What gets transferred during handoff."""
    conversation_summary: str
    active_goals: List[str]
    working_memory_snapshot: Dict[str, Any]
    user_preferences: Dict[str, Any]
    pending_actions: List[PendingAction]
```

### 2.2 Switching Strategies

- [x] **2.2.1** Implement `PersonaSwitchingService`:
  - `initiate_handoff(from_persona, to_persona, context)` - Start transfer
  - `prepare_context(persona_id)` - Serialize current context
  - `restore_context(persona_id, context)` - Restore on new persona
  - `complete_handoff(handoff_id)` - Finalize switch
- [x] **2.2.2** Handoff triggers:
  - Explicit user request
  - Capability mismatch (persona can't handle request)
  - Domain change detection
  - Escalation (complexity exceeds persona scope)
- [x] **2.2.3** Graceful degradation:
  - Fallback persona chain
  - Partial capability handoff
  - Emergency fallback to ATLAS base

### 2.3 Conditional Switching

- [x] **2.3.1** Intent-based switching rules:
  ```python
  class SwitchingRule:
      trigger: str  # Intent pattern or keyword
      from_persona: Optional[str]  # None = any
      to_persona: str
      priority: int
      conditions: List[Condition]
  ```
- [x] **2.3.2** Automatic persona suggestion based on user query
- [x] **2.3.3** User confirmation flow for suggested switches

---

## Phase 3: Multi-Agent Persona Coordination

> **Goal**: Personas can work together, delegate, and orchestrate

### 3.1 Persona as Agent

```python
@dataclass
class PersonaAgentManifest:
    """Persona capabilities for multi-agent orchestration."""
    persona_id: str
    capabilities: List[str]
    specializations: List[str]  # Domains of expertise
    max_parallel_tasks: int
    can_delegate: bool
    can_be_delegated_to: bool
    orchestration_role: OrchestratorRole  # worker, orchestrator, hybrid
```

### 3.2 Orchestrator Patterns

> **Note**: Full orchestrator implementation deferred to 21-multi-agent. Foundation types defined in `core/services/personas/types.py`.

- [ ] **3.2.1** Implement `PersonaOrchestrator`:
  - `spawn_worker(persona_id, task)` - Create worker instance
  - `delegate_task(to_persona, task, context)` - Assign work
  - `collect_results(task_ids)` - Gather worker outputs
  - `synthesize(results)` - Combine into coherent response
- [ ] **3.2.2** Task decomposition with persona assignment
- [ ] **3.2.3** Parallel execution across personas
- [ ] **3.2.4** Result aggregation strategies

### 3.3 Ledger Integration (Magentic-One Style)

> **Note**: Ledger integration deferred to 21-multi-agent.

- [ ] **3.3.1** Per-persona task ledger
- [ ] **3.3.2** Per-persona progress ledger
- [ ] **3.3.3** Cross-persona coordination via shared ledgers

---

## Phase 4: Self-Improvement Capabilities

> **Goal**: Personas improve over time through analytics and refinement

### 4.1 Performance Analytics

```python
@dataclass
class PersonaPerformanceMetrics:
    """Track persona effectiveness."""
    persona_id: str
    period: TimePeriod
    
    # Usage metrics
    total_interactions: int
    avg_response_time_ms: float
    token_usage: TokenUsage
    
    # Quality metrics
    task_success_rate: float
    user_satisfaction_score: Optional[float]  # From feedback
    escalation_rate: float  # How often it hands off
    retry_rate: float  # How often responses need retry
    
    # Capability metrics
    tools_used: Dict[str, int]  # Tool -> usage count
    skills_invoked: Dict[str, int]
    capability_gaps: List[str]  # Requested but unavailable
```

### 4.2 Analytics Service

- [x] **4.2.1** Implement `PersonaAnalyticsService`:
  - `record_interaction(persona_id, interaction)` - Track usage
  - `get_metrics(persona_id, period)` - Retrieve analytics
  - `get_comparison(persona_ids, period)` - Compare personas
  - `identify_improvement_areas(persona_id)` - Suggest improvements
- [ ] **4.2.2** Dashboard integration for GTKUI
- [ ] **4.2.3** Automated reporting

### 4.3 Automatic Prompt Refinement

- [x] **4.3.1** Track prompt effectiveness:
  - Which system prompts lead to better outcomes
  - A/B testing infrastructure
- [x] **4.3.2** Implement `PromptRefinementService`:
  - `suggest_refinement(persona_id, context)` - AI-suggested improvements
  - `apply_refinement(persona_id, refinement)` - Update persona
  - `rollback_refinement(persona_id, version)` - Undo changes
- [x] **4.3.3** Version history for persona prompts
- [x] **4.3.4** Refinement approval workflow

### 4.4 A/B Testing

- [ ] **4.4.1** Persona variation support:
  ```python
  class PersonaVariant:
      base_persona_id: str
      variant_id: str
      modifications: Dict[str, Any]
      traffic_percentage: float
      metrics: PersonaPerformanceMetrics
  ```
- [x] **4.4.2** Traffic splitting
- [x] **4.4.3** Statistical significance testing
- [x] **4.4.4** Automatic winner promotion

---

## Phase 5: Guardrails per Persona

> **Goal**: Each persona has tailored safety policies

### 5.1 Persona Safety Policies

```python
@dataclass
class PersonaSafetyPolicy:
    """Safety configuration for a persona."""
    persona_id: str
    
    # Action controls
    allowed_actions: Set[str]  # Whitelist
    blocked_actions: Set[str]  # Blacklist
    require_approval: Set[str]  # HITL required
    
    # Content policies
    input_filters: List[ContentFilter]
    output_filters: List[ContentFilter]
    pii_handling: PIIPolicy
    
    # Rate limits
    max_actions_per_minute: int
    max_tokens_per_conversation: int
    max_tool_calls_per_turn: int
    
    # Risk tolerance
    risk_threshold: float  # 0.0 - 1.0
    auto_escalation_threshold: float
```

### 5.2 Guardrails Integration

- [x] **5.2.1** Extend PersonaService with safety checks:
  - `check_persona_action(persona_id, action)` - Pre-flight check
  - `get_safety_policy(persona_id)` - Get active policy
  - `update_safety_policy(actor, persona_id, policy)` - Modify policy
- [x] **5.2.2** Per-persona HITL triggers
- [x] **5.2.3** Risk scoring per persona
- [x] **5.2.4** Audit logging

### 5.3 Content Filtering per Persona

- [x] **5.3.1** Persona-specific filter rules:
  - Medical persona: Allow health discussions
  - Code persona: Allow code with security caveats
  - General persona: Standard content policy
- [x] **5.3.2** Output sanitization hooks
- [x] **5.3.3** Dynamic filter adjustment

### 5.4 Human-in-the-Loop per Persona

- [x] **5.4.1** Configurable HITL triggers per persona
- [x] **5.4.2** Approval workflow integration
- [x] **5.4.3** Escalation paths:
  ```python
  class EscalationPath:
      from_persona: str
      to_persona: str  # Or to human
      triggers: List[str]
      auto_escalate: bool
      timeout_seconds: int
  ```

---

## Implementation Priority

| Phase | Priority | Dependencies | Effort |
|-------|----------|--------------|--------|
| Phase 5: Guardrails | 🔴 High | 22-guardrails | 3-4 days |
| Phase 1: Memory | 🔴 High | 20-memory | 3-4 days |
| Phase 4: Self-Improvement | 🟡 Medium | 11-analytics | 3-4 days |
| Phase 2: Switching | 🟡 Medium | Phase 1 | 2-3 days |
| Phase 3: Multi-Agent | 🟢 Lower | 21-multi-agent | 4-5 days |

---

## New Files Created

| File | Purpose | Status |
|------|---------|--------|
| `core/services/personas/memory.py` | PersonaMemoryService | ✅ Created |
| `core/services/personas/switching.py` | PersonaSwitchingService, HandoffProtocol | ✅ Created |
| `core/services/personas/orchestration.py` | PersonaOrchestrator | ⏳ Deferred to 21-multi-agent |
| `core/services/personas/analytics.py` | PersonaAnalyticsService + PromptRefinementService | ✅ Created |
| `core/services/personas/safety.py` | PersonaSafetyService | ✅ Created |
| `core/services/personas/refinement.py` | PromptRefinementService | ✅ Merged into analytics.py |
| `tests/services/personas/test_persona_memory.py` | Memory service tests | ✅ Created |
| `tests/services/personas/test_persona_switching.py` | Switching service tests | ✅ Created |
| `tests/services/personas/test_persona_analytics.py` | Analytics service tests | ✅ Created |
| `tests/services/personas/test_persona_safety.py` | Safety service tests | ✅ Created |

---

## Integration Points

### With Memory Service (20-memory)
- Persona-scoped working memory
- Episodic memory per persona
- Semantic knowledge extraction

### With Multi-Agent (21-multi-agent)
- Personas as agents in orchestration
- Task delegation between personas
- Ledger participation

### With Guardrails (22-guardrails)
- Per-persona safety policies
- Persona-specific HITL triggers
- Content filtering rules

### With Analytics (11-analytics)
- Performance metrics per persona
- Usage tracking
- Quality scoring

---

## Success Criteria

1. Each persona can maintain working memory across sessions
2. Context is preserved during persona switches
3. Personas can delegate to each other
4. Performance analytics available for all personas
5. Each persona has configurable safety policies
6. A/B testing enables data-driven persona improvement
7. >85% test coverage on new components

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Should memory persist across conversation restarts? | Yes / No / Configurable | Configurable |
| Who can modify persona safety policies? | Admin only / Owner / Any user | Admin + Owner |
| Automatic vs. manual prompt refinement? | Auto / Manual / Hybrid with approval | Hybrid |
| How long to retain performance metrics? | 30 days / 90 days / Forever | 90 days default |
