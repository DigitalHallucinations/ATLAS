# Multi-Agent Orchestration (SOTA)

> **Status**: 📋 Planning  
> **Priority**: Medium  
> **Complexity**: High  
> **Effort**: 2-3 weeks  
> **Created**: 2026-01-07

---

## Overview

Implement multi-agent orchestration patterns based on state-of-the-art research:

### Research References

- [Microsoft Magentic-One](https://www.microsoft.com/en-us/research/publication/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - Orchestrator-Worker pattern
- [LangGraph Multi-Agent Systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)

---

## Patterns to Implement

### 1. Orchestrator-Worker Pattern

```Text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Orchestrator                                    │
│  • Task decomposition  • Subtask assignment  • Result synthesis             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Worker 1   │  │  Worker 2   │  │  Worker 3   │  │  Worker N   │        │
│  │  (Code)     │  │  (Research) │  │  (Analysis) │  │  (Custom)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Magentic-One Style (Task/Progress Ledgers)

- **Task Ledger**: Track what needs to be done
- **Progress Ledger**: Track what has been accomplished
- **Specialized Agents**: WebSurfer, FileSurfer, Coder, ComputerTerminal

### 3. Agent Handoff

- Seamless transfer of context between specialized agents
- Handoff triggers and criteria
- Context preservation

---

## Phases

### Phase 1: Agent Registry

- [ ] **1.1** Create `core/services/orchestration/` package
- [ ] **1.2** Implement `AgentRegistry`:
  - `register_agent(manifest)` - Register agent capabilities
  - `list_agents()` - Available agents
  - `get_agent(agent_id)` - Get agent details
  - `get_agents_for_capability(capability)` - Find capable agents
- [ ] **1.3** Agent capability manifests
- [ ] **1.4** Agent health tracking

### Phase 2: Task Decomposition

- [ ] **2.1** Implement `TaskDecomposer`:
  - `decompose(task, strategy)` - Break into subtasks
  - `estimate_complexity(subtask)` - Complexity scoring
  - `suggest_agent(subtask)` - Agent recommendation
- [ ] **2.2** Decomposition strategies:
  - Sequential
  - Parallel
  - Hierarchical
- [ ] **2.3** Dependency graph generation

### Phase 3: Orchestrator Service

- [ ] **3.1** Implement `OrchestratorService`:
  - `execute_plan(plan)` - Run orchestrated execution
  - `assign_task(task, agent)` - Delegate to worker
  - `collect_result(task_id)` - Gather results
  - `synthesize_results(results)` - Combine outputs
- [ ] **3.2** Ledger management (Task + Progress)
- [ ] **3.3** Error recovery and retry
- [ ] **3.4** Parallel execution coordination

### Phase 4: Agent Handoff

- [ ] **4.1** Implement handoff protocol:
  - `initiate_handoff(from_agent, to_agent, context)`
  - `accept_handoff(handoff_id)`
  - `complete_handoff(handoff_id, result)`
- [ ] **4.2** Context serialization
- [ ] **4.3** Handoff triggers
- [ ] **4.4** Rollback on failure

### Phase 5: Specialized Agents

- [ ] **5.1** Define base specialized agents:
  - CoderAgent - Code generation/modification
  - ResearcherAgent - Information gathering
  - AnalystAgent - Data analysis
  - WriterAgent - Content creation
- [ ] **5.2** Agent persona integration
- [ ] **5.3** Tool/skill assignment per agent

### Phase 6: UI Integration

- [ ] **6.1** Orchestration status visualization
- [ ] **6.2** Agent activity dashboard
- [ ] **6.3** Handoff notifications
- [ ] **6.4** Manual intervention points

---

## Architecture

```Text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Orchestration Service                                │
│                    (core/services/orchestration/)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Orchestrator                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ Task Ledger  │  │Progress Ledger│  │Result Synth. │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────┴─────────────────────────────────┐     │
│  │                        Agent Registry                              │     │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │     │
│  │  │ Coder   │ │Research │ │ Analyst │ │ Writer  │ │ Custom  │     │     │
│  │  │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │     │     │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘     │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Handoff Protocol                              │   │
│  │  Context Serialization → Transfer → Accept → Execute → Report       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
| ---------- | ------- | ---------- |
| `orchestration.plan_started` | `PlanEvent` | OrchestratorService |
| `orchestration.task_assigned` | `TaskAssignmentEvent` | OrchestratorService |
| `orchestration.task_completed` | `TaskResultEvent` | OrchestratorService |
| `orchestration.plan_completed` | `PlanResultEvent` | OrchestratorService |
| `orchestration.handoff_initiated` | `HandoffEvent` | HandoffService |
| `orchestration.handoff_completed` | `HandoffEvent` | HandoffService |
| `agent.registered` | `AgentEvent` | AgentRegistry |
| `agent.status_changed` | `AgentStatusEvent` | AgentRegistry |

---

## Files to Create

| File | Purpose |
| ------ | --------- |
| `core/services/orchestration/__init__.py` | Package exports |
| `core/services/orchestration/types.py` | Types and events |
| `core/services/orchestration/registry.py` | AgentRegistry |
| `core/services/orchestration/decomposer.py` | TaskDecomposer |
| `core/services/orchestration/orchestrator.py` | OrchestratorService |
| `core/services/orchestration/handoff.py` | Handoff protocol |
| `core/services/orchestration/ledgers.py` | Task/Progress ledgers |
| `modules/orchestration/agents/` | Specialized agent definitions |
| `tests/services/orchestration/` | Service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- **Prerequisite**: [20-memory](../20-memory/) - Working memory for context
- [06-personas](../06-personas/) - Agent personas
- [08-skills-tools](../08-skills-tools/) - Agent capabilities

---

## Deferred from 06-personas SOTA Enhancements

> The following was deferred from [06-personas/SOTA-ENHANCEMENTS.md](../06-personas/SOTA-ENHANCEMENTS.md) Phase 3.

### Persona as Multi-Agent Workers

The SOTA persona enhancements include types for personas to participate in multi-agent orchestration:

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

### PersonaOrchestrator Requirements

- [ ] **P.1** Implement `PersonaOrchestrator` in `core/services/orchestration/persona_orchestrator.py`:
  - `spawn_worker(persona_id, task)` - Create worker instance from persona
  - `delegate_task(to_persona, task, context)` - Assign work to persona
  - `collect_results(task_ids)` - Gather persona worker outputs
  - `synthesize(results)` - Combine into coherent response
- [ ] **P.2** Task decomposition with persona assignment based on specializations
- [ ] **P.3** Parallel execution across personas with rate limiting
- [ ] **P.4** Result aggregation strategies per persona type

### Persona Ledger Integration (Magentic-One Style)

- [ ] **P.5** Per-persona task ledger tracking assigned subtasks
- [ ] **P.6** Per-persona progress ledger with completion status
- [ ] **P.7** Cross-persona coordination via shared ledgers
- [ ] **P.8** Persona escalation paths when task exceeds capabilities

### Integration with Existing Persona Services

The following services from `core/services/personas/` should be leveraged:

| Service | Integration Point |
| --------- | ------------------- |
| `PersonaSwitchingService` | Use handoff protocol for persona-to-persona delegation |
| `PersonaMemoryService` | Share working memory context during orchestration |
| `PersonaSafetyService` | Apply per-persona safety policies to delegated tasks |
| `PersonaAnalyticsService` | Track orchestration performance metrics per persona |

---

## Success Criteria

1. Complex tasks decomposed automatically
2. Multiple agents collaborate effectively
3. Handoffs preserve context
4. Ledgers track progress accurately
5. Results synthesized coherently
6. Error recovery handles failures gracefully

---

## Open Questions

| Question | Options | Decision |
| ---------- | --------- | ---------- |
| Agent selection strategy? | Capability-based / LLM-guided / Hybrid | TBD |
| Maximum parallel agents? | Fixed limit / Dynamic / Resource-based | TBD |
| Handoff approval requirement? | Always automatic / User approval / Risk-based | TBD |
| Cross-agent communication? | Via orchestrator only / Direct / Both | TBD |
| Agent model assignment? | Same model / Per-agent / Dynamic | TBD |
| Failure escalation? | Retry / Fallback agent / User intervention | TBD |
