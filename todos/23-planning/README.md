# Planning & Reflection Service (SOTA)

> **Status**: 📋 Planning  
> **Priority**: Medium  
> **Complexity**: High  
> **Effort**: 1-2 weeks  
> **Created**: 2026-01-07

---

## Overview

Implement planning and self-reflection capabilities based on state-of-the-art research:

### Research References

- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) - Planning & Reflection
- Chain-of-Thought (CoT) prompting
- Tree-of-Thoughts (ToT) exploration
- ReAct (Reasoning + Acting)
- Reflexion (self-improvement through reflection)
- Microsoft Magentic-One Task/Progress Ledgers

---

## Core Concepts

### 1. Task Decomposition

Break complex tasks into manageable subtasks:
- **Chain-of-Thought**: Step-by-step reasoning
- **Tree-of-Thoughts**: Explore multiple paths
- **Least-to-Most**: Build from simple to complex

### 2. Self-Reflection

Agent evaluates its own performance:
- **Reflexion**: Learn from mistakes
- **Self-consistency**: Verify multiple approaches
- **Critique**: Evaluate solution quality

### 3. Plan Execution & Tracking

- **Task Ledger**: What needs to be done
- **Progress Ledger**: What has been accomplished
- **Replanning**: Adjust based on results

---

## Phases

### Phase 1: Planning Service Core

- [ ] **1.1** Create `core/services/planning/` package
- [ ] **1.2** Implement `PlanningService`:
  - `create_plan(goal, constraints)` - Generate plan
  - `decompose_task(task, strategy)` - Break down task
  - `estimate_effort(plan)` - Effort estimation
  - `validate_plan(plan)` - Feasibility check
- [ ] **1.3** Planning strategies:
  - Sequential
  - Hierarchical
  - DAG (dependency graph)

### Phase 2: Chain-of-Thought

- [ ] **2.1** Implement CoT prompting:
  - `reason_step_by_step(problem)` - Generate reasoning
  - `extract_steps(reasoning)` - Parse steps
  - `validate_reasoning(steps)` - Check logic
- [ ] **2.2** Few-shot example injection
- [ ] **2.3** Domain-specific CoT templates

### Phase 3: Tree-of-Thoughts

- [ ] **3.1** Implement ToT exploration:
  - `generate_thoughts(state)` - Branch generation
  - `evaluate_thought(thought)` - Score branches
  - `select_path(thoughts, strategy)` - Path selection
  - `backtrack(state)` - Return to previous state
- [ ] **3.2** Search strategies:
  - Breadth-first
  - Depth-first
  - Best-first
- [ ] **3.3** Pruning strategies

### Phase 4: ReAct (Reasoning + Acting)

- [ ] **4.1** Implement ReAct loop:
  - `think(observation)` - Generate thought
  - `act(thought)` - Take action
  - `observe(action_result)` - Process result
  - `iterate_until_done(goal)` - Full loop
- [ ] **4.2** Action-observation pairing
- [ ] **4.3** Loop termination criteria

### Phase 5: Reflection & Self-Improvement

- [ ] **5.1** Implement `ReflectionService`:
  - `evaluate_execution(plan, results)` - Assess performance
  - `identify_failures(execution)` - Find issues
  - `generate_insights(evaluation)` - Learn lessons
  - `update_approach(insights)` - Apply learnings
- [ ] **5.2** Reflection triggers:
  - On failure
  - On completion
  - Periodic
- [ ] **5.3** Learning persistence

### Phase 6: Ledger Management

- [ ] **6.1** Implement `TaskLedger`:
  - Track planned tasks
  - Task dependencies
  - Priority ordering
- [ ] **6.2** Implement `ProgressLedger`:
  - Track completed tasks
  - Capture results
  - Note blockers
- [ ] **6.3** Ledger synchronization with jobs/tasks

### Phase 7: Integration

- [ ] **7.1** Integrate with JobService for job planning
- [ ] **7.2** Integrate with Orchestration for multi-agent planning
- [ ] **7.3** Memory integration for learning persistence

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Planning Service                                    │
│                     (core/services/planning/)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Plan Generator                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │     CoT      │  │     ToT      │  │   ReAct      │               │   │
│  │  │  (Linear)    │  │  (Branching) │  │  (Iterative) │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Ledger Management                                │   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐                 │   │
│  │  │    Task Ledger       │  │   Progress Ledger    │                 │   │
│  │  │  • Planned tasks     │  │  • Completed tasks   │                 │   │
│  │  │  • Dependencies      │  │  • Results           │                 │   │
│  │  │  • Priorities        │  │  • Blockers          │                 │   │
│  │  └──────────────────────┘  └──────────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Reflection Engine                                │   │
│  │  Evaluate → Identify Issues → Generate Insights → Update Approach    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `planning.plan_created` | `PlanEvent` | PlanningService |
| `planning.step_started` | `StepEvent` | PlanningService |
| `planning.step_completed` | `StepResultEvent` | PlanningService |
| `planning.plan_completed` | `PlanResultEvent` | PlanningService |
| `planning.replan_triggered` | `ReplanEvent` | PlanningService |
| `reflection.evaluation_started` | `ReflectionEvent` | ReflectionService |
| `reflection.insight_generated` | `InsightEvent` | ReflectionService |
| `reflection.approach_updated` | `ApproachEvent` | ReflectionService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/planning/__init__.py` | Package exports |
| `core/services/planning/types.py` | Types and events |
| `core/services/planning/service.py` | PlanningService |
| `core/services/planning/cot.py` | Chain-of-Thought |
| `core/services/planning/tot.py` | Tree-of-Thoughts |
| `core/services/planning/react.py` | ReAct implementation |
| `core/services/planning/reflection.py` | ReflectionService |
| `core/services/planning/ledgers.py` | Task/Progress ledgers |
| `tests/services/planning/` | Service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- **Prerequisite**: [20-memory](../20-memory/) - Learning persistence
- [05-jobs-tasks](../05-jobs-tasks/) - Job/task integration
- [21-multi-agent](../21-multi-agent/) - Multi-agent planning

---

## Success Criteria

1. Complex tasks decomposed effectively
2. Multiple reasoning strategies available
3. Self-reflection improves performance
4. Ledgers track progress accurately
5. Replanning handles failures
6. Learnings persist across sessions

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Default planning strategy? | CoT / ToT / ReAct / Auto-select | TBD |
| ToT branch limit? | 3 / 5 / Dynamic | TBD |
| Reflection frequency? | Every step / On failure / Configurable | TBD |
| Learning storage? | Memory service / Dedicated store | TBD |
| Plan execution timeout? | Per-step / Total plan / Both | TBD |
| User visibility into planning? | Full transparency / Summary / Minimal | TBD |
