# Memory Service (SOTA)

> **Status**: 📋 Planning  
> **Priority**: High  
> **Complexity**: High  
> **Effort**: 1-2 weeks  
> **Created**: 2026-01-07

---

## Overview

Implement a comprehensive memory system based on state-of-the-art agent architectures:

1. **Short-term Memory** - Context window management
2. **Long-term Memory** - Vector store with retrieval (MIPS)
3. **Episodic Memory** - Past interaction recall
4. **Semantic Memory** - Learned facts and preferences
5. **Working Memory** - Cross-task scratchpad

### Research References

- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- Microsoft Magentic-One Task/Progress Ledgers

---

## Phases

### Phase 1: Short-term Memory (Context Management)

- [ ] **1.1** Enhance `LLMContextManager` with memory abstraction
- [ ] **1.2** Implement sliding window with priority retention
- [ ] **1.3** Token budget tracking per conversation
- [ ] **1.4** Automatic summarization of older context
- [ ] **1.5** Context compression strategies

### Phase 2: Long-term Memory (Vector Store)

- [ ] **2.1** Create `core/services/memory/` package
- [ ] **2.2** Implement `LongTermMemoryService`:
  - `store_memory(content, metadata)` - Store with embedding
  - `recall(query, top_k)` - MIPS retrieval
  - `forget(memory_id)` - Explicit deletion
  - `get_related(memory_id, top_k)` - Similar memories
- [ ] **2.3** Integrate with RAGService for embeddings
- [ ] **2.4** Memory importance scoring
- [ ] **2.5** Automatic memory consolidation

### Phase 3: Episodic Memory (Interaction History)

- [ ] **3.1** Implement `EpisodicMemoryService`:
  - `record_episode(interaction)` - Store interaction
  - `recall_similar(context)` - Find similar past interactions
  - `get_recent_episodes(n)` - Recent history
  - `search_episodes(query)` - Semantic search
- [ ] **3.2** Episode summarization
- [ ] **3.3** Cross-conversation linking
- [ ] **3.4** Privacy controls (user can clear)

### Phase 4: Semantic Memory (Learned Facts)

- [ ] **4.1** Implement `SemanticMemoryService`:
  - `learn_fact(subject, predicate, object, confidence)` - Store fact
  - `query_facts(subject)` - Get known facts
  - `update_belief(fact_id, new_confidence)` - Update certainty
  - `contradiction_check(fact)` - Detect conflicts
- [ ] **4.2** User preference extraction
- [ ] **4.3** Entity knowledge graph
- [ ] **4.4** Fact verification with sources

### Phase 5: Working Memory (Scratchpad)

- [ ] **5.1** Implement `WorkingMemoryService`:
  - `set(key, value, scope)` - Store temporary data
  - `get(key, scope)` - Retrieve data
  - `clear_scope(scope)` - Clear scope
  - `get_all(scope)` - All data in scope
- [ ] **5.2** Scope management (conversation, job, task, global)
- [ ] **5.3** TTL support for temporary data
- [ ] **5.4** Cross-service access

### Phase 6: Memory Orchestration

- [ ] **6.1** Unified `MemoryService` facade
- [ ] **6.2** Memory injection into agent context
- [ ] **6.3** Automatic memory type selection
- [ ] **6.4** Memory-aware prompt construction

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Memory Service                                  │
│                        (core/services/memory/)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Short-term     │  │   Long-term     │  │   Episodic      │             │
│  │  (Context Mgmt) │  │   (Vector DB)   │  │   (Interactions)│             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                                  │
│  │   Semantic      │  │   Working       │                                  │
│  │   (Facts/KB)    │  │   (Scratchpad)  │                                  │
│  └─────────────────┘  └─────────────────┘                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Memory Orchestrator                                  │
│  • Context injection • Retrieval ranking • Memory consolidation             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────┐            ┌──────────────────┐
          │    RAGService   │            │  Vector Store    │
          │   (embeddings)  │            │   (PostgreSQL)   │
          └─────────────────┘            └──────────────────┘
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `memory.stored` | `MemoryEvent` | MemoryService |
| `memory.recalled` | `MemoryRecallEvent` | MemoryService |
| `memory.consolidated` | `MemoryConsolidationEvent` | MemoryService |
| `memory.forgotten` | `MemoryEvent` | MemoryService |
| `memory.fact_learned` | `FactEvent` | SemanticMemoryService |
| `memory.episode_recorded` | `EpisodeEvent` | EpisodicMemoryService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/memory/__init__.py` | Package exports |
| `core/services/memory/types.py` | Memory types, events |
| `core/services/memory/short_term.py` | Context management |
| `core/services/memory/long_term.py` | Vector store memory |
| `core/services/memory/episodic.py` | Interaction history |
| `core/services/memory/semantic.py` | Facts and knowledge |
| `core/services/memory/working.py` | Scratchpad |
| `core/services/memory/orchestrator.py` | Unified facade |
| `modules/memory_store/` | Database models |
| `tests/services/memory/` | Service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `core/services/rag.py` - Embedding generation
- `core/context/llm_context_manager.py` - Context window
- Vector storage (PostgreSQL pgvector or dedicated)

---

## Success Criteria

1. Agent can recall relevant past interactions
2. User preferences persist across sessions
3. Facts accumulate and inform responses
4. Context window optimally utilized
5. Working memory enables complex multi-step tasks
6. Privacy controls respected

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Vector storage backend? | pgvector / Qdrant / Chroma / Weaviate | TBD |
| Memory retention policy? | Infinite / Time-based / Importance-based | TBD |
| Cross-tenant memory isolation? | Strict / Shared facts / Configurable | TBD |
| Memory consolidation frequency? | Real-time / Batch / Hybrid | TBD |
| Embedding model? | Same as RAG / Dedicated / Configurable | TBD |
| Privacy: User memory deletion? | Full wipe / Selective / Audit trail | TBD |
