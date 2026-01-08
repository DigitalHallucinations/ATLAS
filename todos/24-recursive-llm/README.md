# Recursive Language Models (RLMs)

**Research Paper:** arXiv:2512.24601v1 [cs.AI] - Zhang, Kraska, Khattab (Dec 31, 2025)

## Overview

Implement Recursive Language Models capability to handle arbitrarily long prompts through inference-time scaling. RLMs treat long prompts as an external environment, allowing programmatic examination, decomposition, and recursive self-calling over prompt snippets.

## Key Benefits

- Handle inputs up to 2 orders of magnitude beyond context windows
- Outperform base LLMs and long-context scaffolds on quality
- Comparable or cheaper cost per query vs traditional approaches

## Planned Components

### Core Module (`modules/recursive_processing/`)

- **Prompt Decomposer** - Analyzes and chunks input intelligently
- **Recursive Executor** - Manages recursive calling pattern with state
- **Result Synthesizer** - Aggregates partial results
- **Context Window Manager** - Token tracking and recursion decisions

### Integration Points

1. **Core Orchestration**
   - Extend `core/AgentRouter.py` for recursive dispatch
   - Hook into `core/ATLAS.py` main processing loop

2. **Context Management**
   - Leverage `core/context/` for state tracking
   - Extend context handlers for recursive boundaries

3. **Provider Integration**
   - Use `core/provider_manager.py` for uniform LLM calls
   - Support all existing providers (OpenAI, Anthropic, etc.)

4. **Budget Control**
   - Integrate with `modules/budget/` for cost tracking
   - Add recursive call limits and cost guards

## Implementation Phases

### Phase 1: Foundation

- [ ] Create module structure
- [ ] Implement basic prompt decomposition
- [ ] Add configuration options
- [ ] Set up test framework

### Phase 2: Core Functionality

- [ ] Implement recursive executor
- [ ] Add result synthesis
- [ ] Context window management
- [ ] State persistence across recursion

### Phase 3: Integration

- [ ] Hook into ATLAS main loop
- [ ] Provider manager integration
- [ ] Budget tracking
- [ ] Error handling and recovery

### Phase 4: Use Cases

- [ ] Document summarization
- [ ] Long conversation analysis
- [ ] Multi-document synthesis
- [ ] Extended RAG queries

### Phase 5: Optimization

- [ ] Performance benchmarking
- [ ] Cost optimization
- [ ] Latency reduction
- [ ] Quality metrics

## Considerations

### Technical Challenges

- **Cost Control** - Recursive calls multiply quickly
- **Latency** - Multiple round trips vs single long call
- **State Consistency** - Maintaining coherence across boundaries
- **Token Accounting** - Accurate budget tracking

### Safety & Limits

- Maximum recursion depth
- Cost ceiling per query
- Timeout handling
- Graceful degradation

## Testing Strategy

### Unit Tests (`tests/modules/recursive_processing/`)

- Decomposition algorithms
- State management
- Result synthesis logic

### Integration Tests

- End-to-end with mock long documents
- Multi-provider compatibility
- Budget integration

### Benchmarks

- Quality vs baseline
- Cost comparison
- Latency analysis
- Scale testing (10x, 100x context length)

## Configuration

Add to `config.yaml`:

```yaml
recursive_llm:
  enabled: false  # Experimental flag
  max_depth: 5
  chunk_size: auto  # Or explicit token count
  synthesis_strategy: hierarchical  # Or sequential
  cost_limit_per_query: 1.0  # USD
```

## Agent Scope

Per `AGENTS.md`:

- **Backend Agent** - Owns `modules/recursive_processing/` and integration logic
- **Testing Agent** - Implements test suite under `tests/`
- **Docs Agent** - Documents configuration and usage

## References

- Paper: <https://arxiv.org/abs/2512.24601>
- PDF: <https://arxiv.org/pdf/2512.24601>
