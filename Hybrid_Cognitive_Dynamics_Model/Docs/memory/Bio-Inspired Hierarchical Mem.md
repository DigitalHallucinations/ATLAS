# Documentation for `hierarchical_memory.py` Module

## Overview

The `hierarchical_memory.py` module implements a biologically-inspired hierarchical memory system. This system is designed to emulate the processes of human memory, including sensory, working, short-term, and long-term memory. The memory system also incorporates advanced mechanisms for memory consolidation, decay, and context-aware retrieval, inspired by the functions of the hippocampus and neocortex.

## Classes and Methods

### MemoryConfig

#### Description

A configuration class that defines the default parameters for various memory components.

#### Attributes

- `SENSORY_MEMORY_SIZE`: The maximum size of the sensory memory buffer.
- `WORKING_MEMORY_CAPACITY`: The capacity of the working memory.
- `SHORT_TERM_MEMORY_CAPACITY`: The capacity of the short-term memory.
- `INTERMEDIATE_MEMORY_CAPACITY`: The capacity of the intermediate memory.
- `LONG_TERM_SEMANTIC_COMPONENTS`: The number of components for the random projection in semantic memory.
- `MEMORY_DECAY_HALF_LIFE`: The half-life for memory decay, in seconds.
- `CONSOLIDATION_THRESHOLD`: The threshold for memory consolidation.
- `PATTERN_COMPLETION_THRESHOLD`: The threshold for pattern completion.
- `MEMORY_CLEANUP_THRESHOLD`: The threshold for memory cleanup.

### SensoryMemory

#### Description

Represents the sensory memory, briefly holding incoming information.

#### Methods

- `__init__(max_size: int = MemoryConfig.SENSORY_MEMORY_SIZE)`: Initializes the sensory memory with a maximum size.
- `add(input_data: Any) -> None`: Adds new input data to the sensory memory buffer.
- `get_salient_info() -> List[Any]`: Retrieves the most recent (salient) information from the buffer.

### WorkingMemory

#### Description

Represents the working memory, handling active processing of information.

#### Methods

- `__init__(capacity: int = MemoryConfig.WORKING_MEMORY_CAPACITY)`: Initializes the working memory with a specified capacity.
- `process(info: List[Any]) -> List[Any]`: Processes new information, maintaining the capacity limit.

### ShortTermMemory

#### Description

Represents the short-term memory, temporarily storing processed information.

#### Methods

- `__init__(capacity: int = MemoryConfig.SHORT_TERM_MEMORY_CAPACITY)`: Initializes the short-term memory with a specified capacity.
- `add(item: Any) -> None`: Adds a new item to short-term memory, maintaining the capacity limit.
- `get_memories_for_consolidation() -> List[Any]`: Retrieves items for potential consolidation into long-term memory.
- `clear_consolidated() -> None`: Removes items that have been consolidated.

### EnhancedLongTermEpisodicMemory

#### Description

Represents the long-term episodic memory with context-aware retrieval.

#### Methods

- `__init__(state_model)`: Initializes the long-term episodic memory with a state model.
- `add(episode: Any, context: np.ndarray) -> None`: Adds a new episode with its context to long-term episodic memory.
- `get_relevant_episodes(n: int = 10) -> List[Dict[str, Any]]`: Retrieves the most relevant episodes based on the current context.
- `get_recent_episodes(n: int = 10) -> List[Dict[str, Any]]`: Retrieves the most recent episodes.

### EnhancedLongTermSemanticMemory

#### Description

Represents the long-term semantic memory with advanced pattern operations.

#### Methods

- `__init__(n_components: int = MemoryConfig.LONG_TERM_SEMANTIC_COMPONENTS)`: Initializes the semantic memory with a specified number of components.
- `add(concept: str, related_concepts: List[str]) -> None`: Adds a concept and its related concepts to the semantic memory.
- `pattern_separation(concept1: str, concept2: str) -> Optional[float]`: Computes the separation between two concepts.
- `pattern_completion(partial_concept: str, threshold: float = MemoryConfig.PATTERN_COMPLETION_THRESHOLD) -> List[Tuple[str, float]]`: Completes a partial concept based on stored knowledge.
- `query(concept: str, n: int = 5) -> List[Tuple[str, float]]`: Queries related concepts from the semantic memory.

### ContextAwareRetrieval

#### Description

Handles context-aware memory retrieval.

#### Methods

- `__init__(state_model)`: Initializes the context-aware retrieval with a state model.
- `get_context_vector() -> np.ndarray`: Generates the context vector based on the current state.
- `context_similarity(memory_context: np.ndarray, current_context: np.ndarray) -> float`: Computes the similarity between a memory's context and the current context.

### AdaptiveMemoryConsolidation

#### Description

Handles adaptive memory consolidation based on importance and repetition.

#### Methods

- `__init__()`: Initializes the adaptive memory consolidation.
- `update_importance(memory: Any, importance_score: float) -> None`: Updates the importance score of a memory.
- `increment_repetition(memory: Any) -> None`: Increments the repetition count of a memory.
- `consolidation_priority(memory: Any) -> float`: Computes the consolidation priority of a memory.

### MemoryDecay

#### Description

Handles the decay of memories over time.

#### Methods

- `__init__(half_life: float = MemoryConfig.MEMORY_DECAY_HALF_LIFE)`: Initializes the memory decay with a specified half-life.
- `decay_factor(age_in_seconds: float) -> float`: Computes the decay factor for a given age.

### DecayingMemory

#### Description

Represents a memory that decays over time.

#### Methods

- `__init__(content: Any, timestamp: Optional[float] = None)`: Initializes a decaying memory with content and an optional timestamp.
- `get_strength() -> float`: Computes the current strength of the memory.

### IntermediateMemory

#### Description

Represents the intermediate memory between short-term and long-term storage.

#### Methods

- `__init__(capacity: int = MemoryConfig.INTERMEDIATE_MEMORY_CAPACITY)`: Initializes the intermediate memory with a specified capacity.
- `add(memory: Any) -> None`: Adds a new memory to intermediate storage.
- `consolidate_oldest() -> DecayingMemory`: Consolidates the oldest memory in intermediate storage.
- `get_memories_for_consolidation() -> List[DecayingMemory]`: Retrieves memories ready for consolidation into long-term storage.

### EnhancedHierarchicalMemory

#### Description

The main class for the hierarchical memory system, integrating all memory components.

#### Methods

- `__init__(file_path: Optional[str] = None, state_model: Optional[EnhancedStateSpaceModel] = None, provider_manager: ProviderManager = None)`: Initializes the hierarchical memory system.
- `initialize() -> None`: Initializes the state model if needed.
- `set_consciousness_stream(consciousness_stream: 'ContinuousConsciousnessStream') -> None`: Integrates the consciousness stream with the memory system.
- `update_state_model(new_state: Dict[str, Any]) -> None`: Updates the state model with new state information.
- `process_input(input_data: Any) -> Any`: Processes new input data through sensory, working, and short-term memory, and updates the state model.
- `process_input_with_consciousness(input_data: Any) -> Any`: Asynchronously processes input data and adds thoughts to the consciousness stream.
- `retrieve_from_consciousness() -> Dict[str, Any]`: Retrieves the current thought from the consciousness stream.
- `update_memory_from_consciousness() -> None`: Updates memory based on the current thought from the consciousness stream.
- `get_current_state_context() -> np.ndarray`: Gets the current state context for memory operations.
- `consolidate_memory_with_consciousness() -> None`: Consolidates memory while integrating with the consciousness stream.
- `consolidate_memory() -> None`: Consolidates memories from short-term and intermediate memory.
- `_is_episodic(memory: Any) -> bool`: Determines if a memory is episodic.
- `_extract_concepts(memory: Any) -> Set[str]`: Extracts important concepts from the memory content.
- `get_relevant_context(query: str) -> str`: Retrieves the relevant context for a given query.
- `load_memory() -> None`: Loads memory from file.
- `save_memory() -> None`: Saves memory to file.
- `get_memory_stats() -> Dict[str, int]`: Retrieves memory statistics.
- `cleanup_memory(threshold: float = MemoryConfig.MEMORY_CLEANUP_THRESHOLD) -> None`: Cleans up memories based on a decay threshold.
- `update_importance(memory: Any, importance_score: float) -> None`: Updates the importance score of a memory.
- `increment_memory_repetition(memory: Any) -> None`: Increments the repetition count of a memory.