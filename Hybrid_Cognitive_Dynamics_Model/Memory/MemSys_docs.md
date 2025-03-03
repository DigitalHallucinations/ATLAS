# Memory System Module Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Module Overview](#module-overview)
3. [Classes](#classes)
    - [MemorySystem](#memorysystem)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Dependencies](#dependencies)
7. [Logging](#logging)
8. [Error Handling](#error-handling)
9. [Integration with Other Modules](#integration-with-other-modules)
10. [License](#license)

---

## Introduction

The **Memory System** module is a pivotal component of the Hybrid Cognitive Dynamics Model (HCDM). It orchestrates various types of memories, including sensory, short-term, intermediate, and long-term (both episodic and semantic) memories. Leveraging time-aware processing mechanisms, spaced repetition, and memory consolidation strategies, this module ensures efficient storage, retrieval, and management of information. Additionally, it interacts seamlessly with the State Space Model (SSM), Adaptive Leaky Integrate-and-Fire (aLIF) neurons (referred to as TAP), and Cognitive Temporal States to emulate human-like cognitive behaviors.

---

## Module Overview

The `memory_system.py` module encapsulates the following primary components:

1. **Memory Components:**
    - **SensoryMemory:** Handles the initial capture and storage of sensory inputs.
    - **ShortTermMemory:** Manages the transient storage of information for immediate use.
    - **EnhancedWorkingMemory:** Facilitates the manipulation and processing of information.
    - **IntermediateMemory:** Acts as a bridge between short-term and long-term memories.
    - **EnhancedLongTermEpisodicMemory:** Stores detailed, context-rich memories of specific events.
    - **EnhancedLongTermSemanticMemory:** Maintains generalized knowledge and concepts.
  
2. **Processing Components:**
    - **TimeDecay:** Implements decay mechanisms for different memory types based on temporal dynamics.
    - **SpacedRepetition:** Facilitates memory consolidation through spaced intervals.
    - **MemoryConsolidationThread:** Handles asynchronous memory consolidation tasks.
    - **ContextAwareRetrieval:** Enables context-sensitive retrieval of memories.
  
3. **Integration Components:**
    - **StateSpaceModel (SSM):** Provides state estimation and management.
    - **CognitiveTemporalState:** Manages the system's subjective perception of time.
    - **ProviderManager:** Manages interactions with external providers (e.g., Large Language Models).
    - **ConfigManager:** Handles configuration settings for the memory system.

The `MemorySystem` class integrates all these components to provide a cohesive and dynamic memory management framework within the HCDM.

---

## Classes

### MemorySystem

```python
class MemorySystem:
    def __init__(
        self,
        state_model: StateSpaceModel,
        file_path: Optional[str] = None,
        provider_manager: ProviderManager = None,
        config_manager: ConfigManager = None
    ):
        ...
        
    async def initialize(self):
        ...
        
    async def close(self):
        ...
        
    @io_bound_task        
    async def load_memory(self) -> None: 
        ...
    
    def set_consciousness_stream(self, consciousness_stream):
        ...
    
    @io_bound_task
    async def process_input(self, input_data: Any) -> Any:
        ...
    
    def prepare_input_signal(self, input_content: Any) -> np.ndarray:
        ...
    
    def dict_to_array(self, input_dict):
        ...
    
    def calculate_importance(self, input_data: Any) -> float: 
        ...
    
    @io_bound_task
    async def update_state_model(self, state_update: Dict[str, Any]):
        ...
    
    @io_bound_task
    async def process_input_with_consciousness(self, input_data: Any) -> Any:
        ...
    
    @io_bound_task
    async def retrieve_from_consciousness(self) -> Optional[Dict[str, Any]]:
        ...
    
    @io_bound_task
    async def update_memory_from_consciousness(self) -> None:
        ...
    
    async def get_current_state_context(self) -> np.ndarray:
        ...
    
    @cpu_bound_task
    async def consolidate_memory_with_consciousness(self) -> None:
        ...
    
    @cpu_bound_task
    async def consolidate_memory(self) -> None:
        ...
    
    def _is_episodic(self, memory: Any) -> bool:
        ...
    
    def _extract_concepts(self, memory: Any) -> Set[str]:
        ...
    
    def get_relevant_context(self, query: str) -> str:
        ...
    
    @io_bound_task
    async def save_memory(self) -> None:
        ...
    
    def get_memory_stats(self) -> Dict[str, int]:
        ...
    
    @cpu_bound_task
    def cleanup_memory(self, threshold: Optional[float] = None) -> None:
        ...
```

#### Description

The `MemorySystem` class orchestrates the various memory components and integrates them with time-aware processing and cognitive temporal states. It handles the initialization, processing, consolidation, and management of memories, ensuring that information flows seamlessly between different memory types and that memory retention aligns with cognitive dynamics.

#### Initialization

```python
def __init__(
    self,
    state_model: StateSpaceModel,
    file_path: Optional[str] = None,
    provider_manager: ProviderManager = None,
    config_manager: ConfigManager = None
):
    ...
```

- **Parameters:**
    - `state_model` (`StateSpaceModel`): The state space model of the system.
    - `file_path` (`Optional[str]`): Path to the memory storage file. If not provided, a default path is used.
    - `provider_manager` (`ProviderManager`, optional): Manager for external providers.
    - `config_manager` (`ConfigManager`, optional): Handles configuration settings.

- **Attributes:**
    - **Logging:** Initializes a logger for the memory system.
    - **Memory Components:** Initializes instances of SensoryMemory, ShortTermMemory, EnhancedWorkingMemory, IntermediateMemory, EnhancedLongTermEpisodicMemory, EnhancedLongTermSemanticMemory, and ContextAwareRetrieval.
    - **Time-Aware Processing:** Initializes TimeDecay, SpacedRepetition, and MemoryConsolidationThread.
    - **Cognitive Temporal State:** Initializes the current cognitive temporal state based on configuration.

#### Methods

- **`initialize()`**

    Asynchronously initializes memory components, loads existing memory from storage, and preloads long-term semantic memory if available.

- **`close()`**

    Gracefully terminates memory consolidation threads, closes database connections, clears memory structures, and saves any unsaved data.

- **`load_memory()`**

    Asynchronously loads memory data from the specified file path, populating episodic and semantic memories.

- **`save_memory()`**

    Asynchronously saves current memory states (episodic and semantic) to the specified file path.

- **`process_input(input_data)`**

    Processes incoming input data by preparing the input signal, applying attention focus, simulating neural activities, updating various memory components, extracting concepts, updating semantic and episodic memories, and updating the state model.

- **`process_input_with_consciousness(input_data)`**

    Extends `process_input` by integrating with the consciousness stream, adding thoughts, performing pattern completion, and triggering memory consolidation.

- **`retrieve_from_consciousness()`**

    Retrieves the current thought from the consciousness stream, if available.

- **`update_memory_from_consciousness()`**

    Updates memory based on the current thought retrieved from the consciousness stream, performing pattern separation and updating memories accordingly.

- **`consolidate_memory()`**

    Consolidates memories from short-term and intermediate memories into long-term episodic and semantic memories based on consolidation priorities.

- **`consolidate_memory_with_consciousness()`**

    Consolidates memory with consciousness stream integration, updating the state model and performing pattern completion insights.

- **`cleanup_memory(threshold)`**

    Cleans up memories based on a decay threshold, removing outdated or less important memories from various memory types.

- **Utility Methods:**
    - **`prepare_input_signal(input_content)`**: Converts input content into a numpy array suitable for processing.
    - **`dict_to_array(input_dict)`**: Converts dictionaries into numpy arrays by processing their values.
    - **`calculate_importance(input_data)`**: Calculates the importance of input data based on its characteristics.
    - **`_extract_concepts(memory)`**: Extracts significant concepts from memory content using techniques like TF-IDF.
    - **`get_relevant_context(query)`**: Retrieves relevant context for a given query by querying semantic and episodic memories.

---

## Configuration

The `MemorySystem` is highly configurable through the `config.yaml` file. The configuration defines parameters for memory components, state space modeling, attention mechanisms, and time-aware processing. Below is a detailed explanation of the configuration sections relevant to the Memory System.

### Configuration Parameters

#### `memory`

```yaml
memory:
  enabled: True
  sensory:
    enabled: True
    buffer_size: 100
  working:
    enabled: True
    capacity: 7
    total_resources: 1.0
  short_term:
    enabled: True
    capacity: 100
  intermediate:
    enabled: True
    capacity: 1000
  long_term_episodic:
    enabled: True
    max_episodes: 10000
  long_term_semantic:
    enabled: True
    max_concepts: 100000
```

- **`enabled`** (`bool`):  
  Flag to enable or disable the entire memory system.  
  **Default:** `True`

- **`sensory`** (`dict`):  
  Configurations for Sensory Memory.
    - **`enabled`** (`bool`): Enable or disable Sensory Memory.  
      **Default:** `True`
    - **`buffer_size`** (`int`): Maximum number of sensory inputs to store.  
      **Default:** `100`

- **`working`** (`dict`):  
  Configurations for Working Memory.
    - **`enabled`** (`bool`): Enable or disable Working Memory.  
      **Default:** `True`
    - **`capacity`** (`int`): Maximum number of items in Working Memory.  
      **Default:** `7`
    - **`total_resources`** (`float`): Total computational resources allocated.  
      **Default:** `1.0`

- **`short_term`** (`dict`):  
  Configurations for Short-Term Memory.
    - **`enabled`** (`bool`): Enable or disable Short-Term Memory.  
      **Default:** `True`
    - **`capacity`** (`int`): Maximum number of items in Short-Term Memory.  
      **Default:** `100`

- **`intermediate`** (`dict`):  
  Configurations for Intermediate Memory.
    - **`enabled`** (`bool`): Enable or disable Intermediate Memory.  
      **Default:** `True`
    - **`capacity`** (`int`): Maximum number of items in Intermediate Memory.  
      **Default:** `1000`

- **`long_term_episodic`** (`dict`):  
  Configurations for Long-Term Episodic Memory.
    - **`enabled`** (`bool`): Enable or disable Long-Term Episodic Memory.  
      **Default:** `True`
    - **`max_episodes`** (`int`): Maximum number of episodes to store.  
      **Default:** `10000`

- **`long_term_semantic`** (`dict`):  
  Configurations for Long-Term Semantic Memory.
    - **`enabled`** (`bool`): Enable or disable Long-Term Semantic Memory.  
      **Default:** `True`
    - **`max_concepts`** (`int`): Maximum number of semantic concepts to store.  
      **Default:** `100000`

#### `state_space_model`

*(As previously documented in the State Space Model section)*

#### `attention_mechanism`

*(As previously documented in the State Space Model section)*

#### `time_aware_processing`

*(As previously documented in the Cognitive Temporal State section)*

### Example Configuration File

Below is the complete `config.yaml` file incorporating configurations for the Memory System, State Space Model, Attention Mechanism, and Time-Aware Processing.

```yaml
memory:
  enabled: True
  sensory:
    enabled: True
    buffer_size: 100
  working:
    enabled: True
    capacity: 7
    total_resources: 1.0
  short_term:
    enabled: True
    capacity: 100
  intermediate:
    enabled: True
    capacity: 1000
  long_term_episodic:
    enabled: True
    max_episodes: 10000
  long_term_semantic:
    enabled: True
    max_concepts: 100000

state_space_model:
  enabled: True
  dimension: 50  # Dimension size for the state vector
  update_interval: 1.0  # seconds
  pfc_frequency: 5
  striatum_frequency: 40
  learning_rate: 0.001
  ukf_alpha: 0.1
  ukf_beta: 2.0
  ukf_kappa: -1.0
  process_noise: 0.01
  measurement_noise: 0.1
  dt: 0.001  # Time step for HH neurons
  scaling_factor: 2.0
  attention_mlp_hidden_size: 64  
  initial_confidence_threshold: 0.5
  threshold_increment: 0.01
  aLIF_parameters:
    tau_m: 20.0
    tau_ref: 2.0
    learning_rate: 0.001
  default_cognitive_temporal_state: IMMEDIATE  # Possible values: IMMEDIATE, REFLECTIVE, EMOTIONAL, DEEP_LEARNING, SOCIAL, REACTIVE, ANALYTICAL, CREATIVE, FOCUSED

attention_mechanism:
  enabled: True
  update_interval: 0.5  # seconds
  focus_threshold: 0.7
  num_attention_heads: 4
  switch_cooldown: 5  # seconds
  dropout_prob: 0.1
  blending_weights: [0.7, 0.3]
  activation_multiplier: 2.0
  activation_function: 'tanh'  # Options: 'tanh', 'relu', etc.
  consciousness_threshold: 0.2
  cognitive_load_threshold: 0.2
  trigger_words: ['urgent', 'important', 'critical', 'emergency']
  priority_weights: [0.4, 0.3, 0.3]  # Weights for relevance, urgency, importance
  attention_mlp_hidden_size: 64  

time_aware_processing:
  # Default CognitiveTemporalState at system initialization
  default_cognitive_temporal_state: IMMEDIATE  # Possible values: IMMEDIATE, REFLECTIVE, EMOTIONAL, DEEP_LEARNING, SOCIAL, REACTIVE, ANALYTICAL, CREATIVE, FOCUSED
  
  # Base decay rates for each MemoryType
  decay_rates:
    sensory_decay_rate: 0.1
    short_term_decay_rate: 0.01
    long_term_epidolic_decay_rate: 0.001
    long_term_semantic_decay_rate: 0.0001
  
  # Spaced repetition parameters
  spaced_repetition:
    ease_factor: 2.5
    initial_interval: 1  # in days
    initial_repetitions: 0
  
  # Memory consolidation settings
  consolidation:
    consolidation_interval: 3600  # in seconds
  
  # CognitiveTemporalState-specific configurations
  cognitive_temporal_states:
    IMMEDIATE:
      decay_rates_multiplier:
        sensory_decay_rate: 1.0
        short_term_decay_rate: 1.0
        long_term_epidolic_decay_rate: 1.0
        long_term_semantic_decay_rate: 1.0
      consolidation_interval: 3600  # Override if necessary
    REFLECTIVE:
      decay_rates_multiplier:
        sensory_decay_rate: 1.0
        short_term_decay_rate: 1.0
        long_term_epidolic_decay_rate: 0.5  # Slower decay for episodic memories
        long_term_semantic_decay_rate: 1.0
      consolidation_interval: 7200  # 2 hours
    EMOTIONAL:
      decay_rates_multiplier:
        sensory_decay_rate: 1.0
        short_term_decay_rate: 1.5  # Faster decay for short-term memories
        long_term_epidolic_decay_rate: 1.0
        long_term_semantic_decay_rate: 1.0
      consolidation_interval: 1800  # 30 minutes
    DEEP_LEARNING:
      decay_rates_multiplier:
        sensory_decay_rate: 1.0
        short_term_decay_rate: 1.0
        long_term_epidolic_decay_rate: 1.0
        long_term_semantic_decay_rate: 0.4  # Extremely slow decay for semantic memories
      consolidation_interval: 14400  # 4 hours
    SOCIAL:
      decay_rates_multiplier:
        sensory_decay_rate: 1.0
        short_term_decay_rate: 1.0
        long_term_epidolic_decay_rate: 1.0
        long_term_semantic_decay_rate: 1.0
      consolidation_interval: 3600  # 1 hour
    REACTIVE:
      decay_rates_multiplier:
        sensory_decay_rate: 1.3  # Faster decay for sensory memories
        short_term_decay_rate: 1.0
        long_term_epidolic_decay_rate: 1.0
        long_term_semantic_decay_rate: 1.0
      consolidation_interval: 300  # 5 minutes
    ANALYTICAL:
      decay_rates_multiplier:
        sensory_decay_rate: 1.0
        short_term_decay_rate: 1.0
        long_term_epidolic_decay_rate: 0.9  # Slightly slower decay for episodic memories
        long_term_semantic_decay_rate: 1.0
      consolidation_interval: 5400  # 1.5 hours
    CREATIVE:
      decay_rates_multiplier:
        sensory_decay_rate: 1.0
        short_term_decay_rate: 1.1  # Slightly faster decay for short-term memories
        long_term_epidolic_decay_rate: 1.0
        long_term_semantic_decay_rate: 1.0
      consolidation_interval: 2700  # 45 minutes
    FOCUSED:
      decay_rates_multiplier:
        sensory_decay_rate: 0.8  # Slower decay to maintain focus
        short_term_decay_rate: 0.8  # Slower decay for short-term memories
        long_term_epidolic_decay_rate: 1.0
        long_term_semantic_decay_rate: 1.0
      consolidation_interval: 4800  # 1 hour 20 minutes
```

**Note:** Ensure that the `cognitive_temporal_states` section aligns with the states defined in `CognitiveTemporalStateEnum`, including the newly added `FOCUSED` state.

---

## Usage Examples

### Initializing the Memory System

```python
import asyncio
from modules.Config.config import ConfigManager
from modules.Providers.provider_manager import ProviderManager
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.memory_system import MemorySystem
from modules.Hybrid_Cognitive_Dynamics_Model.StateSpaceModel.state_space_model import StateSpaceModel

async def main():
    # Initialize configuration and provider managers
    config_manager = ConfigManager(config_file='config.yaml')
    provider_manager = ProviderManager()

    # Initialize the StateSpaceModel
    state_space_model = StateSpaceModel(provider_manager, config_manager)
    await state_space_model.initialize()

    # Initialize the MemorySystem
    memory_system = MemorySystem(
        state_model=state_space_model,
        file_path='path/to/memory_store.json',
        provider_manager=provider_manager,
        config_manager=config_manager
    )
    await memory_system.initialize()

    # Example input data
    input_data = "The quick brown fox jumps over the lazy dog."

    # Process input data
    processed_info = await memory_system.process_input(input_data)
    print("Processed Information:", processed_info)

    # Retrieve memory statistics
    memory_stats = memory_system.get_memory_stats()
    print("Memory Statistics:", memory_stats)

    # Retrieve relevant context for a query
    query = "Climate change effects"
    context = memory_system.get_relevant_context(query)
    print("Relevant Context:", context)

    # Gracefully close the MemorySystem
    await memory_system.close()

# Run the main function
asyncio.run(main())
```

### Processing Input with Consciousness Stream Integration

```python
import asyncio
from modules.Config.config import ConfigManager
from modules.Providers.provider_manager import ProviderManager
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.memory_system import MemorySystem
from modules.Hybrid_Cognitive_Dynamics_Model.StateSpaceModel.state_space_model import StateSpaceModel
from modules.Consciousness.consciousness_stream import ConsciousnessStream  # Hypothetical module

async def main():
    # Initialize configuration and provider managers
    config_manager = ConfigManager(config_file='config.yaml')
    provider_manager = ProviderManager()

    # Initialize the StateSpaceModel
    state_space_model = StateSpaceModel(provider_manager, config_manager)
    await state_space_model.initialize()

    # Initialize the MemorySystem
    memory_system = MemorySystem(
        state_model=state_space_model,
        file_path='path/to/memory_store.json',
        provider_manager=provider_manager,
        config_manager=config_manager
    )
    await memory_system.initialize()

    # Initialize and set the ConsciousnessStream
    consciousness_stream = ConsciousnessStream()
    memory_system.set_consciousness_stream(consciousness_stream)

    # Example input data
    input_data = "Analyzing the impact of renewable energy sources on global economies."

    # Process input data with consciousness integration
    processed_info = await memory_system.process_input_with_consciousness(input_data)
    print("Processed Information with Consciousness Integration:", processed_info)

    # Retrieve memory statistics
    memory_stats = memory_system.get_memory_stats()
    print("Memory Statistics:", memory_stats)

    # Gracefully close the MemorySystem
    await memory_system.close()

# Run the main function
asyncio.run(main())
```

**Note:** Ensure that the `ConsciousnessStream` class is properly implemented and integrated within your project.

### Cleaning Up Memory

```python
import asyncio
from modules.Config.config import ConfigManager
from modules.Providers.provider_manager import ProviderManager
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.memory_system import MemorySystem
from modules.Hybrid_Cognitive_Dynamics_Model.StateSpaceModel.state_space_model import StateSpaceModel

async def main():
    # Initialize configuration and provider managers
    config_manager = ConfigManager(config_file='config.yaml')
    provider_manager = ProviderManager()

    # Initialize the StateSpaceModel
    state_space_model = StateSpaceModel(provider_manager, config_manager)
    await state_space_model.initialize()

    # Initialize the MemorySystem
    memory_system = MemorySystem(
        state_model=state_space_model,
        file_path='path/to/memory_store.json',
        provider_manager=provider_manager,
        config_manager=config_manager
    )
    await memory_system.initialize()

    # Perform memory cleanup with default threshold
    await memory_system.cleanup_memory()

    # Perform memory cleanup with a specific threshold
    await memory_system.cleanup_memory(threshold=0.2)

    # Retrieve memory statistics after cleanup
    memory_stats = memory_system.get_memory_stats()
    print("Memory Statistics After Cleanup:", memory_stats)

    # Gracefully close the MemorySystem
    await memory_system.close()

# Run the main function
asyncio.run(main())
```

---

## Dependencies

The `memory_system.py` module relies on several external libraries and internal modules. Ensure that all dependencies are installed and properly configured.

### External Libraries

- **Python Standard Libraries:**
    - `re`
    - `os`
    - `time`
    - `json`
    - `traceback`
    - `logging`
    - `enum`
    - `datetime`
    - `typing`

- **Third-Party Libraries:**
    - `numpy`
    - `aiofiles`  
      *Used for asynchronous file operations.*

### Internal Modules

- `ConfigManager`: Handles configuration settings.
- `ProviderManager`: Manages interactions with external providers (e.g., LLMs).
- `SensoryMemory`, `ShortTermMemory`, `EnhancedWorkingMemory`, `IntermediateMemory`: Various memory component classes.
- `ContextAwareRetrieval`: Handles context-sensitive memory retrieval.
- `EnhancedLongTermSemanticMemory`, `EnhancedLongTermEpisodicMemory`: Enhanced long-term memory classes.
- `StateSpaceModel`: Integrates state estimation and cognitive temporal dynamics.
- `CognitiveTemporalState`, `CognitiveTemporalStateEnum`: Manages cognitive temporal states.
- `TimeDecay`, `SpacedRepetition`, `MemoryConsolidationThread`, `MemoryType`: Time-aware processing components.
- `ConsciousnessStream`: Hypothetical module for consciousness stream integration (if applicable).

**Installation Example:**

```bash
pip install numpy aiofiles
```

Ensure that all internal modules are accessible within your project's directory structure.

---

## Logging

The Memory System employs Python's built-in `logging` library to record information, debug messages, warnings, and errors. Proper logging facilitates debugging and monitoring of memory operations and state transitions.

### Logger Configuration Example

```python
import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)  # Set desired logging level
    return logger
```

**Usage in `MemorySystem`:**

```python
self.logger = self.config_manager.setup_logger('MemorySystem') if config_manager else None
```

**Logging Levels:**

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems.
- `INFO`: Confirmation that things are working as expected.
- `WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future.
- `ERROR`: Due to a more serious problem, the software has not been able to perform some function.
- `CRITICAL`: A very serious error, indicating that the program itself may be unable to continue running.

**Adjust the logging level as needed based on the deployment environment and debugging requirements.**

---

## Error Handling

Robust error handling is implemented throughout the Memory System to ensure stability and facilitate debugging. Each method contains `try-except` blocks to catch and log exceptions, preventing unexpected crashes and providing detailed error information.

### Best Practices

- **Specific Exceptions:** Wherever possible, catch specific exception types to handle known error scenarios.
- **Logging:** Always log exceptions with contextual information using `exc_info=True` to include stack traces.
- **Graceful Degradation:** Ensure that failures in non-critical components do not halt the entire system. Implement fallback mechanisms where appropriate.
- **Validation:** Validate inputs and configuration parameters to prevent runtime errors.

### Example

```python
def process_input(self, input_data: Any) -> Any:
    try:
        # Processing logic
        ...
    except SpecificException as e:
        self.logger.error(f"Specific error occurred: {e}", exc_info=True)
        # Handle exception
    except Exception as e:
        self.logger.error(f"Unexpected error in process_input: {e}", exc_info=True)
        raise
```

**Guidelines:**

- Avoid using bare `except` clauses; specify exception types.
- Re-raise exceptions if they cannot be handled meaningfully within the method.
- Provide clear and descriptive log messages to aid in troubleshooting.

---

## Integration with Other Modules

The `MemorySystem` is designed to integrate seamlessly with various components of the HCDM, ensuring cohesive and dynamic cognitive operations. Below are the key integration points:

### 1. State Space Model (SSM)

- **Purpose:** Provides state estimation and management, influencing memory processing based on the system's current state.
  
- **Interaction:**
    - **State Updates:** The `MemorySystem` updates the `StateSpaceModel` with new state information derived from processed memories.
    - **Attention Focus:** Retrieves attention focus vectors from the `StateSpaceModel` to prioritize memory processing.
    - **Context Vector:** Utilizes context vectors from the `StateSpaceModel` to enrich memory consolidation and retrieval processes.

- **Example Integration:**

    ```python
    await self.update_state_model(processed_info)
    ```

### 2. Adaptive Leaky Integrate-and-Fire (aLIF) Neurons (TAP)

- **Purpose:** Simulates temporal dynamics and influences the cognitive temporal state based on neural activities.

- **Interaction:**
    - **Influence Factors:** The `MemorySystem` receives influence factors from the aLIF layer via the `StateSpaceModel`, which are used to update the `CognitiveTemporalState`.
    - **Temporal Dynamics:** Adjusts memory decay rates and consolidation intervals based on the current cognitive temporal state.

- **Example Integration:**

    ```python
    influence_factor = np.mean(tap_output)  # Example influence factor
    self.cognitive_temporal_state.update_state(influence_factor)
    ```

### 3. Cognitive Temporal State

- **Purpose:** Manages the system's subjective perception of time, influencing how memories are processed and retained.

- **Interaction:**
    - **State Transitions:** The `MemorySystem` updates the `CognitiveTemporalState` based on influence factors, which in turn adjusts memory decay rates and consolidation strategies.
    - **Memory Processing:** The current cognitive temporal state determines how different memory types decay and how consolidation intervals are set.

- **Example Integration:**

    ```python
    # Update CognitiveTemporalState based on aLIF output
    influence_factor = np.mean(tap_output)
    self.cognitive_temporal_state.update_state(influence_factor)
    ```

### 4. Time-Aware Processing Components

- **Purpose:** Implements mechanisms like time decay and spaced repetition to manage memory retention over time.

- **Interaction:**
    - **Decay Rates:** Adjusts decay rates for various memory types based on the current cognitive temporal state.
    - **Spaced Repetition:** Determines the scheduling of memory consolidation tasks to optimize retention.

- **Example Integration:**

    ```python
    # Use TimeDecay to determine which episodic memories to keep
    self.long_term_episodic.episodes = [
        ep for ep in self.long_term_episodic.episodes 
        if self.time_decay.decay(
            MemoryType.LONG_TERM_EPISODIC, 
            time_elapsed=current_time - datetime.fromisoformat(ep['timestamp']).timestamp(), 
            importance=ep.get('importance', 1.0)
        ) > threshold
    ]
    ```

### 5. Provider Manager

- **Purpose:** Manages interactions with external providers, such as Large Language Models (LLMs), for tasks like concept extraction and pattern completion.

- **Interaction:**
    - **External Queries:** Utilizes the `ProviderManager` to fetch additional information or perform complex computations that are beyond the system's internal capabilities.
    - **Pattern Completion:** Integrates with external services to complete patterns or retrieve related concepts.

- **Example Integration:**

    ```python
    related_concepts = await self.long_term_semantic.query(concept, n=5)
    ```

### 6. Config Manager

- **Purpose:** Handles configuration settings, allowing for dynamic adjustments of memory system parameters based on configurations.

- **Interaction:**
    - **Initialization:** Retrieves configuration parameters during the initialization of memory components.
    - **Dynamic Updates:** Allows for runtime updates of memory system parameters based on configuration changes.

- **Example Integration:**

    ```python
    self.memory_config = config_manager.get_subsystem_config('memory') if config_manager else {}
    ```

---

## License

This module is part of the Hybrid Cognitive Dynamics Model (HCDM) and is released under the [MIT License](LICENSE).

---

## Conclusion

The `memory_system.py` module is a sophisticated and integral part of the Hybrid Cognitive Dynamics Model (HCDM), providing comprehensive memory management capabilities that emulate human cognitive processes. By integrating various memory types, time-aware processing mechanisms, and cognitive temporal states, it ensures efficient storage, retrieval, and consolidation of information. Its seamless interaction with the State Space Model (SSM) and Adaptive Leaky Integrate-and-Fire (aLIF) neurons enhances its ability to adapt to dynamic cognitive demands.

Proper configuration, robust error handling, and extensive logging facilitate the reliable operation of the Memory System within the broader HCDM framework. For further assistance or contributions, please refer to the project's [Contributing Guidelines](CONTRIBUTING.md) or contact the development team.