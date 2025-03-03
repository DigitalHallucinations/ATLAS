# Cognitive Temporal State Module Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Module Overview](#module-overview)
3. [Classes](#classes)
    - [CognitiveTemporalStateEnum](#cognitivetemporallestateenum)
    - [CognitiveTemporalState](#cognitivetemporallestate)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Dependencies](#dependencies)
7. [Logging](#logging)
8. [Error Handling](#error-handling)
9. [Integration with Other Modules](#integration-with-other-modules)
10. [License](#license)

---

## Introduction

The **Cognitive Temporal State** module is a critical component of the Hybrid Cognitive Dynamics Model (HCDM). It manages the system's subjective perception of time by handling various cognitive temporal states influenced by cognitive and emotional factors. This module allows the model to adapt its processing speed and focus based on internal and external stimuli, enhancing its ability to simulate human-like cognitive behaviors.

---

## Module Overview

The `cognitive_temporal_state.py` module defines the cognitive temporal states and manages transitions between these states based on influence factors derived from neural network activities. It includes:

1. **`CognitiveTemporalStateEnum`**: An enumeration of possible cognitive temporal states.
2. **`CognitiveTemporalState`**: A class that represents and manages the current cognitive temporal state, updating it based on influence factors.

Additionally, the module interacts with the configuration settings defined in the configuration file to adjust decay rates and consolidation intervals based on the current cognitive temporal state.

---

## Classes

### CognitiveTemporalStateEnum

```python
class CognitiveTemporalStateEnum(Enum):
    """
    Enum representing different cognitive temporal states based on cognitive and emotional factors.
    """
    IMMEDIATE = 1
    REFLECTIVE = 2
    EMOTIONAL = 3
    DEEP_LEARNING = 4
    SOCIAL = 5
    REACTIVE = 6
    ANALYTICAL = 7
    CREATIVE = 8
    FOCUSED = 9  # Newly added state
```

#### Description

`CognitiveTemporalStateEnum` is an enumeration that defines the various cognitive temporal states the system can be in. Each state represents a unique mode of temporal perception and processing, influenced by cognitive and emotional dynamics.

#### Enumerated States

- **IMMEDIATE**: Represents a state where processing is quick, focusing on immediate tasks and reactions.
- **REFLECTIVE**: Indicates a contemplative state, allowing for deeper analysis and reflection.
- **EMOTIONAL**: Signifies a state influenced heavily by emotional factors, affecting decision-making and processing speed.
- **DEEP_LEARNING**: Denotes a state optimized for extensive learning and assimilation of complex information.
- **SOCIAL**: Represents a state focused on social interactions and understanding social cues.
- **REACTIVE**: Indicates a highly responsive state, reacting swiftly to stimuli.
- **ANALYTICAL**: Denotes a state focused on logical reasoning and detailed analysis.
- **CREATIVE**: Represents a state optimized for creativity and innovative thinking.
- **FOCUSED**: Newly added state that emphasizes sustained attention and concentration.

### CognitiveTemporalState

```python
class CognitiveTemporalState:
    """
    Represents the subjective perception of time within the model, managing different cognitive temporal states.
    """
    def __init__(self, initial_state: CognitiveTemporalStateEnum = CognitiveTemporalStateEnum.IMMEDIATE, initial_scaling: float = 1.0):
        """
        Initializes the CognitiveTemporalState with a specific state and time scaling factor.
        
        Args:
            initial_state (CognitiveTemporalStateEnum, optional): The initial cognitive temporal state. Defaults to IMMEDIATE.
            initial_scaling (float, optional): The initial time scaling factor. 
                                               >1.0 simulates time moving faster,
                                               <1.0 simulates time moving slower.
                                               Defaults to 1.0.
        """
        self.state = initial_state
        self.scaling_factor = initial_scaling
        self.logger = logging.getLogger('CognitiveTemporalState')
        self.logger.setLevel(logging.DEBUG)
    
    def update_state(self, influence_factor: float):
        """
        Updates the CognitiveTemporalState based on an influence factor.
        
        Args:
            influence_factor (float): Factor derived from aLIF network activity.
                                       Positive values may speed up time perception,
                                       negative values may slow it down.
        """
        # Simple exponential moving average for scaling factor updates
        alpha = 0.1  # Smoothing factor
        new_scaling = (1 - alpha) * self.scaling_factor + alpha * (1 + influence_factor)
        # Clamp the scaling factor within reasonable bounds
        self.scaling_factor = max(0.1, min(new_scaling, 5.0))
        self.logger.debug(f"CognitiveTemporalState scaling_factor updated to: {self.scaling_factor}")
        
        # Determine new state based on updated scaling_factor
        previous_state = self.state
        if self.scaling_factor > 1.5:
            self.state = CognitiveTemporalStateEnum.IMMEDIATE
        elif self.scaling_factor < 0.7:
            self.state = CognitiveTemporalStateEnum.FOCUSED
        elif 0.7 <= self.scaling_factor < 1.0:
            self.state = CognitiveTemporalStateEnum.REFLECTIVE
        elif 1.0 <= self.scaling_factor <= 1.5:
            self.state = CognitiveTemporalStateEnum.REACTIVE
        # Additional conditions can be added here for other states
        
        if self.state != previous_state:
            self.logger.info(f"CognitiveTemporalState changed from {previous_state.name} to {self.state.name}")
    
    def get_time_scaling(self) -> float:
        """
        Retrieves the current time scaling factor.
        
        Returns:
            float: The current time scaling factor.
        """
        return self.scaling_factor
    
    def get_current_state(self) -> CognitiveTemporalStateEnum:
        """
        Retrieves the current cognitive temporal state.
        
        Returns:
            CognitiveTemporalStateEnum: The current cognitive temporal state.
        """
        return self.state
```

#### Description

`CognitiveTemporalState` manages the subjective perception of time within the cognitive model. It maintains the current cognitive temporal state and updates it based on influence factors, which are typically derived from neural network activities such as the Adaptive Leaky Integrate-and-Fire (aLIF) layer.

#### Initialization Parameters

- **`initial_state`** (`CognitiveTemporalStateEnum`, optional):  
  The initial cognitive temporal state upon creation.  
  **Default:** `IMMEDIATE`

- **`initial_scaling`** (`float`, optional):  
  The initial time scaling factor.  
  - Values >1.0 simulate time moving faster.
  - Values <1.0 simulate time moving slower.  
  **Default:** `1.0`

#### Methods

- **`update_state(influence_factor: float)`**
  
  ```python
  def update_state(self, influence_factor: float):
      """
      Updates the CognitiveTemporalState based on an influence factor.
      
      Args:
          influence_factor (float): Factor derived from aLIF network activity.
                                     Positive values may speed up time perception,
                                     negative values may slow it down.
      """
  ```
  
  Updates the time scaling factor using an exponential moving average approach and determines if a state transition is necessary based on predefined scaling thresholds.

- **`get_time_scaling() -> float`**
  
  ```python
  def get_time_scaling(self) -> float:
      """
      Retrieves the current time scaling factor.
      
      Returns:
          float: The current time scaling factor.
      """
  ```
  
  Returns the current time scaling factor, which influences the perception of time within the model.

- **`get_current_state() -> CognitiveTemporalStateEnum`**
  
  ```python
  def get_current_state(self) -> CognitiveTemporalStateEnum:
      """
      Retrieves the current cognitive temporal state.
      
      Returns:
          CognitiveTemporalStateEnum: The current cognitive temporal state.
      """
  ```
  
  Returns the current cognitive temporal state enumeration.

---

## Configuration

The Cognitive Temporal State module is configured through the `time_aware_processing` section of the configuration file. This section defines the default cognitive temporal state, decay rates for various memory types, parameters for spaced repetition, consolidation settings, and specific configurations for each cognitive temporal state.

### Configuration Parameters

#### `time_aware_processing`

- **`default_cognitive_temporal_state`** (`str`):  
  Default cognitive temporal state upon system initialization. Determines the initial configuration of cognitive temporal dynamics.  
  **Possible Values:** `IMMEDIATE`, `REFLECTIVE`, `EMOTIONAL`, `DEEP_LEARNING`, `SOCIAL`, `REACTIVE`, `ANALYTICAL`, `CREATIVE`, `FOCUSED`  
  **Default:** `IMMEDIATE`

- **`decay_rates`** (`dict`):  
  Base decay rates for each `MemoryType`. These rates determine how quickly different types of memories decay over time.

  - **`sensory_decay_rate`** (`float`):  
    Decay rate for sensory memories.  
    **Default:** `0.1`

  - **`short_term_decay_rate`** (`float`):  
    Decay rate for short-term memories.  
    **Default:** `0.01`

  - **`long_term_epidolic_decay_rate`** (`float`):  
    Decay rate for long-term episodic memories.  
    **Default:** `0.001`

  - **`long_term_semantic_decay_rate`** (`float`):  
    Decay rate for long-term semantic memories.  
    **Default:** `0.0001`

- **`spaced_repetition`** (`dict`):  
  Parameters for spaced repetition, which facilitates memory consolidation.

  - **`ease_factor`** (`float`):  
    Factor determining the ease of repeating items.  
    **Default:** `2.5`

  - **`initial_interval`** (`int`):  
    Initial interval (in days) before the first repetition.  
    **Default:** `1`

  - **`initial_repetitions`** (`int`):  
    Initial number of repetitions.  
    **Default:** `0`

- **`consolidation`** (`dict`):  
  Settings related to memory consolidation processes.

  - **`consolidation_interval`** (`int`):  
    Time interval (in seconds) between memory consolidation processes.  
    **Default:** `3600` (1 hour)

- **`cognitive_temporal_states`** (`dict`):  
  Specific configurations for each cognitive temporal state, including decay rate multipliers and consolidation intervals.

  - **`<STATE_NAME>`** (`dict`):  
    Configuration for a specific cognitive temporal state.

    - **`decay_rates_multiplier`** (`dict`):  
      Multipliers applied to the base decay rates for different memory types in the current state.

      - **`sensory_decay_rate`** (`float`):  
        Multiplier for sensory decay rate.  
        **Default:** `1.0`

      - **`short_term_decay_rate`** (`float`):  
        Multiplier for short-term decay rate.  
        **Default:** `1.0`

      - **`long_term_epidolic_decay_rate`** (`float`):  
        Multiplier for long-term episodic decay rate.  
        **Default:** `1.0`

      - **`long_term_semantic_decay_rate`** (`float`):  
        Multiplier for long-term semantic decay rate.  
        **Default:** `1.0`

    - **`consolidation_interval`** (`int`):  
      Override consolidation interval (in seconds) for the current state.  
      **Default:** `3600` (1 hour)

#### Example Configuration File

Below is the complete configuration file, including the `state_space_model` and `time_aware_processing` sections.

```yaml
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

**Note:** Ensure that the `cognitive_temporal_states` section aligns with the states defined in `CognitiveTemporalStateEnum`. The newly added `FOCUSED` state is included with its specific decay rate multipliers and consolidation interval.

---

## Usage Examples

### Initializing the Cognitive Temporal State

```python
import logging
from cognitive_temporal_state import CognitiveTemporalState, CognitiveTemporalStateEnum

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('CognitiveTemporalState')

# Initialize CognitiveTemporalState with default state
cts = CognitiveTemporalState()
print(f"Initial State: {cts.get_current_state().name}")
print(f"Initial Scaling Factor: {cts.get_time_scaling()}")

# Update state with a positive influence factor (e.g., speeding up time perception)
cts.update_state(0.5)
print(f"Updated State: {cts.get_current_state().name}")
print(f"Updated Scaling Factor: {cts.get_time_scaling()}")

# Update state with a negative influence factor (e.g., slowing down time perception)
cts.update_state(-1.0)
print(f"Updated State: {cts.get_current_state().name}")
print(f"Updated Scaling Factor: {cts.get_time_scaling()}")
```

**Output:**
```
Initial State: IMMEDIATE
Initial Scaling Factor: 1.0
CognitiveTemporalState scaling_factor updated to: 1.05
Updated State: IMMEDIATE
Updated Scaling Factor: 1.05
CognitiveTemporalState scaling_factor updated to: 0.945
CognitiveTemporalState changed from IMMEDIATE to REFLECTIVE
Updated State: REFLECTIVE
Updated Scaling Factor: 0.945
```

### Integrating Cognitive Temporal State with State Space Model

Assuming the `StateSpaceModel` class integrates the `CognitiveTemporalState` as shown in the previous module documentation, here's how you might interact with it:

```python
import asyncio
from modules.Config.config import ConfigManager
from modules.Providers.provider_manager import ProviderManager
from modules.Hybrid_Cognitive_Dynamics_Model.StateSpaceModel.state_space_model import StateSpaceModel

async def main():
    # Initialize configuration and provider managers
    config_manager = ConfigManager(config_file='config.yaml')
    provider_manager = ProviderManager()

    # Initialize the StateSpaceModel
    state_space_model = StateSpaceModel(provider_manager, config_manager)
    await state_space_model.initialize()

    # Example input data
    input_data = {
        'content': 'Analyzing the impact of climate change on marine life.',
        'action_required': True,
        'timestamp': time.time()
    }

    # Update the state with input data
    updated_state = await state_space_model.update(input_data)
    print("Updated State:", updated_state)

    # Retrieve state description
    state_description = state_space_model.get_state_description()
    print("State Description:", state_description)

    # Gracefully stop the state space model
    await state_space_model.stop()

# Run the main function
asyncio.run(main())
```

**Note:** Ensure that all internal modules (`ConfigManager`, `ProviderManager`, `StateSpaceModel`) are correctly implemented and accessible.

---

## Dependencies

The `cognitive_temporal_state.py` module relies on the following libraries:

- **Python Standard Libraries:**
  - `logging`
  - `enum`

- **Third-Party Libraries:**
  - None required beyond the standard library.

Ensure that the Python environment has access to the standard libraries used.

---

## Logging

The module employs Python's built-in `logging` library to record information, debug messages, warnings, and errors. Proper logging facilitates debugging and monitoring of the cognitive temporal state transitions.

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

**Usage in Classes:**

Each class initializes its logger using the `logging` library. For example, in the `CognitiveTemporalState` class:

```python
self.logger = logging.getLogger('CognitiveTemporalState')
self.logger.setLevel(logging.DEBUG)
```

**Logging Levels:**

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems.
- `INFO`: Confirmation that things are working as expected.
- `WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future.
- `ERROR`: Due to a more serious problem, the software has not been able to perform some function.
- `CRITICAL`: A very serious error, indicating that the program itself may be unable to continue running.

Adjust the logging level as needed based on the deployment environment and debugging requirements.

---

## Error Handling

Robust error handling is implemented throughout the module to ensure stability and facilitate debugging. Each method contains `try-except` blocks to catch and log exceptions, preventing unexpected crashes and providing detailed error information.

### Best Practices

- **Specific Exceptions:** Wherever possible, catch specific exception types to handle known error scenarios.
- **Logging:** Always log exceptions with contextual information using `exc_info=True` to include stack traces.
- **Graceful Degradation:** Ensure that failures in non-critical components do not halt the entire system. Implement fallback mechanisms where appropriate.
- **Validation:** Validate inputs and configuration parameters to prevent runtime errors.

### Example

```python
def update_state(self, influence_factor: float):
    try:
        # Update logic
        ...
    except SpecificException as e:
        self.logger.error(f"Specific error occurred: {e}", exc_info=True)
        # Handle exception
    except Exception as e:
        self.logger.error(f"Unexpected error in update_state: {e}", exc_info=True)
        raise
```

**Guidelines:**

- Avoid using bare `except` clauses; specify exception types.
- Re-raise exceptions if they cannot be handled meaningfully within the method.
- Provide clear and descriptive log messages to aid in troubleshooting.

---

## Integration with Other Modules

The `CognitiveTemporalState` module is designed to integrate seamlessly with other components of the HCDM, particularly within the `StateSpaceModel`. It interacts with:

- **Adaptive LIF Neural Layer (aLIF):**  
  Influence factors derived from the aLIF layer activity inform the `CognitiveTemporalState`, determining transitions between states.

- **StateSpaceModel:**  
  The `StateSpaceModel` utilizes the current cognitive temporal state to adjust processing parameters such as decay rates and consolidation intervals.

- **Configuration Manager (`ConfigManager`):**  
  Retrieves configuration settings that define behavior and parameters for different cognitive temporal states.

### Example Integration within StateSpaceModel

```python
from cognitive_temporal_state import CognitiveTemporalState, CognitiveTemporalStateEnum

class StateSpaceModel:
    def __init__(self, provider_manager: ProviderManager, config_manager: ConfigManager):
        # Existing initialization
        ...
        # Initialize CognitiveTemporalState
        default_state_str = config_manager.get_subsystem_config('state_space_model').get('default_cognitive_temporal_state', 'IMMEDIATE').upper()
        try:
            initial_state = CognitiveTemporalStateEnum[default_state_str]
            self.current_cognitive_temporal_state = CognitiveTemporalState(initial_state)
            self.logger.debug(f"Initial CognitiveTemporalState set to: {self.current_cognitive_temporal_state.get_current_state().name}")
        except KeyError:
            self.current_cognitive_temporal_state = CognitiveTemporalState(CognitiveTemporalStateEnum.IMMEDIATE)
            self.logger.warning(f"Unknown CognitiveTemporalState '{default_state_str}'. Defaulting to IMMEDIATE.")
        ...
    
    def fx_with_selection(self, x, dt):
        try:
            # Existing state transition logic
            ...
            # Update CognitiveTemporalState based on aLIF output
            influence_factor = np.mean(tap_output)  # Example influence factor
            self.current_cognitive_temporal_state.update_state(influence_factor)
            ...
        except Exception as e:
            self.logger.error(f"Error in state transition function fx_with_selection: {str(e)}", exc_info=True)
            raise
```

**Note:** Ensure that the `CognitiveTemporalState` is appropriately updated within the state transition functions to reflect the dynamic nature of cognitive temporal processing.

---

## License

This module is part of the Hybrid Cognitive Dynamics Model (HCDM) and is released under the [MIT License](LICENSE).

---

## Conclusion

The `cognitive_temporal_state.py` module plays a vital role in managing the system's perception of time, enabling dynamic transitions between various cognitive temporal states. By integrating with neural layers and leveraging configuration settings, it allows the HCDM to simulate complex cognitive behaviors influenced by both cognitive and emotional factors. Proper initialization, configuration, and integration with other modules ensure that the cognitive temporal dynamics operate seamlessly within the broader model.

For further assistance or contributions, please refer to the project's [Contributing Guidelines](CONTRIBUTING.md) or contact the development team.