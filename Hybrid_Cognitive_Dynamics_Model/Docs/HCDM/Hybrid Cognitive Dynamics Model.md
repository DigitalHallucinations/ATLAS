

# Hybrid Cognitive Dynamics Model (HCDM) Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
    - [State Transition Function](#state-transition-function)
    - [Measurement Function](#measurement-function)
    - [Unscented Kalman Filter (UKF)](#unscented-kalman-filter-ukf)
    - [Oscillatory Neural Layers](#oscillatory-neural-layers)
    - [Attention Manager](#attention-manager)
    - [State Measurement](#state-measurement)
    - [Particle Filter](#particle-filter)
4. [Initialization](#initialization)
5. [State Management](#state-management)
6. [Updating the Model](#updating-the-model)
7. [Retrieving the State](#retrieving-the-state)
8. [Parameter Updates](#parameter-updates)
9. [State Interpretation](#state-interpretation)

## Overview
The Hybrid Cognitive Dynamics Model (HCDM) is an advanced cognitive architecture that integrates neural network layers, Kalman filters, and particle filtering techniques. This model aims to emulate human-like cognitive processes by maintaining and updating a dynamic state representation of various cognitive and emotional factors.

## Architecture
HCDM combines several techniques to achieve a comprehensive and dynamic state model:
- **Unscented Kalman Filter (UKF)** for state estimation.
- **Neural Network Layers** to model specific cognitive processes.
- **Attention Manager** to focus on relevant information.
- **Particle Filter** for robust state estimation.
- **State Measurement** for analyzing external input.

## Components

### State Transition Function
The state transition function (`fx`) defines how the state evolves over time:
```python
def fx(x, dt):
    new_x = np.zeros_like(x)
    new_x[:self.dim] = x[:self.dim] + np.sin(x[3*self.dim:4*self.dim]) * dt
    new_x[self.dim:2*self.dim] = x[self.dim:2*self.dim] + np.cos(x[3*self.dim:4*self.dim]) * dt
    new_x[2*self.dim:3*self.dim] = x[2*self.dim:3*self.dim] + dt
    new_x[3*self.dim:4*self.dim] = x[3*self.dim:4*self.dim] + dt
    new_x[4*self.dim:5*self.dim] = x[4*self.dim:5*self.dim] + dt
    new_x[-8:] = x[-8:]
    return new_x
```

### Measurement Function
The measurement function (`hx`) defines how the measurements are generated from the state:
```python
def hx(x):
    return np.concatenate([x[:self.dim], x[-8:]])
```

### Unscented Kalman Filter (UKF)
The UKF is used for state estimation. It uses sigma points to handle non-linear transformations:
```python
points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2., kappa=-1)
self.ukf = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, dt=self.update_interval, fx=fx, hx=hx, points=points)
self.ukf.x = np.zeros(self.dim_x)
self.ukf.P *= 0.2
self.ukf.Q = q
self.ukf.R = np.eye(self.dim_z) * 0.1
```

### Oscillatory Neural Layers
The neural layers model specific cognitive processes:
```python
self.pfc_layer = OscillatoryNeuralLayer(input_size=self.dim, output_size=self.dim, frequency=subsystem_config.get('pfc_frequency', 5))
self.striatum_layer = OscillatoryNeuralLayer(input_size=self.dim, output_size=2, frequency=subsystem_config.get('striatum_frequency', 40))
```

### Attention Manager
The Attention Manager focuses on relevant information based on the current state:
```python
self.attention_manager = AttentionManager(self)
```

### State Measurement
State Measurement analyzes external input and provides necessary information for state updates:
```python
self.state_measurement = StateMeasurement(provider_manager)
```

### Particle Filter
The Particle Filter provides robust state estimation by maintaining a set of particles:
```python
self.particles = [Particle(EnhancedStateSpaceModel(provider_manager, config_manager)) for _ in range(n_particles)]
```

## Initialization
The `EnhancedStateSpaceModel` initializes the UKF, neural layers, and other components:
```python
def __init__(self, provider_manager: ProviderManager, config_manager: ConfigManager):
    self.logger = setup_logger(__name__)
    self.provider_manager = provider_manager
    self.config_manager = config_manager  
    # Other initialization steps
```

## State Management
The state includes various cognitive and emotional components:
```python
@property
def emotional_state(self):
    valence = np.mean(self.ukf.x[self.dim:2*self.dim]) if self.ukf.x[self.dim:2*self.dim].size > 0 else 0.0
    return {'valence': float(valence), 'arousal': float(self.ukf.x[-8]), 'dominance': float(self.ukf.x[-7])}

@property
def consciousness_level(self):
    return float(self.ukf.x[-6])
```

## Updating the Model
The model updates its state based on new data:
```python
async def update(self, data):  
    pfc_input = self.prepare_pfc_input(data)
    striatum_input = self.prepare_striatum_input(data)
    dt = 0.001
    pfc_activation = self.pfc_layer.update(pfc_input, dt)
    striatum_activation = self.striatum_layer.update(striatum_input, dt)
    measurement = np.concatenate([pfc_activation, striatum_activation])
    self.ukf.predict()
    self.ukf.update(measurement)
    self.update_emotional_state(data)
    self.update_attention_focus(pfc_activation)
    self.update_consciousness_level(striatum_activation)
    self.ensure_positive_definite()
    return await self.get_state()
```

## Retrieving the State
The current state can be retrieved asynchronously:
```python
async def get_state(self):  
    state = {
        'ukf_state': self.ukf.x,
        'emotional_state': self.emotional_state,
        'attention_focus': self.attention_focus,
        'consciousness_level': self.consciousness_level
    }
    return state
```

## Parameter Updates
Parameters can be updated dynamically:
```python
def update_parameter(self, param_name, param_value): 
    if param_name == 'process_noise':
        np.fill_diagonal(self.ukf.Q, param_value)
    elif param_name == 'measurement_noise':
        np.fill_diagonal(self.ukf.R, param_value)
    elif param_name == 'alpha':
        self.ukf.points.alpha = param_value
        self.ukf.points.compute_weights()
    elif param_name == 'beta':
        self.ukf.points.beta = param_value
        self.ukf.points.compute_weights()
    elif param_name == 'kappa':
        self.ukf.points.kappa = param_value
        self.ukf.points.compute_weights()
    elif param_name.startswith('state_'):
        index = int(param_name.split('_')[1])
        if 0 <= index < self.dim_x:
            self.ukf.x[index] = param_value
        else:
            raise ValueError(f"Invalid state index: {index}")
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")
    self.ensure_positive_definite()
```

## State Interpretation
The state can be interpreted to provide a human-readable description:
```python
async def interpret_state(state):
    desc = state.get_state_description()
    state_interpretation = f"Focus: {desc['topic_focus']:.2f}, Emotion: (V:{desc['emotional_valence']:.2f}, A:{desc['emotional_arousal']:.2f}, D:{desc['emotional_dominance']:.2f}), " \
           f"Attention: {desc['attention_allocation']:.2f}, Memory: {desc['memory_activation']:.2f}, " \
           f"Cognitive Control: {desc['cognitive_control']:.2f}, Consciousness: {desc['consciousness_level']:.2f}, " \
           f"Cognitive Load: {desc['cognitive_load']:.2f}, Energy: {desc['mental_energy']:.2f}, " \
           f"Curiosity: {desc['curiosity_level']:.2f}, Uncertainty: {desc['uncertainty']:.2f}, " \
           f"Creativity: {desc['creativity_index']:.2f}"
    return await state_interpretation
```

