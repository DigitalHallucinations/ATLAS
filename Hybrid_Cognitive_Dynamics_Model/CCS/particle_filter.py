# modules/Hybrid_Cognitive_Dynamics_Model/CCS/particle_filter.py

import time
import psutil
import numpy as np
from typing import Dict, Any
from modules.Config.config import ConfigManager
from modules.Providers.provider_manager import ProviderManager
from modules.Hybrid_Cognitive_Dynamics_Model.SSM.state_space_model import StateSpaceModel


class Particle:
    """
    Represents a single particle in the particle filter, containing a state model and weight.
    """

    def __init__(self, state_model: StateSpaceModel):
        """
        Initializes a Particle with the given state model and an initial weight.

        Args:
            state_model (StateSpaceModel): The state model associated with this particle.
        """
        self.state = state_model
        self.weight = 1.0  # Initial weight is uniform for all particles

        self.config_manager = ConfigManager()
        self.logger = self.config_manager.setup_logger('Particle')

        self.logger.debug(f"Particle initialized with state: {state_model} and weight: {self.weight}")


class ParticleFilter:
    """
    Implements a particle filter for state estimation across multiple StateSpaceModels.
    Manages a set of particles, each representing a possible state, and updates them based on actions and measurements.
    """

    def __init__(self, n_particles, config_manager: ConfigManager, provider_manager: ProviderManager = None):
        """
        Initializes the ParticleFilter with a specified number of particles and configuration.

        Args:
            n_particles (int, optional): The number of particles to maintain. If None, determined based on system resources.
            config_manager (ConfigManager): The configuration manager for settings.
            provider_manager (ProviderManager): Manages external providers (e.g., LLMs).
        """
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('ParticleFilter')

        # Fetch default particle count from config
        default_particles = self.config_manager.get_subsystem_config('particle_filter').get('default_n_particles', 20)
        max_particles = psutil.cpu_count(logical=True) * 2  # Example: Max particles based on logical CPU cores

        # Use provided value or default if none provided, capped by system resources
        self.n_particles = min(n_particles or default_particles, max_particles)
        self.logger.info(f"Particle Filter initialized with {self.n_particles} particles")

        self.provider_manager = provider_manager

        if provider_manager is None:
            self.logger.error("ProviderManager not provided during ParticleFilter initialization.")
            raise ValueError("provider_manager must be provided")

        subsystem_config = self.config_manager.get_subsystem_config('state_space_model')
        state_model_dim = min(subsystem_config.get('dimension', 100), 50)  # Limit dimension to 50

        # Initialize resource logging thresholds
        resource_config = self.config_manager.get_subsystem_config('resource_monitoring')
        self.cpu_threshold = resource_config.get('cpu_threshold', 80)  # Default CPU threshold: 80%
        self.memory_threshold = resource_config.get('memory_threshold', 80)  # Default Memory threshold: 80%

        # Initialize particles with individual StateSpaceModels
        try:
            self.particles = [
                Particle(StateSpaceModel(provider_manager, config_manager)) for _ in range(self.n_particles)
            ]
            self.logger.info(f"Particle Filter initialized with {self.n_particles} particles")
        except Exception as e:
            self.logger.exception(f"Failed to initialize particles: {str(e)}")
            raise

    def _should_log_resources(self, cpu_percent, memory_percent) -> bool:
        """
        Determines whether to log system resources based on configured thresholds.

        Args:
            cpu_percent (float): Current CPU usage percentage.
            memory_percent (float): Current Memory usage percentage.

        Returns:
            bool: True if either CPU or Memory usage exceeds the threshold, False otherwise.
        """
        return cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold

    def _reinitialize_particle(self, particle_index, retry_count=0, max_retries=3):
        """
        Reinitializes a specific particle after a failure, with retry mechanisms.

        Args:
            particle_index (int): Index of the particle to reinitialize.
            retry_count (int): Current retry attempt.
            max_retries (int): Maximum number of retry attempts.
        """
        if retry_count < max_retries:
            self.logger.warning(f"Reinitializing particle {particle_index} (Attempt {retry_count + 1}/{max_retries})")
            try:
                self.particles[particle_index] = Particle(StateSpaceModel(self.provider_manager, self.config_manager))
            except Exception as e:
                self.logger.error(f"Failed to reinitialize particle {particle_index}: {str(e)}")
                self._reinitialize_particle(particle_index, retry_count + 1, max_retries)  # Retry
        else:
            self.logger.error(f"Exceeded max retries for particle {particle_index}. Reinitializing without retry.")
            self.particles[particle_index] = Particle(StateSpaceModel(self.provider_manager, self.config_manager))  # Force reinitialization

    def predict(self, action, max_retries=3):
        """
        Predicts the next state of all particles based on the given action.

        Args:
            action (dict): The action to apply to each particle.
            max_retries (int): Maximum number of retries for the prediction step.
        """
        self.logger.info(f"Starting prediction with action: {action}")
        retry_count = 0

        while retry_count <= max_retries:
            try:
                # Monitor system resources before prediction
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                if self._should_log_resources(cpu_percent, memory_percent):
                    self.logger.info(
                        f"System resources before prediction - CPU: {cpu_percent}%, Memory: {memory_percent}%"
                    )

                start_time = time.time()
                timeout = 30  # Set a 30-second timeout to prevent infinite loops
                successful_predictions = 0

                for i, particle in enumerate(self.particles):
                    if time.time() - start_time > timeout:
                        self.logger.error(f"Prediction timed out after {timeout} seconds")
                        raise TimeoutError("Prediction step exceeded the time limit.")

                    try:
                        self._apply_action(particle.state, action)
                        successful_predictions += 1
                    except Exception as particle_error:
                        self.logger.error(
                            f"Error applying action to particle {i}: {str(particle_error)}", exc_info=True
                        )
                        # Reinitialize this specific particle in case of error
                        self._reinitialize_particle(i)

                self.logger.info(
                    f"Particle Filter prediction step completed with {successful_predictions}/{self.n_particles} successful predictions"
                )

                # Monitor system resources after prediction
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                if self._should_log_resources(cpu_percent, memory_percent):
                    self.logger.info(
                        f"System resources after prediction - CPU: {cpu_percent}%, Memory: {memory_percent}%"
                    )

                # Check if a significant number of predictions failed
                if successful_predictions / self.n_particles < 0.5:
                    self.logger.warning("Less than 50% of particles were successfully predicted. Initiating retry mechanism.")
                    raise RuntimeError("High failure rate in prediction step.")

                break  # Exit the retry loop if prediction is successful

            except (TimeoutError, RuntimeError) as e:
                retry_count += 1
                self.logger.warning(f"Prediction attempt {retry_count} failed: {str(e)}")
                if retry_count > max_retries:
                    self.logger.error(f"Exceeded maximum retries ({max_retries}) for prediction. Reinitializing all particles.")
                    self._reinitialize_particles()
                    break
                else:
                    self.logger.info(f"Retrying prediction (Attempt {retry_count}/{max_retries})...")
                    time.sleep(1)  # Brief pause before retrying

            except Exception as e:
                self.logger.exception(f"Unexpected error during prediction: {str(e)}")
                self._reinitialize_particles()
                break

    def get_best_particle(self):
        """
        Retrieves the particle with the highest weight, representing the most probable state.

        Returns:
            Particle: The particle with the highest weight.
        """
        try:
            self.logger.debug("Retrieving the particle with the highest weight")
            weights = [p.weight for p in self.particles]
            if not weights:
                self.logger.warning("No particles available to retrieve the best particle.")
                return None
            best_index = np.argmax(weights)
            best_particle = self.particles[best_index]
            self.logger.debug(f"Best particle retrieved: {best_particle}")
            return best_particle
        except Exception as e:
            self.logger.exception(f"Error in get_best_particle: {str(e)}")
            return None

    def _dict_to_array(self, measurement: Dict[str, Any]) -> np.ndarray:
        """
        Converts a measurement dictionary into a numpy array for processing.

        Args:
            measurement (Dict[str, Any]): The measurement data.

        Returns:
            np.ndarray: The measurement as a numpy array.
        """
        array_data = []

        try:
            # Extract values from 'ukf_state' if present
            if 'ukf_state' in measurement and isinstance(measurement['ukf_state'], np.ndarray):
                array_data.extend(measurement['ukf_state'].flatten())

            # Extract values from 'emotional_state' if present
            if 'emotional_state' in measurement and isinstance(measurement['emotional_state'], dict):
                array_data.extend([
                    measurement['emotional_state'].get('valence', 0),
                    measurement['emotional_state'].get('arousal', 0),
                    measurement['emotional_state'].get('dominance', 0)
                ])

            # Extract values from 'attention_focus' if present
            if 'attention_focus' in measurement and isinstance(measurement['attention_focus'], np.ndarray):
                array_data.extend(measurement['attention_focus'].flatten())

            # Add 'consciousness_level' if present
            if 'consciousness_level' in measurement:
                array_data.append(measurement['consciousness_level'])

            self.logger.debug(f"Converted measurement dictionary to array: {array_data}")
        except Exception as e:
            self.logger.error(f"Error converting measurement dictionary to array: {str(e)}", exc_info=True)

        # Convert to numpy array
        return np.array(array_data)

    def update(self, measurement):
        """
        Updates the weights of the particles based on the given measurement.

        Args:
            measurement (np.ndarray or dict): The measurement data to update particle weights.
        """
        self.logger.info(f"Updating particle filter with measurement: {measurement}")

        if measurement is None:
            self.logger.warning("Received None measurement, skipping update")
            return

        try:
            if isinstance(measurement, dict):
                measurement = self._dict_to_array(measurement)

            if not isinstance(measurement, np.ndarray):
                self.logger.warning(f"Measurement is not a numpy array after conversion: {type(measurement)}. Skipping update.")
                return

            expected_shape = (self.particles[0].state.dim_z,)
            if measurement.shape != expected_shape:
                self.logger.warning(
                    f"Measurement shape mismatch. Expected {expected_shape}, got {measurement.shape}. Resizing measurement."
                )
                measurement = np.resize(measurement, expected_shape)

            for i, particle in enumerate(self.particles):
                try:
                    probability = self._measurement_probability(particle.state, measurement)
                    particle.weight *= probability
                    self.logger.debug(f"Updated weight for particle {i}: {particle.weight}")
                except Exception as particle_error:
                    self.logger.error(f"Error updating weight for particle {i}: {str(particle_error)}", exc_info=True)
                    # Optionally, reinitialize the particle if weight update fails
                    self._reinitialize_particle(i)

            self._normalize_weights()
            self.logger.info("Particle weights updated successfully")

        except Exception as e:
            self.logger.exception(f"Error in particle filter update: {str(e)}")
            self._reinitialize_particles()

    def _reinitialize_particles(self):
        """
        Reinitializes all particles in the filter, typically called after a critical error.
        """
        self.logger.warning("Reinitializing particles due to error")
        try:
            self.particles = [
                Particle(StateSpaceModel(self.provider_manager, self.config_manager)) for _ in range(self.n_particles)
            ]
            for particle in self.particles:
                particle.weight = 1.0 / self.n_particles
            self.logger.info("Particles reinitialized successfully")
        except Exception as e:
            self.logger.exception(f"Error in _reinitialize_particles: {str(e)}")

    def resample(self):
        """
        Resamples the particles based on their weights to focus on the most probable states.
        """
        self.logger.info("Resampling particles based on their weights")
        try:
            weights = np.array([p.weight for p in self.particles])
            weight_sum = np.sum(weights)
            if weight_sum == 0:
                self.logger.warning("All particle weights are zero. Reinitializing weights uniformly.")
                weights = np.ones(len(self.particles)) / len(self.particles)
            else:
                weights /= weight_sum

            new_particles = []
            indices = np.random.choice(len(self.particles), size=len(self.particles), p=weights)
            for i in indices:
                new_particle = Particle(self.particles[i].state)
                new_particle.weight = 1.0 / self.n_particles
                new_particles.append(new_particle)
            self.particles = new_particles
            self.logger.debug("Particle resampling completed successfully")
        except Exception as e:
            self.logger.exception(f"Error in particle resampling: {str(e)}")
            self._reinitialize_particles()

    def _apply_action(self, state, action):
        """
        Applies the given action to the state of a particle.

        Args:
            state (StateSpaceModel): The state model of the particle.
            action (dict): The action to apply to the particle's state.
        """
        self.logger.debug(f"Starting _apply_action with action: {action}")
        try:
            if isinstance(action, np.ndarray):
                # Convert numpy array to dictionary if necessary
                action = {
                    'type': 'process_task',
                    'values': action
                }

            if 'type' not in action:
                self.logger.warning(f"Action doesn't have a 'type' key: {action}")
                return

            # Apply different actions based on the action type
            if action['type'] == 'process_task':
                self.logger.debug("Applying 'process_task' action")
                state.ukf.x[-5] += 0.1  # Increase cognitive load
                state.ukf.x[-4] -= 0.05  # Decrease mental energy
            elif action['type'] == 'consolidate_memory':
                self.logger.debug("Applying 'consolidate_memory' action")
                state.ukf.x[-5] += 0.05  # Slight increase in cognitive load
                state.ukf.x[-6] += 0.05  # Slight increase in consciousness level
                state.ukf.x[3 * state.dim:4 * state.dim] += 0.1  # Increase memory activation
            elif action['type'] == 'update_focus':
                self.logger.debug("Applying 'update_focus' action")
                state.ukf.x[:state.dim] = 0.9 * state.ukf.x[:state.dim] + 0.1 * np.random.rand(state.dim)
                state.ukf.x[2 * state.dim:3 * state.dim] += 0.1  # Increase attention allocation
            elif action['type'] == 'relax':
                self.logger.debug("Applying 'relax' action")
                if 'amount' in action:
                    state.ukf.x[-5] = max(0, state.ukf.x[-5] - action['amount'])  # Decrease cognitive load
                state.ukf.x[-4] += 0.05  # Increase mental energy
            else:
                self.logger.warning(f"Unknown action type: {action['type']}")
                return  # Early return for unknown action types

            # Normalize scalar values to stay within reasonable bounds
            self.logger.debug("Normalizing scalar values")
            state.ukf.x[-8:] = np.clip(state.ukf.x[-8:], 0, 1)  # Clip scalar values between 0 and 1

            # Normalize topic focus vector
            self.logger.debug("Normalizing topic focus vector")
            norm = np.linalg.norm(state.ukf.x[:state.dim])
            if norm > 0:
                state.ukf.x[:state.dim] /= norm
            else:
                # Handle the case where the norm is zero
                state.ukf.x[:state.dim] = np.zeros(state.dim)  # Set to zero vector if norm is zero

            self.logger.debug(f"Action applied to particle state: {state.ukf.x}")
        except Exception as e:
            self.logger.error(f"Error in _apply_action: {str(e)}", exc_info=True)
            raise  # Re-raise the exception to be caught in the predict method

    def _measurement_probability(self, state, measurement):
        """
        Computes the probability of the given measurement given the state.

        Args:
            state (StateSpaceModel): The state model of the particle.
            measurement (np.ndarray): The measurement vector.

        Returns:
            float: The probability of the measurement given the state.
        """
        self.logger.debug(f"Computing measurement probability for state and measurement: {measurement}")
        try:
            if isinstance(measurement, np.ndarray):
                if measurement.shape != (state.dim_z,):
                    self.logger.error(f"Shape mismatch: measurement {measurement.shape}, expected {(state.dim_z,)}")
                    return 1.0  # Return neutral probability in case of shape mismatch
                self.logger.debug(f"Adjusted measurement shape: {measurement.shape}")
                diff = state.ukf.x[:state.dim_z] - measurement
                prob = np.exp(-np.sum(diff ** 2) / (2 * 0.1 ** 2))  # Using 0.1 as measurement noise variance
                self.logger.debug(f"Computed measurement probability: {prob}")
                return prob
            else:
                self.logger.warning(f"Unexpected measurement type: {type(measurement)}")
                return 1.0  # Return neutral probability for unexpected types
        except Exception as e:
            self.logger.error(f"Error in _measurement_probability: {str(e)}", exc_info=True)
            return 1.0  # Return neutral probability in case of error

    def _normalize_weights(self):
        """
        Normalizes the weights of all particles to ensure they sum to 1.
        """
        self.logger.debug("Normalizing weights of the particles")
        try:
            total_weight = sum(p.weight for p in self.particles)
            if total_weight > 0:
                for particle in self.particles:
                    particle.weight /= total_weight
                self.logger.debug(f"Particle weights after normalization: {[p.weight for p in self.particles]}")
            else:
                # If all weights are zero, reset to uniform weights
                uniform_weight = 1.0 / len(self.particles)
                for particle in self.particles:
                    particle.weight = uniform_weight
                self.logger.warning("All particle weights were zero. Resetting to uniform weights.")
        except Exception as e:
            self.logger.exception(f"Error in _normalize_weights: {str(e)}")
            # As a fallback, reset all weights uniformly
            try:
                uniform_weight = 1.0 / len(self.particles)
                for particle in self.particles:
                    particle.weight = uniform_weight
                self.logger.info("Particle weights reset to uniform after normalization failure.")
            except Exception as inner_e:
                self.logger.critical(f"Failed to reset particle weights uniformly: {str(inner_e)}")

    def get_estimate(self):
        """
        Computes the estimated state by averaging the states of all particles.

        Returns:
            np.ndarray: The estimated state vector.
        """
        self.logger.debug("Computing estimate of the state based on the particles")
        try:
            states = [p.state.ukf.x for p in self.particles]
            if not states:
                self.logger.warning("No particle states available to compute estimate.")
                return None
            estimate = np.mean(states, axis=0)
            self.logger.debug(f"Estimated state vector: {estimate}")
            return estimate
        except Exception as e:
            self.logger.exception(f"Error in get_estimate: {str(e)}")
            return None


async def interpret_state(state):
    """
    Provides a human-readable interpretation of the state.

    Args:
        state (StateSpaceModel): The state to interpret.

    Returns:
        str: The human-readable interpretation of the state.
    """
    try:
        desc = state.get_state_description()
        state_interpretation = (
            f"Focus: {desc.get('topic_focus', 0):.2f}, "
            f"Emotion: (V:{desc.get('emotional_valence', 0):.2f}, "
            f"A:{desc.get('emotional_arousal', 0):.2f}, "
            f"D:{desc.get('emotional_dominance', 0):.2f}), "
            f"Attention: {desc.get('attention_allocation', 0):.2f}, "
            f"Memory: {desc.get('memory_activation', 0):.2f}, "
            f"Cognitive Control: {desc.get('cognitive_control', 0):.2f}, "
            f"Consciousness: {desc.get('consciousness_level', 0):.2f}, "
            f"Cognitive Load: {desc.get('cognitive_load', 0):.2f}, "
            f"Energy: {desc.get('mental_energy', 0):.2f}, "
            f"Curiosity: {desc.get('curiosity_level', 0):.2f}, "
            f"Uncertainty: {desc.get('uncertainty', 0):.2f}, "
            f"Creativity: {desc.get('creativity_index', 0):.2f}"
        )
        config_manager = ConfigManager()
        logger = config_manager.setup_logger('InterpretState')
        logger.debug(f"State interpretation: {state_interpretation}")
        return state_interpretation
    except Exception as e:
        config_manager = ConfigManager()
        logger = config_manager.setup_logger('InterpretState')
        logger.error(f"Error interpreting state: {str(e)}", exc_info=True)
        return "Error interpreting state."
