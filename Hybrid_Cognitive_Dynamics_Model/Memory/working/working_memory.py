# modules/Hybrid_Cognitive_Dynamics_Model/Memory/Working/working_memory.py

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import logging

from modules.Config.config import ConfigManager
from modules.Hybrid_Cognitive_Dynamics_Model.SSM.state_space_model import (
    OptimizedOscillatoryNeuralLayerHH
)

# Ensure that Numba-compiled functions are compatible
import numba
from numba import njit

class EpisodicBuffer:
    """
    Buffer to store episodic memories up to a specified capacity.
    """
    def __init__(self, capacity: int):
        """
        Initializes the EpisodicBuffer with a given capacity.

        Args:
            capacity (int): Maximum number of episodes to store.
        """
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def add_episode(self, episode: Dict[str, Any]) -> None:
        """
        Adds an episode to the buffer. Removes the oldest episode if capacity is exceeded.

        Args:
            episode (Dict[str, Any]): The episode to add.
        """
        if len(self.buffer) >= self.capacity:
            removed = self.buffer.pop(0)
            logging.debug(f"Removed oldest episode: {removed}")
        self.buffer.append(episode)
        logging.debug(f"Added new episode: {episode}")

    def retrieve_recent_episodes(self, n: int) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent n episodes.

        Args:
            n (int): Number of recent episodes to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of recent episodes.
        """
        return self.buffer[-n:]

class EnhancedWorkingMemory:
    """
    Enhanced Working Memory module that manages short-term memory, gating mechanisms,
    and interacts with neural layers for processing inputs.
    """

    def __init__(
        self,
        capacity: int = 4,
        total_resources: float = 1.0,
        pfc_frequency: float = 5.0,
        striatum_frequency: float = 40.0,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Initializes the EnhancedWorkingMemory with specified parameters.

        Args:
            capacity (int, optional): Maximum number of items in working memory. Defaults to 4.
            total_resources (float, optional): Total cognitive resources available. Defaults to 1.0.
            pfc_frequency (float, optional): Frequency for the PFC neural layer. Defaults to 5.0.
            striatum_frequency (float, optional): Frequency for the Striatum neural layer. Defaults to 40.0.
            config_manager (Optional[ConfigManager], optional): Configuration manager instance. Defaults to None.
        """
        self.config_manager = config_manager or ConfigManager()
        self.logger = self.config_manager.setup_logger('EnhancedWorkingMemory')
        self.capacity = capacity
        self.total_resources = total_resources
        self.contents: OrderedDict[Tuple, Tuple[np.ndarray, float]] = OrderedDict()
        self.gate_state = "closed"
        self.dopamine_level = 0.5
        self.p3b_amplitude = 0.0

        # Initialize neural layers using OptimizedOscillatoryNeuralLayerHH
        self.pfc_layer = OptimizedOscillatoryNeuralLayerHH(
            input_size=capacity,
            output_size=capacity,
            frequency=pfc_frequency,
            dt=0.001,
            learning_rate=0.001
        )
        self.striatum_layer = OptimizedOscillatoryNeuralLayerHH(
            input_size=capacity,
            output_size=2,  # Assuming output_size=2 for go/no-go signals
            frequency=striatum_frequency,
            dt=0.001,
            learning_rate=0.001
        )
        self.prediction_layer = OptimizedOscillatoryNeuralLayerHH(
            input_size=capacity,
            output_size=capacity,
            frequency=10.0,
            dt=0.001,
            learning_rate=0.001
        )

        # Learning rates
        self.lr_weights = 0.01
        self.lr_dopamine = 0.05

        # Attention mechanism
        self.attention_weights = np.ones(capacity) / capacity
        self.current_attention_focus: Optional[np.ndarray] = None

        # Episodic buffer
        self.episodic_buffer = EpisodicBuffer(capacity=10)

        # Chunking
        self.chunks: Dict[str, List[Any]] = {}

        # Input size warnings
        self.input_size_warning_count = 0
        self.max_input_size_warnings = 5  # Limit the number of warnings

        self.logger.debug(f"EnhancedWorkingMemory initialized with capacity={self.capacity}, total_resources={self.total_resources}")

    def get_attention_focus(self) -> Optional[np.ndarray]:
        """
        Retrieves the current attention focus vector.

        Returns:
            Optional[np.ndarray]: The attention focus vector if set, else None.
        """
        return self.current_attention_focus

    def update_attention_focus(self, new_focus: np.ndarray) -> None:
        """
        Updates the attention focus vector.

        Args:
            new_focus (np.ndarray): The new attention focus vector.
        """
        self.current_attention_focus = new_focus
        self.logger.debug(f"Updated attention focus to: {new_focus}")

    def add_to_buffer(self, item: Any) -> None:
        """
        Adds an item to the working memory buffer. If capacity is exceeded, removes the least important item.

        Args:
            item (Any): The item to add.
        """
        try:
            if len(self.contents) < self.capacity:
                item_key = tuple(item.flatten()) if isinstance(item, np.ndarray) else item
                self.contents[item_key] = (item, 1.0)
                self.logger.debug(f"Added item to buffer: {item}")
            else:
                least_important = min(self.contents, key=lambda k: self.contents[k][1])
                del self.contents[least_important]
                item_key = tuple(item.flatten()) if isinstance(item, np.ndarray) else item
                self.contents[item_key] = (item, 1.0)
                self.logger.debug(f"Buffer full. Replaced least important item with: {item}")
        except Exception as e:
            self.logger.error(f"Error adding item to buffer: {str(e)}", exc_info=True)

    def get_item_activation(self, item: Any) -> float:
        """
        Retrieves the activation level of a specific item in working memory.

        Args:
            item (Any): The item to query.

        Returns:
            float: The activation level if item exists, else 0.0.
        """
        if item in self.contents:
            return self.contents[item][1]
        return 0.0

    def open_gate(self, input_signal: np.ndarray) -> bool:
        """
        Attempts to open the gating mechanism based on the Striatum layer's output and dopamine level.

        Args:
            input_signal (np.ndarray): The input signal for the Striatum layer.

        Returns:
            bool: True if the gate is opened, else False.
        """
        try:
            if input_signal.shape[0] != self.capacity:
                self.logger.warning(f"Reshaping input signal from {input_signal.shape} to ({self.capacity},)")
                input_signal = np.resize(input_signal, (self.capacity,))

            striatum_output = self.striatum_layer.forward(input_signal)
            go_signal = striatum_output[0] > striatum_output[1]
            gate_open = go_signal and (np.random.random() < self.dopamine_level)

            self.gate_state = "open" if gate_open else "closed"
            self.logger.debug(f"Gate {'opened' if gate_open else 'closed'} based on Striatum output: {striatum_output}")
            return gate_open
        except Exception as e:
            self.logger.error(f"Error in open_gate: {str(e)}", exc_info=True)
            return False

    def close_gate(self) -> None:
        """
        Closes the gating mechanism.
        """
        self.gate_state = "closed"
        self.logger.debug("Gate closed")

    async def process(self, input_signal: Any) -> Optional[np.ndarray]:
        """
        Processes the raw input signal through the working memory system.

        Args:
            input_signal (Any): The raw input signal.

        Returns:
            Optional[np.ndarray]: The processed output from the PFC layer, or None if processing fails.
        """
        self.logger.debug(f"Raw input received in process: {input_signal}")
        self.logger.debug(f"Type of raw input: {type(input_signal)}")

        try:
            prepared_signal = self.prepare_input_signal(input_signal)
            self.logger.debug(f"Prepared input signal: {prepared_signal}")

            # Handle input size
            if prepared_signal.shape[0] > self.capacity:
                if self.input_size_warning_count < self.max_input_size_warnings:
                    self.logger.warning(f"Input signal larger than capacity. Truncating from {prepared_signal.shape[0]} to {self.capacity}")
                    self.input_size_warning_count += 1
                else:
                    self.logger.warning("Input signal size warnings suppressed after reaching maximum count.")
                prepared_signal = prepared_signal[:self.capacity]
            elif prepared_signal.shape[0] < self.capacity:
                if self.input_size_warning_count < self.max_input_size_warnings:
                    self.logger.warning(f"Input signal smaller than capacity. Padding from {prepared_signal.shape[0]} to {self.capacity}")
                    self.input_size_warning_count += 1
                else:
                    self.logger.warning("Input signal size warnings suppressed after reaching maximum count.")
                prepared_signal = np.pad(prepared_signal, (0, self.capacity - prepared_signal.shape[0]))

            self.logger.debug(f"Final input signal shape: {prepared_signal.shape}")

            # Open gate based on Striatum layer's output
            gate_opened = self.open_gate(prepared_signal)

            if not gate_opened:
                self.logger.debug("Gate is closed. Skipping update.")
                return None

            # Update PFC layer
            pfc_output = self.pfc_layer.forward(prepared_signal)
            self.logger.debug(f"PFC layer output: {pfc_output}")

            # Update neural layers and prediction
            prediction_output = self.prediction_layer.forward(pfc_output)
            self.logger.debug(f"Prediction layer output: {prediction_output}")

            # Close the gate after processing
            self.close_gate()

            return pfc_output
        except Exception as e:
            self.logger.error(f"Error in process method: {str(e)}", exc_info=True)
            return None

    def prepare_input_signal(self, input_content: Any) -> np.ndarray:
        """
        Prepares the input signal by converting it into a compatible NumPy array.

        Args:
            input_content (Any): The raw input content.

        Returns:
            np.ndarray: The prepared input signal as a NumPy array.
        """
        self.logger.debug(f"Preparing input signal. Input content: {input_content}")
        self.logger.debug(f"Input content type: {type(input_content)}")

        try:
            if isinstance(input_content, np.ndarray):
                return input_content.astype(float)
            elif isinstance(input_content, str):
                return np.array([ord(c) for c in input_content], dtype=float)
            elif isinstance(input_content, dict):
                return self.dict_to_array(input_content)
            elif isinstance(input_content, list):
                combined_array = np.array([], dtype=float)
                for item in input_content:
                    if isinstance(item, dict):
                        combined_array = np.concatenate([combined_array, self.dict_to_array(item)])
                    elif isinstance(item, str):
                        combined_array = np.concatenate([combined_array, np.array([ord(c) for c in item], dtype=float)])
                    elif isinstance(item, (int, float)):
                        combined_array = np.append(combined_array, item)
                return combined_array
            else:
                return np.array([hash(str(input_content)) % 1000], dtype=float)
        except Exception as e:
            self.logger.error(f"Error in prepare_input_signal: {str(e)}", exc_info=True)
            return np.array([])

    def dict_to_array(self, input_dict: Dict[Any, Any]) -> np.ndarray:
        """
        Converts a dictionary input into a NumPy array.

        Args:
            input_dict (Dict[Any, Any]): The input dictionary.

        Returns:
            np.ndarray: The converted NumPy array.
        """
        values = []
        try:
            for value in input_dict.values():
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, str):
                    values.append(np.mean([ord(c) for c in value]))
                elif isinstance(value, list):
                    for item in value:
                        sub_array = self.dict_to_array({'item': item})
                        values.extend(sub_array.tolist())
                elif isinstance(value, dict):
                    sub_array = self.dict_to_array(value)
                    values.extend(sub_array.tolist())
                else:
                    values.append(float(hash(str(value)) % 1000))
            return np.array(values, dtype=float)
        except Exception as e:
            self.logger.error(f"Error in dict_to_array: {str(e)}", exc_info=True)
            return np.array([])

    async def update(self, item: np.ndarray, importance: float = 1.0) -> None:
        """
        Updates the working memory with a new item based on the gate state.

        Args:
            item (np.ndarray): The item to update.
            importance (float, optional): The importance level of the item. Defaults to 1.0.
        """
        if self.gate_state != "open":
            self.logger.debug("Gate is closed. Update operation skipped.")
            return

        try:
            item_key = tuple(item.flatten()) if isinstance(item, np.ndarray) else item

            if len(self.contents) < self.capacity:
                self.contents[item_key] = (item, importance)
                self.logger.debug(f"Added new item to working memory. Current size: {len(self.contents)}")
            else:
                least_important = min(self.contents, key=lambda k: self.contents[k][1])
                del self.contents[least_important]
                self.contents[item_key] = (item, importance)
                self.logger.debug(f"Replaced least important item in working memory. Maintained size: {len(self.contents)}")

            await self.reallocate_resources()
            self.logger.debug("Resources reallocated after update.")

            # Update PFC layer with attention weights
            pfc_input = np.zeros(self.capacity)
            pfc_input[len(self.contents) - 1] = 1
            pfc_output = self.pfc_layer.forward(pfc_input)
            self.pfc_layer.output *= self.attention_weights
            self.logger.debug("PFC layer updated and attention weights applied.")

        except Exception as e:
            self.logger.error(f"Error in update method: {str(e)}", exc_info=True)

    def remove(self, item: Any) -> None:
        """
        Removes an item from the working memory.

        Args:
            item (Any): The item to remove.
        """
        try:
            item_key = tuple(item.flatten()) if isinstance(item, np.ndarray) else item
            if item_key in self.contents:
                del self.contents[item_key]
                self.logger.debug(f"Removed item from working memory: {item}")
                asyncio.create_task(self.reallocate_resources())
            else:
                self.logger.warning(f"Attempted to remove non-existent item: {item}")
        except Exception as e:
            self.logger.error(f"Error in remove method: {str(e)}", exc_info=True)

    def get_contents(self) -> List[Any]:
        """
        Retrieves all items currently in working memory.

        Returns:
            List[Any]: A list of items in working memory.
        """
        return [item for item, _ in self.contents.values()]

    async def reallocate_resources(self) -> None:
        """
        Reallocates cognitive resources based on the importance of items in working memory.
        """
        try:
            if self.contents:
                total_importance = sum(importance for _, importance in self.contents.values())
                if total_importance > 0:
                    for key in self.contents:
                        item, importance = self.contents[key]
                        self.contents[key] = (item, importance * self.total_resources / total_importance)
                else:
                    equal_importance = self.total_resources / len(self.contents)
                    for key in self.contents:
                        item, _ = self.contents[key]
                        self.contents[key] = (item, equal_importance)
                self.logger.debug("Resources reallocated based on importance levels.")
            else:
                self.logger.debug("No items in working memory to reallocate resources.")
        except Exception as e:
            self.logger.error(f"Error in reallocate_resources: {str(e)}", exc_info=True)

    async def simulate_p3b(self, stimulus: Any) -> None:
        """
        Simulates the P3b component based on the stimulus and its importance.

        Args:
            stimulus (Any): The stimulus to simulate P3b for.
        """
        try:
            stimulus_key = self.get_stimulus_key(stimulus)
            if stimulus_key in self.contents:
                _, importance = self.contents[stimulus_key]
                self.p3b_amplitude = 0.5 + 0.5 * importance
                self.logger.debug(f"P3b amplitude set to: {self.p3b_amplitude} based on importance.")
            else:
                self.p3b_amplitude = 1.0
                self.logger.debug("P3b amplitude set to default: 1.0.")
        except Exception as e:
            self.logger.error(f"Error in simulate_p3b: {str(e)}", exc_info=True)

    def get_stimulus_key(self, stimulus: Any) -> Any:
        """
        Generates a hashable key for a given stimulus.

        Args:
            stimulus (Any): The stimulus to generate a key for.

        Returns:
            Any: A hashable representation of the stimulus.
        """
        if isinstance(stimulus, np.ndarray):
            return hash(stimulus.tobytes())
        elif isinstance(stimulus, dict):
            return frozenset(stimulus.items())
        elif isinstance(stimulus, list):
            return tuple(stimulus)
        else:
            return stimulus

    def update_dopamine(self, reward: float) -> None:
        """
        Updates the dopamine level based on the received reward.

        Args:
            reward (float): The reward value to update dopamine with.
        """
        try:
            prediction_error = reward - self.dopamine_level
            self.dopamine_level += self.lr_dopamine * prediction_error
            self.logger.debug(f"Dopamine level updated to: {self.dopamine_level}")
        except Exception as e:
            self.logger.error(f"Error in update_dopamine: {str(e)}", exc_info=True)

    def learn(self, input_signal: np.ndarray, reward: float) -> None:
        """
        Updates neural layer weights based on the input signal and received reward.

        Args:
            input_signal (np.ndarray): The input signal used for learning.
            reward (float): The reward value influencing learning.
        """
        try:
            prediction_error = reward - self.dopamine_level
            weight_update = self.lr_weights * prediction_error * np.outer(self.striatum_layer.output, input_signal)
            self.striatum_layer.weights += weight_update
            self.logger.debug(f"Striatum layer weights updated by: {weight_update}")

            weight_update_pfc = self.lr_weights * prediction_error * np.outer(self.pfc_layer.output, input_signal)
            self.pfc_layer.weights += weight_update_pfc
            self.logger.debug(f"PFC layer weights updated by: {weight_update_pfc}")
        except Exception as e:
            self.logger.error(f"Error in learn method: {str(e)}", exc_info=True)

    def focus_attention(self, item_index: int) -> None:
        """
        Focuses attention on a specific item in working memory by updating attention weights.

        Args:
            item_index (int): The index of the item to focus attention on.
        """
        try:
            self.attention_weights = np.zeros(self.capacity)
            self.attention_weights[item_index] = 1.0
            self.logger.debug(f"Focused attention on item index: {item_index}")
        except Exception as e:
            self.logger.error(f"Error in focus_attention: {str(e)}", exc_info=True)

    def distribute_attention(self) -> None:
        """
        Distributes attention evenly across all items in working memory.
        """
        try:
            self.attention_weights = np.ones(self.capacity) / self.capacity
            self.logger.debug("Distributed attention evenly across all items.")
        except Exception as e:
            self.logger.error(f"Error in distribute_attention: {str(e)}", exc_info=True)

    def process_episode(self, episode: Dict[str, Any]) -> None:
        """
        Processes and adds an episode to the episodic buffer.

        Args:
            episode (Dict[str, Any]): The episode to process.
        """
        try:
            self.episodic_buffer.add_episode(episode)
            self.logger.debug(f"Processed and added episode: {episode}")
        except Exception as e:
            self.logger.error(f"Error in process_episode: {str(e)}", exc_info=True)

    def learn_prediction(self, actual_next: np.ndarray) -> float:
        """
        Learns from the prediction by updating weights based on the actual next input.

        Args:
            actual_next (np.ndarray): The actual next input signal.

        Returns:
            float: The mean absolute prediction error.
        """
        try:
            predicted = self.predict_next()
            prediction_error = self.compute_prediction_error(predicted, actual_next)
            learning_rate = self.adaptive_learning_rate(prediction_error)
            weight_update = learning_rate * np.outer(prediction_error, self.pfc_layer.output)
            self.prediction_layer.weights += weight_update
            self.logger.debug(f"Prediction layer weights updated by: {weight_update}")
            return np.mean(np.abs(prediction_error))
        except Exception as e:
            self.logger.error(f"Error in learn_prediction: {str(e)}", exc_info=True)
            return 0.0

    def predict_next(self) -> np.ndarray:
        """
        Generates a prediction for the next input based on the current PFC layer output.

        Returns:
            np.ndarray: The predicted next input signal.
        """
        try:
            current_state = self.pfc_layer.output
            prediction = self.prediction_layer.forward(current_state)
            self.logger.debug(f"Predicted next input: {prediction}")
            return prediction
        except Exception as e:
            self.logger.error(f"Error in predict_next: {str(e)}", exc_info=True)
            return np.array([])

    def compute_prediction_error(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """
        Computes the prediction error between predicted and actual inputs.

        Args:
            predicted (np.ndarray): The predicted input signal.
            actual (np.ndarray): The actual input signal.

        Returns:
            np.ndarray: The prediction error vector.
        """
        try:
            error = actual - predicted
            self.logger.debug(f"Computed prediction error: {error}")
            return error
        except Exception as e:
            self.logger.error(f"Error in compute_prediction_error: {str(e)}", exc_info=True)
            return np.array([])

    def adaptive_learning_rate(self, prediction_error: np.ndarray) -> float:
        """
        Adjusts the learning rate based on the magnitude of the prediction error.

        Args:
            prediction_error (np.ndarray): The prediction error vector.

        Returns:
            float: The adjusted learning rate.
        """
        try:
            base_rate = 0.01
            adjustment = base_rate / (1 + np.mean(np.abs(prediction_error)))
            self.logger.debug(f"Adaptive learning rate set to: {adjustment}")
            return adjustment
        except Exception as e:
            self.logger.error(f"Error in adaptive_learning_rate: {str(e)}", exc_info=True)
            return 0.01

    async def create_chunk(self, items: List[Any], chunk_name: str) -> None:
        """
        Creates a chunk from a list of items and adds it to working memory.

        Args:
            items (List[Any]): The items to include in the chunk.
            chunk_name (str): The name of the chunk.
        """
        try:
            if len(self.chunks) >= self.capacity:
                self.logger.warning("Maximum number of chunks reached. Cannot create new chunk.")
                return

            hashable_items = [tuple(item.flatten()) if isinstance(item, np.ndarray) else item for item in items]
            self.chunks[chunk_name] = hashable_items
            chunk_importance = 0.0

            for item in hashable_items:
                if item in self.contents:
                    _, importance = self.contents[item]
                    chunk_importance += importance
                    del self.contents[item]
                    self.logger.debug(f"Removed item from working memory to form chunk: {item}")

            self.contents[chunk_name] = (hashable_items, chunk_importance)
            await self.reallocate_resources()
            self.logger.debug(f"Created chunk '{chunk_name}' with items: {hashable_items}")
        except Exception as e:
            self.logger.error(f"Error in create_chunk: {str(e)}", exc_info=True)

    async def expand_chunk(self, chunk_name: str) -> List[Any]:
        """
        Expands a previously created chunk by re-adding its constituent items to working memory.

        Args:
            chunk_name (str): The name of the chunk to expand.

        Returns:
            List[Any]: The list of items added back to working memory.
        """
        try:
            if chunk_name not in self.chunks:
                self.logger.warning(f"Chunk '{chunk_name}' not found.")
                return []

            items = self.chunks.pop(chunk_name)
            self.logger.debug(f"Expanding chunk '{chunk_name}' with items: {items}")

            for item in items:
                await self.update(item)
                self.logger.debug(f"Re-added item from chunk to working memory: {item}")

            return items
        except Exception as e:
            self.logger.error(f"Error in expand_chunk: {str(e)}", exc_info=True)
            return []

# Numba-optimized functions for Accelerated Computations
@njit
def compute_dv_hh(voltage: np.ndarray, recovery: np.ndarray, input_signal: np.ndarray, dt: float) -> np.ndarray:
    """
    Computes the change in voltage (dv) using the Hodgkin-Huxley equations.

    Args:
        voltage (np.ndarray): Current membrane potentials.
        recovery (np.ndarray): Current recovery variables.
        input_signal (np.ndarray): Input current.
        dt (float): Time step.

    Returns:
        np.ndarray: Change in voltage.
    """
    return (0.04 * voltage ** 2) + (5 * voltage) + 140 - recovery + input_signal * dt

@njit
def compute_dr_hh(a: float, b: float, voltage: np.ndarray, recovery: np.ndarray, dt: float) -> np.ndarray:
    """
    Computes the change in recovery variable (dr) using the Hodgkin-Huxley equations.

    Args:
        a (float): Parameter a.
        b (float): Parameter b.
        voltage (np.ndarray): Current membrane potentials.
        recovery (np.ndarray): Current recovery variables.
        dt (float): Time step.

    Returns:
        np.ndarray: Change in recovery.
    """
    return a * (b * voltage - recovery) * dt

# ReferenceBackTask and Experiment Functions (Assuming they are part of this module)
class ReferenceBackTask:
    """
    A task to present stimuli and manage reference trials within the working memory system.
    """

    def __init__(self, wm_model: EnhancedWorkingMemory):
        """
        Initializes the ReferenceBackTask with a working memory model.

        Args:
            wm_model (EnhancedWorkingMemory): The working memory model instance.
        """
        self.config_manager = wm_model.config_manager
        self.logger = self.config_manager.setup_logger('ReferenceBackTask')
        self.wm = wm_model
        self.current_stimulus: Optional[Any] = None
        self.is_reference_trial: bool = False
        self.accumulator = EvidenceAccumulator()
        self.stimulus_generator = ComplexStimulusGenerator(n_features=self.wm.capacity)
        self.sequence: List[Tuple[np.ndarray, int]] = []
        self.sequence_length: int = 5

    def generate_sequence(self) -> None:
        """
        Generates a sequence of stimuli for the reference back task.
        """
        self.sequence = [self.stimulus_generator.generate() for _ in range(self.sequence_length)]
        self.logger.debug(f"Generated new stimulus sequence: {self.sequence}")

    async def present_stimulus(self, is_reference: bool) -> Tuple[int, bool, int]:
        """
        Presents a stimulus to the working memory and processes the trial.

        Args:
            is_reference (bool): Indicates whether the trial is a reference trial.

        Returns:
            Tuple[int, bool, int]: Reaction time, decision outcome, and category of the stimulus.
        """
        try:
            if not self.sequence:
                self.generate_sequence()

            self.current_stimulus, category = self.sequence.pop(0)
            self.is_reference_trial = is_reference
            self.logger.debug(f"Presenting stimulus: {self.current_stimulus}, Reference Trial: {is_reference}")

            await self.wm.simulate_p3b(self.current_stimulus)

            # Prepare input signal ensuring it fits within working memory capacity
            input_signal = np.zeros(self.wm.capacity)
            input_length = len(self.current_stimulus)
            input_length = min(input_length, self.wm.capacity - 1)
            input_signal[:input_length] = self.current_stimulus[:input_length]
            input_signal[-1] = int(is_reference)

            if is_reference:
                gate_open = self.wm.open_gate(input_signal)
                if gate_open:
                    await self.wm.update(self.current_stimulus, importance=category + 1)
                    reward = 0.1
                    self.logger.debug("Gate opened and item updated with positive reward.")
                else:
                    reward = -0.1
                    self.logger.debug("Gate failed to open. Negative reward assigned.")
                self.wm.close_gate()
            else:
                reward = 0.0
                self.logger.debug("Non-reference trial. No reward assigned.")

            decision_time, decision = self.make_decision()

            # Learning
            self.wm.learn(input_signal, reward)
            self.wm.update_dopamine(reward)

            # Predictive coding
            prediction_error = self.wm.learn_prediction(input_signal)
            self.logger.debug(f"Prediction error: {prediction_error}")

            # Process episode
            episode = {
                'stimulus': self.current_stimulus,
                'is_reference': is_reference,
                'decision': decision,
                'decision_time': decision_time,
                'prediction_error': prediction_error
            }
            self.wm.process_episode(episode)
            self.logger.debug(f"Processed episode: {episode}")

            return decision_time, decision, category
        except Exception as e:
            self.logger.error(f"Error in present_stimulus: {str(e)}", exc_info=True)
            return 0, False, 0

    def make_decision(self) -> Tuple[int, bool]:
        """
        Makes a decision based on the current contents of working memory.

        Returns:
            Tuple[int, bool]: Reaction time and decision outcome.
        """
        try:
            stimulus_in_contents = any(
                np.array_equal(self.current_stimulus, content) for content in self.wm.get_contents()
            )

            drift_adjustment = 0.05 if stimulus_in_contents else -0.05
            drift_adjustment += 0.02 * self.wm.p3b_amplitude

            self.accumulator.drift_rate = 0.1 + drift_adjustment
            self.logger.debug(f"Decision drift rate adjusted to: {self.accumulator.drift_rate}")

            return self.accumulator.accumulate()
        except Exception as e:
            self.logger.error(f"Error in make_decision: {str(e)}", exc_info=True)
            return 0, False

class EvidenceAccumulator:
    """
    Accumulates evidence over time to make decisions based on drift-diffusion models.
    """

    def __init__(self, drift_rate: float = 0.1, threshold: float = 1.0, noise: float = 0.1):
        """
        Initializes the EvidenceAccumulator with specified parameters.

        Args:
            drift_rate (float, optional): Rate at which evidence accumulates. Defaults to 0.1.
            threshold (float, optional): Threshold for making a decision. Defaults to 1.0.
            noise (float, optional): Noise level in evidence accumulation. Defaults to 0.1.
        """
        self.drift_rate = drift_rate
        self.threshold = threshold
        self.noise = noise

    def accumulate(self) -> Tuple[int, bool]:
        """
        Accumulates evidence until the threshold is reached.

        Returns:
            Tuple[int, bool]: Number of time steps taken and the decision outcome.
        """
        evidence = 0.0
        time_steps = 0
        decision = False

        while abs(evidence) < self.threshold:
            evidence += self.drift_rate + np.random.normal(0, self.noise)
            time_steps += 1
            if evidence >= self.threshold:
                decision = True
            elif evidence <= -self.threshold:
                decision = False

        return time_steps, decision

class ComplexStimulusGenerator:
    """
    Generates complex stimuli with multiple features and categories.
    """

    def __init__(self, n_features: int = 3, n_categories: int = 2):
        """
        Initializes the ComplexStimulusGenerator with specified parameters.

        Args:
            n_features (int, optional): Number of features in each stimulus. Defaults to 3.
            n_categories (int, optional): Number of categories for stimuli. Defaults to 2.
        """
        self.n_features = n_features
        self.n_categories = n_categories

    def generate(self) -> Tuple[np.ndarray, int]:
        """
        Generates a single stimulus with features and a category.

        Returns:
            Tuple[np.ndarray, int]: The generated features and their category.
        """
        features = np.random.randint(0, 2, self.n_features)
        category = np.random.randint(0, self.n_categories)
        return features.astype(float), category

async def run_experiment(num_trials: int = 1000) -> List[Dict[str, Any]]:
    """
    Runs an experiment by executing multiple trials of the reference back task.

    Args:
        num_trials (int, optional): Number of trials to run. Defaults to 1000.

    Returns:
        List[Dict[str, Any]]: A list of results from each trial.
    """
    try:
        config_manager = ConfigManager()
        wm = EnhancedWorkingMemory(config_manager=config_manager)
        task = ReferenceBackTask(wm)

        results: List[Dict[str, Any]] = []
        for trial in range(1, num_trials + 1):
            is_reference = np.random.choice([True, False])
            rt, decision, category = await task.present_stimulus(is_reference)

            results.append({
                'stimulus': task.current_stimulus,
                'category': category,
                'is_reference': is_reference,
                'reaction_time': rt,
                'decision': decision,
                'wm_contents': wm.get_contents(),
                'dopamine_level': wm.dopamine_level,
                'p3b_amplitude': wm.p3b_amplitude
            })

            # Occasionally create and expand chunks
            if trial % 50 == 0:
                contents = wm.get_contents()
                if len(contents) >= 2:
                    chunk_items = contents[:2]
                    await wm.create_chunk(chunk_items, f"chunk_{trial}")
                    wm.logger.debug(f"Chunk created at trial {trial}: {chunk_items}")
            if trial % 100 == 0:
                if wm.chunks:
                    chunk_to_expand = list(wm.chunks.keys())[0]
                    await wm.expand_chunk(chunk_to_expand)
                    wm.logger.debug(f"Chunk expanded at trial {trial}: {chunk_to_expand}")

        return results
    except Exception as e:
        logging.error(f"Error in run_experiment: {str(e)}", exc_info=True)
        return []

async def process_results():
    """
    Processes the results of the experiment by computing statistics and visualizing data.
    """
    try:
        results = await run_experiment()

        # Calculate average reaction times and error rates
        ref_rts = [r['reaction_time'] for r in results if r['is_reference']]
        comp_rts = [r['reaction_time'] for r in results if not r['is_reference']]

        ref_errors = [1 - r['decision'] for r in results if r['is_reference']]
        comp_errors = [1 - r['decision'] for r in results if not r['is_reference']]

        print(f"Average RT (Reference): {np.mean(ref_rts):.2f}")
        print(f"Average RT (Comparison): {np.mean(comp_rts):.2f}")
        print(f"Error Rate (Reference): {np.mean(ref_errors):.2f}")
        print(f"Error Rate (Comparison): {np.mean(comp_errors):.2f}")

        # Analyze learning effects
        dopamine_levels = [r['dopamine_level'] for r in results]
        plt.figure(figsize=(10, 5))
        plt.plot(dopamine_levels)
        plt.title('Dopamine Level Over Time')
        plt.xlabel('Trial')
        plt.ylabel('Dopamine Level')
        plt.show()

        # Analyze P3b amplitude
        p3b_amplitudes = [r['p3b_amplitude'] for r in results]
        plt.figure(figsize=(10, 5))
        plt.plot(p3b_amplitudes)
        plt.title('P3b Amplitude Over Time')
        plt.xlabel('Trial')
        plt.ylabel('P3b Amplitude')
        plt.show()
    except Exception as e:
        logging.error(f"Error in process_results: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(process_results())
