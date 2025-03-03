# modules/Hybrid_Cognitive_Dynamics_Model/Memory/time_aware_processing.py

import math
import time
from enum import Enum
from queue import PriorityQueue
import threading
import asyncio
from typing import Optional, Dict, Any

from modules.Providers.provider_manager import ProviderManager
from modules.Config.config import ConfigManager
from modules.Hybrid_Cognitive_Dynamics_Model.Time_Processing.cognitive_temporal_state import (
    CognitiveTemporalStateEnum
)

from numba import njit
import numpy as np


class MemoryType(Enum):
    """
    Enum representing different types of memory.
    """
    SENSORY = 1
    SHORT_TERM = 2
    LONG_TERM_EPISODIC = 3
    LONG_TERM_SEMANTIC = 4


class TimeDecay:
    """
    Implements time-based decay mechanisms for different types of memory, adapting the rate
    based on cognitive load, attention level, emotional state, memory importance, and CognitiveTemporalState.
    """

    def __init__(self, system_state: Any, config_manager: ConfigManager):
        """
        Initializes the TimeDecay class with the system state and configuration manager.

        Args:
            system_state (Any): The current state of the system (cognitive load, attention, emotions, etc.).
            config_manager (ConfigManager): The configuration manager for retrieving settings.
        """
        self.system_state = system_state
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('TimeDecay')

        # Use settings from ConfigManager for decay rates
        decay_config = config_manager.get_subsystem_config('time_aware_processing') if config_manager else {}
        self.base_decay_rates = {
            MemoryType.SENSORY: decay_config.get('decay_rates', {}).get('sensory_decay_rate', 0.1),
            MemoryType.SHORT_TERM: decay_config.get('decay_rates', {}).get('short_term_decay_rate', 0.01),
            MemoryType.LONG_TERM_EPISODIC: decay_config.get('decay_rates', {}).get('long_term_epidolic_decay_rate', 0.001),
            MemoryType.LONG_TERM_SEMANTIC: decay_config.get('decay_rates', {}).get('long_term_semantic_decay_rate', 0.0001)
        }

        self.logger.debug(f"Initialized TimeDecay with base_decay_rates: {self.base_decay_rates}")

    def decay(self, memory_type: MemoryType, time_elapsed: float, importance: float) -> float:
        """
        Applies decay to a memory type based on the time elapsed and its importance,
        influenced by the current emotional state and CognitiveTemporalState.

        Args:
            memory_type (MemoryType): The type of memory being decayed.
            time_elapsed (float): The amount of time passed since the memory was created.
            importance (float): The importance factor of the memory.

        Returns:
            float: The decayed memory value.
        """
        try:
            base_rate = self.base_decay_rates[memory_type]
            adaptive_rate = self._compute_adaptive_rate(memory_type, importance)

            if memory_type == MemoryType.SENSORY:
                # Sensory memory decays rapidly using exponential decay
                decayed_value = self._exponential_decay(base_rate * adaptive_rate, time_elapsed)
            elif memory_type == MemoryType.SHORT_TERM:
                # Short-term memory uses power law decay for slower fading
                decayed_value = self._power_law_decay(base_rate * adaptive_rate, time_elapsed)
            else:
                # Long-term memories use logarithmic decay to preserve over extended periods
                decayed_value = self._logarithmic_decay(base_rate * adaptive_rate, time_elapsed)

            self.logger.debug(
                f"Decay applied - MemoryType: {memory_type.name}, Time Elapsed: {time_elapsed}, "
                f"Importance: {importance}, Decayed Value: {decayed_value}"
            )
            return decayed_value
        except Exception as e:
            self.logger.error(f"Error in decay method: {str(e)}", exc_info=True)
            return 0.0

    def _exponential_decay(self, rate: float, time: float) -> float:
        """
        Exponential decay function, typically for sensory memory.

        Args:
            rate (float): The decay rate.
            time (float): The time elapsed.

        Returns:
            float: The exponentially decayed value.
        """
        try:
            decayed = math.exp(-rate * time)
            return decayed
        except Exception as e:
            self.logger.error(f"Error in _exponential_decay: {str(e)}", exc_info=True)
            return 0.0

    def _power_law_decay(self, rate: float, time: float) -> float:
        """
        Power law decay function, typically for short-term memory.

        Args:
            rate (float): The decay rate.
            time (float): The time elapsed.

        Returns:
            float: The decayed value based on power law.
        """
        try:
            decayed = 1 / (1 + rate * time)
            return decayed
        except Exception as e:
            self.logger.error(f"Error in _power_law_decay: {str(e)}", exc_info=True)
            return 0.0

    def _logarithmic_decay(self, rate: float, time: float) -> float:
        """
        Logarithmic decay function, typically for long-term memory.

        Args:
            rate (float): The decay rate.
            time (float): The time elapsed.

        Returns:
            float: The decayed value based on a logarithmic function.
        """
        try:
            decayed = 1 - rate * math.log(1 + time)
            return decayed
        except Exception as e:
            self.logger.error(f"Error in _logarithmic_decay: {str(e)}", exc_info=True)
            return 0.0

    def _compute_adaptive_rate(self, memory_type: MemoryType, importance: float) -> float:
        """
        Computes an adaptive decay rate based on cognitive load, attention level, emotional valence,
        memory importance, and current CognitiveTemporalState.

        This method is broken down into smaller components for better testability and debuggability.

        Args:
            memory_type (MemoryType): The type of memory being decayed.
            importance (float): The importance factor of the memory.

        Returns:
            float: The adaptive decay rate.
        """
        try:
            cognitive_load = self.system_state.cognitive_load
            attention_level = self.system_state.consciousness_level
            emotional_valence = self.system_state.emotional_state.get('valence', 0.0)
            current_temporal_state = self.system_state.current_cognitive_temporal_state.get_current_state()

            temporal_adjustment = self._get_temporal_adjustment(current_temporal_state)
            cognitive_influence = self._compute_cognitive_influence(cognitive_load)
            attention_influence = self._compute_attention_influence(attention_level)
            emotional_influence = self._compute_emotional_influence(emotional_valence)

            adaptive_factor = (1 + cognitive_influence - attention_influence + emotional_influence) * temporal_adjustment
            importance_factor = self._compute_importance_factor(importance)

            self.logger.debug(
                f"Adaptive Rate Computation - MemoryType: {memory_type.name}, "
                f"Cognitive Load: {cognitive_load}, Attention Level: {attention_level}, "
                f"Emotional Valence: {emotional_valence}, Temporal Adjustment: {temporal_adjustment}, "
                f"Cognitive Influence: {cognitive_influence}, Attention Influence: {attention_influence}, "
                f"Emotional Influence: {emotional_influence}, Adaptive Factor: {adaptive_factor}, "
                f"Importance Factor: {importance_factor}"
            )

            return adaptive_factor * importance_factor
        except Exception as e:
            self.logger.error(f"Error in _compute_adaptive_rate: {str(e)}", exc_info=True)
            return 1.0

    def _compute_cognitive_influence(self, cognitive_load: float) -> float:
        """
        Computes the influence of cognitive load on the decay rate.

        Args:
            cognitive_load (float): The current cognitive load (e.g., 0.0 to 1.0).

        Returns:
            float: The cognitive influence factor.
        """
        try:
            # Higher cognitive load may slow down decay to retain important information
            influence = cognitive_load * 0.5  # Scaling factor can be adjusted
            return influence
        except Exception as e:
            self.logger.error(f"Error in _compute_cognitive_influence: {str(e)}", exc_info=True)
            return 0.0

    def _compute_attention_influence(self, attention_level: float) -> float:
        """
        Computes the influence of attention level on the decay rate.

        Args:
            attention_level (float): The current attention level (e.g., 0.0 to 1.0).

        Returns:
            float: The attention influence factor.
        """
        try:
            # Higher attention may accelerate decay as information is processed
            influence = (1 - attention_level) * 0.3  # Inverse relation; scaling factor adjustable
            return influence
        except Exception as e:
            self.logger.error(f"Error in _compute_attention_influence: {str(e)}", exc_info=True)
            return 0.0

    def _compute_emotional_influence(self, emotional_valence: float) -> float:
        """
        Computes the influence of emotional valence on the decay rate.

        Args:
            emotional_valence (float): The current emotional valence (-1.0 to 1.0).

        Returns:
            float: The emotional influence factor.
        """
        try:
            # Positive emotions might slow decay, negative might speed it up
            influence = emotional_valence * 0.2  # Scaling factor can be adjusted
            return influence
        except Exception as e:
            self.logger.error(f"Error in _compute_emotional_influence: {str(e)}", exc_info=True)
            return 0.0

    def _compute_importance_factor(self, importance: float) -> float:
        """
        Computes the importance factor influencing the decay rate.

        Args:
            importance (float): The importance factor of the memory.

        Returns:
            float: The importance factor.
        """
        try:
            # More important memories decay slower
            factor = 1 / (1 + importance)
            return factor
        except Exception as e:
            self.logger.error(f"Error in _compute_importance_factor: {str(e)}", exc_info=True)
            return 1.0

    def _get_temporal_adjustment(self, temporal_state: CognitiveTemporalStateEnum) -> float:
        """
        Determines the adjustment factor for decay rates based on the current CognitiveTemporalState.

        Args:
            temporal_state (CognitiveTemporalStateEnum): The current CognitiveTemporalState.

        Returns:
            float: The adjustment factor.
        """
        try:
            adjustments = {
                CognitiveTemporalStateEnum.IMMEDIATE: 1.0,
                CognitiveTemporalStateEnum.REFLECTIVE: 0.8,        # Slower decay
                CognitiveTemporalStateEnum.EMOTIONAL: 1.2,         # Faster decay
                CognitiveTemporalStateEnum.DEEP_LEARNING: 0.6,     # Extremely slow decay for deep learning
                CognitiveTemporalStateEnum.SOCIAL: 1.0,            # Balanced decay rates for social interactions
                CognitiveTemporalStateEnum.REACTIVE: 1.3,          # Faster decay for non-essential memories
                CognitiveTemporalStateEnum.ANALYTICAL: 0.9,        # Slower decay for problem-solving memories
                CognitiveTemporalStateEnum.CREATIVE: 1.1,          # Slightly faster decay to foster creativity
                CognitiveTemporalStateEnum.FOCUSED: 0.7            # Slow decay to maintain focus
            }

            adjustment = adjustments.get(temporal_state, 1.0)
            self.logger.debug(f"CognitiveTemporalState adjustment factor for {temporal_state.name}: {adjustment}")
            return adjustment
        except Exception as e:
            self.logger.error(f"Error in _get_temporal_adjustment: {str(e)}", exc_info=True)
            return 1.0

    def update_cognitive_temporal_state(self, new_temporal_state: CognitiveTemporalStateEnum) -> None:
        """
        Update decay rates or behaviors based on the new CognitiveTemporalState.

        Args:
            new_temporal_state (CognitiveTemporalStateEnum): The new CognitiveTemporalState to adapt to.
        """
        try:
            # Currently, TimeDecay uses CognitiveTemporalState adjustments dynamically in _compute_adaptive_rate
            # Additional behaviors can be implemented here if needed
            self.logger.info(f"TimeDecay received new CognitiveTemporalState: {new_temporal_state.name}")
        except Exception as e:
            self.logger.error(f"Error in update_cognitive_temporal_state: {str(e)}", exc_info=True)


class SpacedRepetition:
    """
    Implements a spaced repetition algorithm to reinforce memory over time based on a review schedule,
    influenced by the emotional state of the system.
    """

    def __init__(self, memory_store: Any, config_manager: ConfigManager):
        """
        Initializes the SpacedRepetition class.

        Args:
            memory_store (Any): The memory store to manage reviews.
            config_manager (ConfigManager): The configuration manager for retrieving settings.
        """
        self.memory_store = memory_store
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('SpacedRepetition')

        # Use settings from ConfigManager
        repetition_config = config_manager.get_subsystem_config('time_aware_processing') if config_manager else {}
        self.review_queue = PriorityQueue()
        self.sm2_params = {
            "ease_factor": repetition_config.get('spaced_repetition', {}).get('ease_factor', 2.5),
            "interval": repetition_config.get('spaced_repetition', {}).get('initial_interval', 1),
            "repetitions": repetition_config.get('spaced_repetition', {}).get('initial_repetitions', 0)
        }

    def schedule_review(self, memory: Any, review_time: float, emotion_factor: float = 1.0) -> None:
        """
        Schedules a memory for review at a given time, adjusted by an emotion factor.

        Args:
            memory (Any): The memory object to review.
            review_time (float): The time at which the memory should be reviewed (epoch time).
            emotion_factor (float, optional): Factor to adjust the review time based on emotion. Defaults to 1.0.
        """
        try:
            adjusted_review_time = review_time / emotion_factor  # Shorten review time for highly emotional memories
            self.review_queue.put((adjusted_review_time, memory))
            self.logger.debug(f"Scheduled review for memory at {adjusted_review_time} with emotion_factor {emotion_factor}")
        except Exception as e:
            self.logger.error(f"Error in schedule_review: {str(e)}", exc_info=True)

    def review(self, memory: Any, quality: int) -> Dict[str, Any]:
        """
        Reviews a memory and updates its spaced repetition parameters based on the quality of recall.

        Args:
            memory (Any): The memory object to review.
            quality (int): The quality of recall (0-5).

        Returns:
            Dict[str, Any]: Updated spaced repetition parameters.
        """
        try:
            params = self.sm2_params.copy()
            if quality >= 3:
                if params["repetitions"] == 0:
                    params["interval"] = 1
                elif params["repetitions"] == 1:
                    params["interval"] = 6
                else:
                    params["interval"] *= params["ease_factor"]

                params["repetitions"] += 1
            else:
                params["repetitions"] = 0
                params["interval"] = 1

            params["ease_factor"] += (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            params["ease_factor"] = max(1.3, params["ease_factor"])

            next_review = time.time() + params["interval"] * 86400  # Convert days to seconds
            self.schedule_review(memory, next_review, emotion_factor=1.0)  # emotion_factor can be adjusted as needed

            self.logger.debug(f"Reviewed memory with quality {quality}. Next review in {params['interval']} days.")
            return params
        except Exception as e:
            self.logger.error(f"Error in review method: {str(e)}", exc_info=True)
            return {}

    def adjust_review_schedule_based_on_emotion(self, memory: Any, emotion_factor: float) -> None:
        """
        Adjusts the review schedule of a memory based on its emotional significance.

        Args:
            memory (Any): The memory object to adjust.
            emotion_factor (float): The factor by which to adjust the review timing.
        """
        try:
            current_time = time.time()
            # Temporarily store items to be reinserted
            temp_queue = PriorityQueue()
            while not self.review_queue.empty():
                review_time, mem = self.review_queue.get()
                if mem == memory:
                    adjusted_time = review_time / emotion_factor
                    temp_queue.put((adjusted_time, mem))
                    self.logger.debug(f"Adjusted review time for memory {mem} to {adjusted_time}")
                else:
                    temp_queue.put((review_time, mem))
            self.review_queue = temp_queue
        except Exception as e:
            self.logger.error(f"Error in adjust_review_schedule_based_on_emotion: {str(e)}", exc_info=True)


class MemoryConsolidationThread(threading.Thread):
    """
    A thread to handle memory consolidation and spaced repetition asynchronously,
    ensuring emotional states influence memory processing.
    """

    def __init__(
        self,
        memory_store: Any,
        spaced_repetition: SpacedRepetition,
        provider_manager: ProviderManager,
        config_manager: ConfigManager,
        system_state: Any
    ):
        """
        Initializes the memory consolidation thread.

        Args:
            memory_store (Any): The memory store to consolidate.
            spaced_repetition (SpacedRepetition): The spaced repetition system for review.
            provider_manager (ProviderManager): The provider manager for generating responses.
            config_manager (ConfigManager): The configuration manager for retrieving settings.
            system_state (Any): The current state of the system (includes emotional state).
        """
        super().__init__()
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('MemoryConsolidationThread')
        self.memory_store = memory_store
        self.spaced_repetition = spaced_repetition
        self.provider_manager = provider_manager
        self.system_state = system_state
        self.running = True
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        consolidation_config = config_manager.get_subsystem_config('time_aware_processing') if config_manager else {}
        self.consolidation_interval = consolidation_config.get('consolidation', {}).get('consolidation_interval', 3600)  # Default to 1 hour

    def run(self):
        """
        Starts the event loop for memory consolidation and review.
        """
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            while self.running:
                self.loop.run_until_complete(self._async_consolidate_and_review())
                time.sleep(self.consolidation_interval)
        except Exception as e:
            self.logger.error(f"Error in MemoryConsolidationThread run: {str(e)}", exc_info=True)

    async def _async_consolidate_and_review(self):
        """
        Consolidates memories and reviews scheduled memories asynchronously.
        """
        try:
            await self.consolidate_memories()
            await self.review_memories()
        except Exception as e:
            self.logger.error(f"Error in _async_consolidate_and_review: {str(e)}", exc_info=True)

    async def consolidate_memories(self):
        """
        Consolidates memories from short-term to long-term memory.
        """
        try:
            self.logger.info("Starting memory consolidation.")
            # Placeholder for actual consolidation logic
            # Example: Move items from short-term to long-term memory
            # This would involve interacting with memory_store's methods
            await asyncio.sleep(0)  # Simulate async operation
            self.logger.info("Memory consolidation completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during memory consolidation: {str(e)}", exc_info=True)

    async def review_memories(self):
        """
        Reviews memories scheduled for review using spaced repetition.
        Adjusts review schedules based on emotional significance.
        """
        try:
            self.logger.info("Starting memory reviews.")
            current_time = time.time()

            while not self.spaced_repetition.review_queue.empty():
                review_time, memory = self.spaced_repetition.review_queue.get()
                if review_time > current_time:
                    self.spaced_repetition.review_queue.put((review_time, memory))
                    break

                # Determine emotion_factor based on memory's emotional weight
                emotion_factor = self._determine_emotion_factor(memory)
                quality = await self.simulate_review_quality(memory, emotion_factor)
                new_params = self.spaced_repetition.review(memory, quality)
                memory.update_review_params(new_params)

            self.logger.info("Memory reviews completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during memory reviews: {str(e)}", exc_info=True)

    def _determine_emotion_factor(self, memory: Any) -> float:
        """
        Determines the emotion factor based on the memory's emotional significance.

        Args:
            memory (Any): The memory object to evaluate.

        Returns:
            float: The emotion factor to adjust review scheduling.
        """
        try:
            emotional_weight = getattr(memory, 'emotional_weight', 0.0)  # Assume memory has an emotional_weight attribute
            # Higher emotional weight leads to faster review scheduling
            emotion_factor = 1 + (emotional_weight * 0.5)  # Adjust multiplier as needed
            self.logger.debug(f"Determined emotion_factor {emotion_factor} based on emotional_weight {emotional_weight}")
            return emotion_factor
        except Exception as e:
            self.logger.error(f"Error in _determine_emotion_factor: {str(e)}", exc_info=True)
            return 1.0

    async def simulate_review_quality(self, memory: Any, emotion_factor: float) -> int:
        """
        Simulates the quality of memory recall during review.

        Args:
            memory (Any): The memory object being reviewed.
            emotion_factor (float): The factor influencing the review timing.

        Returns:
            int: The quality rating (0-5).
        """
        try:
            question = await self.generate_question(memory.content)

            answer = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "Answer the following question based on the given context"},
                    {"role": "user", "content": f"Context: {memory.content}\nQuestion: {question}"}
                ],
                llm_call_type="memory_consolidation"
            )

            quality = await self.evaluate_answer(answer, memory.content)

            if quality >= 4:
                memory.content = await self.update_memory_content(memory.content, answer)

            return quality
        except Exception as e:
            self.logger.error(f"Error in simulate_review_quality: {str(e)}", exc_info=True)
            return 0

    async def generate_question(self, content: str) -> str:
        """
        Generates a question based on the given content to facilitate memory review.

        Args:
            content (str): The content of the memory.

        Returns:
            str: The generated question.
        """
        try:
            response = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "Generate a question based on the following information"},
                    {"role": "user", "content": content}
                ],
                llm_call_type="memory_consolidation"
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error generating question: {str(e)}", exc_info=True)
            return "Could not generate a question"

    async def evaluate_answer(self, answer: str, original_content: str) -> int:
        """
        Evaluates the quality of the provided answer against the original content.

        Args:
            answer (str): The answer generated by the AI.
            original_content (str): The original memory content.

        Returns:
            int: The quality rating (0-5).
        """
        try:
            response = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "Evaluate the following answer based on the original content. Rate from 0 to 5."},
                    {"role": "user", "content": f"Original content: {original_content}\nAnswer: {answer}"}
                ],
                llm_call_type="memory_consolidation"
            )
            quality_str = response.strip()
            quality = int(quality_str)
            quality = max(0, min(5, quality))  # Ensure quality is within 0-5
            self.logger.debug(f"Evaluated answer quality: {quality}")
            return quality
        except ValueError:
            self.logger.error(f"Received non-integer response for quality evaluation: '{response}'")
            return 0
        except Exception as e:
            self.logger.error(f"Error evaluating answer: {str(e)}", exc_info=True)
            return 0

    async def update_memory_content(self, original_content: str, new_information: str) -> str:
        """
        Updates the memory content by merging new information with the original content.

        Args:
            original_content (str): The original memory content.
            new_information (str): The new information to merge.

        Returns:
            str: The updated memory content.
        """
        try:
            response = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "Update the original content with the new information. Provide a concise, merged version."},
                    {"role": "user", "content": f"Original: {original_content}\nNew: {new_information}"}
                ],
                llm_call_type="memory_consolidation"
            )
            updated_content = response.strip()
            self.logger.debug(f"Updated memory content: {updated_content}")
            return updated_content
        except Exception as e:
            self.logger.error(f"Error updating memory content: {str(e)}", exc_info=True)
            return original_content

    def stop(self) -> None:
        """
        Stops the memory consolidation thread gracefully.
        """
        try:
            self.running = False
            self.logger.info("Stopping MemoryConsolidationThread")
        except Exception as e:
            self.logger.error(f"Error in stop method: {str(e)}", exc_info=True)


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
