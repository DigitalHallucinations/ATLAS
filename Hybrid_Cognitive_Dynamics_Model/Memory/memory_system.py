# modules/Hybrid_Cognitive_Dynamics_Model/Memory/memory_system.py

from __future__ import annotations

import re
import os
import time
import json
import aiofiles
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Set

from modules.Config.config import ConfigManager
from modules.Providers.provider_manager import ProviderManager
from modules.thread_orchestrator import cpu_bound_task, io_bound_task

from .Sensory.sensory_memory import SensoryMemory
from .Short_Term.short_term_memory import ShortTermMemory
from .working.working_memory import EnhancedWorkingMemory
from .working.working_memory import ReferenceBackTask  # Ensure ReferenceBackTask is imported if used
from .working.working_memory import run_experiment, process_results  # Import experiment functions if needed
from .working.working_memory import EvidenceAccumulator, ComplexStimulusGenerator  # Additional imports
from .working.working_memory import EpisodicBuffer  # Import EpisodicBuffer if referenced
from .Retrieval.context_aware_retrieval import ContextAwareRetrieval
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.Intermediate_Memory.intermediate_memory import IntermediateMemory
from modules.Hybrid_Cognitive_Dynamics_Model.SSM.state_space_model import StateSpaceModel
from modules.Hybrid_Cognitive_Dynamics_Model.Time_Processing.cognitive_temporal_state import (
    CognitiveTemporalStateEnum
)
from modules.Hybrid_Cognitive_Dynamics_Model.Time_Processing.time_aware_processing import (
    TimeDecay,
    SpacedRepetition,
    MemoryConsolidationThread,
    MemoryType
)
from .Long_Term.Semantic.long_term_semantic_memory import EnhancedLongTermSemanticMemory
from .Long_Term.Episodic.long_term_episodic_memory import EnhancedLongTermEpisodicMemory


class MemorySystem:
    def __init__(
        self,
        state_model: Optional[StateSpaceModel] = None,
        file_path: Optional[str] = None,
        provider_manager: Optional[ProviderManager] = None,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Lightweight initialization of the Memory System.

        Args:
            state_model (Optional[StateSpaceModel], optional): The state space model of the system. Defaults to None.
            file_path (Optional[str], optional): Path to the memory storage file. Defaults to None.
            provider_manager (Optional[ProviderManager], optional): Manager for external providers. Defaults to None.
            config_manager (Optional[ConfigManager], optional): Configuration manager. Defaults to None.
        """
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('MemorySystem') if config_manager else None
        if self.logger:
            self.logger.debug("MemorySystem instance created with lightweight __init__")

        self.file_path = file_path
        self.state_model = state_model
        self.provider_manager = provider_manager

        # Initialize Time-Aware Processing components
        if self.state_model:
            self.time_decay = TimeDecay(system_state=self.state_model, config_manager=config_manager)
            self.spaced_repetition = SpacedRepetition(memory_store=self, config_manager=config_manager)
            if self.logger:
                self.logger.debug("TimeDecay and SpacedRepetition initialized with state_model")
        else:
            if self.logger:
                self.logger.warning("state_model is not provided. TimeDecay and SpacedRepetition may not function correctly.")
            self.time_decay = None
            self.spaced_repetition = None

        # Placeholder for Memory Consolidation Thread
        self.memory_consolidation_thread = None

        # Retrieving memory configuration from the configuration manager.
        self.memory_config = config_manager.get_subsystem_config('memory') if config_manager else {}

        self.working = None
        self.sensory = None
        self.short_term = None
        self.intermediate = None
        self.long_term_episodic = None
        self.long_term_semantic = None
        self.context_retrieval = None

        # Initialize Consciousness Stream (if applicable)
        self.consciousness_stream = None  # This should be set externally if used

    async def initialize(self):
        """Initialize the memory system components."""
        try:
            if self.logger:
                self.logger.info("Initializing memory system components")

            # Set default memory path if not provided
            if not self.file_path:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.file_path = os.path.join(base_dir, 'data', 'memory_store.json')

            if self.logger:
                self.logger.info(f"Using memory file path: {self.file_path}")

            # Initialize memory components
            self.working = EnhancedWorkingMemory(
                capacity=self.memory_config.get('working', {}).get('capacity', 4),
                total_resources=self.memory_config.get('working', {}).get('total_resources', 1.0),
                pfc_frequency=self.memory_config.get('working', {}).get('pfc_frequency', 5),
                striatum_frequency=self.memory_config.get('working', {}).get('striatum_frequency', 40)
            )
            if self.logger:
                self.logger.debug("EnhancedWorkingMemory initialized")

            self.sensory = SensoryMemory(self.config_manager)
            if self.logger:
                self.logger.debug("SensoryMemory initialized")

            self.short_term = ShortTermMemory(config_manager=self.config_manager)
            if self.logger:
                self.logger.debug("ShortTermMemory initialized")

            self.intermediate = IntermediateMemory(self.config_manager)
            if self.logger:
                self.logger.debug("IntermediateMemory initialized")

            self.long_term_episodic = EnhancedLongTermEpisodicMemory(self.state_model, self.config_manager, self)
            if self.logger:
                self.logger.debug("EnhancedLongTermEpisodicMemory initialized")

            self.long_term_semantic = EnhancedLongTermSemanticMemory(self.config_manager)
            if self.logger:
                self.logger.debug("EnhancedLongTermSemanticMemory initialized")

            self.context_retrieval = ContextAwareRetrieval(self.state_model, self.config_manager)
            if self.logger:
                self.logger.debug("ContextAwareRetrieval initialized")

            # Load existing memory or create new memory store
            if self.file_path:
                await self._load_memory()

            # Preload Long-Term Semantic Memory if available
            preload_LTMS = self.config_manager.get_preload_LTMS() if self.config_manager else None
            if preload_LTMS:
                await self.long_term_semantic.preload_LTMS(preload_LTMS)
                if self.logger:
                    self.logger.debug("Preloaded Long-Term Semantic Memory")
            else:
                if self.logger:
                    self.logger.warning("No preload data available for long-term semantic memory")

            if self.state_model is None and self.logger:
                self.logger.warning("state_model is not initialized. Some features may not work properly.")

            # Initialize and start Memory Consolidation Thread
            if self.time_decay and self.spaced_repetition and self.provider_manager:
                self.memory_consolidation_thread = MemoryConsolidationThread(
                    memory_store=self,
                    spaced_repetition=self.spaced_repetition,
                    provider_manager=self.provider_manager,
                    config_manager=self.config_manager,
                    system_state=self.state_model
                )
                self.memory_consolidation_thread.start()
                if self.logger:
                    self.logger.debug("MemoryConsolidationThread started")
            else:
                if self.logger:
                    self.logger.warning("Cannot start MemoryConsolidationThread due to missing components.")

            if self.logger:
                self.logger.info("MemorySystem initialization completed successfully")
                self.logger.debug(f"Available methods after initialization: {[method for method in dir(self) if callable(getattr(self, method))]}")
                self.logger.debug(f"MemorySystem instance: {self}")
            return self
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during MemorySystem initialization: {str(e)}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.error(f"JSON decode error during MemorySystem initialization: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during MemorySystem initialization: {str(e)}", exc_info=True)
            raise

    async def close(self):
        """
        Close and clean up resources used by the MemorySystem.
        This method should be called when shutting down the system.
        """
        try:
            if self.logger:
                self.logger.info("Closing MemorySystem resources")

            # Stop Memory Consolidation Thread
            if self.memory_consolidation_thread:
                await self.memory_consolidation_thread.stop()
                self.memory_consolidation_thread.join()
                if self.logger:
                    self.logger.debug("MemoryConsolidationThread stopped")

            # Close any open database connections
            if hasattr(self, 'db_manager') and self.db_manager:
                await self.db_manager.close()
                if self.logger:
                    self.logger.debug("Database connections closed")
            
            # Clear memory structures
            self.working = None
            self.sensory = None
            self.short_term = None
            self.intermediate = None
            self.long_term_episodic = None
            self.long_term_semantic = None
            
            # Close context retrieval if it has a close method
            if self.context_retrieval and hasattr(self.context_retrieval, 'close'):
                await self.context_retrieval.close()
                if self.logger:
                    self.logger.debug("ContextAwareRetrieval closed")

            # Save any unsaved data
            if self.file_path:
                await self._save_memory()

            if self.logger:
                self.logger.info("MemorySystem resources have been closed and cleaned up.")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during MemorySystem closure: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during MemorySystem closure: {str(e)}", exc_info=True)

    @io_bound_task        
    async def _load_memory(self) -> None: 
        """Load memory from file or create a new memory store if it doesn't exist."""
        try:
            if self.logger:
                self.logger.info(f"Loading memory from {self.file_path}")
            async with aiofiles.open(self.file_path, 'r') as file:
                raw_data = await file.read()
                data = json.loads(raw_data)
                
                # Load episodic memory
                for episode in data.get('episodic', []):
                    context = np.array(episode.get('context', []))
                    await self.long_term_episodic.add(episode['content'], context)
                
                # Load semantic memory
                if self.long_term_semantic:
                    for concept, related in data.get('semantic', {}).items():
                        await self.long_term_semantic.add(concept, related)
                else:
                    if self.logger:
                        self.logger.error("long_term_semantic is not initialized")
                
            if self.logger:
                self.logger.info(f"Memory successfully loaded from {self.file_path}")
        except json.JSONDecodeError as e: 
            if self.logger:
                self.logger.error(f"Error decoding memory file: {str(e)}", exc_info=True)
            raise
        except (IOError, OSError) as e: 
            if self.logger:
                self.logger.error(f"File system error loading or creating memory file: {str(e)}", exc_info=True)
            raise
        except Exception as e: 
            if self.logger:
                self.logger.error(f"Unexpected error loading or creating memory file: {str(e)}", exc_info=True)
                self.logger.error(traceback.format_exc())
            raise

    def set_consciousness_stream(self, consciousness_stream):
        """
        Set the consciousness stream for the memory system.

        Args:
            consciousness_stream: The consciousness stream object to be set.
        """
        self.consciousness_stream = consciousness_stream
        if self.logger:
            self.logger.info("Consciousness stream set for MemorySystem")

    @io_bound_task
    async def process_input(self, input_data: Any) -> Any:
        """
        Processes input data and updates various memory components.

        Args:
            input_data (Any): The input data to process.

        Returns:
            Any: The processed information.
        """
        # Delegate specific tasks to helper methods to avoid over-nesting
        if self.logger:
            self.logger.debug(f"Memory system process_input started with input: {input_data}")

        wrapped_input = await self._try_wrap_input_data(input_data)
        if wrapped_input is None:
            return None

        input_signal = await self._try_prepare_input_signal(wrapped_input.get('content', ''))
        if input_signal is None:
            return None

        input_signal = await self._try_resize_input_signal(input_signal)
        if input_signal is None:
            return None

        attended_input = await self._try_apply_attention_focus(input_signal)
        if attended_input is None:
            return None

        await self._try_simulate_p3b(attended_input)

        await self._try_handle_gate_operations(attended_input, wrapped_input)

        await self._try_add_to_sensory_memory(wrapped_input)

        salient_info = await self._try_get_salient_info()
        if salient_info is None:
            return None

        processed_info = await self._try_process_in_working_memory(salient_info)
        if processed_info is None:
            return None

        processed_info = self._ensure_dict(processed_info)

        await self._try_add_to_short_term_memory(processed_info)
        await self._try_add_to_intermediate_memory(processed_info)

        concepts = self._extract_concepts(processed_info)
        await self._try_process_related_concepts(concepts)

        context_vector = await self._try_retrieve_context_vector()
        if context_vector is None:
            return None

        await self._try_add_to_episodic_memory(processed_info, context_vector)

        await self._try_update_state_model(processed_info)

        if self.logger:
            self.logger.debug(f"Successfully processed input: {input_data}")

        return processed_info

    async def _try_wrap_input_data(self, input_data: Any) -> Optional[Dict[str, Any]]:
        """
        Attempts to wrap the input data into a dictionary.

        Args:
            input_data (Any): The input data to wrap.

        Returns:
            Optional[Dict[str, Any]]: The wrapped input data if successful, else None.
        """
        try:
            return self._wrap_input_data(input_data)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error wrapping input data: {str(e)}", exc_info=True)
            return None

    async def _try_prepare_input_signal(self, content: Any) -> Optional[np.ndarray]:
        """
        Attempts to prepare the input signal from the content.

        Args:
            content (Any): The content to prepare.

        Returns:
            Optional[np.ndarray]: The prepared input signal if successful, else None.
        """
        try:
            return self._prepare_input_signal(content)
        except (TypeError, ValueError) as e:
            if self.logger:
                self.logger.error(f"Error preparing input signal: {str(e)}", exc_info=True)
            return None

    async def _try_resize_input_signal(self, input_signal: np.ndarray) -> Optional[np.ndarray]:
        """
        Attempts to resize the input signal to match working memory capacity.

        Args:
            input_signal (np.ndarray): The input signal to resize.

        Returns:
            Optional[np.ndarray]: The resized input signal if successful, else None.
        """
        try:
            return self._resize_input_signal(input_signal)
        except (ValueError, TypeError) as e:
            if self.logger:
                self.logger.error(f"Error resizing input signal: {str(e)}", exc_info=True)
            return None

    async def _try_apply_attention_focus(self, input_signal: np.ndarray) -> Optional[np.ndarray]:
        """
        Attempts to apply attention focus from the state model to the input signal.

        Args:
            input_signal (np.ndarray): The input signal to modify.

        Returns:
            Optional[np.ndarray]: The attended input signal if successful, else None.
        """
        try:
            return self._apply_attention_focus(input_signal)
        except (ValueError, TypeError) as e:
            if self.logger:
                self.logger.error(f"Error applying attention focus: {str(e)}", exc_info=True)
            return None

    async def _try_simulate_p3b(self, attended_input: np.ndarray) -> None:
        """
        Attempts to simulate the P3b cognitive process.

        Args:
            attended_input (np.ndarray): The attended input signal.
        """
        try:
            await self._simulate_p3b(attended_input)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error simulating P3b: {str(e)}", exc_info=True)

    async def _try_handle_gate_operations(self, attended_input: np.ndarray, input_data: Dict[str, Any]) -> None:
        """
        Attempts to handle gate operations for working memory.

        Args:
            attended_input (np.ndarray): The attended input signal.
            input_data (Dict[str, Any]): The input data.
        """
        try:
            await self._handle_gate_operations(attended_input, input_data)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error handling gate operations: {str(e)}", exc_info=True)

    async def _try_add_to_sensory_memory(self, input_data: Dict[str, Any]) -> None:
        """
        Attempts to add input data to sensory memory.

        Args:
            input_data (Dict[str, Any]): The input data to add.
        """
        try:
            await self._add_to_sensory_memory(input_data)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding to sensory memory: {str(e)}", exc_info=True)

    async def _try_get_salient_info(self) -> Optional[Any]:
        """
        Attempts to retrieve salient information from sensory memory.

        Returns:
            Optional[Any]: The salient information if available, else None.
        """
        try:
            return await self._get_salient_info()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting salient info: {str(e)}", exc_info=True)
            return None

    async def _try_process_in_working_memory(self, salient_info: Any) -> Optional[Any]:
        """
        Attempts to process salient information in working memory.

        Args:
            salient_info (Any): The salient information to process.

        Returns:
            Optional[Any]: The processed information if successful, else None.
        """
        try:
            return await self._process_in_working_memory(salient_info)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing in working memory: {str(e)}", exc_info=True)
            return None

    async def _try_add_to_short_term_memory(self, processed_info: Dict[str, Any]) -> None:
        """
        Attempts to add processed information to short-term memory.

        Args:
            processed_info (Dict[str, Any]): The processed information to add.
        """
        try:
            await self._add_to_short_term_memory(processed_info)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding to short-term memory: {str(e)}", exc_info=True)

    async def _try_add_to_intermediate_memory(self, processed_info: Dict[str, Any]) -> None:
        """
        Attempts to add processed information to intermediate memory.

        Args:
            processed_info (Dict[str, Any]): The processed information to add.
        """
        try:
            await self._add_to_intermediate_memory(processed_info)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding to intermediate memory: {str(e)}", exc_info=True)

    async def _try_process_related_concepts(self, concepts: Set[str]) -> None:
        """
        Attempts to process related concepts by querying and updating semantic memory.

        Args:
            concepts (Set[str]): The set of concepts to process.
        """
        try:
            await self._process_related_concepts(concepts)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing related concepts: {str(e)}", exc_info=True)

    async def _try_retrieve_context_vector(self) -> Optional[np.ndarray]:
        """
        Attempts to retrieve the current context vector from the state model.

        Returns:
            Optional[np.ndarray]: The context vector if available, else None.
        """
        try:
            return await self._retrieve_context_vector()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving context vector: {str(e)}", exc_info=True)
            return None

    async def _try_add_to_episodic_memory(self, processed_info: Dict[str, Any], context_vector: Optional[np.ndarray]) -> None:
        """
        Attempts to add processed information to episodic memory.

        Args:
            processed_info (Dict[str, Any]): The processed information to add.
            context_vector (Optional[np.ndarray]): The context vector.
        """
        try:
            await self._add_to_episodic_memory(processed_info, context_vector)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error adding to episodic memory: {str(e)}", exc_info=True)

    async def _try_update_state_model(self, state_update: Dict[str, Any]) -> None:
        """
        Attempts to update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        try:
            await self._update_state_model(state_update)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating state model: {str(e)}", exc_info=True)

    def _wrap_input_data(self, input_data: Any) -> Dict[str, Any]:
        """
        Wraps the input data into a dictionary if it's a string.

        Args:
            input_data (Any): The input data to wrap.

        Returns:
            Dict[str, Any]: The wrapped input data.
        """
        if isinstance(input_data, str):
            if self.logger:
                self.logger.debug("Wrapped string input into dictionary")
            return {'content': input_data}
        return input_data

    def _prepare_input_signal(self, content: Any) -> np.ndarray:
        """
        Prepares the input data for processing by converting it to a numpy array.

        Args:
            content (Any): The content to prepare.

        Returns:
            np.ndarray: The prepared input signal.
        """
        if self.logger:
            self.logger.debug(f"Preparing input signal for: {content}")

        if isinstance(content, np.ndarray):
            return content
        elif isinstance(content, str):
            return np.array([ord(c) for c in content])
        elif isinstance(content, dict):
            return self._dict_to_array(content)
        elif isinstance(content, list):
            return np.array(content, dtype=float)
        else:
            return np.array([hash(str(content)) % 1000], dtype=float)

    def _dict_to_array(self, input_dict: dict) -> np.ndarray:
        """
        Converts a dictionary to a numpy array by processing its values.

        Args:
            input_dict (dict): The input dictionary.

        Returns:
            np.ndarray: A numpy array representation of the dictionary.
        """
        values = []
        for value in input_dict.values():
            if isinstance(value, (int, float)):
                values.append(value)
            elif isinstance(value, str):
                values.append(np.mean([ord(c) for c in value]))
            elif isinstance(value, list):
                # Flatten nested lists
                for item in value:
                    if isinstance(item, dict):
                        values.extend(self._dict_to_array(item))
                    else:
                        values.append(hash(str(item)) % 1000)
            elif isinstance(value, dict):
                values.extend(self._dict_to_array(value))
            else:
                values.append(hash(str(value)) % 1000)
        return np.array(values, dtype=float)

    def _resize_input_signal(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Resizes the input signal to match working memory capacity.

        Args:
            input_signal (np.ndarray): The input signal to resize.

        Returns:
            np.ndarray: The resized input signal.
        """
        try:
            input_signal = np.resize(input_signal, (self.working.capacity,))
            if self.logger:
                self.logger.debug(f"Resized input signal: {input_signal}")
            return input_signal
        except ValueError as e:
            if self.logger:
                self.logger.error(f"Value error during input signal resizing: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during input signal resizing: {str(e)}", exc_info=True)
            raise

    def _apply_attention_focus(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Applies attention focus from the state model to the input signal.

        Args:
            input_signal (np.ndarray): The input signal to modify.

        Returns:
            np.ndarray: The attended input signal.
        """
        try:
            if self.state_model:
                attention_focus = self.state_model.attention_focus
                if self.logger:
                    self.logger.debug(f"Current attention focus: {attention_focus}")

                attention_focus = np.resize(attention_focus, input_signal.shape)
                attended_input = input_signal * attention_focus
                if self.logger:
                    self.logger.debug(f"Attended input signal: {attended_input}")
                return attended_input
            else:
                if self.logger:
                    self.logger.warning("state_model is None. Using original input signal without attention focus.")
                return input_signal
        except ValueError as e:
            if self.logger:
                self.logger.error(f"Value error during attention focus application: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during attention focus application: {str(e)}", exc_info=True)
            raise

    async def _simulate_p3b(self, attended_input: np.ndarray) -> None:
        """
        Simulates the P3b cognitive process.

        Args:
            attended_input (np.ndarray): The attended input signal.
        """
        try:
            await self.working.simulate_p3b(attended_input)
            if self.logger:
                self.logger.debug("P3b simulation completed")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error in P3b simulation: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error in P3b simulation: {str(e)}", exc_info=True)

    async def _handle_gate_operations(self, attended_input: np.ndarray, input_data: Dict[str, Any]) -> None:
        """
        Handles gate operations for working memory.

        Args:
            attended_input (np.ndarray): The attended input signal.
            input_data (Dict[str, Any]): The input data.
        """
        try:
            if self.working.open_gate(attended_input):
                if self.logger:
                    self.logger.info("Gate opened")
                importance = self._calculate_importance(input_data)
                await self.working.update(attended_input, importance)  
                self.working.close_gate()
                if self.logger:
                    self.logger.debug("Gate closed after updating working memory")
            else:
                if self.logger:
                    self.logger.info("Gate remained closed")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during gate operations: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during gate operations: {str(e)}", exc_info=True)

    async def _add_to_sensory_memory(self, input_data: Dict[str, Any]) -> None:
        """
        Adds input data to sensory memory.

        Args:
            input_data (Dict[str, Any]): The input data to add.
        """
        try:
            await self.sensory.add(input_data)
            if self.logger:
                self.logger.debug("Added to sensory memory")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error adding to sensory memory: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error adding to sensory memory: {str(e)}", exc_info=True)

    async def _get_salient_info(self) -> Optional[Any]:
        """
        Retrieves salient information from sensory memory.

        Returns:
            Optional[Any]: The salient information if available, else None.
        """
        try:
            salient_info = await self.sensory.get_salient_info()  
            if self.logger:
                self.logger.debug(f"Salient info retrieved: {salient_info}")
            return salient_info
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error getting salient info: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error getting salient info: {str(e)}", exc_info=True)
            return None

    async def _process_in_working_memory(self, salient_info: Any) -> Optional[Any]:
        """
        Processes salient information in working memory.

        Args:
            salient_info (Any): The salient information to process.

        Returns:
            Optional[Any]: The processed information if successful, else None.
        """
        try:
            processed_info = await self.working.process(salient_info)  
            if self.logger:
                self.logger.debug(f"Processed info: {processed_info}")
            return processed_info
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error processing in working memory: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error processing in working memory: {str(e)}", exc_info=True)
            return None

    def _ensure_dict(self, processed_info: Any) -> Dict[str, Any]:
        """
        Ensures that the processed information is a dictionary.

        Args:
            processed_info (Any): The processed information.

        Returns:
            Dict[str, Any]: The processed information as a dictionary.
        """
        if not isinstance(processed_info, dict):
            if self.logger:
                self.logger.debug("Processed info is not a dict. Wrapping it.")
            return {'processed_info': processed_info}
        return processed_info

    async def _try_add_to_short_term_memory(self, processed_info: Dict[str, Any]) -> None:
        """
        Attempts to add processed information to short-term memory.

        Args:
            processed_info (Dict[str, Any]): The processed information to add.
        """
        try:
            self.short_term.add(processed_info)
            if self.logger:
                self.logger.debug("Added to short-term memory")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error adding to short-term memory: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error adding to short-term memory: {str(e)}", exc_info=True)

    async def _try_add_to_intermediate_memory(self, processed_info: Dict[str, Any]) -> None:
        """
        Attempts to add processed information to intermediate memory.

        Args:
            processed_info (Dict[str, Any]): The processed information to add.
        """
        try:
            self.intermediate.add(processed_info)
            if self.logger:
                self.logger.debug("Added to intermediate memory")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error adding to intermediate memory: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error adding to intermediate memory: {str(e)}", exc_info=True)

    async def _try_process_related_concepts(self, concepts: Set[str]) -> None:
        """
        Attempts to process related concepts by querying and updating semantic memory.

        Args:
            concepts (Set[str]): The set of concepts to process.
        """
        for concept in concepts:
            try:
                related_concepts = await self.long_term_semantic.query(concept, n=5)
                related_concept_names = [rc[0] for rc in related_concepts]
                await self.long_term_semantic.add(concept, related_concept_names)
                if self.logger:
                    self.logger.debug(f"Added concept to semantic memory: {concept}")
            except (IOError, OSError) as e:
                if self.logger:
                    self.logger.error(f"File system error updating semantic memory for concept {concept}: {str(e)}", exc_info=True)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Unexpected error updating semantic memory for concept {concept}: {str(e)}", exc_info=True)

    async def _try_retrieve_context_vector(self) -> Optional[np.ndarray]:
        """
        Attempts to retrieve the current context vector from the state model.

        Returns:
            Optional[np.ndarray]: The context vector if available, else None.
        """
        try:
            return await self._retrieve_context_vector()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving context vector: {str(e)}", exc_info=True)
            return None

    async def _try_add_to_episodic_memory(self, processed_info: Dict[str, Any], context_vector: Optional[np.ndarray]) -> None:
        """
        Attempts to add processed information to episodic memory.

        Args:
            processed_info (Dict[str, Any]): The processed information to add.
            context_vector (Optional[np.ndarray]): The context vector.
        """
        if context_vector is not None:
            try:
                await self.long_term_episodic.add(processed_info, context_vector)
                if self.logger:
                    self.logger.debug(f"Added to episodic memory: {processed_info}")
            except (IOError, OSError) as e:
                if self.logger:
                    self.logger.error(f"File system error adding to episodic memory: {str(e)}", exc_info=True)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Unexpected error adding to episodic memory: {str(e)}", exc_info=True)

    async def _try_update_state_model(self, state_update: Dict[str, Any]) -> None:
        """
        Attempts to update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        try:
            await self.update_state_model(state_update)
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating state model: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating state model: {str(e)}", exc_info=True)

    def _calculate_importance(self, input_data: Any) -> float: 
        """
        Calculate the importance of the input data.

        Args:
            input_data (Any): The input data.

        Returns:
            float: The calculated importance score.
        """
        # Simple importance calculation based on input length and uniqueness
        if isinstance(input_data, str):
            words = input_data.split()
            if not words:
                return 0.0
            unique_ratio = len(set(words)) / len(words)  # Calculate uniqueness ratio
            importance = min(1.0, (len(input_data) / 1000) * unique_ratio)  # Importance based on length and uniqueness
            if self.logger:
                self.logger.debug(f"Calculated importance for string input: {importance}")
            return importance
        else:
            if self.logger:
                self.logger.debug("Calculated default importance for non-string input: 0.5")
            return 0.5  # Default importance for non-string inputs

    def _extract_concepts(self, memory: Any) -> Set[str]:
        """
        Extract important concepts from the memory content.

        Args:
            memory (Any): The memory content.

        Returns:
            Set[str]: A set of extracted concepts.
        """
        concepts = set()
        try:
            if isinstance(memory, str):
                # Remove punctuation and convert to lowercase
                text = re.sub(r'[^\w\s]', '', memory.lower())
                
                # Split into words and filter out short words
                words = text.split()
                concepts.update(word for word in words if len(word) > 3)
                
                # Add bigrams (pairs of adjacent words) as concepts
                bigrams = set(' '.join(words[i:i+2]) for i in range(len(words)-1))
                concepts.update(bigrams)
                
                # Use TF-IDF to extract important terms as concepts
                if hasattr(self.long_term_semantic, 'vectorizer'):
                    tfidf_matrix = self.long_term_semantic.vectorizer.fit_transform([memory])
                    feature_names = self.long_term_semantic.vectorizer.get_feature_names_out()
                    for _, idx in zip(*tfidf_matrix.nonzero()):
                        concepts.add(feature_names[idx])
            elif isinstance(memory, dict):
                # If memory is a dict, extract concepts from all string values
                string_values = [v for v in memory.values() if isinstance(v, str)]
                for v in string_values:
                    concepts.update(self._extract_concepts(v))
            # Additional types can be handled here as needed
        except (ValueError, TypeError) as e:
            if self.logger:
                self.logger.error(f"Error extracting concepts: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error extracting concepts: {str(e)}", exc_info=True)
        finally:
            if self.logger:
                self.logger.debug(f"Extracted concepts: {concepts}")
            return concepts

    async def update_state_model(self, state_update: Dict[str, Any]):
        """
        Update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        if self.state_model is None:
            if self.logger:
                self.logger.error("Cannot update state model: state_model is None")
            return
        try:
            await self.state_model.update(state_update)
            if self.logger:
                self.logger.debug("HCDM state model updated")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating state model: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating state model: {str(e)}", exc_info=True)

    async def consolidate_memory_with_consciousness(self) -> None:
        """
        Consolidate memory with consciousness stream integration.
        """
        try:
            await self.update_memory_from_consciousness()
            await self.consolidate_memory()
            
            if self.consciousness_stream and self.state_model:
                consciousness_state = await self.consciousness_stream.get_state()
                state_update = await self.state_model.derive_state_update(consciousness_state)
                await self.update_state_model(state_update)
                
                completed_patterns = await self.long_term_semantic.pattern_completion(str(consciousness_state))
                if completed_patterns:
                    if self.logger:
                        self.logger.debug(f"Completed patterns from consciousness state: {completed_patterns}")
                    for pattern, similarity in completed_patterns:
                        thought = {
                            "type": "CONSOLIDATION_INSIGHT",
                            "content": f"Insight from pattern completion: {pattern} (similarity: {similarity})",
                            "timestamp": time.time()
                        }
                        await self.consciousness_stream.add_thought(thought)
                        if self.logger:
                            self.logger.debug(f"Thought added to consciousness stream from consolidation insight: {thought}")
                
                if self.logger:
                    self.logger.debug("Memory consolidated with consciousness stream integration")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during consolidate_memory_with_consciousness: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during consolidate_memory_with_consciousness: {str(e)}", exc_info=True)

    async def update_memory_from_consciousness(self) -> None:
        """
        Update memory based on the current thought from the consciousness stream.
        """
        try:
            current_thought = await self.retrieve_from_consciousness()
            if current_thought:
                await self.process_input(current_thought['content'])  
                
                # Perform pattern separation on current thought
                concepts = self._extract_concepts(current_thought['content'])
                for concept in concepts:
                    for other_concept in self.long_term_semantic.memory_vectors.keys():
                        separation = self.long_term_semantic.pattern_separation(concept, other_concept)
                        if separation is not None and separation < 0.1:  # Threshold for considering concepts as distinct
                            if self.logger:
                                self.logger.debug(f"Low pattern separation between {concept} and {other_concept}: {separation}")
                
                if self.logger:
                    self.logger.debug("Memory updated from consciousness stream")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating memory from consciousness stream: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating memory from consciousness stream: {str(e)}", exc_info=True)

    async def retrieve_from_consciousness(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the current thought from the consciousness stream.

        Returns:
            Optional[Dict[str, Any]]: The current thought if available, else None.
        """
        try:
            if self.consciousness_stream:
                current_thought = await self.consciousness_stream.get_current_thought()
                if self.logger:
                    self.logger.debug(f"Retrieved thought from consciousness stream: {current_thought}")
                return current_thought
            else:
                if self.logger:
                    self.logger.warning("Consciousness stream not set")
                return None
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None

    async def _add_to_episodic_memory(self, processed_info: Dict[str, Any], context_vector: Optional[np.ndarray]) -> None:
        """
        Adds processed information to episodic memory.

        Args:
            processed_info (Dict[str, Any]): The processed information to add.
            context_vector (Optional[np.ndarray]): The context vector.
        """
        if context_vector is not None:
            try:
                await self.long_term_episodic.add(processed_info, context_vector)
                if self.logger:
                    self.logger.debug(f"Added to episodic memory: {processed_info}")
            except (IOError, OSError) as e:
                if self.logger:
                    self.logger.error(f"File system error adding to episodic memory: {str(e)}", exc_info=True)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Unexpected error adding to episodic memory: {str(e)}", exc_info=True)

    async def _try_update_state_model(self, state_update: Dict[str, Any]) -> None:
        """
        Attempts to update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        try:
            await self.update_state_model(state_update)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating state model: {str(e)}", exc_info=True)

    async def get_current_state_context(self) -> np.ndarray:
        """
        Retrieve the current state context vector from context-aware retrieval.

        Returns:
            np.ndarray: The current context vector.
        """
        try:
            context = await self.context_retrieval.get_context_vector()
            if self.logger:
                self.logger.debug(f"Current state context vector retrieved: {context}")
            return context
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])

    @io_bound_task
    async def consolidate_memory(self) -> None:
        """
        Consolidate memories from short-term and intermediate memory into long-term episodic and semantic memory.
        """
        try:
            current_context = await self.get_current_state_context()
            
            memories_to_consolidate = (
                self.short_term.get_memories_for_consolidation() + 
                self.intermediate.get_memories_for_consolidation()
            )
            sorted_memories = sorted(
                memories_to_consolidate, 
                key=self.spaced_repetition.consolidation_priority,
                reverse=True
            )
            
            for memory in sorted_memories:
                content = memory.content if hasattr(memory, 'content') else memory
                consolidated_content = f"Consolidated: {content}"
                await self.long_term_episodic.add(consolidated_content, current_context)
                
                concepts = self._extract_concepts(content)
                for concept in concepts:
                    try:
                        related_concepts = await self.long_term_semantic.query(concept, n=5)
                        related_concept_names = [rc[0] for rc in related_concepts]
                        await self.long_term_semantic.add(concept, related_concept_names)
                        if self.logger:
                            self.logger.debug(f"Added related concepts for {concept} to semantic memory")
                    except (IOError, OSError) as e:
                        if self.logger:
                            self.logger.error(f"File system error updating semantic memory for concept {concept}: {str(e)}", exc_info=True)
                    except Exception as semantic_error:
                        if self.logger:
                            self.logger.error(f"Unexpected error updating semantic memory for concept {concept}: {str(semantic_error)}", exc_info=True)
            
            self.short_term.clear_consolidated()
            self.intermediate.memories = [
                m for m in self.intermediate.memories if m not in sorted_memories
            ]
            await self._save_memory()
            if self.logger:
                self.logger.info("Memory consolidation completed")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during memory consolidation: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during memory consolidation: {str(e)}", exc_info=True)

    def _extract_concepts(self, memory: Any) -> Set[str]:
        # (This method is already defined above; ensure no duplication)
        pass  # Placeholder since the method is defined earlier

    async def get_relevant_context(self, query: str) -> str:
        """
        Retrieve the relevant context for a given query.

        Args:
            query (str): The query string.

        Returns:
            str: The relevant context as a string.
        """
        try:
            # Added pattern completion on the query
            semantic_results = await self.long_term_semantic.query(query)
            recent_episodes = self.long_term_episodic.get_recent_episodes()
            
            context = f"Semantic knowledge: {', '.join(result[0] for result in semantic_results)}\n"
            context += f"Recent experiences: {', '.join(str(episode['content']) for episode in recent_episodes)}"
            
            # Perform pattern completion on the query
            completed_patterns = await self.long_term_semantic.pattern_completion(query)
            if completed_patterns:
                context += f"\nRelated concepts: {', '.join(pattern for pattern, _ in completed_patterns)}"
            
            if self.logger:
                self.logger.debug(f"Retrieved relevant context for query: {query}")
            return context
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving relevant context for query '{query}': {str(e)}", exc_info=True)
            return ""
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving relevant context for query '{query}': {str(e)}", exc_info=True)
            return ""

    @io_bound_task
    async def _save_memory(self) -> None:
        """
        Save memory to file.

        This includes episodic and semantic memories.
        """
        try:
            data = {
                'episodic': self.long_term_episodic.episodes if self.long_term_episodic else [],
                'semantic': {
                    node: list(self.long_term_semantic.knowledge_graph.neighbors(node))
                    for node in self.long_term_semantic.knowledge_graph.nodes()
                } if self.long_term_semantic else {}
            }
            async with aiofiles.open(self.file_path, 'w') as file:
                await file.write(json.dumps(data, indent=2))
            if self.logger:
                self.logger.info(f"Memory saved to {self.file_path}")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error saving memory: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error saving memory: {str(e)}", exc_info=True)

    def get_memory_stats(self) -> Dict[str, int]:
        """
        Retrieve memory statistics.

        Returns:
            Dict[str, int]: A dictionary containing various memory statistics.
        """
        try:
            stats = {
                "sensory_size": len(self.sensory.buffer) if self.sensory else 0,
                "working_memory_size": len(self.working.contents) if self.working else 0, 
                "short_term_memory_size": len(self.short_term.items) if self.short_term else 0,
                "intermediate_memory_size": len(self.intermediate.memories) if self.intermediate else 0,
                "long_term_episodic_size": len(self.long_term_episodic.episodes) if self.long_term_episodic else 0,
                "long_term_semantic_size": len(self.long_term_semantic.knowledge_graph.nodes()) if self.long_term_semantic else 0,
                "long_term_semantic_vectors": len(self.long_term_semantic.memory_vectors) if self.long_term_semantic else 0,
                "working_memory_capacity": self.working.capacity if self.working else 0,
                "short_term_memory_capacity": self.short_term.capacity if self.short_term else 0,
                "intermediate_memory_capacity": self.intermediate.capacity if self.intermediate else 0
            }
            if self.logger:
                self.logger.debug(f"Memory statistics retrieved: {stats}")
            return stats
        except AttributeError as e:
            if self.logger:
                self.logger.error(f"Attribute error retrieving memory statistics: {str(e)}", exc_info=True)
            return {}
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving memory statistics: {str(e)}", exc_info=True)
            return {}

    @cpu_bound_task
    async def cleanup_memory(self, threshold: Optional[float] = None) -> None:
        """
        Clean up memories based on a decay threshold.

        Args:
            threshold (Optional[float], optional): The decay threshold. Defaults to None.
        """
        try:
            # Retrieve threshold from config_manager if not provided
            if threshold is None:
                threshold = self.memory_config.get('cleanup_threshold', 0.1)

            # Use TimeDecay to determine which episodic memories to keep
            if self.long_term_episodic and self.time_decay:
                current_time = time.time()
                original_count = len(self.long_term_episodic.episodes)
                self.long_term_episodic.episodes = [
                    ep for ep in self.long_term_episodic.episodes 
                    if self.time_decay.decay(
                        memory_type=MemoryType.LONG_TERM_EPISODIC, 
                        time_elapsed=current_time - datetime.fromisoformat(ep['timestamp']).timestamp(), 
                        importance=ep.get('importance', 1.0)
                    ) > threshold
                ]
                cleaned_count = original_count - len(self.long_term_episodic.episodes)
                if self.logger:
                    self.logger.info(f"Episodic memory cleaned: {cleaned_count} entries removed")
            else:
                if self.logger:
                    self.logger.warning("TimeDecay or long_term_episodic is not properly initialized.")

            # Clean up intermediate memory
            if self.intermediate:
                original_intermediate = len(self.intermediate.memories)
                self.intermediate.memories = [
                    m for m in self.intermediate.memories if m.get_strength() > threshold
                ]
                cleaned_intermediate = original_intermediate - len(self.intermediate.memories)
                if self.logger:
                    self.logger.info(f"Intermediate memory cleaned: {cleaned_intermediate} entries removed")
            else:
                if self.logger:
                    self.logger.warning("Intermediate memory is not initialized.")

            # Clean up long-term semantic memory
            if self.long_term_semantic:
                nodes_to_remove = []
                for node in self.long_term_semantic.knowledge_graph.nodes():
                    if len(list(self.long_term_semantic.knowledge_graph.neighbors(node))) < 2:
                        nodes_to_remove.append(node)
                
                for node in nodes_to_remove:
                    self.long_term_semantic.knowledge_graph.remove_node(node)
                    self.long_term_semantic.memory_vectors.pop(node, None)
                    if self.logger:
                        self.logger.debug(f"Removed semantic memory node: {node}")
                
                if self.logger:
                    self.logger.info(f"Long-term semantic memory cleaned: {len(nodes_to_remove)} nodes removed")
            else:
                if self.logger:
                    self.logger.warning("long_term_semantic is not properly initialized.")

            if self.logger:
                self.logger.info(f"Memory cleanup completed. Threshold: {threshold}")
        except AttributeError as e:
            if self.logger:
                self.logger.error(f"Attribute error during memory cleanup: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during memory cleanup: {str(e)}", exc_info=True)

    @io_bound_task
    async def update_state_model(self, state_update: Dict[str, Any]):
        """
        Update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        if self.state_model is None:
            if self.logger:
                self.logger.error("Cannot update state model: state_model is None")
            return
        try:
            await self.state_model.update(state_update)
            if self.logger:
                self.logger.debug("HCDM state model updated")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating state model: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating state model: {str(e)}", exc_info=True)

    @io_bound_task
    async def update_memory_from_consciousness(self) -> None:
        """
        Update memory based on the current thought from the consciousness stream.
        """
        try:
            current_thought = await self.retrieve_from_consciousness()
            if current_thought:
                await self.process_input(current_thought['content'])  
                
                # Perform pattern separation on current thought
                concepts = self._extract_concepts(current_thought['content'])
                for concept in concepts:
                    for other_concept in self.long_term_semantic.memory_vectors.keys():
                        separation = self.long_term_semantic.pattern_separation(concept, other_concept)
                        if separation is not None and separation < 0.1:  # Threshold for considering concepts as distinct
                            if self.logger:
                                self.logger.debug(f"Low pattern separation between {concept} and {other_concept}: {separation}")
                
                if self.logger:
                    self.logger.debug("Memory updated from consciousness stream")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating memory from consciousness stream: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating memory from consciousness stream: {str(e)}", exc_info=True)

    @cpu_bound_task
    async def consolidate_memory_with_consciousness(self) -> None:
        """
        Consolidate memory with consciousness stream integration.
        """
        try:
            await self.update_memory_from_consciousness()
            await self.consolidate_memory()
            
            if self.consciousness_stream and self.state_model:
                consciousness_state = await self.consciousness_stream.get_state()
                state_update = await self.state_model.derive_state_update(consciousness_state)
                await self.update_state_model(state_update)
                
                completed_patterns = await self.long_term_semantic.pattern_completion(str(consciousness_state))
                if completed_patterns:
                    if self.logger:
                        self.logger.debug(f"Completed patterns from consciousness state: {completed_patterns}")
                    for pattern, similarity in completed_patterns:
                        thought = {
                            "type": "CONSOLIDATION_INSIGHT",
                            "content": f"Insight from pattern completion: {pattern} (similarity: {similarity})",
                            "timestamp": time.time()
                        }
                        await self.consciousness_stream.add_thought(thought)
                        if self.logger:
                            self.logger.debug(f"Thought added to consciousness stream from consolidation insight: {thought}")
                
                if self.logger:
                    self.logger.debug("Memory consolidated with consciousness stream integration")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during consolidate_memory_with_consciousness: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during consolidate_memory_with_consciousness: {str(e)}", exc_info=True)

    @cpu_bound_task
    async def consolidate_memory(self) -> None:
        """
        Consolidate memories from short-term and intermediate memory into long-term episodic and semantic memory.
        """
        try:
            current_context = await self.get_current_state_context()
            
            memories_to_consolidate = (
                self.short_term.get_memories_for_consolidation() + 
                self.intermediate.get_memories_for_consolidation()
            )
            sorted_memories = sorted(
                memories_to_consolidate, 
                key=self.spaced_repetition.consolidation_priority,
                reverse=True
            )
            
            for memory in sorted_memories:
                content = memory.content if hasattr(memory, 'content') else memory
                consolidated_content = f"Consolidated: {content}"
                await self.long_term_episodic.add(consolidated_content, current_context)
                
                concepts = self._extract_concepts(content)
                for concept in concepts:
                    try:
                        related_concepts = await self.long_term_semantic.query(concept, n=5)
                        related_concept_names = [rc[0] for rc in related_concepts]
                        await self.long_term_semantic.add(concept, related_concept_names)
                        if self.logger:
                            self.logger.debug(f"Added related concepts for {concept} to semantic memory")
                    except (IOError, OSError) as e:
                        if self.logger:
                            self.logger.error(f"File system error updating semantic memory for concept {concept}: {str(e)}", exc_info=True)
                    except Exception as semantic_error:
                        if self.logger:
                            self.logger.error(f"Unexpected error updating semantic memory for concept {concept}: {str(semantic_error)}", exc_info=True)
            
            self.short_term.clear_consolidated()
            self.intermediate.memories = [
                m for m in self.intermediate.memories if m not in sorted_memories
            ]
            await self._save_memory()
            if self.logger:
                self.logger.info("Memory consolidation completed")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during memory consolidation: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during memory consolidation: {str(e)}", exc_info=True)

    async def update_state_model(self, state_update: Dict[str, Any]):
        """
        Update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        if self.state_model is None:
            if self.logger:
                self.logger.error("Cannot update state model: state_model is None")
            return
        try:
            await self.state_model.update(state_update)
            if self.logger:
                self.logger.debug("HCDM state model updated")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating state model: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating state model: {str(e)}", exc_info=True)

    async def update_memory_from_consciousness(self) -> None:
        """
        Update memory based on the current thought from the consciousness stream.
        """
        try:
            current_thought = await self.retrieve_from_consciousness()
            if current_thought:
                await self.process_input(current_thought['content'])  
                
                # Perform pattern separation on current thought
                concepts = self._extract_concepts(current_thought['content'])
                for concept in concepts:
                    for other_concept in self.long_term_semantic.memory_vectors.keys():
                        separation = self.long_term_semantic.pattern_separation(concept, other_concept)
                        if separation is not None and separation < 0.1:  # Threshold for considering concepts as distinct
                            if self.logger:
                                self.logger.debug(f"Low pattern separation between {concept} and {other_concept}: {separation}")
                
                if self.logger:
                    self.logger.debug("Memory updated from consciousness stream")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating memory from consciousness stream: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating memory from consciousness stream: {str(e)}", exc_info=True)

    async def retrieve_from_consciousness(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the current thought from the consciousness stream.

        Returns:
            Optional[Dict[str, Any]]: The current thought if available, else None.
        """
        try:
            if self.consciousness_stream:
                current_thought = await self.consciousness_stream.get_current_thought()
                if self.logger:
                    self.logger.debug(f"Retrieved thought from consciousness stream: {current_thought}")
                return current_thought
            else:
                if self.logger:
                    self.logger.warning("Consciousness stream not set")
                return None
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None

    async def get_current_state_context(self) -> np.ndarray:
        """
        Retrieve the current state context vector from context-aware retrieval.

        Returns:
            np.ndarray: The current context vector.
        """
        try:
            context = await self.context_retrieval.get_context_vector()
            if self.logger:
                self.logger.debug(f"Current state context vector retrieved: {context}")
            return context
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])

    @io_bound_task
    async def consolidate_memory(self) -> None:
        """
        Consolidate memories from short-term and intermediate memory into long-term episodic and semantic memory.
        """
        try:
            current_context = await self.get_current_state_context()
            
            memories_to_consolidate = (
                self.short_term.get_memories_for_consolidation() + 
                self.intermediate.get_memories_for_consolidation()
            )
            sorted_memories = sorted(
                memories_to_consolidate, 
                key=self.spaced_repetition.consolidation_priority,
                reverse=True
            )
            
            for memory in sorted_memories:
                content = memory.content if hasattr(memory, 'content') else memory
                consolidated_content = f"Consolidated: {content}"
                await self.long_term_episodic.add(consolidated_content, current_context)
                
                concepts = self._extract_concepts(content)
                for concept in concepts:
                    try:
                        related_concepts = await self.long_term_semantic.query(concept, n=5)
                        related_concept_names = [rc[0] for rc in related_concepts]
                        await self.long_term_semantic.add(concept, related_concept_names)
                        if self.logger:
                            self.logger.debug(f"Added related concepts for {concept} to semantic memory")
                    except (IOError, OSError) as e:
                        if self.logger:
                            self.logger.error(f"File system error updating semantic memory for concept {concept}: {str(e)}", exc_info=True)
                    except Exception as semantic_error:
                        if self.logger:
                            self.logger.error(f"Unexpected error updating semantic memory for concept {concept}: {str(semantic_error)}", exc_info=True)
            
            self.short_term.clear_consolidated()
            self.intermediate.memories = [
                m for m in self.intermediate.memories if m not in sorted_memories
            ]
            await self._save_memory()
            if self.logger:
                self.logger.info("Memory consolidation completed")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during memory consolidation: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during memory consolidation: {str(e)}", exc_info=True)

    async def update_state_model(self, state_update: Dict[str, Any]):
        """
        Update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        if self.state_model is None:
            if self.logger:
                self.logger.error("Cannot update state model: state_model is None")
            return
        try:
            await self.state_model.update(state_update)
            if self.logger:
                self.logger.debug("HCDM state model updated")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating state model: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating state model: {str(e)}", exc_info=True)

    async def update_memory_from_consciousness(self) -> None:
        """
        Update memory based on the current thought from the consciousness stream.
        """
        try:
            current_thought = await self.retrieve_from_consciousness()
            if current_thought:
                await self.process_input(current_thought['content'])  
                
                # Perform pattern separation on current thought
                concepts = self._extract_concepts(current_thought['content'])
                for concept in concepts:
                    for other_concept in self.long_term_semantic.memory_vectors.keys():
                        separation = self.long_term_semantic.pattern_separation(concept, other_concept)
                        if separation is not None and separation < 0.1:  # Threshold for considering concepts as distinct
                            if self.logger:
                                self.logger.debug(f"Low pattern separation between {concept} and {other_concept}: {separation}")
                
                if self.logger:
                    self.logger.debug("Memory updated from consciousness stream")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating memory from consciousness stream: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating memory from consciousness stream: {str(e)}", exc_info=True)

    async def retrieve_from_consciousness(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the current thought from the consciousness stream.

        Returns:
            Optional[Dict[str, Any]]: The current thought if available, else None.
        """
        try:
            if self.consciousness_stream:
                current_thought = await self.consciousness_stream.get_current_thought()
                if self.logger:
                    self.logger.debug(f"Retrieved thought from consciousness stream: {current_thought}")
                return current_thought
            else:
                if self.logger:
                    self.logger.warning("Consciousness stream not set")
                return None
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None

    async def get_current_state_context(self) -> np.ndarray:
        """
        Retrieve the current state context vector from context-aware retrieval.

        Returns:
            np.ndarray: The current context vector.
        """
        try:
            context = await self.context_retrieval.get_context_vector()
            if self.logger:
                self.logger.debug(f"Current state context vector retrieved: {context}")
            return context
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])

    @io_bound_task
    async def consolidate_memory(self) -> None:
        """
        Consolidate memories from short-term and intermediate memory into long-term episodic and semantic memory.
        """
        try:
            current_context = await self.get_current_state_context()
            
            memories_to_consolidate = (
                self.short_term.get_memories_for_consolidation() + 
                self.intermediate.get_memories_for_consolidation()
            )
            sorted_memories = sorted(
                memories_to_consolidate, 
                key=self.spaced_repetition.consolidation_priority,
                reverse=True
            )
            
            for memory in sorted_memories:
                content = memory.content if hasattr(memory, 'content') else memory
                consolidated_content = f"Consolidated: {content}"
                await self.long_term_episodic.add(consolidated_content, current_context)
                
                concepts = self._extract_concepts(content)
                for concept in concepts:
                    try:
                        related_concepts = await self.long_term_semantic.query(concept, n=5)
                        related_concept_names = [rc[0] for rc in related_concepts]
                        await self.long_term_semantic.add(concept, related_concept_names)
                        if self.logger:
                            self.logger.debug(f"Added related concepts for {concept} to semantic memory")
                    except (IOError, OSError) as e:
                        if self.logger:
                            self.logger.error(f"File system error updating semantic memory for concept {concept}: {str(e)}", exc_info=True)
                    except Exception as semantic_error:
                        if self.logger:
                            self.logger.error(f"Unexpected error updating semantic memory for concept {concept}: {str(semantic_error)}", exc_info=True)
            
            self.short_term.clear_consolidated()
            self.intermediate.memories = [
                m for m in self.intermediate.memories if m not in sorted_memories
            ]
            await self._save_memory()
            if self.logger:
                self.logger.info("Memory consolidation completed")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during memory consolidation: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during memory consolidation: {str(e)}", exc_info=True)

    @io_bound_task
    async def update_state_model(self, state_update: Dict[str, Any]):
        """
        Update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        if self.state_model is None:
            if self.logger:
                self.logger.error("Cannot update state model: state_model is None")
            return
        try:
            await self.state_model.update(state_update)
            if self.logger:
                self.logger.debug("HCDM state model updated")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating state model: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating state model: {str(e)}", exc_info=True)

    async def update_memory_from_consciousness(self) -> None:
        """
        Update memory based on the current thought from the consciousness stream.
        """
        try:
            current_thought = await self.retrieve_from_consciousness()
            if current_thought:
                await self.process_input(current_thought['content'])  
                
                # Perform pattern separation on current thought
                concepts = self._extract_concepts(current_thought['content'])
                for concept in concepts:
                    for other_concept in self.long_term_semantic.memory_vectors.keys():
                        separation = self.long_term_semantic.pattern_separation(concept, other_concept)
                        if separation is not None and separation < 0.1:  # Threshold for considering concepts as distinct
                            if self.logger:
                                self.logger.debug(f"Low pattern separation between {concept} and {other_concept}: {separation}")
                
                if self.logger:
                    self.logger.debug("Memory updated from consciousness stream")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating memory from consciousness stream: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating memory from consciousness stream: {str(e)}", exc_info=True)

    async def retrieve_from_consciousness(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the current thought from the consciousness stream.

        Returns:
            Optional[Dict[str, Any]]: The current thought if available, else None.
        """
        try:
            if self.consciousness_stream:
                current_thought = await self.consciousness_stream.get_current_thought()
                if self.logger:
                    self.logger.debug(f"Retrieved thought from consciousness stream: {current_thought}")
                return current_thought
            else:
                if self.logger:
                    self.logger.warning("Consciousness stream not set")
                return None
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None

    async def get_current_state_context(self) -> np.ndarray:
        """
        Retrieve the current state context vector from context-aware retrieval.

        Returns:
            np.ndarray: The current context vector.
        """
        try:
            context = await self.context_retrieval.get_context_vector()
            if self.logger:
                self.logger.debug(f"Current state context vector retrieved: {context}")
            return context
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])

    @io_bound_task
    async def consolidate_memory(self) -> None:
        """
        Consolidate memories from short-term and intermediate memory into long-term episodic and semantic memory.
        """
        try:
            current_context = await self.get_current_state_context()
            
            memories_to_consolidate = (
                self.short_term.get_memories_for_consolidation() + 
                self.intermediate.get_memories_for_consolidation()
            )
            sorted_memories = sorted(
                memories_to_consolidate, 
                key=self.spaced_repetition.consolidation_priority,
                reverse=True
            )
            
            for memory in sorted_memories:
                content = memory.content if hasattr(memory, 'content') else memory
                consolidated_content = f"Consolidated: {content}"
                await self.long_term_episodic.add(consolidated_content, current_context)
                
                concepts = self._extract_concepts(content)
                for concept in concepts:
                    try:
                        related_concepts = await self.long_term_semantic.query(concept, n=5)
                        related_concept_names = [rc[0] for rc in related_concepts]
                        await self.long_term_semantic.add(concept, related_concept_names)
                        if self.logger:
                            self.logger.debug(f"Added related concepts for {concept} to semantic memory")
                    except (IOError, OSError) as e:
                        if self.logger:
                            self.logger.error(f"File system error updating semantic memory for concept {concept}: {str(e)}", exc_info=True)
                    except Exception as semantic_error:
                        if self.logger:
                            self.logger.error(f"Unexpected error updating semantic memory for concept {concept}: {str(semantic_error)}", exc_info=True)
            
            self.short_term.clear_consolidated()
            self.intermediate.memories = [
                m for m in self.intermediate.memories if m not in sorted_memories
            ]
            await self._save_memory()
            if self.logger:
                self.logger.info("Memory consolidation completed")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during memory consolidation: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during memory consolidation: {str(e)}", exc_info=True)

    def _calculate_importance(self, input_data: Any) -> float: 
        """
        Calculate the importance of the input data.

        Args:
            input_data (Any): The input data.

        Returns:
            float: The calculated importance score.
        """
        # Simple importance calculation based on input length and uniqueness
        if isinstance(input_data, str):
            words = input_data.split()
            if not words:
                return 0.0
            unique_ratio = len(set(words)) / len(words)  # Calculate uniqueness ratio
            importance = min(1.0, (len(input_data) / 1000) * unique_ratio)  # Importance based on length and uniqueness
            if self.logger:
                self.logger.debug(f"Calculated importance for string input: {importance}")
            return importance
        else:
            if self.logger:
                self.logger.debug("Calculated default importance for non-string input: 0.5")
            return 0.5  # Default importance for non-string inputs

    def _extract_concepts(self, memory: Any) -> Set[str]:
        """
        Extract important concepts from the memory content.

        Args:
            memory (Any): The memory content.

        Returns:
            Set[str]: A set of extracted concepts.
        """
        concepts = set()
        try:
            if isinstance(memory, str):
                # Remove punctuation and convert to lowercase
                text = re.sub(r'[^\w\s]', '', memory.lower())
                
                # Split into words and filter out short words
                words = text.split()
                concepts.update(word for word in words if len(word) > 3)
                
                # Add bigrams (pairs of adjacent words) as concepts
                bigrams = set(' '.join(words[i:i+2]) for i in range(len(words)-1))
                concepts.update(bigrams)
                
                # Use TF-IDF to extract important terms as concepts
                if hasattr(self.long_term_semantic, 'vectorizer'):
                    tfidf_matrix = self.long_term_semantic.vectorizer.fit_transform([memory])
                    feature_names = self.long_term_semantic.vectorizer.get_feature_names_out()
                    for _, idx in zip(*tfidf_matrix.nonzero()):
                        concepts.add(feature_names[idx])
            elif isinstance(memory, dict):
                # If memory is a dict, extract concepts from all string values
                string_values = [v for v in memory.values() if isinstance(v, str)]
                for v in string_values:
                    concepts.update(self._extract_concepts(v))
            # Additional types can be handled here as needed
        except (ValueError, TypeError) as e:
            if self.logger:
                self.logger.error(f"Error extracting concepts: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error extracting concepts: {str(e)}", exc_info=True)
        finally:
            if self.logger:
                self.logger.debug(f"Extracted concepts: {concepts}")
            return concepts

    async def update_state_model(self, state_update: Dict[str, Any]):
        """
        Update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        if self.state_model is None:
            if self.logger:
                self.logger.error("Cannot update state model: state_model is None")
            return
        try:
            await self.state_model.update(state_update)
            if self.logger:
                self.logger.debug("HCDM state model updated")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating state model: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating state model: {str(e)}", exc_info=True)

    async def update_memory_from_consciousness(self) -> None:
        """
        Update memory based on the current thought from the consciousness stream.
        """
        try:
            current_thought = await self.retrieve_from_consciousness()
            if current_thought:
                await self.process_input(current_thought['content'])  
                
                # Perform pattern separation on current thought
                concepts = self._extract_concepts(current_thought['content'])
                for concept in concepts:
                    for other_concept in self.long_term_semantic.memory_vectors.keys():
                        separation = self.long_term_semantic.pattern_separation(concept, other_concept)
                        if separation is not None and separation < 0.1:  # Threshold for considering concepts as distinct
                            if self.logger:
                                self.logger.debug(f"Low pattern separation between {concept} and {other_concept}: {separation}")
                
                if self.logger:
                    self.logger.debug("Memory updated from consciousness stream")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating memory from consciousness stream: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating memory from consciousness stream: {str(e)}", exc_info=True)

    async def retrieve_from_consciousness(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the current thought from the consciousness stream.

        Returns:
            Optional[Dict[str, Any]]: The current thought if available, else None.
        """
        try:
            if self.consciousness_stream:
                current_thought = await self.consciousness_stream.get_current_thought()
                if self.logger:
                    self.logger.debug(f"Retrieved thought from consciousness stream: {current_thought}")
                return current_thought
            else:
                if self.logger:
                    self.logger.warning("Consciousness stream not set")
                return None
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving from consciousness stream: {str(e)}", exc_info=True)
            return None

    async def get_current_state_context(self) -> np.ndarray:
        """
        Retrieve the current state context vector from context-aware retrieval.

        Returns:
            np.ndarray: The current context vector.
        """
        try:
            context = await self.context_retrieval.get_context_vector()
            if self.logger:
                self.logger.debug(f"Current state context vector retrieved: {context}")
            return context
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving current state context: {str(e)}", exc_info=True)
            return np.array([])

    @io_bound_task
    async def consolidate_memory(self) -> None:
        """
        Consolidate memories from short-term and intermediate memory into long-term episodic and semantic memory.
        """
        try:
            current_context = await self.get_current_state_context()
            
            memories_to_consolidate = (
                self.short_term.get_memories_for_consolidation() + 
                self.intermediate.get_memories_for_consolidation()
            )
            sorted_memories = sorted(
                memories_to_consolidate, 
                key=self.spaced_repetition.consolidation_priority,
                reverse=True
            )
            
            for memory in sorted_memories:
                content = memory.content if hasattr(memory, 'content') else memory
                consolidated_content = f"Consolidated: {content}"
                await self.long_term_episodic.add(consolidated_content, current_context)
                
                concepts = self._extract_concepts(content)
                for concept in concepts:
                    try:
                        related_concepts = await self.long_term_semantic.query(concept, n=5)
                        related_concept_names = [rc[0] for rc in related_concepts]
                        await self.long_term_semantic.add(concept, related_concept_names)
                        if self.logger:
                            self.logger.debug(f"Added related concepts for {concept} to semantic memory")
                    except (IOError, OSError) as e:
                        if self.logger:
                            self.logger.error(f"File system error updating semantic memory for concept {concept}: {str(e)}", exc_info=True)
                    except Exception as semantic_error:
                        if self.logger:
                            self.logger.error(f"Unexpected error updating semantic memory for concept {concept}: {str(semantic_error)}", exc_info=True)
            
            self.short_term.clear_consolidated()
            self.intermediate.memories = [
                m for m in self.intermediate.memories if m not in sorted_memories
            ]
            await self._save_memory()
            if self.logger:
                self.logger.info("Memory consolidation completed")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during memory consolidation: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during memory consolidation: {str(e)}", exc_info=True)

    @io_bound_task
    async def update_state_model(self, state_update: Dict[str, Any]):
        """
        Update the Hybrid Cognitive Dynamics Model (HCDM) with new state information.

        Args:
            state_update (Dict[str, Any]): The state update information.
        """
        if self.state_model is None:
            if self.logger:
                self.logger.error("Cannot update state model: state_model is None")
            return
        try:
            await self.state_model.update(state_update)
            if self.logger:
                self.logger.debug("HCDM state model updated")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error updating state model: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error updating state model: {str(e)}", exc_info=True)

    async def consolidate_memory_with_consciousness(self) -> None:
        """
        Consolidate memory with consciousness stream integration.
        """
        try:
            await self.update_memory_from_consciousness()
            await self.consolidate_memory()
            
            if self.consciousness_stream and self.state_model:
                consciousness_state = await self.consciousness_stream.get_state()
                state_update = await self.state_model.derive_state_update(consciousness_state)
                await self.update_state_model(state_update)
                
                completed_patterns = await self.long_term_semantic.pattern_completion(str(consciousness_state))
                if completed_patterns:
                    if self.logger:
                        self.logger.debug(f"Completed patterns from consciousness state: {completed_patterns}")
                    for pattern, similarity in completed_patterns:
                        thought = {
                            "type": "CONSOLIDATION_INSIGHT",
                            "content": f"Insight from pattern completion: {pattern} (similarity: {similarity})",
                            "timestamp": time.time()
                        }
                        await self.consciousness_stream.add_thought(thought)
                        if self.logger:
                            self.logger.debug(f"Thought added to consciousness stream from consolidation insight: {thought}")
                
                if self.logger:
                    self.logger.debug("Memory consolidated with consciousness stream integration")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during consolidate_memory_with_consciousness: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during consolidate_memory_with_consciousness: {str(e)}", exc_info=True)

    @cpu_bound_task
    async def consolidate_memory_with_consciousness(self) -> None:
        """
        Consolidate memory with consciousness stream integration.
        """
        try:
            await self.update_memory_from_consciousness()
            await self.consolidate_memory()
            
            if self.consciousness_stream and self.state_model:
                consciousness_state = await self.consciousness_stream.get_state()
                state_update = await self.state_model.derive_state_update(consciousness_state)
                await self.update_state_model(state_update)
                
                completed_patterns = await self.long_term_semantic.pattern_completion(str(consciousness_state))
                if completed_patterns:
                    if self.logger:
                        self.logger.debug(f"Completed patterns from consciousness state: {completed_patterns}")
                    for pattern, similarity in completed_patterns:
                        thought = {
                            "type": "CONSOLIDATION_INSIGHT",
                            "content": f"Insight from pattern completion: {pattern} (similarity: {similarity})",
                            "timestamp": time.time()
                        }
                        await self.consciousness_stream.add_thought(thought)
                        if self.logger:
                            self.logger.debug(f"Thought added to consciousness stream from consolidation insight: {thought}")
                
                if self.logger:
                    self.logger.debug("Memory consolidated with consciousness stream integration")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error during consolidate_memory_with_consciousness: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during consolidate_memory_with_consciousness: {str(e)}", exc_info=True)

    @io_bound_task
    async def _save_memory(self) -> None:
        """
        Save memory to file.

        This includes episodic and semantic memories.
        """
        try:
            data = {
                'episodic': self.long_term_episodic.episodes if self.long_term_episodic else [],
                'semantic': {
                    node: list(self.long_term_semantic.knowledge_graph.neighbors(node))
                    for node in self.long_term_semantic.knowledge_graph.nodes()
                } if self.long_term_semantic else {}
            }
            async with aiofiles.open(self.file_path, 'w') as file:
                await file.write(json.dumps(data, indent=2))
            if self.logger:
                self.logger.info(f"Memory saved to {self.file_path}")
        except (IOError, OSError) as e:
            if self.logger:
                self.logger.error(f"File system error saving memory: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error saving memory: {str(e)}", exc_info=True)

    def get_memory_stats(self) -> Dict[str, int]:
        """
        Retrieve memory statistics.

        Returns:
            Dict[str, int]: A dictionary containing various memory statistics.
        """
        try:
            stats = {
                "sensory_size": len(self.sensory.buffer) if self.sensory else 0,
                "working_memory_size": len(self.working.contents) if self.working else 0, 
                "short_term_memory_size": len(self.short_term.items) if self.short_term else 0,
                "intermediate_memory_size": len(self.intermediate.memories) if self.intermediate else 0,
                "long_term_episodic_size": len(self.long_term_episodic.episodes) if self.long_term_episodic else 0,
                "long_term_semantic_size": len(self.long_term_semantic.knowledge_graph.nodes()) if self.long_term_semantic else 0,
                "long_term_semantic_vectors": len(self.long_term_semantic.memory_vectors) if self.long_term_semantic else 0,
                "working_memory_capacity": self.working.capacity if self.working else 0,
                "short_term_memory_capacity": self.short_term.capacity if self.short_term else 0,
                "intermediate_memory_capacity": self.intermediate.capacity if self.intermediate else 0
            }
            if self.logger:
                self.logger.debug(f"Memory statistics retrieved: {stats}")
            return stats
        except AttributeError as e:
            if self.logger:
                self.logger.error(f"Attribute error retrieving memory statistics: {str(e)}", exc_info=True)
            return {}
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error retrieving memory statistics: {str(e)}", exc_info=True)
            return {}

    @cpu_bound_task
    async def cleanup_memory(self, threshold: Optional[float] = None) -> None:
        """
        Clean up memories based on a decay threshold.

        Args:
            threshold (Optional[float], optional): The decay threshold. Defaults to None.
        """
        try:
            # Retrieve threshold from config_manager if not provided
            if threshold is None:
                threshold = self.memory_config.get('cleanup_threshold', 0.1)

            # Use TimeDecay to determine which episodic memories to keep
            if self.long_term_episodic and self.time_decay:
                current_time = time.time()
                original_count = len(self.long_term_episodic.episodes)
                self.long_term_episodic.episodes = [
                    ep for ep in self.long_term_episodic.episodes 
                    if self.time_decay.decay(
                        memory_type=MemoryType.LONG_TERM_EPISODIC, 
                        time_elapsed=current_time - datetime.fromisoformat(ep['timestamp']).timestamp(), 
                        importance=ep.get('importance', 1.0)
                    ) > threshold
                ]
                cleaned_count = original_count - len(self.long_term_episodic.episodes)
                if self.logger:
                    self.logger.info(f"Episodic memory cleaned: {cleaned_count} entries removed")
            else:
                if self.logger:
                    self.logger.warning("TimeDecay or long_term_episodic is not properly initialized.")

            # Clean up intermediate memory
            if self.intermediate:
                original_intermediate = len(self.intermediate.memories)
                self.intermediate.memories = [
                    m for m in self.intermediate.memories if m.get_strength() > threshold
                ]
                cleaned_intermediate = original_intermediate - len(self.intermediate.memories)
                if self.logger:
                    self.logger.info(f"Intermediate memory cleaned: {cleaned_intermediate} entries removed")
            else:
                if self.logger:
                    self.logger.warning("Intermediate memory is not initialized.")

            # Clean up long-term semantic memory
            if self.long_term_semantic:
                nodes_to_remove = []
                for node in self.long_term_semantic.knowledge_graph.nodes():
                    if len(list(self.long_term_semantic.knowledge_graph.neighbors(node))) < 2:
                        nodes_to_remove.append(node)
                
                for node in nodes_to_remove:
                    self.long_term_semantic.knowledge_graph.remove_node(node)
                    self.long_term_semantic.memory_vectors.pop(node, None)
                    if self.logger:
                        self.logger.debug(f"Removed semantic memory node: {node}")
                
                if self.logger:
                    self.logger.info(f"Long-term semantic memory cleaned: {len(nodes_to_remove)} nodes removed")
            else:
                if self.logger:
                    self.logger.warning("long_term_semantic is not properly initialized.")

            if self.logger:
                self.logger.info(f"Memory cleanup completed. Threshold: {threshold}")
        except AttributeError as e:
            if self.logger:
                self.logger.error(f"Attribute error during memory cleanup: {str(e)}", exc_info=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error during memory cleanup: {str(e)}", exc_info=True)
