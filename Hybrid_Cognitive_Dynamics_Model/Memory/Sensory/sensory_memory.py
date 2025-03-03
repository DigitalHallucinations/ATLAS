# modules/Hybrid_Cognitive_Dynamics_Model/Memory/Sensory/sensory_memory.py

import time
import numpy as np
from typing import Any, List, Dict
from modules.Config.config import ConfigManager

class SensoryMemory:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('ContextAwareRetrieval')
        memory_config = self.config_manager.get_subsystem_config('memory')
        sensory_config = memory_config.get('sensory_memory', {})
        
        self.max_size = sensory_config.get('max_size', 100)
        self.decay_rate = sensory_config.get('decay_rate', 0.1)
        
        self.buffer: List[Dict[str, Any]] = []
        self.logger.info(f"Initialized SensoryMemory with max size: {self.max_size} and decay rate: {self.decay_rate}")

    async def add(self, input_data: Any) -> None:
        """Add new input data to the sensory memory buffer with preprocessing."""
        processed_data = self._preprocess_input(input_data)
        timestamp = time.time()
        self.buffer.append({"data": processed_data, "timestamp": timestamp, "salience": 1.0})
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
        self.logger.debug(f"Added processed input to SensoryMemory: {processed_data}")

    def _preprocess_input(self, input_data: Any) -> Any:
        """Preprocess the input data based on its type."""
        if isinstance(input_data, str):
            return self._process_text(input_data)
        elif isinstance(input_data, np.ndarray):
            return self._process_visual(input_data)
        # Add more type-specific processing as needed
        return input_data

    def _process_text(self, text: str) -> str:
        """Simple text processing (e.g., lowercase, remove punctuation)."""
        return ''.join(char.lower() for char in text if char.isalnum() or char.isspace())

    def _process_visual(self, image: np.ndarray) -> np.ndarray:
        """Simple visual processing (e.g., normalize values)."""
        return (image - np.min(image)) / (np.max(image) - np.min(image))

    def _compute_salience(self, data: Any) -> float:
        """Compute salience of the data (placeholder implementation)."""
        # This is a simplified salience computation
        if isinstance(data, str):
            return len(set(data.split())) / len(data.split())  # Unique word ratio
        elif isinstance(data, np.ndarray):
            return np.std(data)  # Standard deviation as a measure of salience
        return 1.0

    async def update(self) -> None:
        """Update the sensory memory (apply decay and recompute salience)."""
        current_time = time.time()
        for item in self.buffer:
            time_diff = current_time - item["timestamp"]
            item["salience"] *= np.exp(-self.decay_rate * time_diff)
            item["salience"] = max(item["salience"], 0.1)  # Prevent complete decay

    async def get_salient_info(self) -> List[Any]:
        """Retrieve the most salient information from the buffer."""
        await self.update()
        sorted_buffer = sorted(self.buffer, key=lambda x: x["salience"], reverse=True)
        return [item["data"] for item in sorted_buffer[:10]]
