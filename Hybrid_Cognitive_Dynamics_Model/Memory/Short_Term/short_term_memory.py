# modules/Hybrid_Cognitive_Dynamics_Model/Memory/Short_Term/short_term_memory.py

from modules.Config.config import ConfigManager
from typing import List, Any, Optional

class ShortTermMemory:
    """Represents the short-term memory, temporarily storing processed information."""

    def __init__(self, config_manager: ConfigManager, capacity: Optional[int] = None):
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('ShortTermMemory')

        # Retrieve capacity from config_manager or use the provided value
        memory_config = config_manager.get_subsystem_config('memory')
        self.capacity = capacity or memory_config.get('short_term_memory', {}).get('capacity', 100)
        
        # Ensure capacity is valid
        if self.capacity <= 0:
            raise ValueError("ShortTermMemory capacity must be a positive integer")
        
        # Log whether the capacity was taken from config or overridden
        source = "provided directly" if capacity else "from configuration"
        self.logger.info(f"Initialized ShortTermMemory with capacity: {self.capacity} ({source})")

        self.items = []

    def add(self, item: Any) -> None:
        """Add a new item to short-term memory, maintaining the capacity limit."""
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]
        self.logger.debug(f"Added item to ShortTermMemory: {item}")

    def get_memories_for_consolidation(self) -> List[Any]:
        """Retrieve items for potential consolidation into long-term memory."""
        return self.items.copy()  # Return a copy to prevent unintended modifications

    def clear_consolidated(self) -> None:
        """Remove items that have been consolidated."""
        consolidated_count = len(self.items) // 2  # Consolidate half of the items
        self.items = self.items[consolidated_count:]
        self.logger.debug(f"Cleared {consolidated_count} consolidated memories from ShortTermMemory")

    def __len__(self) -> int:
        """Return the current number of items in short-term memory."""
        return len(self.items)