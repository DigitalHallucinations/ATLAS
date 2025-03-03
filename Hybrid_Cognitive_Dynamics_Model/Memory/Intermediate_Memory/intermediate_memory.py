# modules/Hybrid_Cognitive_Dynamics_Model/Memory/Intermediate_Memory/intermediate_memory.py

from modules.Config.config import ConfigManager
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.Intermediate_Memory.intermediate_memory import TimeDecay, SpacedRepetition, MemoryType
from typing import List, Any, Dict
import time

class IntermediateMemory:
    """Represents the intermediate memory between short-term and long-term storage."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('IntermediateMemory')
        memory_config = config_manager.get_subsystem_config('memory')
        self.capacity = memory_config.get('intermediate_memory', {}).get('capacity', 1000)
        self.consolidation_threshold = memory_config.get('consolidation_threshold', 0.7)
        self.memories: List[Dict[str, Any]] = []
        self.logger.info(f"Initialized IntermediateMemory with capacity: {self.capacity}")

        # Initialize TimeDecay and SpacedRepetition components
        self.time_decay = TimeDecay(system_state=None, config_manager=self.config_manager)
        self.spaced_repetition = SpacedRepetition(memory_store=self, config_manager=self.config_manager)

    async def add(self, memory: Any, importance: float = 1.0) -> None:
        """
        Add a new memory to intermediate storage with time-aware decay.

        Args:
            memory (Any): The memory content to add.
            importance (float): The importance factor of the memory (default is 1.0).
        """
        if len(self.memories) >= self.capacity:
            await self.consolidate_oldest()
        memory_entry = {
            'content': memory,
            'timestamp': time.time(),
            'importance': importance
        }
        self.memories.append(memory_entry)
        self.logger.debug(f"Added memory to IntermediateMemory: {memory[:50]}... with importance: {importance}")

    async def consolidate_oldest(self) -> Dict[str, Any]:
        """
        Consolidate the oldest memory in intermediate storage based on time decay.

        Returns:
            Dict[str, Any]: The consolidated memory entry.
        """
        if not self.memories:
            self.logger.warning("No memories to consolidate.")
            return {}
        
        oldest_memory = min(self.memories, key=lambda m: m['timestamp'])
        self.memories.remove(oldest_memory)
        self.logger.debug(f"Consolidated oldest memory: {oldest_memory['content'][:50]}...")

        # Determine if the memory should be moved to long-term memory based on decay
        time_elapsed = time.time() - oldest_memory['timestamp']
        decayed_strength = self.time_decay.decay(
            memory_type=MemoryType.LONG_TERM_EPISODIC,  # Assuming episodic memory
            time_elapsed=time_elapsed,
            importance=oldest_memory.get('importance', 1.0)
        )

        if decayed_strength > self.consolidation_threshold:
            # Schedule for long-term consolidation using spaced repetition
            self.spaced_repetition.schedule_review(
                memory=oldest_memory,
                review_time=time.time() + self.spaced_repetition.sm2_params["interval"] * 86400,  # days to seconds
                emotion_factor=1.0  # Can be adjusted based on emotional weight
            )
            self.logger.debug(f"Memory scheduled for spaced repetition: {oldest_memory['content'][:50]}...")
        else:
            self.logger.debug(f"Memory discarded due to low strength: {oldest_memory['content'][:50]}...")

        return oldest_memory

    async def get_memories_for_consolidation(self) -> List[Dict[str, Any]]:
        """
        Retrieve memories ready for consolidation into long-term storage based on decay strength.

        Returns:
            List[Dict[str, Any]]: A list of memory entries ready for consolidation.
        """
        consolidated_memories = []
        current_time = time.time()

        for memory in self.memories[:]:  # Iterate over a shallow copy
            time_elapsed = current_time - memory['timestamp']
            decayed_strength = self.time_decay.decay(
                memory_type=MemoryType.LONG_TERM_EPISODIC,  # Assuming episodic memory
                time_elapsed=time_elapsed,
                importance=memory.get('importance', 1.0)
            )
            if decayed_strength > self.consolidation_threshold:
                consolidated_memories.append(memory)
                self.memories.remove(memory)
                self.logger.debug(f"Memory marked for consolidation: {memory['content'][:50]}...")

        return consolidated_memories

    async def clear_consolidated(self) -> None:
        """
        Remove a portion of memories that have been consolidated to free up space.
        """
        # Example strategy: retain only the most recent half of the memories
        retained_memories = self.memories[-(self.capacity // 2):]
        self.memories = retained_memories
        self.logger.debug("Cleared consolidated memories from IntermediateMemory")

    async def process_memories(self) -> None:
        """
        Process and consolidate memories based on time-aware decay and spaced repetition.
        """
        memories_to_consolidate = await self.get_memories_for_consolidation()
        for memory in memories_to_consolidate:
            quality = await self.simulate_review_quality(memory)
            if quality >= 3:
                self.spaced_repetition.review(memory, quality)
                self.logger.debug(f"Memory reviewed with quality {quality}: {memory['content'][:50]}...")
            else:
                self.logger.debug(f"Memory review failed with quality {quality}: {memory['content'][:50]}...")

        await self.clear_consolidated()

    async def simulate_review_quality(self, memory: Dict[str, Any]) -> int:
        """
        Simulates the quality of memory recall during review.

        Args:
            memory (Dict[str, Any]): The memory entry being reviewed.

        Returns:
            int: The quality rating (0-5).
        """
        # Placeholder for actual review simulation logic
        # For demonstration, we'll return a random quality score
        import random
        quality = random.randint(0, 5)
        self.logger.debug(f"Simulated review quality for memory: {quality}")
        return quality
