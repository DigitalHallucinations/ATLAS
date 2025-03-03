# modules/Hybrid_Cognitive_Dynamics_Model/Memory/Long_Term/Episodic/long_term_episodic_memory.py

import asyncio
import numpy as np
from datetime import datetime
from modules.Config.config import ConfigManager
from typing import List, Dict, Any, Optional
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.Retrieval.context_aware_retrieval import ContextAwareRetrieval
from modules.Hybrid_Cognitive_Dynamics_Model.Memory.Retrieval.simplified_context_retrieval import SimplifiedContextRetrieval

class EnhancedLongTermEpisodicMemory:
    """Represents the long-term episodic memory with context-aware retrieval."""

    def __init__(self, state_model: Optional[Any], config_manager: ConfigManager = None, memory_system=None):
        self.episodes = []
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('EnhancedLongTermEpisodicMemory')
        self.context_retrieval = (ContextAwareRetrieval(state_model, config_manager) 
                          if state_model is not None 
                          else SimplifiedContextRetrieval(config_manager))
        
        self.logger.debug(f"Initialized EnhancedLongTermEpisodicMemory with {type(self.context_retrieval).__name__}")
        
        memory_config = self.config_manager.get_subsystem_config('memory')
        self.consolidation_interval = memory_config.get('consolidation_interval', 3600)  # Default to 1 hour
        self.last_consolidation_time = datetime.now()
        self.memory_system = memory_system

    def set_memory_system(self, memory_system):
        """Set the memory system for this component."""
        self.memory_system = memory_system
        self.logger.info("Memory system set for EnhancedLongTermEpisodicMemory")

    async def add(self, episode: Any, context: np.ndarray) -> None:
        """Add a new episode with its context to long-term episodic memory."""
        self.episodes.append({
            'timestamp': datetime.now().isoformat(),
            'content': episode,
            'context': context.tolist() if isinstance(context, np.ndarray) else context,
            'importance': 0.5  # Default importance
        })
        self.logger.debug(f"Added episode to EnhancedLongTermEpisodicMemory: {episode}")
        
        # Check if it's time to consolidate
        if (datetime.now() - self.last_consolidation_time).total_seconds() >= self.consolidation_interval:
            await self.consolidate_memory()
        
        # Ensure the episode is immediately available for retrieval
        if self.memory_system:
            await self.memory_system.process_input(f"New episodic memory: {episode}")
        else:
            self.logger.warning("memory_system is not set, skipping immediate processing of new episode")

    async def get_relevant_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        try:
            self.logger.debug("Attempting to get relevant episodes")
            
            current_context = await self.context_retrieval.get_context_vector()
            
            scored_episodes = []
            for episode in self.episodes:
                memory_context = np.asarray(episode['context'])
                
                similarity = await self.context_retrieval.context_similarity(memory_context, current_context)
                scored_episodes.append((episode, similarity * episode['importance']))
            
            relevant_episodes = [episode for episode, _ in sorted(scored_episodes, key=lambda x: x[1], reverse=True)[:n]]
            return [{'content': str(episode['content'])} for episode in relevant_episodes]
        except Exception as e:
            self.logger.error(f"Error in get_relevant_episodes: {str(e)}", exc_info=True)
            return []

    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent episodes."""
        return sorted(self.episodes, key=lambda x: x['timestamp'], reverse=True)[:n]

    async def remove_episode(self, episode_id: str):
        """Remove an episode from memory by its ID."""
        self.episodes = [ep for ep in self.episodes if ep.get('id') != episode_id]
        self.logger.debug(f"Removed episode with ID: {episode_id}")

    async def update_episode(self, episode_id: str, new_content: Any):
        """Update the content of an existing episode."""
        for ep in self.episodes:
            if ep.get('id') == episode_id:
                ep['content'] = new_content
                ep['last_updated'] = datetime.now().isoformat()
                self.logger.debug(f"Updated episode with ID: {episode_id}")
                break

    async def consolidate_memory(self):
        """
        Consolidate memory by performing necessary processing on recently added episodes.
        This method implements more sophisticated consolidation strategies.
        """
        self.logger.debug("Performing episodic memory consolidation")
        
        # Sort episodes by timestamp (most recent first)
        self.episodes.sort(key=lambda x: datetime.fromisoformat(x['timestamp']), reverse=True)
        
        # Merge similar episodes
        merged_episodes = []
        for episode in self.episodes:
            if not any(self._are_episodes_similar(episode, merged) for merged in merged_episodes):
                merged_episodes.append(episode)
        
        # Extract common themes or patterns
        common_themes = self._extract_common_themes(merged_episodes)
        
        # Remove or archive old, less relevant episodes
        current_time = datetime.now()
        consolidated_episodes = [
            ep for ep in merged_episodes 
            if (current_time - datetime.fromisoformat(ep['timestamp'])).days <= 30  # Keep episodes from last 30 days
        ]
        
        # Update the context or importance of episodes based on recent experiences
        for episode in consolidated_episodes:
            episode['importance'] = self._calculate_episode_importance(episode, common_themes)
        
        # Update the episodes list with the consolidated episodes
        self.episodes = consolidated_episodes
        
        self.last_consolidation_time = datetime.now()
        self.logger.debug(f"Episodic memory consolidation completed. Total episodes after consolidation: {len(self.episodes)}")

    def _are_episodes_similar(self, episode1: Dict[str, Any], episode2: Dict[str, Any]) -> bool:
        """Helper method to determine if two episodes are similar."""
        content_similarity = self.context_retrieval.context_similarity(
            np.array(episode1['context']), np.array(episode2['context'])
        )
        return content_similarity > 0.8 

    def _extract_common_themes(self, episodes: List[Dict[str, Any]]) -> List[str]:
        """Helper method to extract common themes from episodes."""
        # Implement theme extraction logic
        all_content = " ".join([ep['content'] for ep in episodes])
        # This is a placeholder. Use NLP techniques
        # such as topic modeling or keyword extraction
        return list(set(all_content.split()))[:10]  # Return top 10 most common words as themes

    def _calculate_episode_importance(self, episode: Dict[str, Any], common_themes: List[str]) -> float:
        """Helper method to calculate the importance of an episode."""
        importance = episode.get('importance', 0.5)  # Start with existing importance or default
        for theme in common_themes:
            if theme in str(episode['content']):
                importance += 0.1
        time_factor = 1 / (1 + (datetime.now() - datetime.fromisoformat(episode['timestamp'])).days)
        return min(importance * time_factor, 1.0)
    
    async def start_periodic_consolidation(self):
        while True:
            await asyncio.sleep(self.consolidation_interval)
            await self.consolidate_memory()