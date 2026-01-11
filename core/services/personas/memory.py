"""
Persona memory service for ATLAS.

Provides working memory, episodic memory, and semantic knowledge
scoped to individual personas per user.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from core.services.common import Actor, OperationResult
from core.services.common.exceptions import PermissionDeniedError

from .exceptions import PersonaError, PersonaNotFoundError
from .permissions import PersonaPermissionChecker
from .types import (
    # Memory types
    PersonaMemoryContext,
    PersonaKnowledge,
    LearnedFact,
    EpisodicMemory,
    # Events
    PersonaMemoryUpdated,
)

if TYPE_CHECKING:
    from core.config import ConfigManager
    from core.messaging import MessageBus


logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _make_memory_key(persona_id: str, user_id: str) -> str:
    """Create a key for persona-user memory lookup."""
    return f"{persona_id}:{user_id}"


class PersonaMemoryError(PersonaError):
    """Error related to persona memory operations."""
    pass


class PersonaMemoryService:
    """
    Service for managing persona-scoped memory.
    
    Provides:
    - Working memory (scratchpad) per persona-user pair
    - Episodic memory for interaction history
    - Semantic knowledge extraction and storage
    - Memory injection into agent context
    """
    
    # Default TTL for working memory entries (24 hours)
    DEFAULT_MEMORY_TTL_HOURS = 24
    
    # Maximum episodes to retain per persona-user
    MAX_EPISODES = 1000
    
    # Maximum facts per persona-user
    MAX_FACTS = 500
    
    def __init__(
        self,
        config_manager: Optional["ConfigManager"] = None,
        message_bus: Optional["MessageBus"] = None,
        permission_checker: Optional[PersonaPermissionChecker] = None,
    ) -> None:
        """
        Initialize the PersonaMemoryService.
        
        Args:
            config_manager: Configuration manager
            message_bus: Message bus for publishing events
            permission_checker: Permission checker for authorization
        """
        self._config_manager = config_manager
        self._message_bus = message_bus
        self._permission_checker = permission_checker or PersonaPermissionChecker()
        
        # In-memory storage (would be persisted in production)
        # Key: "{persona_id}:{user_id}"
        self._working_memory: Dict[str, PersonaMemoryContext] = {}
        self._knowledge: Dict[str, PersonaKnowledge] = {}
        self._episodes: Dict[str, List[EpisodicMemory]] = {}
    
    # =========================================================================
    # Working Memory (Scratchpad)
    # =========================================================================
    
    async def get_working_memory(
        self,
        actor: Actor,
        persona_id: str,
        user_id: Optional[str] = None,
    ) -> OperationResult[PersonaMemoryContext]:
        """
        Get the working memory for a persona-user pair.
        
        Args:
            actor: The actor requesting memory
            persona_id: ID of the persona
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing the memory context
        """
        try:
            user_id = user_id or actor.id
            key = _make_memory_key(persona_id, user_id)
            
            # Check permission
            if actor.id != user_id and not self._permission_checker.can_admin(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot access memory for user {user_id}"
                )
            
            memory = self._working_memory.get(key)
            if memory is None:
                memory = PersonaMemoryContext(
                    persona_id=persona_id,
                    user_id=user_id,
                )
                self._working_memory[key] = memory
            
            # Update last accessed
            memory.last_accessed = _now_utc()
            
            return OperationResult.success(memory)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error getting working memory for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to get working memory: {e}")
    
    async def set_working_memory(
        self,
        actor: Actor,
        persona_id: str,
        key: str,
        value: Any,
        user_id: Optional[str] = None,
        ttl_hours: Optional[int] = None,
    ) -> OperationResult[PersonaMemoryContext]:
        """
        Set a value in working memory.
        
        Args:
            actor: The actor setting the value
            persona_id: ID of the persona
            key: Key to set
            value: Value to store
            user_id: ID of the user (defaults to actor.id)
            ttl_hours: Optional TTL for this entry
            
        Returns:
            OperationResult containing the updated memory context
        """
        try:
            user_id = user_id or actor.id
            mem_key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_write(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot modify memory for user {user_id}"
                )
            
            memory = self._working_memory.get(mem_key)
            if memory is None:
                memory = PersonaMemoryContext(
                    persona_id=persona_id,
                    user_id=user_id,
                )
                self._working_memory[mem_key] = memory
            
            memory.scratchpad[key] = value
            memory.last_accessed = _now_utc()
            
            if ttl_hours:
                memory.expires_at = _now_utc() + timedelta(hours=ttl_hours)
            
            await self._emit_memory_event(
                persona_id, user_id, "working", "set", key
            )
            
            return OperationResult.success(memory)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error setting working memory for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to set working memory: {e}")
    
    async def get_scratchpad_value(
        self,
        actor: Actor,
        persona_id: str,
        key: str,
        user_id: Optional[str] = None,
        default: Any = None,
    ) -> OperationResult[Any]:
        """
        Get a specific value from the scratchpad.
        
        Args:
            actor: The actor requesting the value
            persona_id: ID of the persona
            key: Key to retrieve
            user_id: ID of the user (defaults to actor.id)
            default: Default value if key not found
            
        Returns:
            OperationResult containing the value
        """
        try:
            result = await self.get_working_memory(actor, persona_id, user_id)
            if not result.is_success:
                return result
            
            memory = result.value
            value = memory.scratchpad.get(key, default)
            
            return OperationResult.success(value)
            
        except Exception as e:
            logger.exception(f"Error getting scratchpad value {key}")
            return OperationResult.failure(f"Failed to get scratchpad value: {e}")
    
    async def delete_scratchpad_key(
        self,
        actor: Actor,
        persona_id: str,
        key: str,
        user_id: Optional[str] = None,
    ) -> OperationResult[bool]:
        """
        Delete a key from the scratchpad.
        
        Args:
            actor: The actor deleting the key
            persona_id: ID of the persona
            key: Key to delete
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing True if deleted, False if not found
        """
        try:
            user_id = user_id or actor.id
            mem_key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_write(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot modify memory for user {user_id}"
                )
            
            memory = self._working_memory.get(mem_key)
            if memory is None or key not in memory.scratchpad:
                return OperationResult.success(False)
            
            del memory.scratchpad[key]
            
            return OperationResult.success(True)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error deleting scratchpad key {key}")
            return OperationResult.failure(f"Failed to delete scratchpad key: {e}")
    
    async def clear_working_memory(
        self,
        actor: Actor,
        persona_id: str,
        user_id: Optional[str] = None,
    ) -> OperationResult[None]:
        """
        Clear all working memory for a persona-user pair.
        
        Args:
            actor: The actor clearing memory
            persona_id: ID of the persona
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult indicating success/failure
        """
        try:
            user_id = user_id or actor.id
            mem_key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_admin(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot clear memory for user {user_id}"
                )
            
            if mem_key in self._working_memory:
                del self._working_memory[mem_key]
            
            await self._emit_memory_event(
                persona_id, user_id, "working", "clear"
            )
            
            logger.info(f"Cleared working memory for {persona_id}/{user_id}")
            return OperationResult.success(None)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error clearing working memory for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to clear working memory: {e}")
    
    async def set_active_goals(
        self,
        actor: Actor,
        persona_id: str,
        goals: List[str],
        user_id: Optional[str] = None,
    ) -> OperationResult[PersonaMemoryContext]:
        """
        Set the active goals for a persona-user pair.
        
        Args:
            actor: The actor setting goals
            persona_id: ID of the persona
            goals: List of active goals
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing the updated memory context
        """
        try:
            user_id = user_id or actor.id
            mem_key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_write(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot modify goals for user {user_id}"
                )
            
            memory = self._working_memory.get(mem_key)
            if memory is None:
                memory = PersonaMemoryContext(
                    persona_id=persona_id,
                    user_id=user_id,
                )
                self._working_memory[mem_key] = memory
            
            memory.active_goals = goals
            memory.last_accessed = _now_utc()
            
            return OperationResult.success(memory)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error setting active goals for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to set active goals: {e}")
    
    async def set_task_context(
        self,
        actor: Actor,
        persona_id: str,
        task_context: Optional[str],
        user_id: Optional[str] = None,
    ) -> OperationResult[PersonaMemoryContext]:
        """
        Set the current task context.
        
        Args:
            actor: The actor setting context
            persona_id: ID of the persona
            task_context: Current task description
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing the updated memory context
        """
        try:
            user_id = user_id or actor.id
            mem_key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_write(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot modify task context for user {user_id}"
                )
            
            memory = self._working_memory.get(mem_key)
            if memory is None:
                memory = PersonaMemoryContext(
                    persona_id=persona_id,
                    user_id=user_id,
                )
                self._working_memory[mem_key] = memory
            
            memory.task_context = task_context
            memory.last_accessed = _now_utc()
            
            return OperationResult.success(memory)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error setting task context for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to set task context: {e}")
    
    # =========================================================================
    # Episodic Memory
    # =========================================================================
    
    async def record_episode(
        self,
        actor: Actor,
        persona_id: str,
        summary: str,
        conversation_id: Optional[str] = None,
        outcome: Optional[str] = None,
        key_entities: Optional[List[str]] = None,
        key_topics: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> OperationResult[EpisodicMemory]:
        """
        Record an episode (interaction) for episodic memory.
        
        Args:
            actor: The actor recording the episode
            persona_id: ID of the persona
            summary: Summary of what happened
            conversation_id: Optional conversation ID
            outcome: Optional outcome (success, failure, etc.)
            key_entities: Optional list of key entities
            key_topics: Optional list of key topics
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing the recorded episode
        """
        try:
            user_id = user_id or actor.id
            key = _make_memory_key(persona_id, user_id)
            
            episode = EpisodicMemory(
                persona_id=persona_id,
                user_id=user_id,
                conversation_id=conversation_id,
                summary=summary,
                outcome=outcome,
                key_entities=key_entities or [],
                key_topics=key_topics or [],
            )
            
            if key not in self._episodes:
                self._episodes[key] = []
            
            self._episodes[key].append(episode)
            
            # Trim to max episodes
            if len(self._episodes[key]) > self.MAX_EPISODES:
                self._episodes[key] = self._episodes[key][-self.MAX_EPISODES:]
            
            await self._emit_memory_event(
                persona_id, user_id, "episodic", "record"
            )
            
            return OperationResult.success(episode)
            
        except Exception as e:
            logger.exception(f"Error recording episode for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to record episode: {e}")
    
    async def get_recent_episodes(
        self,
        actor: Actor,
        persona_id: str,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> OperationResult[List[EpisodicMemory]]:
        """
        Get recent episodes for a persona-user pair.
        
        Args:
            actor: The actor requesting episodes
            persona_id: ID of the persona
            limit: Maximum number of episodes to return
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing list of recent episodes
        """
        try:
            user_id = user_id or actor.id
            key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_read(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot read episodes for user {user_id}"
                )
            
            episodes = self._episodes.get(key, [])
            recent = episodes[-limit:] if episodes else []
            
            return OperationResult.success(list(reversed(recent)))
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error getting recent episodes for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to get recent episodes: {e}")
    
    async def search_episodes(
        self,
        actor: Actor,
        persona_id: str,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> OperationResult[List[EpisodicMemory]]:
        """
        Search episodes by keyword (simple text search).
        
        For production, this would use vector similarity search.
        
        Args:
            actor: The actor searching
            persona_id: ID of the persona
            query: Search query
            limit: Maximum results
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing matching episodes
        """
        try:
            user_id = user_id or actor.id
            key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_read(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot search episodes for user {user_id}"
                )
            
            episodes = self._episodes.get(key, [])
            query_lower = query.lower()
            
            # Simple keyword matching (would use embeddings in production)
            matches = []
            for episode in episodes:
                score = 0
                if query_lower in episode.summary.lower():
                    score += 3
                for entity in episode.key_entities:
                    if query_lower in entity.lower():
                        score += 2
                for topic in episode.key_topics:
                    if query_lower in topic.lower():
                        score += 2
                if score > 0:
                    matches.append((score, episode))
            
            # Sort by score descending
            matches.sort(key=lambda x: x[0], reverse=True)
            
            return OperationResult.success([ep for _, ep in matches[:limit]])
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error searching episodes for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to search episodes: {e}")
    
    async def clear_episodes(
        self,
        actor: Actor,
        persona_id: str,
        user_id: Optional[str] = None,
        before_date: Optional[datetime] = None,
    ) -> OperationResult[int]:
        """
        Clear episodic memory.
        
        Args:
            actor: The actor clearing episodes
            persona_id: ID of the persona
            user_id: ID of the user (defaults to actor.id)
            before_date: Optional cutoff date (clear only older episodes)
            
        Returns:
            OperationResult containing number of episodes cleared
        """
        try:
            user_id = user_id or actor.id
            key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_admin(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot clear episodes for user {user_id}"
                )
            
            if key not in self._episodes:
                return OperationResult.success(0)
            
            if before_date:
                original_count = len(self._episodes[key])
                self._episodes[key] = [
                    ep for ep in self._episodes[key]
                    if ep.timestamp >= before_date
                ]
                cleared = original_count - len(self._episodes[key])
            else:
                cleared = len(self._episodes[key])
                del self._episodes[key]
            
            await self._emit_memory_event(
                persona_id, user_id, "episodic", "clear"
            )
            
            return OperationResult.success(cleared)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error clearing episodes for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to clear episodes: {e}")
    
    # =========================================================================
    # Semantic Knowledge
    # =========================================================================
    
    async def get_persona_knowledge(
        self,
        actor: Actor,
        persona_id: str,
        user_id: Optional[str] = None,
    ) -> OperationResult[PersonaKnowledge]:
        """
        Get semantic knowledge for a persona-user pair.
        
        Args:
            actor: The actor requesting knowledge
            persona_id: ID of the persona
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing persona knowledge
        """
        try:
            user_id = user_id or actor.id
            key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_read(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot read knowledge for user {user_id}"
                )
            
            knowledge = self._knowledge.get(key)
            if knowledge is None:
                knowledge = PersonaKnowledge(
                    persona_id=persona_id,
                    user_id=user_id,
                )
                self._knowledge[key] = knowledge
            
            return OperationResult.success(knowledge)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error getting persona knowledge for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to get persona knowledge: {e}")
    
    async def learn_fact(
        self,
        actor: Actor,
        persona_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> OperationResult[LearnedFact]:
        """
        Learn a new fact for a persona-user pair.
        
        Args:
            actor: The actor recording the fact
            persona_id: ID of the persona
            subject: Subject of the fact
            predicate: Predicate (relationship)
            obj: Object of the fact
            confidence: Confidence score (0.0 - 1.0)
            source: Where this fact was learned
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing the learned fact
        """
        try:
            user_id = user_id or actor.id
            key = _make_memory_key(persona_id, user_id)
            
            fact = LearnedFact(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                source=source or "interaction",
            )
            
            knowledge = self._knowledge.get(key)
            if knowledge is None:
                knowledge = PersonaKnowledge(
                    persona_id=persona_id,
                    user_id=user_id,
                )
                self._knowledge[key] = knowledge
            
            # Check for existing fact with same subject/predicate
            existing = next(
                (f for f in knowledge.learned_facts
                 if f.subject == subject and f.predicate == predicate),
                None
            )
            
            if existing:
                # Update existing fact
                existing.object = obj
                existing.confidence = confidence
                existing.last_verified = _now_utc()
                existing.verification_count += 1
                fact = existing
            else:
                knowledge.learned_facts.append(fact)
            
            # Trim to max facts
            if len(knowledge.learned_facts) > self.MAX_FACTS:
                # Remove oldest, lowest confidence facts
                knowledge.learned_facts.sort(
                    key=lambda f: (f.confidence, f.learned_at),
                    reverse=True
                )
                knowledge.learned_facts = knowledge.learned_facts[:self.MAX_FACTS]
            
            knowledge.last_updated = _now_utc()
            
            await self._emit_memory_event(
                persona_id, user_id, "semantic", "learn"
            )
            
            return OperationResult.success(fact)
            
        except Exception as e:
            logger.exception(f"Error learning fact for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to learn fact: {e}")
    
    async def query_facts(
        self,
        actor: Actor,
        persona_id: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        min_confidence: float = 0.0,
        user_id: Optional[str] = None,
    ) -> OperationResult[List[LearnedFact]]:
        """
        Query learned facts.
        
        Args:
            actor: The actor querying
            persona_id: ID of the persona
            subject: Optional subject filter
            predicate: Optional predicate filter
            min_confidence: Minimum confidence threshold
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing matching facts
        """
        try:
            user_id = user_id or actor.id
            key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_read(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot query facts for user {user_id}"
                )
            
            knowledge = self._knowledge.get(key)
            if knowledge is None:
                return OperationResult.success([])
            
            facts = knowledge.learned_facts
            
            if subject:
                facts = [f for f in facts if f.subject.lower() == subject.lower()]
            if predicate:
                facts = [f for f in facts if f.predicate.lower() == predicate.lower()]
            if min_confidence > 0:
                facts = [f for f in facts if f.confidence >= min_confidence]
            
            return OperationResult.success(facts)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error querying facts for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to query facts: {e}")
    
    async def set_user_preference(
        self,
        actor: Actor,
        persona_id: str,
        preference_key: str,
        preference_value: Any,
        user_id: Optional[str] = None,
    ) -> OperationResult[PersonaKnowledge]:
        """
        Set a user preference for a persona-user pair.
        
        Args:
            actor: The actor setting the preference
            persona_id: ID of the persona
            preference_key: Preference key
            preference_value: Preference value
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing updated knowledge
        """
        try:
            user_id = user_id or actor.id
            key = _make_memory_key(persona_id, user_id)
            
            if actor.id != user_id and not self._permission_checker.can_write(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot set preferences for user {user_id}"
                )
            
            knowledge = self._knowledge.get(key)
            if knowledge is None:
                knowledge = PersonaKnowledge(
                    persona_id=persona_id,
                    user_id=user_id,
                )
                self._knowledge[key] = knowledge
            
            knowledge.user_preferences[preference_key] = preference_value
            knowledge.last_updated = _now_utc()
            
            return OperationResult.success(knowledge)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error setting preference for {persona_id}/{user_id}")
            return OperationResult.failure(f"Failed to set user preference: {e}")
    
    async def get_user_preferences(
        self,
        actor: Actor,
        persona_id: str,
        user_id: Optional[str] = None,
    ) -> OperationResult[Dict[str, Any]]:
        """
        Get all user preferences for a persona-user pair.
        
        Args:
            actor: The actor requesting preferences
            persona_id: ID of the persona
            user_id: ID of the user (defaults to actor.id)
            
        Returns:
            OperationResult containing preferences dict
        """
        try:
            result = await self.get_persona_knowledge(actor, persona_id, user_id)
            if not result.is_success:
                return result
            
            return OperationResult.success(result.value.user_preferences)
            
        except Exception as e:
            logger.exception(f"Error getting preferences for {persona_id}")
            return OperationResult.failure(f"Failed to get user preferences: {e}")
    
    # =========================================================================
    # Memory Injection
    # =========================================================================
    
    async def get_context_injection(
        self,
        actor: Actor,
        persona_id: str,
        user_id: Optional[str] = None,
        include_goals: bool = True,
        include_task: bool = True,
        include_recent_episodes: int = 3,
        include_preferences: bool = True,
    ) -> OperationResult[Dict[str, Any]]:
        """
        Get memory context for injection into agent prompts.
        
        Args:
            actor: The actor requesting context
            persona_id: ID of the persona
            user_id: ID of the user (defaults to actor.id)
            include_goals: Include active goals
            include_task: Include current task context
            include_recent_episodes: Number of recent episodes to include
            include_preferences: Include user preferences
            
        Returns:
            OperationResult containing context dict for injection
        """
        try:
            user_id = user_id or actor.id
            
            context: Dict[str, Any] = {}
            
            # Get working memory
            mem_result = await self.get_working_memory(actor, persona_id, user_id)
            if mem_result.is_success:
                memory = mem_result.value
                if include_goals and memory.active_goals:
                    context["active_goals"] = memory.active_goals
                if include_task and memory.task_context:
                    context["current_task"] = memory.task_context
                if memory.scratchpad:
                    context["scratchpad"] = memory.scratchpad
            
            # Get recent episodes
            if include_recent_episodes > 0:
                ep_result = await self.get_recent_episodes(
                    actor, persona_id, include_recent_episodes, user_id
                )
                if ep_result.is_success and ep_result.value:
                    context["recent_interactions"] = [
                        {"summary": ep.summary, "outcome": ep.outcome}
                        for ep in ep_result.value
                    ]
            
            # Get preferences
            if include_preferences:
                pref_result = await self.get_user_preferences(actor, persona_id, user_id)
                if pref_result.is_success and pref_result.value:
                    context["user_preferences"] = pref_result.value
            
            return OperationResult.success(context)
            
        except Exception as e:
            logger.exception(f"Error getting context injection for {persona_id}")
            return OperationResult.failure(f"Failed to get context injection: {e}")
    
    # =========================================================================
    # Event Emission
    # =========================================================================
    
    async def _emit_memory_event(
        self,
        persona_id: str,
        user_id: str,
        memory_type: str,
        operation: str,
        key: Optional[str] = None,
    ) -> None:
        """Emit a memory update event."""
        if self._message_bus:
            try:
                event = PersonaMemoryUpdated(
                    persona_id=persona_id,
                    user_id=user_id,
                    memory_type=memory_type,
                    operation=operation,
                    key=key,
                )
                await self._message_bus.publish(event.event_type, event.to_dict())
            except Exception as e:
                logger.warning(f"Failed to emit memory event: {e}")
