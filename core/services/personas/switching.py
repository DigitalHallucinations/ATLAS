"""
Persona switching service for ATLAS.

Provides handoff protocols, conditional switching, and
context preservation when transitioning between personas.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from core.services.common import Actor, OperationResult
from core.services.common.exceptions import PermissionDeniedError

from .exceptions import PersonaError, PersonaNotFoundError
from .permissions import PersonaPermissionChecker
from .types import (
    # Switching types
    PersonaHandoff,
    HandoffContext,
    HandoffStatus,
    SwitchingRule,
    SwitchTriggerType,
    FallbackChain,
    # Events
    PersonaHandoffInitiated,
    PersonaHandoffCompleted,
)

if TYPE_CHECKING:
    from core.config import ConfigManager
    from core.messaging import MessageBus
    from .memory import PersonaMemoryService


logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class PersonaSwitchingError(PersonaError):
    """Error related to persona switching operations."""
    pass


class HandoffFailedError(PersonaSwitchingError):
    """Raised when a handoff fails."""
    
    def __init__(self, message: str, handoff: PersonaHandoff):
        super().__init__(message)
        self.handoff = handoff


class PersonaSwitchingService:
    """
    Service for managing persona transitions and handoffs.
    
    Provides:
    - Handoff protocol for context transfer
    - Conditional switching based on rules
    - Fallback chains for graceful degradation
    - Context preservation across switches
    """
    
    # Default timeout for handoffs
    DEFAULT_HANDOFF_TIMEOUT_SECONDS = 300
    
    # Maximum rules per persona
    MAX_RULES_PER_PERSONA = 50
    
    def __init__(
        self,
        config_manager: Optional["ConfigManager"] = None,
        message_bus: Optional["MessageBus"] = None,
        permission_checker: Optional[PersonaPermissionChecker] = None,
        memory_service: Optional["PersonaMemoryService"] = None,
    ) -> None:
        """
        Initialize the PersonaSwitchingService.
        
        Args:
            config_manager: Configuration manager
            message_bus: Message bus for publishing events
            permission_checker: Permission checker for authorization
            memory_service: Memory service for context retrieval
        """
        self._config_manager = config_manager
        self._message_bus = message_bus
        self._permission_checker = permission_checker or PersonaPermissionChecker()
        self._memory_service = memory_service
        
        # In-memory storage (would be persisted in production)
        self._handoffs: Dict[str, PersonaHandoff] = {}
        self._switching_rules: Dict[str, List[SwitchingRule]] = {}  # persona_id -> rules
        self._fallback_chains: Dict[str, FallbackChain] = {}  # chain_id -> chain
        self._active_persona: Dict[str, str] = {}  # user_id -> current_persona_id
    
    # =========================================================================
    # Handoff Protocol
    # =========================================================================
    
    async def initiate_handoff(
        self,
        actor: Actor,
        from_persona: str,
        to_persona: str,
        reason: str,
        trigger_type: str = SwitchTriggerType.EXPLICIT,
        context: Optional[HandoffContext] = None,
        conversation_id: Optional[str] = None,
    ) -> OperationResult[PersonaHandoff]:
        """
        Initiate a handoff from one persona to another.
        
        Args:
            actor: The actor initiating the handoff
            from_persona: ID of the source persona
            to_persona: ID of the target persona
            reason: Reason for the handoff
            trigger_type: What triggered the handoff
            context: Optional pre-built context
            conversation_id: Optional conversation ID
            
        Returns:
            OperationResult containing the handoff record
        """
        try:
            # Build context if not provided
            if context is None:
                context = await self._build_handoff_context(actor, from_persona)
            
            handoff = PersonaHandoff(
                from_persona=from_persona,
                to_persona=to_persona,
                user_id=actor.id,
                conversation_id=conversation_id,
                context=context,
                trigger_type=trigger_type,
                reason=reason,
                status=HandoffStatus.PENDING,
            )
            
            self._handoffs[handoff.handoff_id] = handoff
            
            # Emit event
            await self._emit_handoff_initiated(handoff)
            
            logger.info(
                f"Handoff initiated: {from_persona} -> {to_persona} "
                f"(user: {actor.id}, reason: {reason})"
            )
            
            return OperationResult.success(handoff)
            
        except Exception as e:
            logger.exception(f"Error initiating handoff from {from_persona} to {to_persona}")
            return OperationResult.failure(f"Failed to initiate handoff: {e}")
    
    async def accept_handoff(
        self,
        handoff_id: str,
    ) -> OperationResult[PersonaHandoff]:
        """
        Accept a pending handoff.
        
        Args:
            handoff_id: ID of the handoff to accept
            
        Returns:
            OperationResult containing the updated handoff
        """
        try:
            if handoff_id not in self._handoffs:
                return OperationResult.failure(f"Handoff {handoff_id} not found")
            
            handoff = self._handoffs[handoff_id]
            
            if handoff.status != HandoffStatus.PENDING:
                return OperationResult.failure(
                    f"Handoff is not pending (status: {handoff.status})"
                )
            
            handoff.status = HandoffStatus.ACCEPTED
            handoff.accepted_at = _now_utc()
            
            logger.info(f"Handoff {handoff_id} accepted")
            return OperationResult.success(handoff)
            
        except Exception as e:
            logger.exception(f"Error accepting handoff {handoff_id}")
            return OperationResult.failure(f"Failed to accept handoff: {e}")
    
    async def complete_handoff(
        self,
        handoff_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> OperationResult[PersonaHandoff]:
        """
        Complete a handoff.
        
        Args:
            handoff_id: ID of the handoff to complete
            success: Whether the handoff was successful
            error_message: Error message if failed
            
        Returns:
            OperationResult containing the completed handoff
        """
        try:
            if handoff_id not in self._handoffs:
                return OperationResult.failure(f"Handoff {handoff_id} not found")
            
            handoff = self._handoffs[handoff_id]
            
            if handoff.status not in (HandoffStatus.PENDING, HandoffStatus.ACCEPTED, HandoffStatus.IN_PROGRESS):
                return OperationResult.failure(
                    f"Handoff cannot be completed (status: {handoff.status})"
                )
            
            handoff.status = HandoffStatus.COMPLETED if success else HandoffStatus.FAILED
            handoff.completed_at = _now_utc()
            handoff.error_message = error_message
            
            # Update active persona tracking
            if success:
                self._active_persona[handoff.user_id] = handoff.to_persona
            
            # Calculate duration
            duration_ms = 0
            if handoff.initiated_at:
                delta = handoff.completed_at - handoff.initiated_at
                duration_ms = int(delta.total_seconds() * 1000)
            
            # Emit event
            await self._emit_handoff_completed(handoff, duration_ms)
            
            logger.info(f"Handoff {handoff_id} completed (success: {success})")
            return OperationResult.success(handoff)
            
        except Exception as e:
            logger.exception(f"Error completing handoff {handoff_id}")
            return OperationResult.failure(f"Failed to complete handoff: {e}")
    
    async def cancel_handoff(
        self,
        actor: Actor,
        handoff_id: str,
        reason: str = "",
    ) -> OperationResult[PersonaHandoff]:
        """
        Cancel a pending or in-progress handoff.
        
        Args:
            actor: The actor cancelling the handoff
            handoff_id: ID of the handoff to cancel
            reason: Reason for cancellation
            
        Returns:
            OperationResult containing the cancelled handoff
        """
        try:
            if handoff_id not in self._handoffs:
                return OperationResult.failure(f"Handoff {handoff_id} not found")
            
            handoff = self._handoffs[handoff_id]
            
            # Only the user who initiated or an admin can cancel
            if handoff.user_id != actor.id and not self._permission_checker.can_admin(actor, handoff.from_persona):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot cancel handoff {handoff_id}"
                )
            
            if handoff.status in (HandoffStatus.COMPLETED, HandoffStatus.FAILED, HandoffStatus.CANCELLED):
                return OperationResult.failure(
                    f"Handoff is already finalized (status: {handoff.status})"
                )
            
            handoff.status = HandoffStatus.CANCELLED
            handoff.completed_at = _now_utc()
            handoff.error_message = reason or "Cancelled by user"
            
            logger.info(f"Handoff {handoff_id} cancelled")
            return OperationResult.success(handoff)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error cancelling handoff {handoff_id}")
            return OperationResult.failure(f"Failed to cancel handoff: {e}")
    
    async def get_handoff(
        self,
        handoff_id: str,
    ) -> OperationResult[PersonaHandoff]:
        """
        Get a handoff by ID.
        
        Args:
            handoff_id: ID of the handoff
            
        Returns:
            OperationResult containing the handoff
        """
        try:
            if handoff_id not in self._handoffs:
                return OperationResult.failure(f"Handoff {handoff_id} not found")
            
            return OperationResult.success(self._handoffs[handoff_id])
            
        except Exception as e:
            logger.exception(f"Error getting handoff {handoff_id}")
            return OperationResult.failure(f"Failed to get handoff: {e}")
    
    async def get_pending_handoffs(
        self,
        user_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> OperationResult[List[PersonaHandoff]]:
        """
        Get all pending handoffs, optionally filtered.
        
        Args:
            user_id: Optional user filter
            persona_id: Optional persona filter (from or to)
            
        Returns:
            OperationResult containing list of pending handoffs
        """
        try:
            handoffs = [
                h for h in self._handoffs.values()
                if h.status in (HandoffStatus.PENDING, HandoffStatus.ACCEPTED, HandoffStatus.IN_PROGRESS)
            ]
            
            if user_id:
                handoffs = [h for h in handoffs if h.user_id == user_id]
            
            if persona_id:
                handoffs = [
                    h for h in handoffs
                    if h.from_persona == persona_id or h.to_persona == persona_id
                ]
            
            return OperationResult.success(handoffs)
            
        except Exception as e:
            logger.exception("Error getting pending handoffs")
            return OperationResult.failure(f"Failed to get pending handoffs: {e}")
    
    async def _build_handoff_context(
        self,
        actor: Actor,
        persona_id: str,
    ) -> HandoffContext:
        """Build handoff context from memory service."""
        context = HandoffContext()
        
        if self._memory_service:
            try:
                # Get working memory
                mem_result = await self._memory_service.get_working_memory(
                    actor, persona_id, actor.id
                )
                if mem_result.is_success:
                    memory = mem_result.value
                    context.active_goals = memory.active_goals
                    context.working_memory_snapshot = memory.scratchpad
                    context.pending_actions = memory.pending_actions
                    if memory.conversation_summary:
                        context.conversation_summary = memory.conversation_summary
                
                # Get user preferences
                pref_result = await self._memory_service.get_user_preferences(
                    actor, persona_id, actor.id
                )
                if pref_result.is_success:
                    context.user_preferences = pref_result.value
                    
            except Exception as e:
                logger.warning(f"Error building handoff context: {e}")
        
        return context
    
    # =========================================================================
    # Switching Rules
    # =========================================================================
    
    async def add_switching_rule(
        self,
        actor: Actor,
        rule: SwitchingRule,
    ) -> OperationResult[SwitchingRule]:
        """
        Add a switching rule.
        
        Args:
            actor: The actor adding the rule
            rule: The switching rule to add
            
        Returns:
            OperationResult containing the added rule
        """
        try:
            persona_id = rule.from_persona or "_global"
            
            if persona_id != "_global" and not self._permission_checker.can_admin(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot add switching rules for persona {persona_id}"
                )
            
            if persona_id not in self._switching_rules:
                self._switching_rules[persona_id] = []
            
            if len(self._switching_rules[persona_id]) >= self.MAX_RULES_PER_PERSONA:
                return OperationResult.failure(
                    f"Maximum rules ({self.MAX_RULES_PER_PERSONA}) reached for persona {persona_id}"
                )
            
            self._switching_rules[persona_id].append(rule)
            
            # Sort by priority
            self._switching_rules[persona_id].sort(key=lambda r: r.priority)
            
            logger.info(f"Added switching rule {rule.rule_id} for persona {persona_id}")
            return OperationResult.success(rule)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception("Error adding switching rule")
            return OperationResult.failure(f"Failed to add switching rule: {e}")
    
    async def remove_switching_rule(
        self,
        actor: Actor,
        rule_id: str,
    ) -> OperationResult[None]:
        """
        Remove a switching rule.
        
        Args:
            actor: The actor removing the rule
            rule_id: ID of the rule to remove
            
        Returns:
            OperationResult indicating success/failure
        """
        try:
            # Find the rule
            for persona_id, rules in self._switching_rules.items():
                for i, rule in enumerate(rules):
                    if rule.rule_id == rule_id:
                        if persona_id != "_global" and not self._permission_checker.can_admin(actor, persona_id):
                            raise PermissionDeniedError(
                                f"Actor {actor.id} cannot remove switching rules for persona {persona_id}"
                            )
                        
                        del rules[i]
                        logger.info(f"Removed switching rule {rule_id}")
                        return OperationResult.success(None)
            
            return OperationResult.failure(f"Rule {rule_id} not found")
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error removing switching rule {rule_id}")
            return OperationResult.failure(f"Failed to remove switching rule: {e}")
    
    async def get_switching_rules(
        self,
        actor: Actor,
        persona_id: Optional[str] = None,
    ) -> OperationResult[List[SwitchingRule]]:
        """
        Get switching rules.
        
        Args:
            actor: The actor requesting rules
            persona_id: Optional persona filter
            
        Returns:
            OperationResult containing list of rules
        """
        try:
            rules = []
            
            # Global rules
            if "_global" in self._switching_rules:
                rules.extend(self._switching_rules["_global"])
            
            # Persona-specific rules
            if persona_id and persona_id in self._switching_rules:
                if self._permission_checker.can_read(actor, persona_id):
                    rules.extend(self._switching_rules[persona_id])
            elif persona_id is None:
                # Get all rules actor can read
                for pid, prules in self._switching_rules.items():
                    if pid == "_global" or self._permission_checker.can_read(actor, pid):
                        rules.extend(prules)
            
            # Sort by priority
            rules.sort(key=lambda r: r.priority)
            
            return OperationResult.success(rules)
            
        except Exception as e:
            logger.exception("Error getting switching rules")
            return OperationResult.failure(f"Failed to get switching rules: {e}")
    
    async def evaluate_switching_rules(
        self,
        current_persona: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OperationResult[Optional[SwitchingRule]]:
        """
        Evaluate switching rules to determine if a switch should occur.
        
        Args:
            current_persona: ID of the current persona
            user_input: The user's input
            context: Optional additional context
            
        Returns:
            OperationResult containing matching rule or None
        """
        try:
            context = context or {}
            
            # Get applicable rules
            rules = []
            if "_global" in self._switching_rules:
                rules.extend(self._switching_rules["_global"])
            if current_persona in self._switching_rules:
                rules.extend(self._switching_rules[current_persona])
            
            # Sort by priority (lower = higher priority)
            rules.sort(key=lambda r: r.priority)
            
            for rule in rules:
                if not rule.enabled:
                    continue
                
                # Check from_persona filter
                if rule.from_persona and rule.from_persona != current_persona:
                    continue
                
                # Check trigger pattern
                if rule.trigger_pattern:
                    try:
                        if not re.search(rule.trigger_pattern, user_input, re.IGNORECASE):
                            continue
                    except re.error:
                        logger.warning(f"Invalid regex in rule {rule.rule_id}: {rule.trigger_pattern}")
                        continue
                
                # Check additional conditions
                if not self._evaluate_conditions(rule.conditions, context):
                    continue
                
                # Rule matches
                return OperationResult.success(rule)
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception("Error evaluating switching rules")
            return OperationResult.failure(f"Failed to evaluate switching rules: {e}")
    
    def _evaluate_conditions(
        self,
        conditions: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> bool:
        """Evaluate additional rule conditions."""
        for condition in conditions:
            condition_type = condition.get("type")
            
            if condition_type == "context_contains":
                key = condition.get("key")
                value = condition.get("value")
                if key not in context or context[key] != value:
                    return False
            
            elif condition_type == "context_exists":
                key = condition.get("key")
                if key not in context:
                    return False
            
            elif condition_type == "time_of_day":
                # Check if current time is within range
                start = condition.get("start_hour", 0)
                end = condition.get("end_hour", 24)
                current_hour = _now_utc().hour
                if not (start <= current_hour < end):
                    return False
        
        return True
    
    async def suggest_persona_switch(
        self,
        current_persona: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OperationResult[Optional[Dict[str, Any]]]:
        """
        Suggest a persona switch based on input analysis.
        
        Args:
            current_persona: ID of the current persona
            user_input: The user's input
            context: Optional additional context
            
        Returns:
            OperationResult containing switch suggestion or None
        """
        try:
            # First check explicit rules
            rule_result = await self.evaluate_switching_rules(
                current_persona, user_input, context
            )
            if rule_result.is_success and rule_result.value:
                rule = rule_result.value
                return OperationResult.success({
                    "suggested_persona": rule.to_persona,
                    "reason": f"Matched rule: {rule.name}",
                    "trigger_type": SwitchTriggerType.RULE_BASED,
                    "require_confirmation": rule.require_confirmation,
                    "preserve_context": rule.preserve_context,
                })
            
            # In production, this could use ML to detect domain/intent
            # and suggest appropriate persona switches
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception("Error suggesting persona switch")
            return OperationResult.failure(f"Failed to suggest persona switch: {e}")
    
    # =========================================================================
    # Fallback Chains
    # =========================================================================
    
    async def set_fallback_chain(
        self,
        actor: Actor,
        chain: FallbackChain,
    ) -> OperationResult[FallbackChain]:
        """
        Set a fallback chain for a persona.
        
        Args:
            actor: The actor setting the chain
            chain: The fallback chain
            
        Returns:
            OperationResult containing the chain
        """
        try:
            if not self._permission_checker.can_admin(actor, chain.primary_persona):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot set fallback chain for persona {chain.primary_persona}"
                )
            
            self._fallback_chains[chain.chain_id] = chain
            
            logger.info(f"Set fallback chain {chain.chain_id} for persona {chain.primary_persona}")
            return OperationResult.success(chain)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception("Error setting fallback chain")
            return OperationResult.failure(f"Failed to set fallback chain: {e}")
    
    async def get_fallback_chain(
        self,
        primary_persona: str,
    ) -> OperationResult[Optional[FallbackChain]]:
        """
        Get the fallback chain for a persona.
        
        Args:
            primary_persona: ID of the primary persona
            
        Returns:
            OperationResult containing the chain or None
        """
        try:
            for chain in self._fallback_chains.values():
                if chain.primary_persona == primary_persona and chain.enabled:
                    return OperationResult.success(chain)
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Error getting fallback chain for {primary_persona}")
            return OperationResult.failure(f"Failed to get fallback chain: {e}")
    
    async def get_next_fallback(
        self,
        primary_persona: str,
        failed_personas: List[str],
        trigger: str = "error",
    ) -> OperationResult[Optional[str]]:
        """
        Get the next persona in the fallback chain.
        
        Args:
            primary_persona: ID of the primary persona
            failed_personas: List of personas that have already failed
            trigger: What triggered the fallback
            
        Returns:
            OperationResult containing next persona ID or None
        """
        try:
            chain_result = await self.get_fallback_chain(primary_persona)
            if not chain_result.is_success or not chain_result.value:
                return OperationResult.success(None)
            
            chain = chain_result.value
            
            # Check trigger conditions
            if trigger == "error" and not chain.trigger_on_error:
                return OperationResult.success(None)
            if trigger == "timeout" and not chain.trigger_on_timeout:
                return OperationResult.success(None)
            if trigger == "capability_gap" and not chain.trigger_on_capability_gap:
                return OperationResult.success(None)
            
            # Check max fallbacks
            if len(failed_personas) >= chain.max_fallbacks:
                logger.warning(f"Max fallbacks reached for chain {chain.chain_id}")
                return OperationResult.success(None)
            
            # Find next available fallback
            for fallback_persona in chain.fallback_personas:
                if fallback_persona not in failed_personas:
                    return OperationResult.success(fallback_persona)
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Error getting next fallback for {primary_persona}")
            return OperationResult.failure(f"Failed to get next fallback: {e}")
    
    # =========================================================================
    # Active Persona Tracking
    # =========================================================================
    
    async def get_active_persona(
        self,
        user_id: str,
    ) -> OperationResult[Optional[str]]:
        """
        Get the currently active persona for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            OperationResult containing persona ID or None
        """
        try:
            return OperationResult.success(self._active_persona.get(user_id))
        except Exception as e:
            logger.exception(f"Error getting active persona for user {user_id}")
            return OperationResult.failure(f"Failed to get active persona: {e}")
    
    async def set_active_persona(
        self,
        user_id: str,
        persona_id: str,
    ) -> OperationResult[None]:
        """
        Set the currently active persona for a user.
        
        Args:
            user_id: ID of the user
            persona_id: ID of the persona
            
        Returns:
            OperationResult indicating success/failure
        """
        try:
            self._active_persona[user_id] = persona_id
            return OperationResult.success(None)
        except Exception as e:
            logger.exception(f"Error setting active persona for user {user_id}")
            return OperationResult.failure(f"Failed to set active persona: {e}")
    
    # =========================================================================
    # Context Transfer
    # =========================================================================
    
    async def transfer_context(
        self,
        actor: Actor,
        from_persona: str,
        to_persona: str,
        context: HandoffContext,
    ) -> OperationResult[None]:
        """
        Transfer context from one persona to another.
        
        Args:
            actor: The actor performing the transfer
            from_persona: Source persona
            to_persona: Target persona
            context: Context to transfer
            
        Returns:
            OperationResult indicating success/failure
        """
        try:
            if not self._memory_service:
                logger.warning("No memory service configured, context transfer skipped")
                return OperationResult.success(None)
            
            # Transfer working memory
            for key, value in context.working_memory_snapshot.items():
                await self._memory_service.set_working_memory(
                    actor, to_persona, key, value, actor.id
                )
            
            # Transfer goals
            if context.active_goals:
                await self._memory_service.set_active_goals(
                    actor, to_persona, context.active_goals, actor.id
                )
            
            # Transfer user preferences
            for key, value in context.user_preferences.items():
                await self._memory_service.set_user_preference(
                    actor, to_persona, key, value, actor.id
                )
            
            logger.info(f"Context transferred from {from_persona} to {to_persona}")
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Error transferring context from {from_persona} to {to_persona}")
            return OperationResult.failure(f"Failed to transfer context: {e}")
    
    # =========================================================================
    # Event Emission
    # =========================================================================
    
    async def _emit_handoff_initiated(self, handoff: PersonaHandoff) -> None:
        """Emit handoff initiated event."""
        if self._message_bus:
            try:
                event = PersonaHandoffInitiated(
                    handoff_id=handoff.handoff_id,
                    from_persona=handoff.from_persona,
                    to_persona=handoff.to_persona,
                    user_id=handoff.user_id,
                    reason=handoff.reason,
                    trigger_type=handoff.trigger_type,
                )
                await self._message_bus.publish(event.event_type, event.to_dict())
            except Exception as e:
                logger.warning(f"Failed to emit handoff initiated event: {e}")
    
    async def _emit_handoff_completed(
        self,
        handoff: PersonaHandoff,
        duration_ms: int,
    ) -> None:
        """Emit handoff completed event."""
        if self._message_bus:
            try:
                event = PersonaHandoffCompleted(
                    handoff_id=handoff.handoff_id,
                    from_persona=handoff.from_persona,
                    to_persona=handoff.to_persona,
                    user_id=handoff.user_id,
                    success=handoff.status == HandoffStatus.COMPLETED,
                    duration_ms=duration_ms,
                )
                await self._message_bus.publish(event.event_type, event.to_dict())
            except Exception as e:
                logger.warning(f"Failed to emit handoff completed event: {e}")
