"""
Integration adapter for DomainEvents and ATLAS messaging system.

Provides utilities to publish DomainEvents through the AgentBus
and subscribe to events using the existing channel infrastructure.

Author: ATLAS Team
Date: Jan 7, 2026
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from core.messaging.agent_bus import AgentBus
from core.messaging.channels import (
    BLACKBOARD_EVENT,
    CONVERSATION_END,
    CONVERSATION_MESSAGE,
    SKILL_ACTIVITY,
    TASK_COMPLETE,
    TASK_CREATE,
    TASK_UPDATE,
)
from core.messaging.messages import AgentMessage

from .protocols import EventPublisher, EventSubscriber
from .types import DomainEvent


logger = logging.getLogger(__name__)


# Domain event to channel mapping
# Services can publish to these channels using standard event types
DOMAIN_EVENT_CHANNELS: Dict[str, str] = {
    # Conversation events  
    "conversation.created": CONVERSATION_MESSAGE.name,
    "conversation.updated": CONVERSATION_MESSAGE.name,
    "conversation.deleted": CONVERSATION_END.name,
    "conversation.message.added": CONVERSATION_MESSAGE.name,
    
    # Task events
    "task.created": TASK_CREATE.name,
    "task.updated": TASK_UPDATE.name,
    "task.completed": TASK_COMPLETE.name,
    "task.failed": TASK_COMPLETE.name,
    
    # Blackboard/context events  
    "blackboard.created": BLACKBOARD_EVENT.name,
    "blackboard.updated": BLACKBOARD_EVENT.name,
    "blackboard.deleted": BLACKBOARD_EVENT.name,
    
    # Skill events
    "skill.started": SKILL_ACTIVITY.name,
    "skill.completed": SKILL_ACTIVITY.name,
    "skill.failed": SKILL_ACTIVITY.name,
    
    # Budget events
    "budget.policy_created": BLACKBOARD_EVENT.name,
    "budget.policy_updated": BLACKBOARD_EVENT.name,
    "budget.policy_deleted": BLACKBOARD_EVENT.name,
    "budget.check_requested": BLACKBOARD_EVENT.name,
    "budget.usage_recorded": BLACKBOARD_EVENT.name,
    "budget.threshold_reached": BLACKBOARD_EVENT.name,
    "budget.alert_triggered": BLACKBOARD_EVENT.name,
    "budget.alert_acknowledged": BLACKBOARD_EVENT.name,
    "budget.limit_exceeded": BLACKBOARD_EVENT.name,
    "budget.approaching_limit": BLACKBOARD_EVENT.name,
    
    # Generic service events (default to blackboard channel)
    "entity.created": BLACKBOARD_EVENT.name,
    "entity.updated": BLACKBOARD_EVENT.name,
    "entity.deleted": BLACKBOARD_EVENT.name,
}


class DomainEventPublisher(EventPublisher):
    """
    Publisher that adapts DomainEvents to the ATLAS AgentBus.
    
    Maps domain events to appropriate channels and converts them
    to AgentMessages for transport through the existing messaging infrastructure.
    
    Example:
        publisher = DomainEventPublisher(agent_bus)
        
        event = DomainEvent.create(
            event_type="conversation.created",
            entity_id=conversation_id,
            tenant_id="org_123",
            actor="user"
        )
        
        await publisher.publish(event)
    """
    
    def __init__(self, agent_bus: AgentBus) -> None:
        self._bus = agent_bus
    
    async def publish(self, event: DomainEvent) -> None:
        """
        Publish a single domain event.
        
        Args:
            event: Domain event to publish
        """
        try:
            channel = self._get_channel_for_event(event.event_type)
            message = self._event_to_message(event, channel)
            
            logger.debug(
                f"Publishing domain event {event.event_type} to channel {channel}",
                extra={
                    "event_type": event.event_type,
                    "entity_id": str(event.entity_id),
                    "tenant_id": event.tenant_id,
                    "channel": channel,
                }
            )
            
            await self._bus.publish(message)
            
        except Exception as e:
            logger.error(
                f"Failed to publish domain event {event.event_type}: {e}",
                extra={
                    "event_type": event.event_type,
                    "entity_id": str(event.entity_id),
                    "error": str(e),
                },
                exc_info=True
            )
            raise
    
    async def publish_many(self, events: List[DomainEvent]) -> None:
        """
        Publish multiple domain events.
        
        Args:
            events: List of events to publish
        """
        messages = []
        
        for event in events:
            try:
                channel = self._get_channel_for_event(event.event_type)
                message = self._event_to_message(event, channel)
                messages.append(message)
            except Exception as e:
                logger.error(
                    f"Failed to convert event {event.event_type} to message: {e}",
                    extra={"event_type": event.event_type, "error": str(e)}
                )
                # Continue with other events
        
        if messages:
            logger.debug(f"Publishing {len(messages)} domain events")
            await self._bus.publish_many(messages)
    
    def _get_channel_for_event(self, event_type: str) -> str:
        """
        Map event type to appropriate channel.
        
        Args:
            event_type: Domain event type (e.g., "conversation.created")
            
        Returns:
            Channel name to publish to
        """
        # Direct mapping
        if event_type in DOMAIN_EVENT_CHANNELS:
            return DOMAIN_EVENT_CHANNELS[event_type]
        
        # Pattern matching for unknown event types
        if event_type.startswith("conversation."):
            return CONVERSATION_MESSAGE.name
        elif event_type.startswith("task."):
            return TASK_CREATE.name  # Default task channel
        elif event_type.startswith("skill."):
            return SKILL_ACTIVITY.name
        elif event_type.startswith("blackboard."):
            return BLACKBOARD_EVENT.name
        
        # Default fallback
        logger.warning(f"No channel mapping for event type {event_type}, using blackboard.event")
        return BLACKBOARD_EVENT.name
    
    def _event_to_message(self, event: DomainEvent, channel: str) -> AgentMessage:
        """
        Convert DomainEvent to AgentMessage.
        
        Args:
            event: Domain event to convert
            channel: Target channel name
            
        Returns:
            AgentMessage ready for transport
        """
        return AgentMessage(
            channel=channel,
            payload=event.to_dict(),
            # Map event context to message fields
            agent_id=f"{event.actor}_{event.entity_id}",  # Synthetic agent ID
            conversation_id=str(event.entity_id) if "conversation" in event.event_type else None,
            user_id=event.tenant_id,  # Map tenant to user for routing
            trace_id=None,  # Could be added to DomainEvent if needed
            headers={
                "domain_event": "true",
                "event_type": event.event_type,
                "tenant_id": event.tenant_id,
                "actor": event.actor,
            }
        )


class DomainEventSubscriber(EventSubscriber):
    """
    Base class for services that want to subscribe to domain events.
    
    Handles the conversion from AgentMessages back to DomainEvents
    and provides filtering by event type.
    
    Example:
        class ConversationEventHandler(DomainEventSubscriber):
            def get_subscribed_event_types(self) -> List[str]:
                return ["conversation.created", "conversation.updated"]
                
            async def handle_event(self, event: DomainEvent) -> None:
                if event.event_type == "conversation.created":
                    await self._handle_conversation_created(event)
    """
    
    def __init__(self) -> None:
        self._subscribed_types = set(self.get_subscribed_event_types())
    
    async def handle_agent_message(self, message: AgentMessage) -> None:
        """
        Handle an AgentMessage and convert to DomainEvent if applicable.
        
        Args:
            message: AgentMessage from the bus
        """
        # Check if this is a domain event
        if not message.headers.get("domain_event") == "true":
            return
        
        try:
            # Convert message payload back to DomainEvent
            event = DomainEvent.from_dict(message.payload)
            
            # Filter by subscribed event types
            if self._subscribed_types and event.event_type not in self._subscribed_types:
                return
            
            await self.handle_event(event)
            
        except Exception as e:
            logger.error(
                f"Error handling domain event message: {e}",
                extra={
                    "message_id": message.id,
                    "channel": message.channel,
                    "error": str(e),
                },
                exc_info=True
            )
    
    async def handle_event(self, event: DomainEvent) -> None:
        """
        Handle a domain event.
        
        Subclasses should override this method.
        
        Args:
            event: Domain event to handle
        """
        raise NotImplementedError("Subclasses must implement handle_event")
    
    def get_subscribed_event_types(self) -> List[str]:
        """
        Get list of event types this subscriber handles.
        
        Subclasses should override this method.
        
        Returns:
            List of event type strings
        """
        raise NotImplementedError("Subclasses must implement get_subscribed_event_types")


def register_domain_event_channel(event_type: str, channel_name: str) -> None:
    """
    Register a custom mapping from event type to channel.
    
    Allows services to define their own event-to-channel mappings
    beyond the default ones.
    
    Args:
        event_type: Domain event type (e.g., "user.login") 
        channel_name: Channel to publish to
    """
    DOMAIN_EVENT_CHANNELS[event_type] = channel_name
    logger.info(f"Registered domain event mapping: {event_type} -> {channel_name}")


def get_event_channel_mappings() -> Dict[str, str]:
    """
    Get all current event type to channel mappings.
    
    Returns:
        Dictionary mapping event types to channel names
    """
    return DOMAIN_EVENT_CHANNELS.copy()


def create_domain_event_publisher(agent_bus: AgentBus) -> EventPublisher:
    """
    Factory function to create a DomainEventPublisher.
    
    Args:
        agent_bus: AgentBus instance to publish through
        
    Returns:
        EventPublisher instance
    """
    return DomainEventPublisher(agent_bus)