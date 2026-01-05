"""
ATLAS Messaging System
======================

Provides the AgentBus as the primary interface for agent communication,
backed by the Neural Cognitive Bus (NCB) for multi-channel async messaging.

Primary exports:
- AgentBus: High-level messaging API with typed messages
- get_agent_bus: Get the global bus instance
- MessagePriority: Priority levels for messages
- AgentMessage: Base message type with ATLAS context fields

Channel subscriptions:
    bus = get_agent_bus()
    await bus.start()
    await bus.on_tool_invoke(my_handler)
    await bus.invoke_tool("calculator", {"x": 1}, agent_id="main")

External transports (Kafka, Redis) are also available for bridging.
"""

from __future__ import annotations

from typing import Any

# Primary API - AgentBus
from .agent_bus import (
    AgentBus,
    Subscription,
    get_agent_bus,
    configure_agent_bus,
    shutdown_agent_bus,
)

# Channels and priorities
from .channels import (
    ChannelConfig,
    MessagePriority,
    ALL_CHANNELS,
    CHANNEL_BY_NAME,
    get_channel,
    get_all_channel_names,
    # Individual channel configs for direct reference
    USER_INPUT,
    USER_OUTPUT,
    LLM_REQUEST,
    LLM_STREAM,
    LLM_COMPLETE,
    TOOL_INVOKE,
    TOOL_RESULT,
    AGENT_DELEGATE,
    AGENT_SPAWN,
    AGENT_STATUS,
    TASK_CREATE,
    TASK_UPDATE,
    TASK_COMPLETE,
    JOB_SCHEDULE,
    JOB_CREATE,
    JOB_UPDATE,
    JOB_EXECUTE,
    JOB_RESULT,
    CONVERSATION_START,
    CONVERSATION_MESSAGE,
    CONVERSATION_END,
    CONTEXT_UPDATE,
    FOLLOWUP,
    SKILL_ACTIVITY,
    BLACKBOARD_EVENT,
    BUDGET_USAGE,
    BUDGET_ALERT,
    BUDGET_POLICY,
    SYSTEM_CONTROL,
    SYSTEM_METRICS,
    SYSTEM_LOG,
    SYSTEM_DLQ,
)

# Message types
from .messages import (
    AgentMessage,
    UserInputMessage,
    UserOutputMessage,
    LLMRequestMessage,
    LLMStreamChunk,
    LLMCompleteMessage,
    ToolInvokeMessage,
    ToolResultMessage,
    AgentDelegateMessage,
    AgentSpawnMessage,
    AgentStatusMessage,
    TaskCreateMessage,
    TaskUpdateMessage,
    SystemControlMessage,
    ErrorMessage,
    create_message,
)

# Low-level NCB (for advanced use cases)
from .NCB import NeuralCognitiveBus, Message as NCBMessage

# Idempotency support
from .idempotency import IdempotencyStore

__all__ = [
    # Primary API
    "AgentBus",
    "Subscription",
    "get_agent_bus",
    "configure_agent_bus",
    "shutdown_agent_bus",
    # Channels
    "ChannelConfig",
    "MessagePriority",
    "ALL_CHANNELS",
    "CHANNEL_BY_NAME",
    "get_channel",
    "get_all_channel_names",
    "USER_INPUT",
    "USER_OUTPUT",
    "LLM_REQUEST",
    "LLM_STREAM",
    "LLM_COMPLETE",
    "TOOL_INVOKE",
    "TOOL_RESULT",
    "AGENT_DELEGATE",
    "AGENT_SPAWN",
    "AGENT_STATUS",
    "TASK_CREATE",
    "TASK_UPDATE",
    "TASK_COMPLETE",
    "JOB_SCHEDULE",
    "JOB_CREATE",
    "JOB_UPDATE",
    "JOB_EXECUTE",
    "JOB_RESULT",
    "CONVERSATION_START",
    "CONVERSATION_MESSAGE",
    "CONVERSATION_END",
    "CONTEXT_UPDATE",
    "FOLLOWUP",
    "SKILL_ACTIVITY",
    "BLACKBOARD_EVENT",
    "BUDGET_USAGE",
    "BUDGET_ALERT",
    "BUDGET_POLICY",
    "SYSTEM_CONTROL",
    "SYSTEM_METRICS",
    "SYSTEM_LOG",
    "SYSTEM_DLQ",
    # Messages
    "AgentMessage",
    "UserInputMessage",
    "UserOutputMessage",
    "LLMRequestMessage",
    "LLMStreamChunk",
    "LLMCompleteMessage",
    "ToolInvokeMessage",
    "ToolResultMessage",
    "AgentDelegateMessage",
    "AgentSpawnMessage",
    "AgentStatusMessage",
    "TaskCreateMessage",
    "TaskUpdateMessage",
    "SystemControlMessage",
    "ErrorMessage",
    "create_message",
    # Low-level
    "NeuralCognitiveBus",
    "NCBMessage",
    "IdempotencyStore",
    # External transports (lazy loaded)
    "KafkaSink",
    "KafkaSinkUnavailable",
    "RedisToKafkaBridge",
    "build_bridge_from_settings",
]


def __getattr__(name: str) -> Any:
    """Lazy load external transport modules."""
    if name in {"KafkaSink", "KafkaSinkUnavailable"}:
        from .kafka_sink import KafkaSink, KafkaSinkUnavailable  # pylint: disable=import-outside-toplevel

        globals().update({"KafkaSink": KafkaSink, "KafkaSinkUnavailable": KafkaSinkUnavailable})
        return globals()[name]

    if name in {"RedisToKafkaBridge", "build_bridge_from_settings"}:
        from .bridge_redis_to_kafka import (  # pylint: disable=import-outside-toplevel
            RedisToKafkaBridge,
            build_bridge_from_settings,
        )

        globals().update(
            {
                "RedisToKafkaBridge": RedisToKafkaBridge,
                "build_bridge_from_settings": build_bridge_from_settings,
            }
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
