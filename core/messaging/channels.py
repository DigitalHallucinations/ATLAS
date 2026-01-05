"""
Channel Definitions for Agent Communication
============================================

Fine-grained semantic channels aligned with ATLAS agent workflows.
Each channel has specific backpressure, priority, and TTL configurations.

Author: ATLAS Team
Date: Jan 1, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional


class MessagePriority(IntEnum):
    """
    Priority levels for agent messages.
    Lower values = higher priority (processed first).
    """

    CRITICAL = 0    # System shutdown, security alerts, emergencies
    HIGH = 2        # User commands, tool calls, direct requests
    NORMAL = 5      # Agent responses, delegations, status updates
    LOW = 8         # Streaming chunks, incremental updates
    BACKGROUND = 10 # Telemetry, logging, cleanup tasks


@dataclass(frozen=True)
class ChannelConfig:
    """
    Configuration for a single NCB channel.
    
    Attributes:
        name: Unique channel identifier (e.g., 'user.input', 'llm.stream')
        maxsize: Maximum queue depth before backpressure applies
        priority_levels: Number of priority buckets (1-10)
        ttl_seconds: Message time-to-live (None = no expiry)
        drop_policy: 'drop_new' rejects incoming when full, 'drop_old' evicts oldest
        default_priority: Priority assigned when not specified
        persistent: Whether to persist messages to SQLite
        dead_letter_channel: Channel for failed/expired messages (None = discard)
        description: Human-readable purpose
        idempotency_key_field: Payload field to use for duplicate detection (None = disabled)
        idempotency_ttl_seconds: How long to remember processed keys
    """

    name: str
    maxsize: int = 1000
    priority_levels: int = 3
    ttl_seconds: Optional[float] = None
    drop_policy: str = "drop_new"
    default_priority: int = MessagePriority.NORMAL
    persistent: bool = False
    dead_letter_channel: Optional[str] = None
    description: str = ""
    idempotency_key_field: Optional[str] = None
    idempotency_ttl_seconds: float = 60.0

    def to_ncb_config(self) -> Dict:
        """Convert to NCB channel configuration dict."""
        return {
            "max_queue_size": self.maxsize,
            "drop_policy": self.drop_policy,
            "default_priority": self.default_priority,
            "persistent": self.persistent,
            "dead_letter_channel": self.dead_letter_channel,
            "idempotency_key_field": self.idempotency_key_field,
            "idempotency_ttl_seconds": self.idempotency_ttl_seconds,
        }


# =============================================================================
# Channel Definitions - Organized by Communication Domain
# =============================================================================

# -----------------------------------------------------------------------------
# User Interface Channels
# -----------------------------------------------------------------------------

USER_INPUT = ChannelConfig(
    name="user.input",
    maxsize=100,
    priority_levels=3,
    ttl_seconds=30.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    description="User commands and queries entering the system",
)

USER_OUTPUT = ChannelConfig(
    name="user.output",
    maxsize=500,
    priority_levels=2,
    ttl_seconds=60.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.NORMAL,
    description="Final responses and updates sent to the user",
)

USER_NOTIFICATION = ChannelConfig(
    name="user.notification",
    maxsize=200,
    priority_levels=3,
    ttl_seconds=120.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.NORMAL,
    description="Non-blocking notifications (status, progress, alerts)",
)

# -----------------------------------------------------------------------------
# LLM Provider Channels
# -----------------------------------------------------------------------------

LLM_REQUEST = ChannelConfig(
    name="llm.request",
    maxsize=50,
    priority_levels=2,
    ttl_seconds=120.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    description="Outbound requests to LLM providers",
)

LLM_STREAM = ChannelConfig(
    name="llm.stream",
    maxsize=10000,
    priority_levels=1,
    ttl_seconds=10.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.LOW,
    description="Token-by-token streaming chunks from LLM",
)

LLM_COMPLETE = ChannelConfig(
    name="llm.complete",
    maxsize=200,
    priority_levels=2,
    ttl_seconds=300.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Complete LLM responses",
)

LLM_ERROR = ChannelConfig(
    name="llm.error",
    maxsize=100,
    priority_levels=2,
    ttl_seconds=600.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    dead_letter_channel="system.dlq",
    description="LLM provider errors and failures",
)

# -----------------------------------------------------------------------------
# Tool Execution Channels
# -----------------------------------------------------------------------------

TOOL_INVOKE = ChannelConfig(
    name="tool.invoke",
    maxsize=100,
    priority_levels=3,
    ttl_seconds=300.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    description="Tool invocation requests",
)

TOOL_RESULT = ChannelConfig(
    name="tool.result",
    maxsize=100,
    priority_levels=2,
    ttl_seconds=300.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Tool execution results",
)

TOOL_PROGRESS = ChannelConfig(
    name="tool.progress",
    maxsize=500,
    priority_levels=1,
    ttl_seconds=30.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.LOW,
    description="Tool execution progress updates",
)

TOOL_ERROR = ChannelConfig(
    name="tool.error",
    maxsize=100,
    priority_levels=2,
    ttl_seconds=600.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    dead_letter_channel="system.dlq",
    description="Tool execution errors",
)

# -----------------------------------------------------------------------------
# Inter-Agent Coordination Channels
# -----------------------------------------------------------------------------

AGENT_SPAWN = ChannelConfig(
    name="agent.spawn",
    maxsize=50,
    priority_levels=2,
    ttl_seconds=60.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    description="Sub-agent creation requests",
)

AGENT_DELEGATE = ChannelConfig(
    name="agent.delegate",
    maxsize=100,
    priority_levels=3,
    ttl_seconds=120.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    description="Task delegation between agents",
)

AGENT_STATUS = ChannelConfig(
    name="agent.status",
    maxsize=500,
    priority_levels=1,
    ttl_seconds=30.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.LOW,
    description="Agent health and status updates",
)

AGENT_TERMINATE = ChannelConfig(
    name="agent.terminate",
    maxsize=50,
    priority_levels=2,
    ttl_seconds=30.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.CRITICAL,
    description="Agent termination signals",
)

AGENT_RESULT = ChannelConfig(
    name="agent.result",
    maxsize=100,
    priority_levels=2,
    ttl_seconds=300.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Results from delegated agent tasks",
)

# -----------------------------------------------------------------------------
# Task & Job Orchestration Channels
# -----------------------------------------------------------------------------

TASK_CREATE = ChannelConfig(
    name="task.create",
    maxsize=100,
    priority_levels=3,
    ttl_seconds=120.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    description="New task creation requests",
)

TASK_UPDATE = ChannelConfig(
    name="task.update",
    maxsize=200,
    priority_levels=2,
    ttl_seconds=60.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.NORMAL,
    description="Task status updates and modifications",
)

TASK_COMPLETE = ChannelConfig(
    name="task.complete",
    maxsize=100,
    priority_levels=2,
    ttl_seconds=300.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Task completion events",
)

JOB_SCHEDULE = ChannelConfig(
    name="job.schedule",
    maxsize=50,
    priority_levels=2,
    ttl_seconds=None,  # Jobs can be scheduled far in advance
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Job scheduling requests",
)

JOB_CREATE = ChannelConfig(
    name="job.create",
    maxsize=50,
    priority_levels=2,
    ttl_seconds=120.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Job creation events",
)

JOB_UPDATE = ChannelConfig(
    name="job.update",
    maxsize=100,
    priority_levels=2,
    ttl_seconds=60.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.NORMAL,
    description="Job status updates and modifications",
)

JOB_EXECUTE = ChannelConfig(
    name="job.execute",
    maxsize=50,
    priority_levels=3,
    ttl_seconds=300.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    description="Job execution triggers",
)

JOB_RESULT = ChannelConfig(
    name="job.result",
    maxsize=100,
    priority_levels=2,
    ttl_seconds=600.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Job completion results",
)

# -----------------------------------------------------------------------------
# Conversation & Context Channels
# -----------------------------------------------------------------------------

CONVERSATION_START = ChannelConfig(
    name="conversation.start",
    maxsize=50,
    priority_levels=2,
    ttl_seconds=60.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    description="New conversation initialization",
)

CONVERSATION_MESSAGE = ChannelConfig(
    name="conversation.message",
    maxsize=500,
    priority_levels=2,
    ttl_seconds=120.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Conversation message events",
)

CONVERSATION_END = ChannelConfig(
    name="conversation.end",
    maxsize=50,
    priority_levels=2,
    ttl_seconds=60.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Conversation termination events",
)

CONTEXT_UPDATE = ChannelConfig(
    name="context.update",
    maxsize=200,
    priority_levels=2,
    ttl_seconds=60.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.NORMAL,
    description="Blackboard and context state updates",
)

FOLLOWUP = ChannelConfig(
    name="followup",
    maxsize=100,
    priority_levels=2,
    ttl_seconds=300.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Conversation follow-up events",
)

SKILL_ACTIVITY = ChannelConfig(
    name="skill.activity",
    maxsize=200,
    priority_levels=2,
    ttl_seconds=60.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.NORMAL,
    description="Skill execution activity events",
)

BLACKBOARD_EVENT = ChannelConfig(
    name="blackboard.event",
    maxsize=200,
    priority_levels=2,
    ttl_seconds=60.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.NORMAL,
    description="Blackboard CRUD events",
)

# -----------------------------------------------------------------------------
# Budget Management Channels
# -----------------------------------------------------------------------------

BUDGET_USAGE = ChannelConfig(
    name="budget.usage",
    maxsize=500,
    priority_levels=1,
    ttl_seconds=60.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.BACKGROUND,
    description="Usage tracking events for cost monitoring",
)

BUDGET_ALERT = ChannelConfig(
    name="budget.alert",
    maxsize=100,
    priority_levels=3,
    ttl_seconds=300.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.HIGH,
    persistent=True,
    description="Budget threshold alerts and warnings",
)

BUDGET_POLICY = ChannelConfig(
    name="budget.policy",
    maxsize=50,
    priority_levels=2,
    ttl_seconds=120.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.NORMAL,
    persistent=True,
    description="Budget policy changes and updates",
)

# -----------------------------------------------------------------------------
# System Control Plane Channels
# -----------------------------------------------------------------------------

SYSTEM_CONTROL = ChannelConfig(
    name="system.control",
    maxsize=50,
    priority_levels=1,
    ttl_seconds=30.0,
    drop_policy="drop_new",
    default_priority=MessagePriority.CRITICAL,
    description="System-wide control commands (shutdown, pause, resume)",
)

SYSTEM_METRICS = ChannelConfig(
    name="system.metrics",
    maxsize=1000,
    priority_levels=1,
    ttl_seconds=60.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.BACKGROUND,
    description="Telemetry and performance metrics",
)

SYSTEM_LOG = ChannelConfig(
    name="system.log",
    maxsize=2000,
    priority_levels=1,
    ttl_seconds=300.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.BACKGROUND,
    description="Structured log events",
)

SYSTEM_DLQ = ChannelConfig(
    name="system.dlq",
    maxsize=500,
    priority_levels=1,
    ttl_seconds=None,  # DLQ messages don't expire
    drop_policy="drop_new",
    default_priority=MessagePriority.LOW,
    persistent=True,
    description="Dead letter queue for failed messages",
)

SYSTEM_HEALTH = ChannelConfig(
    name="system.health",
    maxsize=100,
    priority_levels=1,
    ttl_seconds=30.0,
    drop_policy="drop_old",
    default_priority=MessagePriority.LOW,
    description="Component health check events",
)

# =============================================================================
# Channel Registry
# =============================================================================

ALL_CHANNELS: tuple[ChannelConfig, ...] = (
    # User Interface
    USER_INPUT,
    USER_OUTPUT,
    USER_NOTIFICATION,
    # LLM
    LLM_REQUEST,
    LLM_STREAM,
    LLM_COMPLETE,
    LLM_ERROR,
    # Tools
    TOOL_INVOKE,
    TOOL_RESULT,
    TOOL_PROGRESS,
    TOOL_ERROR,
    # Agents
    AGENT_SPAWN,
    AGENT_DELEGATE,
    AGENT_STATUS,
    AGENT_TERMINATE,
    AGENT_RESULT,
    # Tasks & Jobs
    TASK_CREATE,
    TASK_UPDATE,
    TASK_COMPLETE,
    JOB_SCHEDULE,
    JOB_CREATE,
    JOB_UPDATE,
    JOB_EXECUTE,
    JOB_RESULT,
    # Conversation
    CONVERSATION_START,
    CONVERSATION_MESSAGE,
    CONVERSATION_END,
    CONTEXT_UPDATE,
    FOLLOWUP,
    # Skills
    SKILL_ACTIVITY,
    # Blackboard
    BLACKBOARD_EVENT,
    # Budget
    BUDGET_USAGE,
    BUDGET_ALERT,
    BUDGET_POLICY,
    # System
    SYSTEM_CONTROL,
    SYSTEM_METRICS,
    SYSTEM_LOG,
    SYSTEM_DLQ,
    SYSTEM_HEALTH,
)

CHANNEL_BY_NAME: Dict[str, ChannelConfig] = {ch.name: ch for ch in ALL_CHANNELS}


def get_channel(name: str) -> Optional[ChannelConfig]:
    """Look up a channel configuration by name."""
    return CHANNEL_BY_NAME.get(name)


def get_all_channel_names() -> list[str]:
    """Return all registered channel names."""
    return list(CHANNEL_BY_NAME.keys())
