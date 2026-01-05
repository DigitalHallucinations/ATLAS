"""
Message Types for Agent Communication
======================================

Extended Message dataclass with ATLAS-specific fields and typed subclasses
for different message categories.

Author: ATLAS Team
Date: Jan 1, 2026
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from .channels import MessagePriority


# =============================================================================
# Base Message
# =============================================================================

@dataclass(frozen=True)
class AgentMessage:
    """
    Base message envelope for all agent communication.
    
    Extends NCB Message concept with ATLAS-specific routing and context fields.
    Immutable (frozen) to ensure message integrity during async processing.
    
    Attributes:
        id: Unique message identifier (UUID)
        channel: Target channel name (e.g., 'user.input', 'tool.invoke')
        payload: Message content (domain-specific data)
        priority: Message priority (lower = more urgent)
        ts: Timestamp when message was created
        ttl: Time-to-live in seconds (None = no expiry)
        
        # ATLAS Context Fields
        agent_id: Source agent identifier
        conversation_id: Associated conversation (for context tracking)
        request_id: Correlation ID for request/response pairing
        user_id: Originating user (for access control, audit)
        trace_id: Distributed tracing identifier
        
        # Metadata
        headers: Arbitrary key-value headers
        source_channel: Channel that originated this message (for replies)
    """

    # Core message fields
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    channel: str = ""
    payload: Any = None
    priority: int = MessagePriority.NORMAL
    ts: float = field(default_factory=time.time)
    ttl: Optional[float] = None

    # ATLAS context fields
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = field(default_factory=lambda: uuid.uuid4().hex)

    # Metadata
    headers: Dict[str, str] = field(default_factory=dict)
    source_channel: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "id": self.id,
            "channel": self.channel,
            "payload": self.payload,
            "priority": self.priority,
            "ts": self.ts,
            "ttl": self.ttl,
            "agent_id": self.agent_id,
            "conversation_id": self.conversation_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "trace_id": self.trace_id,
            "headers": self.headers,
            "source_channel": self.source_channel,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Deserialize message from dictionary."""
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            channel=str(data.get("channel") or ""),
            payload=data.get("payload"),
            priority=int(data.get("priority") or MessagePriority.NORMAL),
            ts=float(data.get("ts") or time.time()),
            ttl=data.get("ttl"),
            agent_id=data.get("agent_id"),
            conversation_id=data.get("conversation_id"),
            request_id=data.get("request_id"),
            user_id=data.get("user_id"),
            trace_id=data.get("trace_id") or uuid.uuid4().hex,
            headers=dict(data.get("headers") or {}),
            source_channel=data.get("source_channel"),
        )

    def to_ncb_meta(self) -> Dict[str, Any]:
        """Convert ATLAS fields to NCB message meta dict."""
        meta: Dict[str, Any] = {}
        if self.agent_id:
            meta["agent_id"] = self.agent_id
        if self.conversation_id:
            meta["conversation_id"] = self.conversation_id
        if self.request_id:
            meta["request_id"] = self.request_id
        if self.user_id:
            meta["user_id"] = self.user_id
        if self.trace_id:
            meta["trace_id"] = self.trace_id
        if self.source_channel:
            meta["source_channel"] = self.source_channel
        if self.headers:
            meta["headers"] = self.headers
        return meta

    def with_reply_context(self, reply_channel: str) -> "AgentMessage":
        """Create a copy with source_channel set for reply routing."""
        return AgentMessage(
            id=self.id,
            channel=self.channel,
            payload=self.payload,
            priority=self.priority,
            ts=self.ts,
            ttl=self.ttl,
            agent_id=self.agent_id,
            conversation_id=self.conversation_id,
            request_id=self.request_id,
            user_id=self.user_id,
            trace_id=self.trace_id,
            headers=self.headers,
            source_channel=reply_channel,
        )

    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.ts) > self.ttl


# =============================================================================
# Typed Message Subclasses
# =============================================================================

T = TypeVar("T", bound="AgentMessage")


@dataclass(frozen=True)
class UserInputMessage(AgentMessage):
    """Message carrying user input to the system."""

    channel: str = field(default="user.input")
    priority: int = field(default=MessagePriority.HIGH)

    # Typed payload fields (in addition to generic payload)
    content: str = ""
    input_type: str = "text"  # text, voice, image, file

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["content"] = self.content
        d["input_type"] = self.input_type
        return d


@dataclass(frozen=True)
class UserOutputMessage(AgentMessage):
    """Message carrying output to the user."""

    channel: str = field(default="user.output")
    priority: int = field(default=MessagePriority.NORMAL)

    content: str = ""
    output_type: str = "text"  # text, markdown, html, error
    is_final: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["content"] = self.content
        d["output_type"] = self.output_type
        d["is_final"] = self.is_final
        d["metadata"] = self.metadata
        return d


@dataclass(frozen=True)
class LLMRequestMessage(AgentMessage):
    """Request to an LLM provider."""

    channel: str = field(default="llm.request")
    priority: int = field(default=MessagePriority.HIGH)
    ttl: Optional[float] = field(default=120.0)

    prompt: str = ""
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = True
    tools: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["prompt"] = self.prompt
        d["system_prompt"] = self.system_prompt
        d["model"] = self.model
        d["provider"] = self.provider
        d["temperature"] = self.temperature
        d["max_tokens"] = self.max_tokens
        d["stream"] = self.stream
        d["tools"] = self.tools
        return d


@dataclass(frozen=True)
class LLMStreamChunk(AgentMessage):
    """Streaming chunk from LLM."""

    channel: str = field(default="llm.stream")
    priority: int = field(default=MessagePriority.LOW)
    ttl: Optional[float] = field(default=10.0)

    chunk: str = ""
    chunk_index: int = 0
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["chunk"] = self.chunk
        d["chunk_index"] = self.chunk_index
        d["is_final"] = self.is_final
        return d


@dataclass(frozen=True)
class LLMCompleteMessage(AgentMessage):
    """Complete LLM response."""

    channel: str = field(default="llm.complete")
    priority: int = field(default=MessagePriority.NORMAL)

    content: str = ""
    model: Optional[str] = None
    provider: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["content"] = self.content
        d["model"] = self.model
        d["provider"] = self.provider
        d["usage"] = self.usage
        d["finish_reason"] = self.finish_reason
        d["tool_calls"] = self.tool_calls
        return d


@dataclass(frozen=True)
class ToolInvokeMessage(AgentMessage):
    """Tool invocation request."""

    channel: str = field(default="tool.invoke")
    priority: int = field(default=MessagePriority.HIGH)
    ttl: Optional[float] = field(default=300.0)

    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["tool_name"] = self.tool_name
        d["tool_args"] = self.tool_args
        d["tool_call_id"] = self.tool_call_id
        return d


@dataclass(frozen=True)
class ToolResultMessage(AgentMessage):
    """Tool execution result."""

    channel: str = field(default="tool.result")
    priority: int = field(default=MessagePriority.NORMAL)

    tool_name: str = ""
    tool_call_id: Optional[str] = None
    result: Any = None
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["tool_name"] = self.tool_name
        d["tool_call_id"] = self.tool_call_id
        d["result"] = self.result
        d["success"] = self.success
        d["error"] = self.error
        d["execution_time_ms"] = self.execution_time_ms
        return d


@dataclass(frozen=True)
class AgentDelegateMessage(AgentMessage):
    """Task delegation between agents."""

    channel: str = field(default="agent.delegate")
    priority: int = field(default=MessagePriority.HIGH)
    ttl: Optional[float] = field(default=120.0)

    target_agent: str = ""
    source_agent: str = ""
    task: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["target_agent"] = self.target_agent
        d["source_agent"] = self.source_agent
        d["task"] = self.task
        d["context"] = self.context
        d["timeout_seconds"] = self.timeout_seconds
        return d


@dataclass(frozen=True)
class AgentSpawnMessage(AgentMessage):
    """Sub-agent spawn request."""

    channel: str = field(default="agent.spawn")
    priority: int = field(default=MessagePriority.HIGH)

    agent_type: str = ""
    parent_agent: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    initial_task: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["agent_type"] = self.agent_type
        d["parent_agent"] = self.parent_agent
        d["config"] = self.config
        d["initial_task"] = self.initial_task
        return d


@dataclass(frozen=True)
class AgentStatusMessage(AgentMessage):
    """Agent status/health update."""

    channel: str = field(default="agent.status")
    priority: int = field(default=MessagePriority.LOW)
    ttl: Optional[float] = field(default=30.0)

    status: str = "running"  # running, idle, busy, error, terminated
    load: Optional[float] = None  # 0.0 - 1.0
    current_task: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["status"] = self.status
        d["load"] = self.load
        d["current_task"] = self.current_task
        d["metrics"] = self.metrics
        return d


@dataclass(frozen=True)
class TaskCreateMessage(AgentMessage):
    """Task creation request."""

    channel: str = field(default="task.create")
    priority: int = field(default=MessagePriority.HIGH)

    task_type: str = ""
    title: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["task_type"] = self.task_type
        d["title"] = self.title
        d["description"] = self.description
        d["parameters"] = self.parameters
        d["parent_task_id"] = self.parent_task_id
        return d


@dataclass(frozen=True)
class TaskUpdateMessage(AgentMessage):
    """Task status update."""

    channel: str = field(default="task.update")
    priority: int = field(default=MessagePriority.NORMAL)

    task_id: str = ""
    status: str = ""  # pending, running, completed, failed, cancelled
    progress: Optional[float] = None  # 0.0 - 1.0
    message: Optional[str] = None
    result: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["task_id"] = self.task_id
        d["status"] = self.status
        d["progress"] = self.progress
        d["message"] = self.message
        d["result"] = self.result
        return d


@dataclass(frozen=True)
class SystemControlMessage(AgentMessage):
    """System control command."""

    channel: str = field(default="system.control")
    priority: int = field(default=MessagePriority.CRITICAL)

    command: str = ""  # shutdown, pause, resume, reload_config
    target: Optional[str] = None  # specific component or None for all
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["command"] = self.command
        d["target"] = self.target
        d["parameters"] = self.parameters
        return d


@dataclass(frozen=True)
class ErrorMessage(AgentMessage):
    """Error notification message."""

    priority: int = field(default=MessagePriority.HIGH)

    error_type: str = ""
    error_message: str = ""
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    recoverable: bool = False
    original_message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["error_type"] = self.error_type
        d["error_message"] = self.error_message
        d["error_code"] = self.error_code
        d["stack_trace"] = self.stack_trace
        d["recoverable"] = self.recoverable
        d["original_message_id"] = self.original_message_id
        return d


# =============================================================================
# Message Type Registry
# =============================================================================

MESSAGE_TYPES: Dict[str, Type[AgentMessage]] = {
    "user.input": UserInputMessage,
    "user.output": UserOutputMessage,
    "llm.request": LLMRequestMessage,
    "llm.stream": LLMStreamChunk,
    "llm.complete": LLMCompleteMessage,
    "tool.invoke": ToolInvokeMessage,
    "tool.result": ToolResultMessage,
    "agent.delegate": AgentDelegateMessage,
    "agent.spawn": AgentSpawnMessage,
    "agent.status": AgentStatusMessage,
    "task.create": TaskCreateMessage,
    "task.update": TaskUpdateMessage,
    "system.control": SystemControlMessage,
}


def create_message(channel: str, **kwargs: Any) -> AgentMessage:
    """
    Factory function to create appropriate message type for a channel.
    
    Falls back to base AgentMessage if no typed class exists.
    """
    msg_class = MESSAGE_TYPES.get(channel, AgentMessage)
    return msg_class(channel=channel, **kwargs)
