"""LLM context representation for model interactions.

This module defines the LLMContext dataclass that represents everything
an LLM needs to generate a response: system prompt, conversation history,
available tools, and token budget information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class ToolDefinition:
    """Represents a tool available to the LLM.
    
    Attributes:
        name: Unique tool identifier (e.g., "web_search", "mcp.server.tool").
        description: Human-readable description for the LLM.
        parameters: JSON Schema describing the tool's parameters.
        source: Origin of the tool ("native", "mcp", "persona", "skill").
        server: MCP server name if source is "mcp".
        enabled: Whether the tool is currently enabled.
        requires_consent: Whether the tool requires user consent before execution.
        metadata: Additional tool metadata.
    """
    
    name: str
    description: str
    parameters: dict[str, Any]
    source: str = "native"
    server: Optional[str] = None
    enabled: bool = True
    requires_consent: bool = False
    metadata: Optional[dict[str, Any]] = None
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tools format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
    
    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tools format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to a generic dictionary representation."""
        result = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source": self.source,
            "enabled": self.enabled,
        }
        if self.server:
            result["server"] = self.server
        if self.requires_consent:
            result["requires_consent"] = self.requires_consent
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolDefinition:
        """Create a ToolDefinition from a dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            source=data.get("source", "native"),
            server=data.get("server"),
            enabled=data.get("enabled", True),
            requires_consent=data.get("requires_consent", False),
            metadata=data.get("metadata"),
        )


@dataclass(slots=True)
class TokenBudget:
    """Token allocation information for context assembly.
    
    Attributes:
        model_limit: Maximum context window for the target model.
        system_tokens: Estimated tokens used by system prompt.
        history_tokens: Estimated tokens used by conversation history.
        tools_tokens: Estimated tokens used by tool definitions.
        reserved_output: Tokens reserved for model output.
        available: Tokens available for additional context.
        truncated: Whether history was truncated to fit budget.
        summarized: Whether history was summarized to fit budget.
    """
    
    model_limit: int = 128000
    system_tokens: int = 0
    history_tokens: int = 0
    tools_tokens: int = 0
    reserved_output: int = 4096
    available: int = 0
    truncated: bool = False
    summarized: bool = False
    
    @property
    def total_used(self) -> int:
        """Total tokens used by context components."""
        return self.system_tokens + self.history_tokens + self.tools_tokens
    
    @property
    def utilization(self) -> float:
        """Context window utilization as a percentage."""
        usable = self.model_limit - self.reserved_output
        if usable <= 0:
            return 1.0
        return min(1.0, self.total_used / usable)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_limit": self.model_limit,
            "system_tokens": self.system_tokens,
            "history_tokens": self.history_tokens,
            "tools_tokens": self.tools_tokens,
            "reserved_output": self.reserved_output,
            "available": self.available,
            "total_used": self.total_used,
            "utilization": round(self.utilization, 3),
            "truncated": self.truncated,
            "summarized": self.summarized,
        }


@dataclass(slots=True)
class MessageEntry:
    """A single message in the conversation history.
    
    Attributes:
        role: Message role ("system", "user", "assistant", "tool").
        content: Message content (text or structured).
        name: Optional name for tool messages.
        tool_call_id: Tool call ID for tool response messages.
        tool_calls: Tool calls made by assistant.
        metadata: Additional message metadata (thinking, audio, etc.).
    """
    
    role: str
    content: Any
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    metadata: Optional[dict[str, Any]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for provider APIs."""
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageEntry:
        """Create a MessageEntry from a dictionary."""
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=data.get("tool_calls"),
            metadata=data.get("metadata"),
        )


@dataclass(slots=True)
class LLMContext:
    """Complete context for an LLM interaction.
    
    Represents everything the model needs to generate a response:
    - System prompt (persona instructions, injected context)
    - Conversation history (possibly truncated/summarized)
    - Available tools (native + MCP)
    - Token budget information
    - Execution metadata
    
    Attributes:
        system_prompt: The complete system prompt including persona and injections.
        messages: Conversation history as a list of MessageEntry objects.
        tools: Available tool definitions.
        token_budget: Token allocation and usage information.
        persona_name: Name of the active persona.
        model: Target model identifier.
        provider: Target provider name.
        conversation_id: Associated conversation ID.
        injected_context: Additional context that was injected (blackboard, tasks).
        metadata: Additional context metadata.
    """
    
    system_prompt: str = ""
    messages: list[MessageEntry] = field(default_factory=list)
    tools: list[ToolDefinition] = field(default_factory=list)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    persona_name: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    conversation_id: Optional[str] = None
    injected_context: Optional[dict[str, str]] = None
    metadata: Optional[dict[str, Any]] = None
    
    @property
    def has_tools(self) -> bool:
        """Whether any tools are available."""
        return bool(self.tools)
    
    @property
    def enabled_tools(self) -> list[ToolDefinition]:
        """Tools that are currently enabled."""
        return [t for t in self.tools if t.enabled]
    
    @property
    def message_count(self) -> int:
        """Number of messages in history."""
        return len(self.messages)
    
    def get_messages_as_dicts(self) -> list[dict[str, Any]]:
        """Get messages as list of dictionaries for provider APIs."""
        return [m.to_dict() for m in self.messages]
    
    def get_tools_for_provider(self, provider: str) -> list[dict[str, Any]]:
        """Get tool definitions formatted for a specific provider.
        
        Args:
            provider: Provider name ("OpenAI", "Anthropic", "Mistral", etc.)
            
        Returns:
            List of tool definitions in provider-specific format.
        """
        enabled = self.enabled_tools
        if not enabled:
            return []
        
        if provider in ("OpenAI", "Mistral", "Grok"):
            return [t.to_openai_format() for t in enabled]
        elif provider == "Anthropic":
            return [t.to_anthropic_format() for t in enabled]
        else:
            # Generic format
            return [t.to_dict() for t in enabled]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization or logging."""
        result: dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.messages],
            "tools": [t.to_dict() for t in self.tools],
            "token_budget": self.token_budget.to_dict(),
        }
        if self.persona_name:
            result["persona_name"] = self.persona_name
        if self.model:
            result["model"] = self.model
        if self.provider:
            result["provider"] = self.provider
        if self.conversation_id:
            result["conversation_id"] = self.conversation_id
        if self.injected_context:
            result["injected_context"] = self.injected_context
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMContext:
        """Create an LLMContext from a dictionary."""
        messages = [
            MessageEntry.from_dict(m) if isinstance(m, dict) else m
            for m in data.get("messages", [])
        ]
        tools = [
            ToolDefinition.from_dict(t) if isinstance(t, dict) else t
            for t in data.get("tools", [])
        ]
        
        budget_data = data.get("token_budget", {})
        token_budget = TokenBudget(
            model_limit=budget_data.get("model_limit", 128000),
            system_tokens=budget_data.get("system_tokens", 0),
            history_tokens=budget_data.get("history_tokens", 0),
            tools_tokens=budget_data.get("tools_tokens", 0),
            reserved_output=budget_data.get("reserved_output", 4096),
            available=budget_data.get("available", 0),
            truncated=budget_data.get("truncated", False),
            summarized=budget_data.get("summarized", False),
        )
        
        return cls(
            system_prompt=data.get("system_prompt", ""),
            messages=messages,
            tools=tools,
            token_budget=token_budget,
            persona_name=data.get("persona_name"),
            model=data.get("model"),
            provider=data.get("provider"),
            conversation_id=data.get("conversation_id"),
            injected_context=data.get("injected_context"),
            metadata=data.get("metadata"),
        )
    
    def summary(self) -> str:
        """Return a brief summary of the context for logging."""
        tool_count = len(self.enabled_tools)
        injections = list(self.injected_context.keys()) if self.injected_context else []
        
        parts = [
            f"messages={self.message_count}",
            f"tools={tool_count}",
            f"tokens={self.token_budget.total_used}/{self.token_budget.model_limit}",
        ]
        if self.token_budget.truncated:
            parts.append("truncated")
        if self.token_budget.summarized:
            parts.append("summarized")
        if injections:
            parts.append(f"injected={injections}")
        
        return f"LLMContext({', '.join(parts)})"
