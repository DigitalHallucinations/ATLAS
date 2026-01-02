"""LLM context manager for assembling model context.

This module provides the LLMContextManager which orchestrates the assembly
of everything an LLM needs to generate a response: system prompts, conversation
history, tool definitions (native + MCP), blackboard state, task context,
and intelligent token budgeting.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

from .llm_context import (
    LLMContext,
    MessageEntry,
    TokenBudget,
    ToolDefinition,
)

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager
    from ATLAS.persona_manager import PersonaManager
    from modules.orchestration.blackboard import BlackboardStore

logger = setup_logger(__name__)

# Default context window sizes for common models
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # OpenAI
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "o1-preview": 128000,
    "o3-mini": 200000,
    # Anthropic
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    # Google
    "gemini-1.5-pro": 2000000,
    "gemini-1.5-flash": 1000000,
    "gemini-2.0-flash": 1000000,
    # Mistral
    "mistral-large-latest": 128000,
    "mistral-medium-latest": 32000,
    "mistral-small-latest": 32000,
    # Default fallback
    "default": 128000,
}

# Approximate tokens per character for estimation
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.
    
    Uses a simple character-based approximation. For more accurate
    counting, integrate tiktoken or the provider's tokenizer.
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_tool_tokens(tools: List[ToolDefinition]) -> int:
    """Estimate token count for tool definitions."""
    if not tools:
        return 0
    
    total = 0
    for tool in tools:
        # Estimate based on serialized JSON size
        tool_json = json.dumps(tool.to_dict(), separators=(",", ":"))
        total += estimate_tokens(tool_json)
    
    return total


def estimate_messages_tokens(messages: List[MessageEntry]) -> int:
    """Estimate token count for conversation messages."""
    if not messages:
        return 0
    
    total = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Handle structured content (e.g., vision messages)
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if text:
                        total += estimate_tokens(text)
                elif isinstance(part, str):
                    total += estimate_tokens(part)
        
        # Add overhead for role and structure
        total += 4  # Approximate per-message overhead
        
        # Tool calls add tokens
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total += estimate_tokens(json.dumps(tc, separators=(",", ":")))
    
    return total


def get_model_context_limit(model: Optional[str]) -> int:
    """Get the context window limit for a model."""
    if not model:
        return MODEL_CONTEXT_LIMITS["default"]
    
    # Exact match
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]
    
    # Prefix match (e.g., "gpt-4o-2024-11-20" -> "gpt-4o")
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if model.startswith(prefix):
            return limit
    
    return MODEL_CONTEXT_LIMITS["default"]


class LLMContextManager:
    """Orchestrates assembly of LLM context.
    
    Responsible for:
    - Assembling system prompts from personas
    - Resolving available tools (native + MCP)
    - Injecting blackboard state and task context
    - Managing token budget and history truncation
    - Formatting context for specific providers
    
    Example:
        context_manager = LLMContextManager(
            persona_manager=persona_manager,
            config_manager=config_manager,
        )
        
        llm_context = await context_manager.build_context(
            conversation_id="conv-123",
            messages=chat_history,
            model="gpt-4o",
        )
        
        # Use with provider
        response = await provider.generate_response(
            messages=llm_context.get_messages_as_dicts(),
            system=llm_context.system_prompt,
            tools=llm_context.get_tools_for_provider("OpenAI"),
        )
    """
    
    def __init__(
        self,
        *,
        persona_manager: Optional[PersonaManager] = None,
        config_manager: Optional[ConfigManager] = None,
        blackboard: Optional[BlackboardStore] = None,
        tool_resolver: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        mcp_tool_resolver: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    ):
        """Initialize the LLM context manager.
        
        Args:
            persona_manager: PersonaManager for system prompt generation.
            config_manager: ConfigManager for settings access.
            blackboard: BlackboardStore for shared context injection.
            tool_resolver: Optional callable that returns native tool definitions.
            mcp_tool_resolver: Optional callable that returns MCP tool definitions.
        """
        self._persona_manager = persona_manager
        self._config_manager = config_manager
        self._blackboard = blackboard
        self._tool_resolver = tool_resolver
        self._mcp_tool_resolver = mcp_tool_resolver
        self._logger = logger
    
    async def build_context(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        include_tools: bool = True,
        include_blackboard: bool = True,
        include_task_context: bool = True,
        task_context: Optional[Dict[str, Any]] = None,
        max_history_tokens: Optional[int] = None,
        reserved_output_tokens: int = 4096,
        system_prompt_override: Optional[str] = None,
        additional_system_context: Optional[str] = None,
    ) -> LLMContext:
        """Build complete LLM context for a request.
        
        Args:
            conversation_id: The active conversation ID.
            messages: Raw conversation history as list of dicts.
            model: Target model identifier for token budgeting.
            provider: Target provider name.
            include_tools: Whether to include tool definitions.
            include_blackboard: Whether to inject blackboard summary.
            include_task_context: Whether to inject active task state.
            task_context: Explicit task context to inject.
            max_history_tokens: Maximum tokens for conversation history.
            reserved_output_tokens: Tokens to reserve for model output.
            system_prompt_override: Override persona system prompt entirely.
            additional_system_context: Extra context to append to system prompt.
            
        Returns:
            LLMContext containing assembled context ready for LLM call.
        """
        # Determine context limits
        model_limit = get_model_context_limit(model)
        
        # 1. Build system prompt
        system_prompt = await self._build_system_prompt(
            system_prompt_override=system_prompt_override,
            additional_context=additional_system_context,
        )
        
        injected_context: Dict[str, str] = {}
        
        # 2. Inject blackboard context if requested
        if include_blackboard and self._blackboard:
            blackboard_content = self._get_blackboard_context(conversation_id)
            if blackboard_content:
                injected_context["blackboard"] = blackboard_content
                system_prompt = f"{system_prompt}\n\n## Shared Context\n{blackboard_content}"
        
        # 3. Inject task context if provided
        if include_task_context and task_context:
            task_content = self._format_task_context(task_context)
            if task_content:
                injected_context["task"] = task_content
                system_prompt = f"{system_prompt}\n\n## Current Task\n{task_content}"
        
        # 4. Resolve tools
        tools: List[ToolDefinition] = []
        if include_tools:
            tools = await self._resolve_tools()
        
        # 5. Convert messages to MessageEntry objects
        message_entries = self._convert_messages(messages)
        
        # 6. Calculate token budget and potentially truncate history
        token_budget, message_entries = self._apply_token_budget(
            system_prompt=system_prompt,
            messages=message_entries,
            tools=tools,
            model_limit=model_limit,
            reserved_output=reserved_output_tokens,
            max_history_tokens=max_history_tokens,
        )
        
        # 7. Get persona name
        persona_name = None
        if self._persona_manager:
            current = getattr(self._persona_manager, "current_persona", None)
            if current:
                persona_name = current.get("name")
        
        return LLMContext(
            system_prompt=system_prompt,
            messages=message_entries,
            tools=tools,
            token_budget=token_budget,
            persona_name=persona_name,
            model=model,
            provider=provider,
            conversation_id=conversation_id,
            injected_context=injected_context if injected_context else None,
        )
    
    async def _build_system_prompt(
        self,
        *,
        system_prompt_override: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """Build the system prompt from persona or override."""
        if system_prompt_override:
            base_prompt = system_prompt_override
        elif self._persona_manager:
            base_prompt = self._persona_manager.get_current_persona_prompt() or ""
        else:
            base_prompt = ""
        
        if additional_context:
            base_prompt = f"{base_prompt}\n\n{additional_context}"
        
        return base_prompt.strip()
    
    def _get_blackboard_context(self, conversation_id: str) -> Optional[str]:
        """Get formatted blackboard summary for injection."""
        if not self._blackboard:
            return None
        
        try:
            summary = self._blackboard.get_summary(
                conversation_id,
                scope_type="conversation",
            )
        except Exception as exc:
            self._logger.warning(
                "Failed to get blackboard summary for %s: %s",
                conversation_id,
                exc,
            )
            return None
        
        entries = summary.get("entries", [])
        if not entries:
            return None
        
        # Format entries for LLM consumption
        parts = []
        for entry in entries:
            category = entry.get("category", "item")
            title = entry.get("title", "Untitled")
            content = entry.get("content", "")
            parts.append(f"**{category.title()}: {title}**\n{content}")
        
        return "\n\n".join(parts)
    
    def _format_task_context(self, task_context: Dict[str, Any]) -> Optional[str]:
        """Format task context for injection into system prompt."""
        if not task_context:
            return None
        
        parts = []
        
        task_id = task_context.get("id") or task_context.get("task_id")
        if task_id:
            parts.append(f"Task ID: {task_id}")
        
        title = task_context.get("title") or task_context.get("name")
        if title:
            parts.append(f"Title: {title}")
        
        description = task_context.get("description")
        if description:
            parts.append(f"Description: {description}")
        
        status = task_context.get("status")
        if status:
            parts.append(f"Status: {status}")
        
        objectives = task_context.get("objectives", [])
        if objectives:
            obj_list = "\n".join(f"- {obj}" for obj in objectives)
            parts.append(f"Objectives:\n{obj_list}")
        
        return "\n".join(parts) if parts else None
    
    async def _resolve_tools(self) -> List[ToolDefinition]:
        """Resolve all available tools (native + MCP)."""
        tools: List[ToolDefinition] = []
        
        # Get native tools
        if self._tool_resolver:
            try:
                native_tools = self._tool_resolver()
                tools.extend(self._convert_tool_entries(native_tools, source="native"))
            except Exception as exc:
                self._logger.warning("Failed to resolve native tools: %s", exc)
        
        # Get MCP tools
        if self._mcp_tool_resolver:
            try:
                mcp_tools = self._mcp_tool_resolver()
                tools.extend(self._convert_tool_entries(mcp_tools, source="mcp"))
            except Exception as exc:
                self._logger.warning("Failed to resolve MCP tools: %s", exc)
        
        # Deduplicate by name (prefer native over MCP)
        seen: set[str] = set()
        unique_tools: List[ToolDefinition] = []
        for tool in tools:
            if tool.name not in seen:
                seen.add(tool.name)
                unique_tools.append(tool)
        
        return unique_tools
    
    def _convert_tool_entries(
        self,
        entries: List[Dict[str, Any]],
        source: str = "native",
    ) -> List[ToolDefinition]:
        """Convert raw tool entries to ToolDefinition objects."""
        tools = []
        for entry in entries:
            name = entry.get("name")
            if not name:
                continue
            
            # Handle both OpenAI-style and flat formats
            if "function" in entry:
                func = entry["function"]
                name = func.get("name", name)
                description = func.get("description", "")
                parameters = func.get("parameters", {})
            else:
                description = entry.get("description", "")
                parameters = entry.get("parameters", entry.get("input_schema", {}))
            
            tools.append(ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                source=source,
                server=entry.get("server"),
                enabled=entry.get("enabled", True),
                requires_consent=entry.get("requires_consent", False),
                metadata=entry.get("metadata"),
            ))
        
        return tools
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[MessageEntry]:
        """Convert raw message dicts to MessageEntry objects."""
        entries = []
        for msg in messages:
            entries.append(MessageEntry(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                name=msg.get("name"),
                tool_call_id=msg.get("tool_call_id"),
                tool_calls=msg.get("tool_calls"),
                metadata=msg.get("metadata"),
            ))
        return entries
    
    def _apply_token_budget(
        self,
        *,
        system_prompt: str,
        messages: List[MessageEntry],
        tools: List[ToolDefinition],
        model_limit: int,
        reserved_output: int,
        max_history_tokens: Optional[int] = None,
    ) -> tuple[TokenBudget, List[MessageEntry]]:
        """Apply token budget, truncating history if needed.
        
        Returns:
            Tuple of (TokenBudget, potentially truncated messages)
        """
        # Estimate fixed costs
        system_tokens = estimate_tokens(system_prompt)
        tools_tokens = estimate_tool_tokens(tools)
        
        # Calculate available budget for history
        usable_limit = model_limit - reserved_output
        fixed_cost = system_tokens + tools_tokens
        available_for_history = usable_limit - fixed_cost
        
        # Apply max_history_tokens cap if specified
        if max_history_tokens is not None:
            available_for_history = min(available_for_history, max_history_tokens)
        
        # Estimate current history tokens
        history_tokens = estimate_messages_tokens(messages)
        
        truncated = False
        final_messages = messages
        
        # Truncate if needed (remove oldest messages, keep system/recent)
        if history_tokens > available_for_history and available_for_history > 0:
            final_messages, history_tokens, truncated = self._truncate_history(
                messages,
                available_for_history,
            )
        
        # Build budget info
        available = max(0, usable_limit - fixed_cost - history_tokens)
        
        budget = TokenBudget(
            model_limit=model_limit,
            system_tokens=system_tokens,
            history_tokens=history_tokens,
            tools_tokens=tools_tokens,
            reserved_output=reserved_output,
            available=available,
            truncated=truncated,
            summarized=False,  # Summarization not yet implemented
        )
        
        return budget, final_messages
    
    def _truncate_history(
        self,
        messages: List[MessageEntry],
        max_tokens: int,
    ) -> tuple[List[MessageEntry], int, bool]:
        """Truncate message history to fit token budget.
        
        Uses a sliding window approach, keeping the most recent messages.
        Always preserves the first system message if present.
        
        Returns:
            Tuple of (truncated messages, token count, was_truncated)
        """
        if not messages:
            return messages, 0, False
        
        total_tokens = estimate_messages_tokens(messages)
        if total_tokens <= max_tokens:
            return messages, total_tokens, False
        
        # Try to preserve system messages at the start
        preserved: List[MessageEntry] = []
        remainder: List[MessageEntry] = []
        
        for msg in messages:
            if msg.role == "system" and not remainder:
                preserved.append(msg)
            else:
                remainder.append(msg)
        
        preserved_tokens = estimate_messages_tokens(preserved)
        available = max_tokens - preserved_tokens
        
        if available <= 0:
            # Can only fit system messages
            return preserved, preserved_tokens, True
        
        # Take messages from the end until we hit the limit
        selected: List[MessageEntry] = []
        selected_tokens = 0
        
        for msg in reversed(remainder):
            msg_tokens = estimate_messages_tokens([msg])
            if selected_tokens + msg_tokens <= available:
                selected.insert(0, msg)
                selected_tokens += msg_tokens
            else:
                break
        
        final = preserved + selected
        final_tokens = preserved_tokens + selected_tokens
        
        return final, final_tokens, True
    
    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------
    @classmethod
    def from_atlas(cls, atlas: Any) -> LLMContextManager:
        """Create an LLMContextManager from an ATLAS instance.
        
        Args:
            atlas: The ATLAS application instance.
            
        Returns:
            Configured LLMContextManager.
        """
        persona_manager = getattr(atlas, "_persona_manager", None) or getattr(atlas, "persona_manager", None)
        config_manager = getattr(atlas, "config_manager", None)
        
        # Try to get blackboard
        blackboard = None
        try:
            from modules.orchestration.blackboard import get_blackboard
            blackboard = get_blackboard()
        except Exception:
            pass
        
        # Create tool resolver that uses ToolManager
        def tool_resolver() -> List[Dict[str, Any]]:
            try:
                from ATLAS.tools.manifests import load_function_map_from_current_persona
                current_persona = None
                if persona_manager:
                    current_persona = getattr(persona_manager, "current_persona", None)
                
                func_map = load_function_map_from_current_persona(
                    current_persona,
                    config_manager=config_manager,
                )
                if not func_map:
                    return []
                
                # Convert function map entries to tool definitions
                tools = []
                for name, meta in func_map.items():
                    if isinstance(meta, Mapping):
                        tools.append({
                            "name": name,
                            "description": meta.get("description", ""),
                            "parameters": meta.get("parameters", {}),
                            "enabled": meta.get("enabled", True),
                        })
                return tools
            except Exception as exc:
                logger.warning("Failed to load tools from persona: %s", exc)
                return []
        
        return cls(
            persona_manager=persona_manager,
            config_manager=config_manager,
            blackboard=blackboard,
            tool_resolver=tool_resolver,
        )
