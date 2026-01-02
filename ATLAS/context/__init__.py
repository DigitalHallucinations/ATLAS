"""Context management for ATLAS.

This package provides two complementary context management systems:

1. ExecutionContext - Request-scoped context carrying tenant_id, user_id,
   conversation_id, and trace_id. Propagates automatically across async
   boundaries using contextvars.

2. LLMContextManager - Assembles everything the LLM needs: system prompts,
   conversation history, tools (native + MCP), blackboard state, and
   task context with intelligent token budgeting.
"""

from ATLAS.context.execution import (
    ExecutionContext,
    get_current_context,
    get_context_or_default,
    set_current_context,
    clear_current_context,
    execution_context,
    require_context,
    ContextNotSetError,
    context_to_legacy_dict,
    context_from_legacy_dict,
)

from ATLAS.context.llm_context import (
    LLMContext,
    MessageEntry,
    TokenBudget,
    ToolDefinition,
)

from ATLAS.context.llm_context_manager import (
    LLMContextManager,
    estimate_tokens,
    get_model_context_limit,
    MODEL_CONTEXT_LIMITS,
)

__all__ = [
    # ExecutionContext
    "ExecutionContext",
    "get_current_context",
    "get_context_or_default",
    "set_current_context",
    "clear_current_context",
    "execution_context",
    "require_context",
    "ContextNotSetError",
    "context_to_legacy_dict",
    "context_from_legacy_dict",
    # LLMContext
    "LLMContext",
    "MessageEntry",
    "TokenBudget",
    "ToolDefinition",
    # LLMContextManager
    "LLMContextManager",
    "estimate_tokens",
    "get_model_context_limit",
    "MODEL_CONTEXT_LIMITS",
]
