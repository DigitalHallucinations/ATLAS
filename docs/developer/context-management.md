---
audience: Contributors building features that need request-scoped or LLM context management
status: draft
last_verified: 2026-01-01
source_of_truth: ATLAS/context/; ATLAS/provider_manager.py
---

# Context Management

ATLAS provides two complementary context management systems:

1. **ExecutionContext** — Request-scoped context for tenant, user, and conversation tracking
2. **LLMContext / LLMContextManager** — Complete context assembly for LLM API calls

## ExecutionContext

`ExecutionContext` is an immutable dataclass that carries request-scoped information across async boundaries using Python's `contextvars`.

### Basic Usage

```python
from ATLAS.context import ExecutionContext, execution_context, get_current_context

# Create a context
ctx = ExecutionContext(
    tenant_id="acme-corp",
    user_id="user123",
    conversation_id="conv456",
    roles=("admin", "user"),
)

# Use as a context manager (recommended)
with execution_context(ctx):
    # All code here can access the context
    current = get_current_context()
    print(current.tenant_id)  # "acme-corp"
```

### Async Propagation

Context automatically propagates across `await` boundaries:

```python
async def inner_function():
    ctx = get_current_context()
    return ctx.tenant_id

async def outer_function():
    ctx = ExecutionContext(tenant_id="test")
    with execution_context(ctx):
        result = await inner_function()
        assert result == "test"
```

### Immutable Updates

Create modified contexts using `with_*` methods:

```python
ctx = ExecutionContext(tenant_id="acme")

# Each returns a NEW context
ctx2 = ctx.with_user("user123")
ctx3 = ctx2.with_conversation("conv456")
ctx4 = ctx3.with_roles(("system",))
ctx5 = ctx4.with_metadata(source="api")
```

### Decorator for Required Context

Use `@require_context` to enforce context presence:

```python
from ATLAS.context import require_context

@require_context
def process_request():
    ctx = get_current_context()
    # Guaranteed to have a context here
    return ctx.tenant_id
```

### Server Context Conversion

Convert between ExecutionContext and server RequestContext:

```python
# From ExecutionContext to server context
server_ctx = exec_ctx.to_server_context()

# From server context to ExecutionContext
exec_ctx = ExecutionContext.from_server_context(server_ctx)
```

### ATLAS Core Integration

The ATLAS class provides helper methods:

```python
# Get current execution context (creates one if needed)
# Automatically includes user identity and roles from config
ctx = atlas.get_execution_context(conversation_id="conv123")

# Get a system-level context for privileged operations
sys_ctx = atlas.get_system_context()  # roles=("system",)

# Convert to dict for APIs that need it
context_dict = ctx.to_dict()
```

## LLMContext and LLMContextManager

`LLMContextManager` assembles everything needed for an LLM API call:

- System prompt with injected context
- Conversation history (token-budgeted)
- Tool definitions (local and MCP)
- Blackboard state
- Active task context

### Building LLM Context

```python
from ATLAS.context import LLMContextManager

manager = LLMContextManager(
    model_name="gpt-4o",
    system_prompt="You are a helpful assistant.",
    history=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ],
    tools=[
        {"name": "search", "description": "Search the web", "parameters": {...}},
    ],
    mcp_client=mcp_client,           # Optional: MCP tool discovery
    blackboard_facade=blackboard,     # Optional: Inject blackboard state
    active_task=task_dict,            # Optional: Inject task context
)

# Build the complete context
llm_context = await manager.build_context(max_history_tokens=4000)
```

### Token Budgeting

LLMContextManager automatically:

1. Estimates token usage for system prompt, tools, and history
2. Truncates history from the oldest messages to fit within limits
3. Preserves a configurable reserve for the model's response

```python
# Model limits are looked up automatically
# GPT-4o: 128K, Claude-3.5: 200K, etc.

# Override history budget if needed
context = await manager.build_context(max_history_tokens=8000)
```

### Provider-Specific Formatting

LLMContext formats messages and tools for different providers:

```python
# Format for OpenAI
messages = llm_context.format_messages_for_provider("OpenAI")
tools = llm_context.format_tools_for_provider("OpenAI")

# Format for Anthropic
messages = llm_context.format_messages_for_provider("Anthropic")
tools = llm_context.format_tools_for_provider("Anthropic")
```

### ProviderManager Integration

Use ProviderManager's convenience methods:

```python
# Build context
llm_context = await provider_manager.build_llm_context(
    model="gpt-4o",
    system_prompt="You are helpful.",
    history=messages,
    tools=tool_definitions,
    mcp_client=mcp,
    blackboard_facade=blackboard,
)

# Generate with pre-built context
response = await provider_manager.generate_with_context(
    llm_context,
    provider="OpenAI",
    stream=True,
    conversation_id="conv123",
)
```

## MCP Tool Resolution

LLMContextManager resolves tools from MCP (Model Context Protocol) servers:

```python
manager = LLMContextManager(
    model_name="gpt-4o",
    system_prompt="...",
    mcp_client=mcp_client,  # Async MCP client
)

# MCP tools are discovered and merged with local tools
context = await manager.build_context()
print(context.mcp_tools)  # List of ToolDefinition from MCP
```

## Blackboard State Injection

When a blackboard facade is provided, its state is injected into the system prompt:

```python
manager = LLMContextManager(
    model_name="gpt-4o",
    system_prompt="Base instructions.",
    blackboard_facade=blackboard,
)

context = await manager.build_context()
# System prompt now includes:
# "Base instructions.\n\n## Shared Context\n\n..."
```

## API Reference

### ExecutionContext Fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `tenant_id` | `str` | Tenant identifier (default: "default") |
| `user_id` | `str \| None` | User identifier |
| `conversation_id` | `str \| None` | Conversation identifier |
| `trace_id` | `str` | Unique trace ID (auto-generated) |
| `roles` | `tuple[str, ...]` | Authorization roles |
| `metadata` | `dict \| None` | Additional metadata |

### LLMContext

| Field | Type | Description |
| ----- | ---- | ----------- |
| `model_name` | `str` | Model identifier |
| `system_prompt` | `str` | Complete system prompt |
| `history` | `list[MessageEntry]` | Conversation messages |
| `tools` | `list[ToolDefinition]` | Local tool definitions |
| `mcp_tools` | `list[ToolDefinition]` | MCP-discovered tools |
| `blackboard_state` | `str \| None` | Injected blackboard context |
| `token_budget` | `TokenBudget` | Token allocation info |

### Context Functions

| Function | Description |
| -------- | ----------- |
| `get_current_context()` | Get active context or None |
| `get_context_or_default()` | Get active context or create default |
| `set_current_context(ctx)` | Set the active context |
| `clear_current_context()` | Clear the active context |
| `execution_context(ctx)` | Context manager for scoped context |
| `require_context` | Decorator requiring active context |

## See Also

- [Architecture Overview](../architecture-overview.md) — System design
- [AtlasServer API](../server/api.md) — Server routes using context
- [Tool Manifest](../tool-manifest.md) — Tool definition format
