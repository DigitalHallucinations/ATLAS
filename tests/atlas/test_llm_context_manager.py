"""Tests for LLMContext and LLMContextManager."""

import pytest

from ATLAS.context import (
    LLMContext,
    LLMContextManager,
    MessageEntry,
    TokenBudget,
    ToolDefinition,
    estimate_tokens,
    get_model_context_limit,
    MODEL_CONTEXT_LIMITS,
)


class TestToolDefinition:
    """Test ToolDefinition dataclass."""

    def test_basic_creation(self):
        """Test creating a basic tool definition."""
        tool = ToolDefinition(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        assert tool.name == "web_search"
        assert tool.description == "Search the web"
        assert tool.source == "native"
        assert tool.enabled is True

    def test_mcp_tool(self):
        """Test MCP tool with server specified."""
        tool = ToolDefinition(
            name="mcp.server.search",
            description="MCP search tool",
            parameters={},
            source="mcp",
            server="myserver",
        )
        assert tool.source == "mcp"
        assert tool.server == "myserver"

    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        result = tool.to_openai_format()
        
        assert result["type"] == "function"
        assert result["function"]["name"] == "test_tool"
        assert result["function"]["description"] == "A test tool"

    def test_to_anthropic_format(self):
        """Test conversion to Anthropic format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        result = tool.to_anthropic_format()
        
        assert result["name"] == "test_tool"
        assert result["description"] == "A test tool"
        assert "input_schema" in result

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = ToolDefinition(
            name="test",
            description="desc",
            parameters={"type": "object"},
            source="mcp",
            server="srv",
            enabled=True,
            requires_consent=True,
        )
        data = original.to_dict()
        restored = ToolDefinition.from_dict(data)
        
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.source == original.source
        assert restored.server == original.server


class TestTokenBudget:
    """Test TokenBudget dataclass."""

    def test_default_values(self):
        """Test default token budget values."""
        budget = TokenBudget()
        assert budget.model_limit == 128000
        assert budget.reserved_output == 4096
        assert budget.truncated is False

    def test_total_used(self):
        """Test total_used calculation."""
        budget = TokenBudget(
            system_tokens=1000,
            history_tokens=5000,
            tools_tokens=500,
        )
        assert budget.total_used == 6500

    def test_utilization(self):
        """Test utilization percentage."""
        budget = TokenBudget(
            model_limit=10000,
            reserved_output=2000,
            system_tokens=4000,  # 50% of usable (8000)
        )
        assert budget.utilization == 0.5

    def test_to_dict(self):
        """Test serialization."""
        budget = TokenBudget(
            model_limit=100000,
            system_tokens=500,
            truncated=True,
        )
        data = budget.to_dict()
        
        assert data["model_limit"] == 100000
        assert data["system_tokens"] == 500
        assert data["truncated"] is True
        assert "utilization" in data


class TestMessageEntry:
    """Test MessageEntry dataclass."""

    def test_basic_message(self):
        """Test basic message creation."""
        msg = MessageEntry(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_tool_message(self):
        """Test tool response message."""
        msg = MessageEntry(
            role="tool",
            content="Tool result",
            tool_call_id="call_123",
        )
        assert msg.tool_call_id == "call_123"

    def test_assistant_with_tool_calls(self):
        """Test assistant message with tool calls."""
        msg = MessageEntry(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "function": {"name": "test"}}],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_to_dict(self):
        """Test serialization."""
        msg = MessageEntry(role="user", content="Test")
        data = msg.to_dict()
        
        assert data["role"] == "user"
        assert data["content"] == "Test"


class TestLLMContext:
    """Test LLMContext dataclass."""

    def test_basic_creation(self):
        """Test basic LLM context creation."""
        ctx = LLMContext(
            system_prompt="You are helpful.",
            messages=[MessageEntry(role="user", content="Hi")],
        )
        assert ctx.system_prompt == "You are helpful."
        assert len(ctx.messages) == 1

    def test_has_tools(self):
        """Test has_tools property."""
        ctx_no_tools = LLMContext()
        assert ctx_no_tools.has_tools is False
        
        ctx_with_tools = LLMContext(tools=[
            ToolDefinition(name="test", description="", parameters={})
        ])
        assert ctx_with_tools.has_tools is True

    def test_enabled_tools(self):
        """Test enabled_tools filter."""
        ctx = LLMContext(tools=[
            ToolDefinition(name="enabled", description="", parameters={}, enabled=True),
            ToolDefinition(name="disabled", description="", parameters={}, enabled=False),
        ])
        enabled = ctx.enabled_tools
        
        assert len(enabled) == 1
        assert enabled[0].name == "enabled"

    def test_get_messages_as_dicts(self):
        """Test message conversion for provider API."""
        ctx = LLMContext(messages=[
            MessageEntry(role="user", content="Hello"),
            MessageEntry(role="assistant", content="Hi there"),
        ])
        dicts = ctx.get_messages_as_dicts()
        
        assert len(dicts) == 2
        assert dicts[0]["role"] == "user"
        assert dicts[1]["role"] == "assistant"

    def test_get_tools_for_provider_openai(self):
        """Test tool formatting for OpenAI."""
        ctx = LLMContext(tools=[
            ToolDefinition(name="test", description="Test tool", parameters={}),
        ])
        tools = ctx.get_tools_for_provider("OpenAI")
        
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    def test_get_tools_for_provider_anthropic(self):
        """Test tool formatting for Anthropic."""
        ctx = LLMContext(tools=[
            ToolDefinition(name="test", description="Test tool", parameters={}),
        ])
        tools = ctx.get_tools_for_provider("Anthropic")
        
        assert len(tools) == 1
        assert "input_schema" in tools[0]

    def test_summary(self):
        """Test context summary for logging."""
        ctx = LLMContext(
            messages=[MessageEntry(role="user", content="Test")],
            tools=[ToolDefinition(name="t", description="", parameters={})],
            token_budget=TokenBudget(
                model_limit=10000,
                system_tokens=100,
                history_tokens=200,
                tools_tokens=50,
            ),
        )
        summary = ctx.summary()
        
        assert "messages=1" in summary
        assert "tools=1" in summary
        assert "tokens=" in summary

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = LLMContext(
            system_prompt="Test prompt",
            messages=[MessageEntry(role="user", content="Hello")],
            tools=[ToolDefinition(name="test", description="", parameters={})],
            persona_name="TestPersona",
            model="gpt-4o",
            conversation_id="conv-123",
        )
        data = original.to_dict()
        restored = LLMContext.from_dict(data)
        
        assert restored.system_prompt == original.system_prompt
        assert restored.persona_name == original.persona_name
        assert restored.model == original.model
        assert len(restored.messages) == 1
        assert len(restored.tools) == 1


class TestTokenEstimation:
    """Test token estimation utilities."""

    def test_estimate_tokens_empty(self):
        """Test estimation for empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        """Test estimation for short string."""
        # "Hello" is 5 chars, ~1-2 tokens
        tokens = estimate_tokens("Hello")
        assert tokens >= 1

    def test_estimate_tokens_long(self):
        """Test estimation for longer string."""
        text = "a" * 400  # Should be ~100 tokens
        tokens = estimate_tokens(text)
        assert 80 <= tokens <= 120

    def test_get_model_context_limit_known(self):
        """Test getting limit for known model."""
        assert get_model_context_limit("gpt-4o") == 128000
        assert get_model_context_limit("claude-3-5-sonnet-20241022") == 200000

    def test_get_model_context_limit_prefix_match(self):
        """Test prefix matching for versioned models."""
        limit = get_model_context_limit("gpt-4o-2024-11-20")
        assert limit == 128000

    def test_get_model_context_limit_unknown(self):
        """Test fallback for unknown model."""
        limit = get_model_context_limit("unknown-model-xyz")
        assert limit == MODEL_CONTEXT_LIMITS["default"]


class TestLLMContextManager:
    """Test LLMContextManager."""

    def test_basic_instantiation(self):
        """Test basic manager creation."""
        manager = LLMContextManager()
        assert manager._persona_manager is None
        assert manager._config_manager is None

    @pytest.mark.asyncio
    async def test_build_context_basic(self):
        """Test basic context building."""
        manager = LLMContextManager()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        ctx = await manager.build_context(
            conversation_id="test-conv",
            messages=messages,
            model="gpt-4o",
            include_tools=False,
            include_blackboard=False,
        )
        
        assert ctx.conversation_id == "test-conv"
        assert ctx.model == "gpt-4o"
        assert len(ctx.messages) == 2

    @pytest.mark.asyncio
    async def test_build_context_with_system_override(self):
        """Test context building with system prompt override."""
        manager = LLMContextManager()
        
        ctx = await manager.build_context(
            conversation_id="test",
            messages=[],
            system_prompt_override="You are a helpful assistant.",
        )
        
        assert ctx.system_prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_build_context_with_additional_context(self):
        """Test context building with additional context injection."""
        manager = LLMContextManager()
        
        ctx = await manager.build_context(
            conversation_id="test",
            messages=[],
            system_prompt_override="Base prompt.",
            additional_system_context="Extra info here.",
        )
        
        assert "Base prompt." in ctx.system_prompt
        assert "Extra info here." in ctx.system_prompt

    @pytest.mark.asyncio
    async def test_build_context_with_task(self):
        """Test context building with task context injection."""
        manager = LLMContextManager()
        
        task_context = {
            "id": "task-123",
            "title": "Research topic",
            "description": "Find information about X",
            "status": "in_progress",
        }
        
        ctx = await manager.build_context(
            conversation_id="test",
            messages=[],
            task_context=task_context,
        )
        
        assert ctx.injected_context is not None
        assert "task" in ctx.injected_context
        assert "Task ID: task-123" in ctx.injected_context["task"]

    @pytest.mark.asyncio
    async def test_build_context_token_budget(self):
        """Test token budget is calculated."""
        manager = LLMContextManager()
        
        ctx = await manager.build_context(
            conversation_id="test",
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o",
        )
        
        assert ctx.token_budget.model_limit == 128000
        assert ctx.token_budget.history_tokens > 0

    @pytest.mark.asyncio
    async def test_context_truncation(self):
        """Test that long history is truncated."""
        manager = LLMContextManager()
        
        # Create many messages to exceed a small budget
        messages = [
            {"role": "user", "content": "x" * 1000}
            for _ in range(100)
        ]
        
        ctx = await manager.build_context(
            conversation_id="test",
            messages=messages,
            model="gpt-4o",
            max_history_tokens=1000,  # Force truncation
        )
        
        # Should have fewer messages due to truncation
        assert len(ctx.messages) < 100
        assert ctx.token_budget.truncated is True

    @pytest.mark.asyncio
    async def test_tool_resolver_integration(self):
        """Test custom tool resolver."""
        def mock_resolver():
            return [
                {
                    "name": "mock_tool",
                    "description": "A mock tool",
                    "parameters": {"type": "object"},
                }
            ]
        
        manager = LLMContextManager(tool_resolver=mock_resolver)
        
        ctx = await manager.build_context(
            conversation_id="test",
            messages=[],
            include_tools=True,
        )
        
        assert len(ctx.tools) == 1
        assert ctx.tools[0].name == "mock_tool"

    @pytest.mark.asyncio
    async def test_tool_deduplication(self):
        """Test that duplicate tools are deduplicated."""
        def native_resolver():
            return [{"name": "tool1", "description": "Native", "parameters": {}}]
        
        def mcp_resolver():
            return [{"name": "tool1", "description": "MCP", "parameters": {}}]
        
        manager = LLMContextManager(
            tool_resolver=native_resolver,
            mcp_tool_resolver=mcp_resolver,
        )
        
        ctx = await manager.build_context(
            conversation_id="test",
            messages=[],
        )
        
        # Should only have one tool (native preferred)
        assert len(ctx.tools) == 1
        assert ctx.tools[0].source == "native"
