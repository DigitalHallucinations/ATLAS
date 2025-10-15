import asyncio
import importlib
import json
import os
import sys
import types
from pathlib import Path
import shutil
import textwrap


def _clear_provider_env(monkeypatch):
    for key in [
        "OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "GOOGLE_API_KEY",
        "HUGGINGFACE_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROK_API_KEY",
        "XI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


def _ensure_yaml(monkeypatch):
    if "yaml" in sys.modules:
        return

    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda stream: {}
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)


def _ensure_dotenv(monkeypatch):
    if "dotenv" in sys.modules:
        return

    dotenv_stub = types.ModuleType("dotenv")

    def _load_dotenv(*_args, **_kwargs):  # pragma: no cover - stub helper
        return True

    def _set_key(*_args, **_kwargs):  # pragma: no cover - stub helper
        return None

    def _find_dotenv(*_args, **_kwargs):  # pragma: no cover - stub helper
        return ""

    dotenv_stub.load_dotenv = _load_dotenv
    dotenv_stub.set_key = _set_key
    dotenv_stub.find_dotenv = _find_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)


def test_tool_manager_import_without_credentials(monkeypatch):
    """Importing ToolManager should not require configuration credentials."""

    _clear_provider_env(monkeypatch)
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    sys.modules.pop("ATLAS.ToolManager", None)

    module = importlib.import_module("ATLAS.ToolManager")

    assert hasattr(module, "use_tool"), "ToolManager module should expose use_tool"


def test_use_tool_prefers_supplied_config_manager(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(self, user, conversation_id, response, timestamp):
            self.responses.append((user, conversation_id, response, timestamp))
            self.history.append({"role": "function", "name": "tool", "content": str(response)})

        def add_message(self, user, conversation_id, role, content, timestamp, **kwargs):
            self.messages.append((user, conversation_id, role, content, timestamp, kwargs))
            entry = {"role": role, "content": content}
            if kwargs:
                entry.update(kwargs)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "model-output"

    class DummyConfigManager:
        def __init__(self):
            self.provider_manager = DummyProviderManager()

    conversation_history = DummyConversationHistory()
    dummy_config = DummyConfigManager()

    async def echo_tool(value):
        return value

    message = {
        "function_call": {
            "name": "echo_tool",
            "arguments": json.dumps({"value": "ping"}),
        }
    }

    function_map = {"echo_tool": echo_tool}

    async def run_test():
        response = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map=function_map,
            functions=None,
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=dummy_config.provider_manager,
            config_manager=dummy_config,
        )

        assert response == "model-output"
        assert (
            dummy_config.provider_manager.generate_calls
        ), "Supplied config manager should be used"
        payload = dummy_config.provider_manager.generate_calls[0]
        assert payload["conversation_manager"] is conversation_history
        assert payload["conversation_id"] == "conversation"
        assert payload["user"] == "user"
        recorded_message = conversation_history.messages[-1]
        assert isinstance(recorded_message[3], str)
        assert recorded_message[3] == "model-output"

    asyncio.run(run_test())


def test_use_tool_handles_multiple_tool_calls(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(self, user, conversation_id, response, timestamp):
            self.responses.append((user, conversation_id, response, timestamp))
            self.history.append({"role": "function", "name": "tool", "content": str(response)})

        def add_message(self, user, conversation_id, role, content, timestamp, **kwargs):
            self.messages.append((user, conversation_id, role, content, timestamp, kwargs))
            entry = {"role": role, "content": content}
            if kwargs:
                entry.update(kwargs)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "model-output"

    conversation_history = DummyConversationHistory()
    provider = DummyProviderManager()
    tool_manager._tool_activity_log.clear()

    call_sequence = []

    async def async_tool(value):
        call_sequence.append(("async_tool", value))
        await asyncio.sleep(0)
        return f"async:{value}"

    def sync_tool(value):
        call_sequence.append(("sync_tool", value))
        return f"sync:{value}"

    message = {
        "tool_calls": [
            {"function": {"name": "async_tool", "arguments": json.dumps({"value": "one"})}},
            {"function": {"name": "sync_tool", "arguments": {"value": "two"}}},
        ]
    }

    function_map = {"async_tool": async_tool, "sync_tool": sync_tool}

    async def run_test():
        response = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map=function_map,
            functions=None,
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=None,
        )

        assert response == "model-output"
        assert call_sequence == [("async_tool", "one"), ("sync_tool", "two")]
        assert len(conversation_history.responses) == 2
        assert conversation_history.responses[0][2] == "async:one"
        assert conversation_history.responses[1][2] == "sync:two"

        assert provider.generate_calls, "Model should be called after executing tools"
        prompt = provider.generate_calls[0]["messages"][-1]["content"]
        assert "1. async_tool: async:one" in prompt
        assert "2. sync_tool: sync:two" in prompt

        log_entries = tool_manager.get_tool_activity_log()
        assert len(log_entries) >= 2
        assert log_entries[-2]["tool_name"] == "async_tool"
        assert log_entries[-1]["tool_name"] == "sync_tool"

    asyncio.run(run_test())


def test_call_model_with_new_prompt_collects_stream(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class StreamingProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)

            async def _generator():
                for chunk in ["Hello", " ", "world"]:
                    yield chunk

            return _generator()

    provider = StreamingProviderManager()

    result = asyncio.run(
        tool_manager.call_model_with_new_prompt(
            "Please continue",
            current_persona=None,
            messages=[{"role": "user", "content": "Hi"}],
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            functions=None,
            provider_manager=provider,
            conversation_manager=None,
            conversation_id="conversation",
            user="user",
        )
    )

    assert result == "Hello world"
    assert provider.generate_calls
    assert provider.generate_calls[0].get("stream") is False


def test_load_function_map_caches_by_persona(monkeypatch):
    persona_name = "CachePersona"
    persona_dir = Path("modules/Personas") / persona_name / "Toolbox"
    maps_path = persona_dir / "maps.py"
    module_name = f"persona_{persona_name}_maps"

    persona_dir.mkdir(parents=True, exist_ok=True)
    maps_path.write_text(
        textwrap.dedent(
            """
            EXECUTION_COUNTER = globals().get("EXECUTION_COUNTER", 0) + 1

            def sample_tool():
                return "ok"

            function_map = {"sample_tool": sample_tool}
            """
        )
    )

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )
    monkeypatch.setitem(tool_manager.__dict__, "_function_map_cache", {})
    sys.modules.pop(module_name, None)

    spec_calls = {"count": 0}
    original_spec = importlib.util.spec_from_file_location

    def counting_spec(*args, **kwargs):
        spec_calls["count"] += 1
        return original_spec(*args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", counting_spec)

    persona_payload = {"name": persona_name}

    try:
        first_map = tool_manager.load_function_map_from_current_persona(persona_payload)
        second_map = tool_manager.load_function_map_from_current_persona(persona_payload)

        assert first_map is second_map
        assert spec_calls["count"] == 1

        module = sys.modules[module_name]
        assert getattr(module, "EXECUTION_COUNTER", 0) == 1
    finally:
        sys.modules.pop(module_name, None)
        shutil.rmtree(Path("modules/Personas") / persona_name, ignore_errors=True)
