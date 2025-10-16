import asyncio
import importlib
import json
import os
import sys
import types
import time
from collections.abc import AsyncIterator as AsyncIteratorABC
from pathlib import Path
import shutil
import textwrap


def _normalize_tool_response_payload(response):
    def _clone(value):
        if isinstance(value, dict):
            return {key: _clone(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_clone(item) for item in value]
        if isinstance(value, tuple):
            return [_clone(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    if isinstance(response, dict):
        return _clone(response)

    if isinstance(response, list):
        normalized = []
        for item in response:
            if isinstance(item, (dict, list, tuple)):
                normalized.append(_normalize_tool_response_payload(item))
            else:
                normalized.append(
                    {"type": "output_text", "text": "" if item is None else str(item)}
                )
        return normalized

    if isinstance(response, tuple):
        return _normalize_tool_response_payload(list(response))

    return {"type": "output_text", "text": "" if response is None else str(response)}


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

def _ensure_pytz(monkeypatch):
    if "pytz" in sys.modules:
        return

    import datetime

    pytz_stub = types.SimpleNamespace(
        timezone=lambda *_args, **_kwargs: datetime.timezone.utc,
        utc=datetime.timezone.utc,
    )
    monkeypatch.setitem(sys.modules, "pytz", pytz_stub)


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
    _ensure_pytz(monkeypatch)
    sys.modules.pop("ATLAS.ToolManager", None)

    module = importlib.import_module("ATLAS.ToolManager")

    assert hasattr(module, "use_tool"), "ToolManager module should expose use_tool"


def test_use_tool_prefers_supplied_config_manager(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
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
            "id": "call-echo",
            "name": "echo_tool",
            "arguments": json.dumps({"value": "ping"}),
        }
    }

    function_map = {"echo_tool": echo_tool}
    tool_manager._tool_activity_log.clear()

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
        assert payload["messages"] == [
            {"role": "user", "content": "Hi"},
            {
                "role": "tool",
                "content": {"type": "output_text", "text": "ping"},
                "tool_call_id": "call-echo",
            },
        ]
        recorded_message = conversation_history.messages[-1]
        assert recorded_message["content"] == [
            {"type": "output_text", "text": "model-output"}
        ]
        assert conversation_history.responses[-1]["tool_call_id"] == "call-echo"
        log_entries = tool_manager.get_tool_activity_log()
        assert log_entries[-1]["tool_call_id"] == "call-echo"

    asyncio.run(run_test())


def test_use_tool_records_structured_follow_up(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
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
            return {
                "content": [
                    {"type": "output_text", "text": "Here is data:"},
                    {"type": "output_json", "json": {"value": 1}},
                ],
                "audio": {"voice": "tester"},
            }

    conversation_history = DummyConversationHistory()
    provider = DummyProviderManager()

    async def echo_tool(value):
        return value

    message = {
        "function_call": {
            "id": "call-echo",
            "name": "echo_tool",
            "arguments": json.dumps({"value": "ping"}),
        }
    }

    async def run_test():
        response = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map={"echo_tool": echo_tool},
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

        assert response == "Here is data:"
        assert provider.generate_calls
        stored_message = conversation_history.messages[-1]
        assert stored_message["content"] == [
            {"type": "output_text", "text": "Here is data:"},
            {"type": "output_json", "json": {"value": 1}},
        ]
        assert stored_message.get("audio") == {"voice": "tester"}

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

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
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
            {
                "id": "call-async",
                "function": {"name": "async_tool", "arguments": json.dumps({"value": "one"})},
            },
            {
                "id": "call-sync",
                "function": {"name": "sync_tool", "arguments": {"value": "two"}},
            },
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
        assert conversation_history.responses[0]["content"]["text"] == "async:one"
        assert conversation_history.responses[0]["tool_call_id"] == "call-async"
        assert conversation_history.responses[1]["content"]["text"] == "sync:two"
        assert conversation_history.responses[1]["tool_call_id"] == "call-sync"

        tool_entries = [
            entry for entry in conversation_history.history if entry.get("role") == "tool"
        ]
        assert [entry.get("tool_call_id") for entry in tool_entries[-2:]] == [
            "call-async",
            "call-sync",
        ]

        assert provider.generate_calls, "Model should be called after executing tools"
        assert provider.generate_calls[0]["messages"] == [
            {"role": "user", "content": "Hi"},
            {
                "role": "tool",
                "content": {"type": "output_text", "text": "async:one"},
                "tool_call_id": "call-async",
            },
            {
                "role": "tool",
                "content": {"type": "output_text", "text": "sync:two"},
                "tool_call_id": "call-sync",
            },
        ]

        log_entries = tool_manager.get_tool_activity_log()
        assert len(log_entries) >= 2
        assert log_entries[-2]["tool_name"] == "async_tool"
        assert log_entries[-2]["tool_call_id"] == "call-async"
        assert log_entries[-1]["tool_name"] == "sync_tool"
        assert log_entries[-1]["tool_call_id"] == "call-sync"

    asyncio.run(run_test())


def test_use_tool_runs_sync_tool_in_thread(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
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

    def slow_tool(value):
        time.sleep(0.2)
        return f"slow:{value}"

    message = {
        "function_call": {
            "id": "call-slow",
            "name": "slow_tool",
            "arguments": json.dumps({"value": "one"}),
        }
    }

    function_map = {"slow_tool": slow_tool}

    async def run_test():
        task = asyncio.create_task(
            tool_manager.use_tool(
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
        )

        await asyncio.sleep(0.05)
        assert not task.done(), "use_tool should not block the event loop when running sync tools"

        response = await task
        assert response == "model-output"
        assert conversation_history.responses[-1]["content"]["text"] == "slow:one"
        assert conversation_history.responses[-1]["tool_call_id"] == "call-slow"
        log_entries = tool_manager.get_tool_activity_log()
        assert log_entries[-1]["tool_call_id"] == "call-slow"
        assert provider.generate_calls, "Provider should be invoked after tool execution"
        assert provider.generate_calls[0]["messages"] == [
            {"role": "user", "content": "Hi"},
            {
                "role": "tool",
                "content": {"type": "output_text", "text": "slow:one"},
                "tool_call_id": "call-slow",
            },
        ]

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
            messages=[{"role": "user", "content": "Hi"}],
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            functions=None,
            provider_manager=provider,
            conversation_manager=None,
            conversation_id="conversation",
            user="user",
            prompt="Please continue",
        )
    )

    assert result == "Hello world"
    assert provider.generate_calls
    assert provider.generate_calls[0].get("stream") is False


def test_use_tool_streams_final_response_when_enabled(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class StreamingProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)

            async def _generator():
                for chunk in ["stream", "-", "result"]:
                    yield chunk

            return _generator()

    provider = StreamingProviderManager()

    async def tool_fn(value):
        return f"tool:{value}"

    conversation_history = DummyConversationHistory()

    message = {
        "function_call": {
            "id": "call-stream",
            "name": "tool_fn",
            "arguments": json.dumps({"value": "input"}),
        }
    }

    tool_manager._tool_activity_log.clear()

    persona = types.SimpleNamespace(name="Persona-Alpha")

    async def run_test():
        response = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map={"tool_fn": tool_fn},
            functions=None,
            current_persona=persona,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=None,
            stream=True,
        )

        assert isinstance(response, AsyncIteratorABC)
        chunks = []
        async for piece in response:
            chunks.append(piece)
        return "".join(chunks), list(chunks)

    collected, chunk_list = asyncio.run(run_test())

    assert collected == "stream-result"
    assert chunk_list == ["stream", "-", "result"]
    assert provider.generate_calls
    assert provider.generate_calls[0].get("stream") is True
    assert conversation_history.responses
    assert conversation_history.messages, "Assistant follow-up should be stored after streaming"
    recorded_message = conversation_history.messages[-1]
    assert recorded_message["content"] == [
        {"type": "output_text", "text": "stream"},
        {"type": "output_text", "text": "-"},
        {"type": "output_text", "text": "result"},
    ]
    assert recorded_message.get("metadata", {}).get("tool_call_ids") == [
        "call-stream"
    ]
    assert recorded_message.get("metadata", {}).get("persona") == "Persona-Alpha"


def test_use_tool_consumes_async_generator_tool(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
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

    async def generator_tool(value):
        print("tool-start")
        yield {"type": "output_text", "text": value}
        print("tool-middle")
        await asyncio.sleep(0)
        print("tool-warning", file=sys.stderr)
        yield {"type": "output_text", "text": value.upper()}
        print("tool-end")

    message = {
        "function_call": {
            "id": "call-generator",
            "name": "generator_tool",
            "arguments": json.dumps({"value": "chunk"}),
        }
    }

    tool_manager._tool_activity_log.clear()

    async def run_test():
        return await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map={"generator_tool": generator_tool},
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

    response = asyncio.run(run_test())

    assert response == "model-output"
    assert conversation_history.responses
    assert conversation_history.messages

    tool_entry = conversation_history.responses[0]
    expected_chunks = [
        {"type": "output_text", "text": "chunk"},
        {"type": "output_text", "text": "CHUNK"},
    ]
    assert tool_entry["content"] == _normalize_tool_response_payload(expected_chunks)
    assert conversation_history.messages[0]["content"] == [
        {"type": "output_text", "text": "model-output"}
    ]

    assert provider.generate_calls

    log_entry = tool_manager._tool_activity_log[-1]
    assert log_entry["result"] == expected_chunks
    assert "tool-start" in log_entry["stdout"]
    assert "tool-middle" in log_entry["stdout"]
    assert "tool-end" in log_entry["stdout"]
    assert "tool-warning" in log_entry["stderr"]


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

        maps_path.write_text(
            textwrap.dedent(
                """
                EXECUTION_COUNTER = globals().get("EXECUTION_COUNTER", 0) + 1

                def updated_tool():
                    return "updated"

                function_map = {"updated_tool": updated_tool}
                """
            )
        )

        new_time = os.path.getmtime(maps_path) + 1
        os.utime(maps_path, (new_time, new_time))

        third_map = tool_manager.load_function_map_from_current_persona(persona_payload)

        assert spec_calls["count"] == 2
        assert third_map is not first_map
        assert "updated_tool" in third_map

    finally:
        sys.modules.pop(module_name, None)
        shutil.rmtree(Path("modules/Personas") / persona_name, ignore_errors=True)


def test_load_function_map_falls_back_to_default(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    persona_name = "MissingPersonaFallback"
    persona_dir = Path("modules/Personas") / persona_name
    shutil.rmtree(persona_dir, ignore_errors=True)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    tool_manager._function_map_cache.clear()
    tool_manager._default_function_map_cache = None
    sys.modules.pop(f"persona_{persona_name}_maps", None)
    sys.modules.pop("modules.Tools.tool_maps.maps", None)

    try:
        function_map = tool_manager.load_function_map_from_current_persona({"name": persona_name})

        assert isinstance(function_map, dict)
        assert "google_search" in function_map
        assert persona_name not in tool_manager._function_map_cache
        assert tool_manager._default_function_map_cache is not None
    finally:
        sys.modules.pop(f"persona_{persona_name}_maps", None)
        tool_manager._default_function_map_cache = None
        shutil.rmtree(persona_dir, ignore_errors=True)


def test_use_tool_resolves_google_search_with_default_map(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    persona_name = "PersonaWithoutToolbox"
    persona_dir = Path("modules/Personas") / persona_name
    shutil.rmtree(persona_dir, ignore_errors=True)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    tool_manager._function_map_cache.clear()
    tool_manager._default_function_map_cache = None
    sys.modules.pop(f"persona_{persona_name}_maps", None)
    sys.modules.pop("modules.Tools.tool_maps.maps", None)

    persona_payload = {"name": persona_name}

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "hello"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
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
            return "final-response"

    shared_map = tool_manager.load_function_map_from_current_persona(persona_payload, refresh=True)

    assert isinstance(shared_map, dict)
    assert "google_search" in shared_map

    captured_args = {}

    async def fake_google_search(query, k=None):
        captured_args["query"] = query
        captured_args["k"] = k
        return {"query": query, "k": k}

    function_map = dict(shared_map)
    function_map["google_search"] = fake_google_search

    conversation_history = DummyConversationHistory()
    provider_manager = DummyProviderManager()

    message = {
        "tool_calls": [
            {
                "id": "call-google",
                "type": "tool",
                "function": {
                    "name": "google_search",
                    "arguments": json.dumps({"query": "atlas project", "k": 3}),
                },
            }
        ]
    }

    tool_manager._tool_activity_log.clear()

    async def run_test():
        return await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map=function_map,
            functions=None,
            current_persona=persona_payload,
            temperature_var=0.1,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider_manager,
            config_manager=None,
        )

    try:
        result = asyncio.run(run_test())

        assert result == "final-response"
        assert captured_args == {"query": "atlas project", "k": 3}
        assert conversation_history.responses
        response_entry = conversation_history.responses[-1]
        assert response_entry["tool_call_id"] == "call-google"
        assert response_entry["content"] == {"query": "atlas project", "k": 3}
        assert provider_manager.generate_calls
        final_entry = conversation_history.messages[-1]
        assert final_entry["content"] == [
            {"type": "output_text", "text": "final-response"}
        ]
        assert persona_name not in tool_manager._function_map_cache
        assert tool_manager._default_function_map_cache is not None
    finally:
        sys.modules.pop(f"persona_{persona_name}_maps", None)
        tool_manager._default_function_map_cache = None
        shutil.rmtree(persona_dir, ignore_errors=True)

