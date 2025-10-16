import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest


class _StubChat:
    last_complete_kwargs = None
    last_stream_kwargs = None

    def complete(self, **kwargs):
        _StubChat.last_complete_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    def stream(self, **kwargs):
        _StubChat.last_stream_kwargs = kwargs

        class _Iterator:
            def __iter__(self):
                return iter([])

        return _Iterator()


class _StubMistral:
    init_kwargs = None

    def __init__(self, **_kwargs):
        _StubMistral.init_kwargs = dict(_kwargs)
        self.chat = _StubChat()

    APIError = Exception


if "mistralai" not in sys.modules:
    sys.modules["mistralai"] = SimpleNamespace(
        Mistral=_StubMistral,
        APIError=Exception,
    )

if "tenacity" not in sys.modules or not hasattr(sys.modules.get("tenacity"), "AsyncRetrying"):
    class _TenacityStub(SimpleNamespace):
        class _AsyncRetrying:
            def __init__(self, **_kwargs):
                self._kwargs = _kwargs

            def __aiter__(self):
                class _Attempt(SimpleNamespace):
                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, *_args):
                        return False

                async def _iterator():
                    yield _Attempt()

                return _iterator()

        @staticmethod
        def retry(*_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

    tenacity_stub = _TenacityStub(
        AsyncRetrying=_TenacityStub._AsyncRetrying,
        stop_after_attempt=lambda *_a, **_k: None,
        wait_exponential=lambda *_a, **_k: None,
    )
    sys.modules["tenacity"] = tenacity_stub

if "yaml" not in sys.modules:
    sys.modules["yaml"] = SimpleNamespace(
        safe_load=lambda *_a, **_k: {},
        dump=lambda *_a, **_k: None,
    )

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = SimpleNamespace(
        load_dotenv=lambda *_a, **_k: None,
        set_key=lambda *_a, **_k: None,
        find_dotenv=lambda *_a, **_k: "",
    )

os.environ.setdefault("OPENAI_API_KEY", "test-openai")
os.environ.setdefault("MISTRAL_API_KEY", "test-mistral")

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "Providers"
    / "Mistral"
    / "Mistral_gen_response.py"
)
spec = importlib.util.spec_from_file_location("mistral_module", MODULE_PATH)
mistral_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mistral_module)


class DummyModelManager:
    def __init__(self, _config):
        self.current_model = None

    def get_current_model(self):
        return self.current_model

    def set_model(self, model, _provider):
        self.current_model = model


class RecordingConversation:
    def __init__(self):
        self.records = []

    def add_message(
        self,
        user,
        conversation_id,
        role,
        content,
        timestamp=None,
        metadata=None,
        **extra,
    ):
        entry = {
            "user": user,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
        }
        if metadata is not None:
            entry["metadata"] = metadata
        entry.update({key: value for key, value in extra.items() if value is not None})
        self.records.append(entry)
        return entry

    def get_history(self, *_args, **_kwargs):  # pragma: no cover - helper for compatibility
        return list(self.records)


def _build_generator(config):
    model_manager = DummyModelManager(config)
    return mistral_module.MistralGenerator(config, model_manager=model_manager)


class DummyConfig:
    def __init__(self, settings):
        self._settings = settings

    def get_mistral_api_key(self):
        return "unit-test-key"

    def get_mistral_llm_settings(self):
        return dict(self._settings)

    @staticmethod
    def _coerce_stop_sequences(value):
        if value is None or value == "":
            return []
        if isinstance(value, str):
            tokens = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple, set)):
            tokens = []
            for item in value:
                if item in {None, ""}:
                    continue
                if not isinstance(item, str):
                    raise ValueError("Stop sequences must be strings.")
                cleaned = item.strip()
                if cleaned:
                    tokens.append(cleaned)
        else:
            raise ValueError("Invalid stop sequences payload.")
        if len(tokens) > 4:
            raise ValueError("Stop sequences cannot contain more than 4 entries.")
        return tokens


def test_mistral_generator_uses_supplied_model_manager(monkeypatch):
    settings = {
        "model": "mistral-unit-test",
    }
    config = DummyConfig(settings)
    provided_manager = DummyModelManager(config)

    class _FailingModelManager(DummyModelManager):
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Fallback ModelManager should not be instantiated")

    monkeypatch.setattr(mistral_module, "ModelManager", _FailingModelManager)

    generator = mistral_module.MistralGenerator(config, model_manager=provided_manager)

    assert generator.model_manager is provided_manager
    assert provided_manager.get_current_model() == "mistral-unit-test"


def _reset_stubs():
    _StubChat.last_complete_kwargs = None
    _StubChat.last_stream_kwargs = None
    _StubMistral.init_kwargs = None


@pytest.fixture(autouse=True)
def _patches(monkeypatch):
    monkeypatch.setattr(
        mistral_module,
        "load_functions_from_json",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mistral_module,
        "load_function_map_from_current_persona",
        lambda *_args, **_kwargs: {},
    )
    _reset_stubs()


def test_mistral_generator_applies_config_defaults():
    settings = {
        "model": "mistral-unit-test",
        "temperature": 0.42,
        "top_p": 0.55,
        "max_tokens": 321,
        "safe_prompt": True,
        "stream": False,
        "random_seed": 17,
        "frequency_penalty": 0.3,
        "presence_penalty": -0.2,
        "tool_choice": "none",
        "parallel_tool_calls": False,
        "stop_sequences": ["<END>", "<STOP>"],
    }

    generator = _build_generator(DummyConfig(settings))

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    kwargs = _StubChat.last_complete_kwargs
    assert kwargs["model"] == "mistral-unit-test"
    assert kwargs["temperature"] == 0.42
    assert kwargs["top_p"] == 0.55
    assert kwargs["max_tokens"] == 321
    assert kwargs["safe_prompt"] is True
    assert kwargs["random_seed"] == 17
    assert kwargs["frequency_penalty"] == 0.3
    assert kwargs["presence_penalty"] == -0.2
    assert kwargs["tool_choice"] == "none"
    assert kwargs["stop"] == ["<END>", "<STOP>"]
    assert _StubChat.last_stream_kwargs is None


def test_mistral_generator_includes_prompt_mode_when_configured():
    settings = {
        "model": "mistral-unit-test",
        "stream": False,
        "prompt_mode": "reasoning",
    }

    generator = _build_generator(DummyConfig(settings))

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    kwargs = _StubChat.last_complete_kwargs
    assert kwargs["prompt_mode"] == "reasoning"


def test_mistral_generator_respects_base_url_changes():
    settings = {
        "model": "mistral-unit-test",
        "stream": False,
        "base_url": "https://custom.mistral/v1",
    }

    config = DummyConfig(settings)
    generator = _build_generator(config)

    assert _StubMistral.init_kwargs["server_url"] == "https://custom.mistral/v1"

    settings["base_url"] = "https://alt.mistral/v2"

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Ping"}],
        )

    result = asyncio.run(exercise())
    assert result == "ok"
    assert _StubMistral.init_kwargs["server_url"] == "https://alt.mistral/v2"


def test_mistral_generator_streams_when_configured_by_default():
    settings = {
        "model": "mistral-stream-test",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": None,
        "safe_prompt": False,
        "stream": True,
        "random_seed": None,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stop_sequences": ["END"],
        "max_retries": 4,
        "retry_min_seconds": 2,
        "retry_max_seconds": 8,
    }

    generator = _build_generator(DummyConfig(settings))

    async def exercise():
        stream = await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )
        chunks = [chunk async for chunk in stream]
        return chunks

    chunks = asyncio.run(exercise())

    assert chunks == []
    assert _StubChat.last_complete_kwargs is None
    stream_kwargs = _StubChat.last_stream_kwargs
    assert stream_kwargs["model"] == "mistral-stream-test"
    assert stream_kwargs["temperature"] == 0.0
    assert stream_kwargs["stop"] == ["END"]


def test_mistral_generator_configures_retry_policy(monkeypatch):
    settings = {
        "model": "mistral-large-latest",
        "stream": False,
        "max_retries": 5,
        "retry_min_seconds": 3,
        "retry_max_seconds": 12,
    }

    generator = _build_generator(DummyConfig(settings))

    recorded: Dict[str, Any] = {}

    class _FakeStop:
        def __init__(self, attempts: int):
            self.attempts = attempts

    def fake_stop_after_attempt(value: int):
        recorded["stop_value"] = value
        return _FakeStop(value)

    class _FakeWait:
        def __init__(self, *, multiplier: int, min: int, max: int):
            self.multiplier = multiplier
            self.min = min
            self.max = max

    def fake_wait_exponential(*, multiplier: int, min: int, max: int):
        recorded["wait"] = {"multiplier": multiplier, "min": min, "max": max}
        return _FakeWait(multiplier=multiplier, min=min, max=max)

    class _FakeAttempt:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, _exc, _tb):
            return False

    class _FakeAsyncRetrying:
        last_kwargs: Optional[Dict[str, Any]] = None

        def __init__(self, **kwargs: Any):
            _FakeAsyncRetrying.last_kwargs = kwargs

        def __aiter__(self):
            async def _iterator():
                yield _FakeAttempt()

            return _iterator()

    monkeypatch.setattr(mistral_module, "stop_after_attempt", fake_stop_after_attempt)
    monkeypatch.setattr(mistral_module, "wait_exponential", fake_wait_exponential)
    monkeypatch.setattr(mistral_module, "AsyncRetrying", _FakeAsyncRetrying)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    assert recorded["stop_value"] == 5
    assert recorded["wait"] == {"multiplier": 1, "min": 3, "max": 12}
    assert isinstance(_FakeAsyncRetrying.last_kwargs["stop"], _FakeStop)
    assert isinstance(_FakeAsyncRetrying.last_kwargs["wait"], _FakeWait)
    assert _FakeAsyncRetrying.last_kwargs["reraise"] is True


def test_mistral_generator_retries_on_api_error(monkeypatch):
    settings = {
        "model": "mistral-large-latest",
        "stream": False,
        "max_retries": 2,
        "retry_min_seconds": 1,
        "retry_max_seconds": 5,
    }

    generator = _build_generator(DummyConfig(settings))

    call_counts = {"complete": 0}

    class _FakeStop:
        def __init__(self, attempts: int):
            self.max_attempts = attempts

    class _RetryingAttempt:
        def __init__(self, retrying: "_RetryingLoop", number: int):
            self.retrying = retrying
            self.number = number

        def __enter__(self):
            return self

        def __exit__(self, exc_type, _exc, _tb):
            if exc_type is None:
                return False
            if self.number >= self.retrying.stop.max_attempts:
                return False
            return True

    class _RetryingLoop:
        def __init__(self, *, stop: _FakeStop, wait: Any, reraise: bool):
            self.stop = stop
            self.wait = wait
            self.reraise = reraise
            self._attempt = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._attempt >= self.stop.max_attempts:
                raise StopAsyncIteration
            self._attempt += 1
            return _RetryingAttempt(self, self._attempt)

    def fake_stop_after_attempt(value: int):
        return _FakeStop(value)

    def fake_wait_exponential(**_kwargs: Any):
        return SimpleNamespace()

    def failing_complete(**kwargs):
        call_counts["complete"] += 1
        if call_counts["complete"] == 1:
            raise mistral_module.Mistral.APIError("boom")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    generator.client.chat.complete = failing_complete

    monkeypatch.setattr(mistral_module, "stop_after_attempt", fake_stop_after_attempt)
    monkeypatch.setattr(mistral_module, "wait_exponential", fake_wait_exponential)
    monkeypatch.setattr(mistral_module, "AsyncRetrying", _RetryingLoop)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    assert call_counts["complete"] == 2

def test_mistral_generator_omits_max_tokens_when_using_provider_default():
    settings = {
        "model": "mistral-large-latest",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": None,
        "safe_prompt": False,
        "stream": False,
        "random_seed": None,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stop_sequences": [],
    }

    generator = _build_generator(DummyConfig(settings))

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    kwargs = _StubChat.last_complete_kwargs
    assert "max_tokens" not in kwargs


def test_mistral_generator_treats_zero_override_as_provider_default():
    settings = {
        "model": "mistral-large-latest",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "safe_prompt": False,
        "stream": False,
        "random_seed": None,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stop_sequences": [],
    }

    generator = _build_generator(DummyConfig(settings))

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=0,
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    kwargs = _StubChat.last_complete_kwargs
    assert "max_tokens" not in kwargs


def test_mistral_generator_translates_functions_to_tools():
    settings = {
        "model": "mistral-large-latest",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "stream": False,
        "stop_sequences": [],
    }

    generator = _build_generator(DummyConfig(settings))

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "call tool"}],
            functions=[
                {
                    "name": "do_something",
                    "description": "Performs an action",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            top_p=0.25,
            parallel_tool_calls=False,
            tool_choice={"type": "function", "function": {"name": "do_something"}},
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    kwargs = _StubChat.last_complete_kwargs
    assert kwargs["top_p"] == 0.25
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "do_something"}}
    tools = kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "do_something"
    assert tools[0]["function"]["description"] == "Performs an action"
    assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_mistral_generator_returns_structured_tool_error(monkeypatch):
    settings = {"model": "mistral-large-latest", "stream": False}
    generator = _build_generator(DummyConfig(settings))

    error_entry = {
        "role": "tool",
        "tool_call_id": "call-error",
        "content": [{"type": "output_text", "text": "boom"}],
        "metadata": {"status": "error", "name": "broken", "error_type": "execution_error"},
    }

    async def failing_use_tool(**_kwargs):
        raise mistral_module.ToolExecutionError(
            "boom",
            tool_call_id="call-error",
            function_name="broken",
            error_type="execution_error",
            entry=error_entry,
        )

    monkeypatch.setattr(mistral_module, "use_tool", failing_use_tool)

    tool_messages = [
        {
            "type": "function",
            "function_call": {
                "id": "call-error",
                "name": "broken",
                "arguments": "{}",
            },
        }
    ]

    async def exercise():
        return await generator._handle_tool_messages(
            tool_messages,
            user="user",
            conversation_id="conv",
            conversation_manager=None,
            function_map=None,
            functions=None,
            current_persona=None,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=False,
        )

    result = asyncio.run(exercise())

    assert result == error_entry


def test_mistral_generator_preserves_message_metadata_in_payload():
    settings = {
        "model": "mistral-large-latest",
        "stream": False,
    }

    generator = _build_generator(DummyConfig(settings))

    conversation = [
        {
            "role": "system",
            "name": "system-priming",
            "content": [{"type": "text", "text": "You are helpful."}],
            "metadata": {"channel": "system"},
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Call the fetch_data tool."}
            ],
        },
        {
            "role": "assistant",
            "id": "assistant-msg-1",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "fetch_data",
                        "arguments": '{"foo": "bar"}',
                    },
                }
            ],
            "function_call": {
                "name": "fetch_data",
                "arguments": '{"foo": "bar"}',
            },
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": {"type": "json", "output": {"foo": "bar"}},
        },
    ]

    async def exercise():
        return await generator.generate_response(messages=conversation)

    result = asyncio.run(exercise())

    assert result == "ok"
    payload = _StubChat.last_complete_kwargs["messages"]

    assert payload[0]["name"] == "system-priming"
    assert payload[0]["content"] == conversation[0]["content"]
    assert payload[0]["content"] is not conversation[0]["content"]
    assert payload[0]["metadata"] == {"channel": "system"}

    assert payload[2]["id"] == "assistant-msg-1"
    assert payload[2]["content"] == ""
    assert payload[2]["function_call"] == {
        "name": "fetch_data",
        "arguments": '{"foo": "bar"}',
    }
    assert payload[2]["tool_calls"][0]["id"] == "call_1"
    assert (
        payload[2]["tool_calls"][0]["function"]["arguments"]
        == '{"foo": "bar"}'
    )
    assert payload[2]["tool_calls"][0] is not conversation[2]["tool_calls"][0]

    assert payload[3]["tool_call_id"] == "call_1"
    assert payload[3]["content"] == conversation[3]["content"]
    assert payload[3]["content"] is not conversation[3]["content"]


def test_mistral_generator_overrides_stop_sequences_argument():
    settings = {
        "model": "mistral-large-latest",
        "stream": False,
        "stop_sequences": ["DEFAULT"],
    }

    generator = _build_generator(DummyConfig(settings))

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["CUSTOM", "HALT"],
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    kwargs = _StubChat.last_complete_kwargs
    assert kwargs["stop"] == ["CUSTOM", "HALT"]


def test_mistral_generator_applies_json_mode_response_format():
    settings = {
        "model": "mistral-large-latest",
        "stream": False,
        "json_mode": True,
    }

    generator = _build_generator(DummyConfig(settings))

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    kwargs = _StubChat.last_complete_kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_mistral_generator_passes_json_schema_response_format():
    schema_payload = {
        "name": "atlas_response",
        "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
    }
    settings = {
        "model": "mistral-large-latest",
        "stream": True,
        "json_schema": schema_payload,
    }

    generator = _build_generator(DummyConfig(settings))

    async def exercise():
        stream = await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(exercise())

    assert chunks == []
    stream_kwargs = _StubChat.last_stream_kwargs
    assert stream_kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": schema_payload,
    }


def test_mistral_generator_executes_tool_call_from_complete(monkeypatch):
    settings = {
        "model": "mistral-large-latest",
        "stream": False,
    }

    generator = _build_generator(DummyConfig(settings))
    recorded_messages = []
    conversation = RecordingConversation()

    async def fake_use_tool(*args, **_kwargs):
        if "message" in _kwargs:
            recorded_messages.append(_kwargs["message"])
        elif len(args) > 2:
            recorded_messages.append(args[2])
        else:  # pragma: no cover - defensive fallback
            recorded_messages.append(None)
        assert len(conversation.records) == 1
        entry = conversation.records[0]
        assert entry["role"] == "assistant"
        return "tool-result"

    monkeypatch.setattr(mistral_module, "use_tool", fake_use_tool)

    def fake_complete(**kwargs):
        _StubChat.last_complete_kwargs = kwargs
        message = SimpleNamespace(
            content="ignored",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    type="function",
                    function=SimpleNamespace(
                        name="lookup",
                        arguments='{"query":"value"}',
                    ),
                )
            ],
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    generator.client.chat.complete = fake_complete

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "call"}],
            current_persona={"name": "tester"},
            user="user-1",
            conversation_id="conv-1",
            conversation_manager=conversation,
        )

    result = asyncio.run(exercise())

    assert result == "tool-result"
    assert recorded_messages == [
        {
            "function_call": {
                "name": "lookup",
                "arguments": '{"query":"value"}',
            }
        }
    ]
    assert conversation.records
    history_entry = conversation.records[0]
    assert history_entry["tool_calls"][0]["function"]["name"] == "lookup"
    assert history_entry["tool_calls"][0]["id"] == "call_1"


def test_mistral_generator_executes_tool_call_from_stream(monkeypatch):
    settings = {
        "model": "mistral-large-latest",
        "stream": True,
    }

    generator = _build_generator(DummyConfig(settings))
    recorded_messages = []
    conversation = RecordingConversation()

    async def fake_use_tool(*args, **_kwargs):
        if "message" in _kwargs:
            recorded_messages.append(_kwargs["message"])
        elif len(args) > 2:
            recorded_messages.append(args[2])
        else:  # pragma: no cover - defensive fallback
            recorded_messages.append(None)
        assert len(conversation.records) == 1
        entry = conversation.records[0]
        assert entry["role"] == "assistant"
        return "stream-tool"

    monkeypatch.setattr(mistral_module, "use_tool", fake_use_tool)

    events = [
        SimpleNamespace(
            data=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="lookup",
                                        arguments='{"query":',
                                    ),
                                )
                            ]
                        ),
                        finish_reason=None,
                    )
                ]
            )
        ),
        SimpleNamespace(
            data=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    function=SimpleNamespace(arguments='"value"}')
                                )
                            ]
                        ),
                        finish_reason=None,
                    )
                ]
            )
        ),
        SimpleNamespace(
            data=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(),
                        finish_reason="stop",
                    )
                ]
            )
        ),
    ]

    class _Stream:
        def __init__(self, payload):
            self._payload = list(payload)
            self.closed = False

        def __iter__(self):
            return iter(self._payload)

        def close(self):
            self.closed = True

    captured_kwargs = {}

    def fake_stream(**kwargs):
        captured_kwargs.update(kwargs)
        return _Stream(events)

    generator.client.chat.stream = fake_stream

    async def exercise():
        stream = await generator.generate_response(
            messages=[{"role": "user", "content": "call"}],
            current_persona={"name": "tester"},
            user="user-2",
            conversation_id="conv-2",
            conversation_manager=conversation,
            stream=True,
        )
        parts = []
        async for chunk in stream:
            parts.append(chunk)
        return parts

    chunks = asyncio.run(exercise())

    assert chunks == ["stream-tool"]
    assert recorded_messages == [
        {
            "function_call": {
                "name": "lookup",
                "arguments": '{"query":"value"}',
            }
        }
    ]
    assert conversation.records
    history_entry = conversation.records[0]
    assert history_entry["tool_calls"][0]["function"]["name"] == "lookup"
    assert history_entry["tool_calls"][0]["id"] == "call_1"
