import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
    def __init__(self, **_kwargs):
        self.chat = _StubChat()

    APIError = Exception


if "mistralai" not in sys.modules:
    sys.modules["mistralai"] = SimpleNamespace(
        Mistral=_StubMistral,
        APIError=Exception,
    )

if "tenacity" not in sys.modules:
    class _TenacityStub(SimpleNamespace):
        @staticmethod
        def retry(*_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

    tenacity_stub = _TenacityStub(
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


def _reset_stubs():
    _StubChat.last_complete_kwargs = None
    _StubChat.last_stream_kwargs = None


@pytest.fixture(autouse=True)
def _patches(monkeypatch):
    monkeypatch.setattr(mistral_module, "ModelManager", DummyModelManager)
    monkeypatch.setattr(
        mistral_module,
        "load_functions_from_json",
        lambda *_args, **_kwargs: None,
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

    generator = mistral_module.MistralGenerator(DummyConfig(settings))

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
    }

    generator = mistral_module.MistralGenerator(DummyConfig(settings))

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

    generator = mistral_module.MistralGenerator(DummyConfig(settings))

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

    generator = mistral_module.MistralGenerator(DummyConfig(settings))

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

    generator = mistral_module.MistralGenerator(DummyConfig(settings))

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


def test_mistral_generator_overrides_stop_sequences_argument():
    settings = {
        "model": "mistral-large-latest",
        "stream": False,
        "stop_sequences": ["DEFAULT"],
    }

    generator = mistral_module.MistralGenerator(DummyConfig(settings))

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["CUSTOM", "HALT"],
        )

    result = asyncio.run(exercise())

    assert result == "ok"
    kwargs = _StubChat.last_complete_kwargs
    assert kwargs["stop"] == ["CUSTOM", "HALT"]
