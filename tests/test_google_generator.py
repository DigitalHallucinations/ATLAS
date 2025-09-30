import asyncio
import copy
import importlib.util
import os
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


class _DummyGenerationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyFunctionDeclaration:
    def __init__(self, *, name: str, description: str, parameters=None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}


class _DummyTool:
    def __init__(self, function_declarations):
        self.function_declarations = list(function_declarations)


genai_module = types.ModuleType("google.generativeai")
genai_types_module = types.ModuleType("google.generativeai.types")
genai_module.configure = lambda **_: None
genai_module.types = genai_types_module
genai_module.GenerativeModel = None
genai_types_module.GenerationConfig = _DummyGenerationConfig
genai_types_module.FunctionDeclaration = _DummyFunctionDeclaration
genai_types_module.Tool = _DummyTool
genai_types_module.ContentDict = dict
genai_types_module.PartDict = dict

sys.modules["google.generativeai"] = genai_module
sys.modules["google.generativeai.types"] = genai_types_module

if "tenacity" not in sys.modules:
    tenacity_stub = types.ModuleType("tenacity")

    def _retry_stub(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    tenacity_stub.retry = _retry_stub
    tenacity_stub.stop_after_attempt = lambda *_args, **_kwargs: None
    tenacity_stub.wait_exponential = lambda *_args, **_kwargs: None
    sys.modules["tenacity"] = tenacity_stub

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.dump = lambda *_args, **_kwargs: None
    sys.modules["yaml"] = yaml_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
    dotenv_stub.set_key = lambda *_args, **_kwargs: None
    dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ""
    sys.modules["dotenv"] = dotenv_stub

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

import google.generativeai as genai


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "Providers"
    / "Google"
    / "GG_gen_response.py"
)
spec = importlib.util.spec_from_file_location("google_module", MODULE_PATH)
google_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(google_module)


class DummyConfig:
    def __init__(self, settings=None):
        self._settings = settings or {}
        self.notifications = []

    def get_google_api_key(self):
        return "test-key"

    def get_google_llm_settings(self):
        return copy.deepcopy(self._settings)

    def notify_ui_warning(self, message):
        self.notifications.append(message)


def test_google_generator_streams_function_call(monkeypatch):
    captured = {}

    monkeypatch.setattr(genai, "configure", lambda **_: None)

    class DummyResponse:
        def __init__(self):
            function_call_part = SimpleNamespace(
                function_call=SimpleNamespace(name="tool_action", args={"value": 1})
            )
            text_chunk = SimpleNamespace(text="Hello", candidates=[])
            function_chunk = SimpleNamespace(
                text="",
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[function_call_part])
                    )
                ],
            )
            self._chunks = [text_chunk, function_chunk]

        def __iter__(self):
            return iter(self._chunks)

        @property
        def candidates(self):
            return []

        @property
        def text(self):
            return "Hello"

    class DummyModel:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def generate_content(self, contents, **kwargs):
            captured["contents"] = contents
            captured["kwargs"] = kwargs
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    generator = google_module.GoogleGeminiGenerator(
        DummyConfig(
            settings={
                "function_call_mode": "require",
                "allowed_function_names": ["tool_action"],
            }
        )
    )

    async def exercise():
        stream = await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemini-unit-test",
            stream=True,
            functions=[
                {
                    "name": "tool_action",
                    "description": "A sample tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                }
            ],
        )

        chunks = []
        async for item in stream:
            chunks.append(item)
        return chunks

    chunks = asyncio.run(exercise())

    assert chunks[0] == "Hello"
    assert chunks[1] == {"function_call": {"name": "tool_action", "arguments": "{\"value\": 1}"}}

    kwargs = captured["kwargs"]
    tools = kwargs["tools"]
    assert len(tools) == 1
    declarations = list(tools[0].function_declarations)
    assert declarations[0].name == "tool_action"
    assert declarations[0].description == "A sample tool"
    tool_config = kwargs["tool_config"]
    assert tool_config["function_calling_config"]["mode"] == "REQUIRE"
    assert tool_config["function_calling_config"]["allowed_function_names"] == [
        "tool_action"
    ]


def test_google_generator_defaults_json_mime_when_schema_present(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda **_: None)

    call_kwargs = []

    class DummyResponse:
        def __init__(self):
            self.text = "done"
            self.candidates = []

    class DummyModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, contents, **kwargs):
            call_kwargs.append(kwargs)
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    config = DummyConfig(
        settings={
            "response_schema": {"type": "object"},
            "response_mime_type": "",
            "stream": False,
        }
    )
    generator = google_module.GoogleGeminiGenerator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

    result = asyncio.run(exercise())
    assert result == "done"
    assert call_kwargs[-1]["response_mime_type"] == "application/json"

    async def exercise_with_explicit_schema():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            response_schema={"type": "object"},
            response_mime_type=None,
        )

    asyncio.run(exercise_with_explicit_schema())
    assert call_kwargs[-1]["response_mime_type"] == "application/json"


def test_google_generator_disables_tool_config_when_functions_off(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda **_: None)

    captured_kwargs = {}

    class DummyResponse:
        def __init__(self):
            self.text = "done"
            self.candidates = []

    class DummyModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, contents, **kwargs):
            captured_kwargs.update(kwargs)
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    config = DummyConfig(settings={"function_call_mode": "any"})
    generator = google_module.GoogleGeminiGenerator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            enable_functions=False,
            stream=False,
        )

    result = asyncio.run(exercise())
    assert result == "done"
    tool_config = captured_kwargs["tool_config"]
    assert tool_config["function_calling_config"]["mode"] == "NONE"
    assert "allowed_function_names" not in tool_config["function_calling_config"]


def test_google_generator_prunes_allowlist_to_declared_tools(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda **_: None)

    captured_kwargs = {}

    class DummyResponse:
        def __init__(self):
            self.text = "done"
            self.candidates = []

    class DummyModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, contents, **kwargs):
            captured_kwargs.update(kwargs)
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    config = DummyConfig(
        settings={
            "function_call_mode": "require",
            "allowed_function_names": [
                "alpha_tool",
                "ghost_tool",
                "beta_tool",
            ],
        }
    )
    generator = google_module.GoogleGeminiGenerator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            functions=[
                {"name": "alpha_tool", "description": "Alpha"},
                {"name": "beta_tool", "description": "Beta"},
            ],
            stream=False,
        )

    result = asyncio.run(exercise())

    assert result == "done"
    tool_config = captured_kwargs["tool_config"]["function_calling_config"]
    assert tool_config["mode"] == "REQUIRE"
    assert tool_config["allowed_function_names"] == ["alpha_tool", "beta_tool"]
    assert config.notifications[-1] == (
        "Removed unsupported Google Gemini tool allowlist entries: ghost_tool"
    )


def test_google_generator_omits_allowlist_when_every_entry_pruned(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda **_: None)

    captured_kwargs = {}

    class DummyResponse:
        def __init__(self):
            self.text = "done"
            self.candidates = []

    class DummyModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, contents, **kwargs):
            captured_kwargs.update(kwargs)
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    config = DummyConfig(
        settings={
            "function_call_mode": "auto",
            "allowed_function_names": ["orphan_tool"],
        }
    )
    generator = google_module.GoogleGeminiGenerator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            functions=[{"name": "different_tool", "description": "Other"}],
            stream=False,
        )

    result = asyncio.run(exercise())

    assert result == "done"
    tool_config = captured_kwargs["tool_config"]["function_calling_config"]
    assert tool_config["mode"] == "AUTO"
    assert "allowed_function_names" not in tool_config
    assert config.notifications[-1] == (
        "Removed unsupported Google Gemini tool allowlist entries: orphan_tool"
    )


def test_google_generator_streaming_runs_off_event_loop(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda **_: None)

    iteration_threads = []

    class DummyResponse:
        def __iter__(self):
            iteration_threads.append(threading.get_ident())
            for index in range(3):
                time.sleep(0.01)
                yield SimpleNamespace(text=f"chunk-{index}", candidates=[])

        @property
        def candidates(self):
            return []

        @property
        def text(self):
            return ""

    class DummyModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, contents, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    generator = google_module.GoogleGeminiGenerator(DummyConfig())

    async def exercise():
        loop_thread_id = threading.get_ident()
        stream = await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemini-streaming-test",
            stream=True,
        )

        observed = []

        async for item in stream:
            observed.append(item)

        return loop_thread_id, observed

    loop_thread, observed_chunks = asyncio.run(exercise())

    assert observed_chunks == ["chunk-0", "chunk-1", "chunk-2"]
    assert iteration_threads, "Streaming iterator was not consumed"
    assert all(tid != loop_thread for tid in iteration_threads)


def test_google_generator_applies_persisted_settings(monkeypatch):
    captured = {}

    monkeypatch.setattr(genai, "configure", lambda **_: None)

    class DummyResponse:
        def __init__(self):
            self.text = "Hello"
            self.candidates = []

    class DummyModel:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def generate_content(self, contents, **kwargs):
            captured["contents"] = contents
            captured["kwargs"] = kwargs
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    settings = {
        "model": "gemini-from-config",
        "temperature": 0.65,
        "top_p": 0.8,
        "top_k": 32,
        "candidate_count": 3,
        "stop_sequences": ["###"],
        "safety_settings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"}
        ],
        "response_mime_type": "application/json",
        "system_instruction": "Respond in JSON.",
        "stream": False,
        "response_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
    }
    config = DummyConfig(settings=settings)
    generator = google_module.GoogleGeminiGenerator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )

    result = asyncio.run(exercise())

    assert result == "Hello"
    assert captured["model_name"] == "gemini-from-config"
    generation_config = captured["kwargs"]["generation_config"]
    assert generation_config.kwargs["temperature"] == 0.65
    assert generation_config.kwargs["top_p"] == 0.8
    assert generation_config.kwargs["top_k"] == 32
    assert generation_config.kwargs["candidate_count"] == 3
    assert generation_config.kwargs["stop_sequences"] == ["###"]
    assert captured["kwargs"]["safety_settings"] == settings["safety_settings"]
    assert captured["kwargs"]["response_mime_type"] == "application/json"
    assert captured["kwargs"]["system_instruction"] == "Respond in JSON."
    assert captured["kwargs"]["stream"] is False
    assert captured["kwargs"]["generation_config"].kwargs["response_schema"] == {
        "type": "object",
        "properties": {"message": {"type": "string"}},
    }
    assert config._settings["stop_sequences"] == ["###"]


def test_google_generator_omits_max_tokens_when_disabled(monkeypatch):
    captured = {}

    monkeypatch.setattr(genai, "configure", lambda **_: None)

    class DummyResponse:
        def __init__(self):
            self.text = "Hello"
            self.candidates = []

    class DummyModel:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def generate_content(self, contents, **kwargs):
            captured["contents"] = contents
            captured["kwargs"] = kwargs
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    config = DummyConfig(settings={"max_output_tokens": None, "stream": False})
    generator = google_module.GoogleGeminiGenerator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
        )

    result = asyncio.run(exercise())

    assert result == "Hello"
    generation_config = captured["kwargs"]["generation_config"]
    assert "max_output_tokens" not in generation_config.kwargs


def test_google_generator_prefers_call_overrides(monkeypatch):
    captured = {}

    monkeypatch.setattr(genai, "configure", lambda **_: None)

    class DummyResponse:
        def __init__(self):
            self.text = "Call"
            self.candidates = []
            self._chunks = [SimpleNamespace(text="Call", candidates=[])]

        def __iter__(self):
            return iter(self._chunks)

    class DummyModel:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def generate_content(self, contents, **kwargs):
            captured["contents"] = contents
            captured["kwargs"] = kwargs
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    base_settings = {
        "model": "gemini-config",
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 64,
        "candidate_count": 5,
        "stop_sequences": ["CONFIG"],
        "safety_settings": [
            {"category": "CONFIG", "threshold": "BLOCK_ONLY_HIGH"}
        ],
        "response_mime_type": "application/json",
        "system_instruction": "Config instruction",
        "stream": False,
        "response_schema": {
            "type": "object",
            "properties": {"config": {"type": "string"}},
        },
    }
    config = DummyConfig(settings=base_settings)
    generator = google_module.GoogleGeminiGenerator(config)

    override_safety = [
        {"category": "CALL", "threshold": "BLOCK_LOW_AND_ABOVE"}
    ]

    async def exercise():
        response = await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            model="call-model",
            temperature=0.1,
            top_p=0.3,
            top_k=8,
            candidate_count=2,
            stop_sequences=["CALL"],
            stream=True,
            safety_settings=override_safety,
            response_mime_type="text/plain",
            system_instruction="Follow call input.",
            response_schema={
                "type": "object",
                "properties": {"call": {"type": "number"}},
            },
        )
        if hasattr(response, "__anext__"):
            chunks = []
            async for item in response:
                chunks.append(item)
            return chunks
        return response

    result = asyncio.run(exercise())

    assert result == ["Call"]
    assert captured["model_name"] == "call-model"
    generation_config = captured["kwargs"]["generation_config"]
    assert generation_config.kwargs["temperature"] == 0.1
    assert generation_config.kwargs["top_p"] == 0.3
    assert generation_config.kwargs["top_k"] == 8
    assert generation_config.kwargs["candidate_count"] == 2
    assert generation_config.kwargs["stop_sequences"] == ["CALL"]
    assert captured["kwargs"]["safety_settings"] == override_safety
    assert captured["kwargs"]["response_mime_type"] == "text/plain"
    assert captured["kwargs"]["system_instruction"] == "Follow call input."
    assert captured["kwargs"]["stream"] is True
    assert captured["kwargs"]["generation_config"].kwargs["response_schema"] == {
        "type": "object",
        "properties": {"call": {"type": "number"}},
    }
    assert config._settings["stop_sequences"] == ["CONFIG"]


def test_google_generator_rejects_invalid_schema(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda **_: None)

    class DummyResponse:
        def __init__(self):
            self.text = "Hello"
            self.candidates = []

    class DummyModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, contents, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    config = DummyConfig(settings={"model": "gemini"})
    generator = google_module.GoogleGeminiGenerator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            response_schema="{invalid",
        )

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(exercise())

    assert "Response schema" in str(excinfo.value)


def test_google_generator_skips_tool_payload_when_disabled(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda **_: None)

    class DummyResponse:
        def __init__(self):
            self.text = "No tools"
            self.candidates = []

    class DummyModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, contents, **kwargs):
            DummyModel.captured = {
                "contents": contents,
                "kwargs": kwargs,
            }
            return DummyResponse()

    monkeypatch.setattr(genai, "GenerativeModel", DummyModel)

    seen = {"called": False}

    def _unexpected_build(self, functions, current_persona):
        seen["called"] = True
        return ["unexpected"]

    monkeypatch.setattr(
        google_module.GoogleGeminiGenerator,
        "_build_tools_payload",
        _unexpected_build,
    )

    generator = google_module.GoogleGeminiGenerator(DummyConfig())

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            enable_functions=False,
            functions=[{"name": "persona_tool"}],
        )

    result = asyncio.run(exercise())

    assert result == "No tools"
    assert seen["called"] is False
    kwargs = DummyModel.captured["kwargs"]
    assert "tools" not in kwargs

