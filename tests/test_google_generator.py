import asyncio
import importlib.util
import os
import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace


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

    def get_google_api_key(self):
        return "test-key"

    def get_google_llm_settings(self):
        return copy.deepcopy(self._settings)


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

    generator = google_module.GoogleGeminiGenerator(DummyConfig())

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
    }
    config = DummyConfig(settings=settings)
    generator = google_module.GoogleGeminiGenerator(config)

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
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
    assert config._settings["stop_sequences"] == ["###"]


def test_google_generator_prefers_call_overrides(monkeypatch):
    captured = {}

    monkeypatch.setattr(genai, "configure", lambda **_: None)

    class DummyResponse:
        def __init__(self):
            self.text = "Call"
            self.candidates = []

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
    }
    config = DummyConfig(settings=base_settings)
    generator = google_module.GoogleGeminiGenerator(config)

    override_safety = [
        {"category": "CALL", "threshold": "BLOCK_LOW_AND_ABOVE"}
    ]

    async def exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "Hi"}],
            model="call-model",
            temperature=0.1,
            top_p=0.3,
            top_k=8,
            candidate_count=2,
            stop_sequences=["CALL"],
            stream=False,
            safety_settings=override_safety,
            response_mime_type="text/plain",
            system_instruction="Follow call input.",
        )

    result = asyncio.run(exercise())

    assert result == "Call"
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
    assert config._settings["stop_sequences"] == ["CONFIG"]

