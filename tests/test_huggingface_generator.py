import asyncio
import sys
from types import ModuleType, SimpleNamespace

if "tenacity" not in sys.modules:
    tenacity_stub = ModuleType("tenacity")

    def _identity_decorator(*_args, **_kwargs):
        def _wrapper(func):
            return func

        return _wrapper

    tenacity_stub.retry = _identity_decorator
    tenacity_stub.stop_after_attempt = lambda *_args, **_kwargs: None
    tenacity_stub.wait_exponential = lambda *_args, **_kwargs: None
    sys.modules["tenacity"] = tenacity_stub

if "modules.logging.logger" not in sys.modules:
    logger_stub = ModuleType("modules.logging.logger")

    class _DummyLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    logger_stub.setup_logger = lambda *_args, **_kwargs: _DummyLogger()
    sys.modules["modules.logging.logger"] = logger_stub

tool_manager_module = "ATLAS.ToolManager"
if tool_manager_module not in sys.modules:
    tool_manager_stub = ModuleType(tool_manager_module)

    class _ToolExecutionError(Exception):
        def __init__(self, message: str = "", function_name: str | None = None):
            super().__init__(message)
            self.function_name = function_name

    async def _noop_use_tool(**_kwargs):  # pragma: no cover - default stub
        return None

    tool_manager_stub.ToolExecutionError = _ToolExecutionError
    tool_manager_stub.load_function_map_from_current_persona = (
        lambda *_args, **_kwargs: {}
    )
    tool_manager_stub.load_functions_from_json = lambda *_args, **_kwargs: []
    tool_manager_stub.use_tool = _noop_use_tool
    sys.modules[tool_manager_module] = tool_manager_stub

hf_manager_module = "modules.Providers.HuggingFace.components.huggingface_model_manager"
if hf_manager_module not in sys.modules:
    manager_stub = ModuleType(hf_manager_module)

    class _PlaceholderManager:
        def __init__(self, *args, **kwargs):
            self.base_config = SimpleNamespace(model_settings={})
            self.current_model = None
            self.ort_sessions = {}

        async def load_model(self, *_args, **_kwargs):
            return None

    manager_stub.HuggingFaceModelManager = _PlaceholderManager
    sys.modules[hf_manager_module] = manager_stub

from modules.Providers.HuggingFace.components import response_generator as hf_response_module
from modules.Providers.HuggingFace.components.response_generator import ResponseGenerator


class _StubCacheManager:
    def __init__(self):
        self._store = {}

    def generate_cache_key(self, *args, **kwargs):
        return "cache-key"

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


class _StubModelManager:
    def __init__(self, config_manager=None):
        base_config = SimpleNamespace(
            model_settings={
                "max_tokens": 128,
                "temperature": 0.3,
                "top_p": 0.95,
            },
            config_manager=config_manager,
        )
        self.base_config = base_config
        self.current_model = "stub-model"
        self.ort_sessions = {}

    async def load_model(self, model: str, *_args, **_kwargs):
        self.current_model = model


class _ConversationManager:
    def __init__(self, provider_manager=None):
        self.ATLAS = SimpleNamespace(provider_manager=provider_manager)


def _build_generator(monkeypatch, *, response_text: str, use_tool_result):
    provider_manager = object()
    config_manager = SimpleNamespace(provider_manager=provider_manager)
    model_manager = _StubModelManager(config_manager)
    cache_manager = _StubCacheManager()
    generator = ResponseGenerator(
        model_manager,
        cache_manager,
        config_manager=config_manager,
    )

    async def fake_generate_text(self, _prompt, _model):
        return response_text

    monkeypatch.setattr(
        ResponseGenerator,
        "_generate_text",
        fake_generate_text,
        raising=False,
    )

    monkeypatch.setattr(
        hf_response_module,
        "load_functions_from_json",
        lambda *_args, **_kwargs: [{"name": "lookup"}],
    )
    monkeypatch.setattr(
        hf_response_module,
        "load_function_map_from_current_persona",
        lambda *_args, **_kwargs: {"lookup": object()},
    )

    recorded_calls = {}

    async def fake_use_tool(**kwargs):
        recorded_calls.update(kwargs)
        return use_tool_result

    monkeypatch.setattr(hf_response_module, "use_tool", fake_use_tool)

    return generator, recorded_calls, _ConversationManager(provider_manager=provider_manager)


def test_huggingface_generator_executes_tool_call(monkeypatch):
    response_payload = '{"function_call": {"name": "lookup", "arguments": "{}"}}'
    generator, recorded, conversation_manager = _build_generator(
        monkeypatch,
        response_text=response_payload,
        use_tool_result="tool-output",
    )

    result = asyncio.run(
        generator.generate_response(
            messages=[{"role": "user", "content": "hello"}],
            model="stub-model",
            stream=False,
            current_persona={"name": "Persona"},
            functions=[{"name": "provided"}],
            conversation_manager=conversation_manager,
            user="alice",
            conversation_id="conv-1",
        )
    )

    assert result == "tool-output"
    assert recorded["message"]["function_call"]["name"] == "lookup"
    assert isinstance(recorded["functions"], list)
    names = {entry.get("name") for entry in recorded["functions"]}
    assert names == {"lookup", "provided"}
    assert recorded["user"] == "alice"
    assert recorded["conversation_id"] == "conv-1"


def test_huggingface_generator_streams_tool_result(monkeypatch):
    response_payload = (
        '{"tool_calls": ['
        '{"id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}'
        ']}'
    )
    generator, recorded, conversation_manager = _build_generator(
        monkeypatch,
        response_text=response_payload,
        use_tool_result="stream-value",
    )

    async def exercise():
        stream = await generator.generate_response(
            messages=[{"role": "user", "content": "hello"}],
            model="stub-model",
            stream=True,
            current_persona={"name": "Persona"},
            conversation_manager=conversation_manager,
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(exercise())

    assert chunks == ["stream-value"]
    assert recorded["message"]["function_call"]["name"] == "lookup"
    assert recorded["stream"] is True
