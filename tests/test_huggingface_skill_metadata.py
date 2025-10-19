import asyncio
import sys
from types import SimpleNamespace, ModuleType

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
        ...

    manager_stub.HuggingFaceModelManager = _PlaceholderManager
    sys.modules[hf_manager_module] = manager_stub

from modules.Providers.HuggingFace.components.response_generator import ResponseGenerator
from modules.Providers.HuggingFace.utils.cache_manager import CacheManager


def test_cache_key_includes_skill_metadata(tmp_path):
    cache_file = tmp_path / "cache.json"
    manager = CacheManager(str(cache_file))

    messages = [{"role": "user", "content": "Hello"}]
    settings = {"temperature": 0.5}

    baseline = manager.generate_cache_key(messages, "test-model", settings)
    version_key = manager.generate_cache_key(
        messages,
        "test-model",
        settings,
        skill_version="2024.05",
    )
    capability_key = manager.generate_cache_key(
        messages,
        "test-model",
        settings,
        capability_tags=["analysis", "compliance"],
    )
    capability_key_permuted = manager.generate_cache_key(
        messages,
        "test-model",
        settings,
        capability_tags=["compliance", "analysis"],
    )

    assert version_key != baseline
    assert capability_key != baseline
    assert capability_key == capability_key_permuted


class _StubCacheManager:
    def __init__(self):
        self.last_kwargs = None
        self._store = {}

    def generate_cache_key(self, messages, model, settings, **kwargs):
        self.last_kwargs = {"messages": messages, "model": model, "settings": settings, **kwargs}
        return "cache-key"

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


class _StubTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "np"):
        return {"input_ids": [[1]], "attention_mask": [[1]]}

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "decoded"


class _StubModelManager:
    def __init__(self):
        self.base_config = SimpleNamespace(model_settings={"max_tokens": 8})
        self.current_model = None
        self.ort_sessions = {}
        self.tokenizer = _StubTokenizer()

    async def load_model(self, model: str):
        self.current_model = model

    def pipeline(self, prompt: str, **_kwargs):
        return [{"generated_text": f"response:{prompt}"}]


def test_response_generator_passes_skill_metadata(tmp_path):
    cache_manager = _StubCacheManager()
    model_manager = _StubModelManager()
    generator = ResponseGenerator(model_manager, cache_manager)

    messages = [{"role": "user", "content": "Hello"}]
    skill_signature = SimpleNamespace(
        version="1.0.0",
        required_capabilities=("alpha", "beta"),
    )

    result = asyncio.run(
        generator.generate_response(
            messages,
            "demo-model",
            stream=False,
            skill_signature=skill_signature,
        )
    )

    assert isinstance(result, str)
    assert cache_manager.last_kwargs is not None
    assert cache_manager.last_kwargs["skill_version"] == "1.0.0"
    assert cache_manager.last_kwargs["capability_tags"] == skill_signature.required_capabilities
