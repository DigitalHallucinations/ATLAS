import asyncio
import sys
import types

if "tenacity" not in sys.modules:
    tenacity_stub = types.ModuleType("tenacity")

    def _identity_decorator(*_args, **_kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    tenacity_stub.retry = _identity_decorator
    tenacity_stub.stop_after_attempt = lambda *args, **kwargs: None
    tenacity_stub.wait_exponential = lambda *args, **kwargs: None
    sys.modules["tenacity"] = tenacity_stub

hf_module_name = "modules.Providers.HuggingFace.HF_gen_response"
if hf_module_name not in sys.modules:
    hf_stub = types.ModuleType(hf_module_name)

    class _PlaceholderGenerator:
        async def load_model(self, *_args, **_kwargs):
            return None

        def unload_model(self):
            return None

        def get_installed_models(self):
            return []

        async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - placeholder
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - placeholder
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

    hf_stub.HuggingFaceGenerator = _PlaceholderGenerator
    sys.modules[hf_module_name] = hf_stub

grok_module_name = "modules.Providers.Grok.grok_generate_response"
if grok_module_name not in sys.modules:
    grok_stub = types.ModuleType(grok_module_name)

    class _PlaceholderGrokGenerator:
        def __init__(self, *_args, **_kwargs):
            pass

        async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - placeholder
            return ""

        async def process_streaming_response(self, response):  # pragma: no cover - placeholder
            collected = []
            async for chunk in response:
                collected.append(chunk)
            return "".join(collected)

        async def unload_model(self):  # pragma: no cover - placeholder
            return None

    grok_stub.GrokGenerator = _PlaceholderGrokGenerator
    sys.modules[grok_module_name] = grok_stub


def _ensure_async_function_module(module_name: str, return_value: str):
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    async def _generate_response(*_args, **_kwargs):  # pragma: no cover - placeholder
        return return_value

    stub.generate_response = _generate_response
    sys.modules[module_name] = stub


_ensure_async_function_module("modules.Providers.OpenAI.OA_gen_response", "openai")
_ensure_async_function_module("modules.Providers.Mistral.Mistral_gen_response", "mistral")
_ensure_async_function_module("modules.Providers.Google.GG_gen_response", "google")
_ensure_async_function_module("modules.Providers.Anthropic.Anthropic_gen_response", "anthropic")

import pytest

import ATLAS.provider_manager as provider_manager_module
from ATLAS.provider_manager import ProviderManager


class DummyConfig:
    def __init__(self, root_path):
        self._root_path = root_path

    def get_default_provider(self):
        return "OpenAI"

    def get_app_root(self):
        return self._root_path

    def get_model_cache_dir(self):
        return self._root_path

    def get_llm_config(self, *_args, **_kwargs):
        return {
            "provider": "OpenAI",
            "model": "gpt-4o",
            "max_tokens": 4000,
            "temperature": 0.0,
            "stream": True,
        }

    def get_config(self, *_args, **_kwargs):
        return None


class FakeHFModelManager:
    def __init__(self):
        self.installed_models = ["alpha", "beta"]
        self.current_model = None

    def get_installed_models(self):
        return list(self.installed_models)

    def remove_installed_model(self, model_name: str):
        if model_name in self.installed_models:
            self.installed_models.remove(model_name)
        if self.current_model == model_name:
            self.current_model = None


class FakeHFGenerator:
    def __init__(self, _config_manager):
        self.model_manager = FakeHFModelManager()
        self.loaded_models = []
        self.unload_calls = 0

    async def load_model(self, model_name: str, force_download: bool = False):
        await asyncio.sleep(0)
        self.loaded_models.append((model_name, force_download))
        self.model_manager.current_model = model_name

    def unload_model(self):
        self.unload_calls += 1
        self.model_manager.current_model = None

    def get_installed_models(self):
        return self.model_manager.get_installed_models()

    def get_current_model(self):
        return self.model_manager.current_model

    async def generate_response(self, *_args, **_kwargs):  # pragma: no cover - not exercised
        return "stubbed"

    async def process_streaming_response(self, response):  # pragma: no cover - not exercised
        collected = []
        async for chunk in response:
            collected.append(chunk)
        return "".join(collected)


@pytest.fixture
def provider_manager(tmp_path, monkeypatch):
    ProviderManager._instance = None

    def fake_load_models(self):
        self.models = {
            "OpenAI": ["gpt-4o"],
            "HuggingFace": ["alpha", "beta"],
            "Mistral": [],
            "Google": [],
            "Anthropic": [],
            "Grok": [],
        }
        self.current_model = None
        self.current_provider = None

    monkeypatch.setattr(provider_manager_module.ModelManager, "load_models", fake_load_models, raising=False)
    monkeypatch.setattr(provider_manager_module, "HuggingFaceGenerator", FakeHFGenerator)

    config = DummyConfig(tmp_path.as_posix())
    manager = asyncio.run(ProviderManager.create(config))
    yield manager

    ProviderManager._instance = None


def test_huggingface_facade_handles_model_lifecycle(provider_manager):
    ensure_first = provider_manager.ensure_huggingface_ready()
    assert ensure_first["success"] is True
    generator = provider_manager.huggingface_generator
    assert isinstance(generator, FakeHFGenerator)

    ensure_second = provider_manager.ensure_huggingface_ready()
    assert ensure_second["success"] is True
    assert provider_manager.huggingface_generator is generator

    models_result = provider_manager.list_hf_models()
    assert models_result["success"] is True
    assert models_result["data"] == ["alpha", "beta"]

    async def exercise():
        await provider_manager.switch_llm_provider("HuggingFace")
        assert generator.loaded_models[0] == ("alpha", False)

        load_result = await provider_manager.load_hf_model("beta", force_download=True)
        assert load_result["success"] is True
        assert ("beta", True) in generator.loaded_models
        assert provider_manager.current_model == "beta"

        unload_result = await provider_manager.unload_hf_model()
        assert unload_result["success"] is True
        assert generator.unload_calls == 1
        assert provider_manager.current_model is None

        remove_result = await provider_manager.remove_hf_model("beta")
        assert remove_result["success"] is True

    asyncio.run(exercise())

    refreshed = provider_manager.list_hf_models()
    assert refreshed["success"] is True
    assert refreshed["data"] == ["alpha"]
