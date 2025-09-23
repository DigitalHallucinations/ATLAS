import asyncio
import importlib
import sys
import types


class _DummyConfig:
    def __init__(self, tmp_path, settings):
        self._tmp_path = tmp_path
        self._settings = dict(settings)

    def get_openai_api_key(self):
        return "test-key"

    def get_openai_llm_settings(self):
        return dict(self._settings)

    def get_model_cache_dir(self):
        cache_dir = self._tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir.as_posix()

    def get_app_root(self):
        return self._tmp_path.as_posix()

    def get_config(self, key, default=None):
        if key == "DEFAULT_MODEL":
            return self._settings.get("model", default)
        if key == "OPENAI_BASE_URL":
            return self._settings.get("base_url")
        if key == "OPENAI_ORGANIZATION":
            return self._settings.get("organization")
        return default


def test_generate_response_passes_json_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")
    sys.modules.pop("modules.Providers.OpenAI.OA_gen_response", None)
    oa_module = importlib.import_module("modules.Providers.OpenAI.OA_gen_response")
    settings = {
        "model": "gpt-4o",
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 128,
        "stream": False,
        "function_calling": True,
        "json_mode": True,
        "base_url": None,
        "organization": None,
    }

    config = _DummyConfig(tmp_path, settings)
    recorded = {}

    class _RecorderCompletions:
        def __init__(self):
            self.last_kwargs = None

        async def create(self, **kwargs):  # pragma: no cover - exercised in test
            self.last_kwargs = kwargs
            message = types.SimpleNamespace(content="{\"ok\": true}", function_call=None)
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice], model=kwargs.get("model", "gpt-4o"))

    class _RecorderClient:
        def __init__(self, **kwargs):
            recorded["client_kwargs"] = kwargs
            completions = _RecorderCompletions()
            recorded["completions"] = completions
            self.chat = types.SimpleNamespace(completions=completions)

    monkeypatch.setattr(oa_module, "AsyncOpenAI", _RecorderClient)

    generator = oa_module.OpenAIGenerator(config)

    async def _exercise():
        return await generator.generate_response(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )

    result = asyncio.run(_exercise())

    assert result == "{\"ok\": true}"
    assert recorded["completions"].last_kwargs["response_format"] == {"type": "json_object"}
    assert recorded["completions"].last_kwargs["stream"] is False
