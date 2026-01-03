import asyncio
from types import SimpleNamespace

from ATLAS.providers import anthropic as anthropic_adapter
from ATLAS.providers import huggingface as huggingface_adapter
from ATLAS.providers import openai as openai_adapter
from ATLAS.providers.base import build_result


class _RecorderLogger:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.infos = []
        self.debugs = []

    def error(self, *args, **kwargs):  # pragma: no cover - passthrough for tests
        self.errors.append((args, kwargs))

    def warning(self, *args, **kwargs):  # pragma: no cover - passthrough for tests
        self.warnings.append((args, kwargs))

    def info(self, *args, **kwargs):  # pragma: no cover - passthrough for tests
        self.infos.append((args, kwargs))

    def debug(self, *args, **kwargs):  # pragma: no cover - passthrough for tests
        self.debugs.append((args, kwargs))


class _StubModelManager:
    def __init__(self):
        self.records = []

    def update_models_for_provider(self, provider, models):
        self.records.append((provider, list(models)))
        return list(models)


def test_huggingface_search_invokes_generator(monkeypatch):
    sentinel = object()
    ensure_calls = []

    def ensure_ready():
        ensure_calls.append(True)
        return build_result(True, message="ready")

    def supplier():
        return sentinel

    async def fake_search(generator, query, filters=None, limit=10):
        assert generator is sentinel
        assert query == "llama"
        assert filters == {"owner": "meta"}
        assert limit == 5
        return [
            {"id": "meta/llama", "tags": [], "downloads": 1, "likes": 1},
        ]

    monkeypatch.setattr(huggingface_adapter, "_hf_search_models", fake_search)

    result = asyncio.run(
        huggingface_adapter.search_models(
            ensure_ready,
            supplier,
            "llama",
            filters={"owner": "meta"},
            limit=5,
            logger=_RecorderLogger(),
        )
    )

    assert result["success"] is True
    assert result["data"][0]["id"] == "meta/llama"
    assert ensure_calls == [True]


def test_huggingface_search_returns_failure(monkeypatch):
    def ensure_ready():
        return build_result(False, error="not ready")

    result = asyncio.run(
        huggingface_adapter.search_models(
            ensure_ready,
            lambda: None,
            "any",
        )
    )

    assert result["success"] is False
    assert result["error"] == "not ready"


def test_huggingface_download_handles_error(monkeypatch):
    sentinel = object()
    logger = _RecorderLogger()

    def ensure_ready():
        return build_result(True)

    def supplier():
        return sentinel

    async def fake_download(*_args, **_kwargs):  # pragma: no cover - helper
        raise RuntimeError("boom")

    monkeypatch.setattr(huggingface_adapter, "_hf_download_model", fake_download)

    result = asyncio.run(
        huggingface_adapter.download_model(
            ensure_ready,
            supplier,
            "meta/llama",
            logger=logger,
        )
    )

    assert result["success"] is False
    assert "boom" in result["error"]
    assert logger.errors


def test_huggingface_update_and_clear(monkeypatch):
    sentinel = object()
    logger = _RecorderLogger()

    def ensure_ready():
        return build_result(True)

    def supplier():
        return sentinel

    async def fake_to_thread(callable_, *args, **kwargs):
        return callable_(*args, **kwargs)

    def fake_update(generator, settings):
        assert generator is sentinel
        return {"temperature": settings["temperature"]}

    def fake_clear(generator):
        assert generator is sentinel
        fake_clear.called = True  # type: ignore[attr-defined]

    fake_clear.called = False  # type: ignore[attr-defined]

    monkeypatch.setattr(huggingface_adapter.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(huggingface_adapter, "_hf_update_model_settings", fake_update)
    monkeypatch.setattr(huggingface_adapter, "_hf_clear_cache", fake_clear)

    update_result = asyncio.run(
        huggingface_adapter.update_settings(
            ensure_ready,
            supplier,
            {"temperature": 0.5},
            logger=logger,
        )
    )

    assert update_result["success"] is True
    assert update_result["data"]["temperature"] == 0.5

    clear_result = asyncio.run(
        huggingface_adapter.clear_cache(
            ensure_ready,
            supplier,
            logger=logger,
        )
    )

    assert clear_result["success"] is True
    assert fake_clear.called is True  # type: ignore[attr-defined]


def test_openai_list_models_success(monkeypatch):
    config = SimpleNamespace(
        get_openai_api_key=lambda: "sk-test",
        get_openai_llm_settings=lambda: {"base_url": "https://api.example/v1"},
    )
    model_manager = _StubModelManager()
    logger = _RecorderLogger()

    async def fake_to_thread(callable_, *args, **kwargs):
        return {"data": [{"id": "gpt-4o"}, {"id": "text-embedding-3-small"}]}

    monkeypatch.setattr(openai_adapter.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(openai_adapter.list_models(config, model_manager, logger))

    assert result["models"] == ["gpt-4o", "text-embedding-3-small"]
    assert model_manager.records[-1][0] == "OpenAI"
    assert result["error"] is None


def test_openai_list_models_requires_key():
    config = SimpleNamespace(
        get_openai_api_key=lambda: "",
        get_openai_llm_settings=lambda: {},
    )
    model_manager = _StubModelManager()
    logger = _RecorderLogger()

    result = asyncio.run(openai_adapter.list_models(config, model_manager, logger))

    assert result["models"] == []
    assert "API key" in (result["error"] or "")


def test_anthropic_list_models_handles_errors(monkeypatch):
    config = SimpleNamespace(get_anthropic_api_key=lambda: "sk-anthropic")
    model_manager = _StubModelManager()
    logger = _RecorderLogger()

    async def fake_to_thread(callable_, *args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(anthropic_adapter.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(anthropic_adapter.list_models(config, model_manager, logger))

    assert result["models"] == []
    assert "network down" in (result["error"] or "")
    assert logger.errors


def test_anthropic_list_models_success(monkeypatch):
    config = SimpleNamespace(get_anthropic_api_key=lambda: "sk-anthropic")
    model_manager = _StubModelManager()
    logger = _RecorderLogger()

    payload = {"data": [{"id": "claude-3-sonnet"}, {"model": "claude-3-opus"}]}

    async def fake_to_thread(callable_, *args, **kwargs):
        return payload

    monkeypatch.setattr(anthropic_adapter.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(anthropic_adapter.list_models(config, model_manager, logger))

    assert result["models"] == ["claude-3-sonnet", "claude-3-opus"]
    assert model_manager.records[-1][0] == "Anthropic"
    assert result["error"] is None
