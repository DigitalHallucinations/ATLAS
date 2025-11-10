import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import ATLAS.services.providers as providers_module
from ATLAS.services.providers import ProviderService


def _make_service(**overrides):
    manager = overrides.get(
        "provider_manager",
        SimpleNamespace(
            set_current_provider=AsyncMock(),
            get_current_provider=Mock(return_value=None),
            get_current_model=Mock(return_value=None),
        ),
    )
    if "config_manager" in overrides:
        config = overrides["config_manager"]
    else:
        config = SimpleNamespace(
            get_pending_provider_warnings=Mock(return_value={}),
            get_default_provider=Mock(return_value=""),
        )
    logger = overrides.get("logger", Mock())
    chat_session = overrides.get("chat_session", SimpleNamespace())
    speech_manager = overrides.get("speech_manager", None)
    return ProviderService(
        provider_manager=manager,
        config_manager=config,
        logger=logger,
        chat_session=chat_session,
        speech_manager=speech_manager,
    )


def test_run_in_background_delegates_to_helper(monkeypatch):
    logger = Mock()
    sentinel = object()
    captured = {}

    def fake_runner(
        factory,
        *,
        on_success=None,
        on_error=None,
        logger=None,
        thread_name=None,
    ):
        captured.update(
            {
                "factory": factory,
                "on_success": on_success,
                "on_error": on_error,
                "logger": logger,
                "thread_name": thread_name,
            }
        )
        return sentinel

    monkeypatch.setattr(providers_module, "run_async_in_thread", fake_runner)

    service = _make_service(logger=logger)

    async def sample():  # pragma: no cover - executed via helper
        return "ok"

    future = service.run_in_background(sample, thread_name="provider-thread")

    assert future is sentinel
    assert captured["factory"] is sample
    assert captured["logger"] is logger
    assert captured["thread_name"] == "provider-thread"
    assert captured["on_success"] is None
    assert captured["on_error"] is None


def test_set_current_provider_updates_chat_session_and_listeners():
    state = {"provider": None, "model": "gpt-4o"}

    async def set_current(provider):
        state["provider"] = provider

    manager = SimpleNamespace(
        set_current_provider=AsyncMock(side_effect=set_current),
        get_current_provider=Mock(side_effect=lambda: state["provider"]),
        get_current_model=Mock(side_effect=lambda: state["model"]),
    )

    chat_session = SimpleNamespace(set_provider=Mock(), set_model=Mock())
    config = SimpleNamespace(
        get_pending_provider_warnings=Mock(return_value={}),
        get_default_provider=Mock(return_value=""),
    )
    service = _make_service(
        provider_manager=manager,
        config_manager=config,
        chat_session=chat_session,
    )

    listener = Mock()
    service.add_provider_change_listener(listener)

    asyncio.run(service.set_current_provider("OpenAI"))

    manager.set_current_provider.assert_awaited_once_with("OpenAI")
    chat_session.set_provider.assert_called_once_with("OpenAI")
    chat_session.set_model.assert_called_once_with("gpt-4o")
    listener.assert_called_once()
    summary = listener.call_args.args[0]
    assert summary["llm_provider"] == "OpenAI"
    assert summary["llm_model"] == "gpt-4o"


def test_set_current_provider_in_background_uses_runner():
    service = _make_service()
    sentinel = object()
    success = Mock()
    error = Mock()
    captured = {}

    service.set_current_provider = AsyncMock()

    def fake_runner(factory, *, on_success=None, on_error=None, thread_name=None):
        captured.update(
            {
                "factory": factory,
                "on_success": on_success,
                "on_error": on_error,
                "thread_name": thread_name,
            }
        )
        return sentinel

    service.run_in_background = fake_runner

    future = service.set_current_provider_in_background(
        "OpenAI", on_success=success, on_error=error
    )

    assert future is sentinel
    assert captured["on_success"] is success
    assert captured["on_error"] is error
    assert captured["thread_name"] == "set-provider-OpenAI"

    factory = captured["factory"]
    coroutine = factory()
    assert inspect.isawaitable(coroutine)
    asyncio.run(coroutine)
    service.set_current_provider.assert_awaited_once_with("OpenAI")


def test_update_api_key_in_background_uses_runner():
    manager = SimpleNamespace(update_provider_api_key=AsyncMock())
    service = _make_service(provider_manager=manager)
    sentinel = object()
    success = Mock()
    error = Mock()
    captured = {}

    def fake_runner(factory, *, on_success=None, on_error=None, thread_name=None):
        captured.update(
            {
                "factory": factory,
                "on_success": on_success,
                "on_error": on_error,
                "thread_name": thread_name,
            }
        )
        return sentinel

    service.run_in_background = fake_runner

    future = service.update_provider_api_key_in_background(
        "OpenAI", "token", on_success=success, on_error=error
    )

    assert future is sentinel
    assert captured["thread_name"] == "update-api-key-OpenAI"
    assert captured["on_success"] is success
    assert captured["on_error"] is error

    coroutine = captured["factory"]()
    assert inspect.isawaitable(coroutine)
    asyncio.run(coroutine)
    manager.update_provider_api_key.assert_awaited_once_with("OpenAI", "token")


def test_refresh_current_provider_behaviour():
    state = {"provider": "OpenAI"}

    async def reset_current(provider):
        state["provider"] = provider

    manager = SimpleNamespace(
        get_current_provider=Mock(side_effect=lambda: state["provider"]),
        set_current_provider=AsyncMock(side_effect=reset_current),
    )
    service = _make_service(provider_manager=manager)

    result = asyncio.run(service.refresh_current_provider("OpenAI"))
    assert result == {
        "success": True,
        "message": "Provider OpenAI refreshed.",
        "provider": "OpenAI",
    }
    manager.set_current_provider.assert_awaited_once_with("OpenAI")

    result_inactive = asyncio.run(service.refresh_current_provider("Anthropic"))
    assert result_inactive == {
        "success": False,
        "error": "Provider 'Anthropic' is not the active provider.",
        "active_provider": "OpenAI",
    }


def test_refresh_current_provider_without_active():
    manager = SimpleNamespace(
        get_current_provider=Mock(return_value=""),
        set_current_provider=AsyncMock(),
    )
    service = _make_service(provider_manager=manager)

    result = asyncio.run(service.refresh_current_provider())
    assert result == {"success": False, "error": "No active provider is configured."}
    manager.set_current_provider.assert_not_called()


def test_settings_delegation_methods():
    setters = {
        "set_openai_llm_settings": Mock(return_value={"ok": True}),
        "set_google_llm_settings": Mock(return_value={"ok": True}),
        "set_anthropic_settings": Mock(return_value={"ok": True}),
    }
    manager = SimpleNamespace(
        get_openai_llm_settings=Mock(return_value={"model": "gpt-4o"}),
        get_google_llm_settings=Mock(return_value={"model": "gemini"}),
        get_anthropic_settings=Mock(return_value={"model": "claude"}),
        **setters,
    )
    service = _make_service(provider_manager=manager)

    assert service.get_openai_llm_settings() == {"model": "gpt-4o"}
    assert service.get_google_llm_settings() == {"model": "gemini"}
    assert service.get_anthropic_settings() == {"model": "claude"}

    result = service.set_openai_llm_settings(model="gpt-4o")
    assert result == {"ok": True}
    setters["set_openai_llm_settings"].assert_called_once_with(model="gpt-4o")

    result = service.set_google_llm_settings(model="gemini")
    assert result == {"ok": True}
    setters["set_google_llm_settings"].assert_called_once_with(model="gemini")

    result = service.set_anthropic_settings(model="claude")
    assert result == {"ok": True}
    setters["set_anthropic_settings"].assert_called_once_with(model="claude")


def test_chat_status_summary_includes_warning():
    manager = SimpleNamespace(
        get_current_provider=Mock(return_value="OpenAI"),
        get_current_model=Mock(return_value="gpt-4o"),
    )
    speech_manager = SimpleNamespace(
        get_active_tts_summary=Mock(return_value=("ElevenLabs", "Alloy"))
    )
    config = SimpleNamespace(
        get_pending_provider_warnings=Mock(return_value={"OpenAI": "Missing key"}),
        get_default_provider=Mock(return_value="OpenAI"),
    )

    service = _make_service(
        provider_manager=manager,
        config_manager=config,
        speech_manager=speech_manager,
    )

    summary = service.get_chat_status_summary()
    assert summary["llm_provider"] == "OpenAI"
    assert summary["llm_model"] == "gpt-4o"
    assert summary["tts_provider"] == "ElevenLabs"
    assert summary["tts_voice"] == "Alloy"
    assert summary["llm_warning"] == "Missing key"


def test_format_chat_status_uses_summary():
    service = _make_service()
    summary = {
        "llm_provider": "OpenAI",
        "llm_model": "gpt-4o",
        "tts_provider": "ElevenLabs",
        "tts_voice": "Alloy",
        "llm_warning": "Missing key",
    }

    text = service.format_chat_status(summary)
    assert "LLM: OpenAI" in text
    assert "Warning: Missing key" in text


def test_get_models_for_provider_delegates():
    getter = Mock(return_value=["gpt-4o"])
    manager = SimpleNamespace(get_models_for_provider=getter)
    service = _make_service(provider_manager=manager)

    models = service.get_models_for_provider("OpenAI")
    assert models == ["gpt-4o"]
    getter.assert_called_once_with("OpenAI")

