import asyncio
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import pytest

from ATLAS.services.speech import SpeechService


def _make_service(manager, summary=None, logger=None):
    if summary is None:
        summary = {}
    summary_getter = Mock(return_value=summary)
    logger = logger or Mock()
    service = SpeechService(
        speech_manager=manager,
        logger=logger,
        status_summary_getter=summary_getter,
        default_status_tooltip="tooltip",
    )
    return service, summary_getter, logger


def test_start_stt_listening_success_includes_status_summary():
    manager = SimpleNamespace(
        get_active_stt_provider=Mock(return_value="openai"),
        listen=Mock(return_value=True),
    )
    summary = {"provider": "OpenAI"}
    service, summary_getter, logger = _make_service(manager, summary=summary)

    payload = service.start_stt_listening()

    assert payload == {
        "ok": True,
        "status_text": "Listening…",
        "provider": "openai",
        "listening": True,
        "spinner": False,
        "error": None,
        "status_tooltip": "Listening via openai",
        "status_summary": summary,
    }
    summary_getter.assert_called_once()
    logger.debug.assert_called_with("Listening started using provider '%s'.", "openai")


def test_start_stt_listening_missing_provider_uses_default_tooltip():
    manager = SimpleNamespace(
        get_active_stt_provider=Mock(return_value=None),
        listen=Mock(return_value=True),
    )
    service, summary_getter, logger = _make_service(manager, summary={"x": "y"})

    payload = service.start_stt_listening()

    assert payload == {
        "ok": False,
        "status_text": "No STT service configured.",
        "provider": None,
        "listening": False,
        "spinner": False,
        "error": "No STT service configured.",
        "status_tooltip": "tooltip",
        "status_summary": {"x": "y"},
    }
    logger.error.assert_called_with("No STT service configured.", exc_info=None)
    summary_getter.assert_called_once()


def test_start_stt_listening_handles_summary_failure():
    failing_getter = Mock(side_effect=RuntimeError("boom"))
    manager = SimpleNamespace(
        get_active_stt_provider=Mock(return_value="openai"),
        listen=Mock(return_value=True),
    )
    logger = Mock()
    service = SpeechService(
        speech_manager=manager,
        logger=logger,
        status_summary_getter=failing_getter,
        default_status_tooltip="tooltip",
    )

    payload = service.start_stt_listening()

    assert payload["status_summary"] == {}
    logger.error.assert_called_with(
        "Status summary callback failed: %s", failing_getter.side_effect, exc_info=True
    )


def test_stop_stt_and_transcribe_success_sets_future_result():
    captured = {}

    class _Manager:
        def get_active_stt_provider(self):
            return "openai"

        def stop_and_transcribe_in_background(self, provider, *, on_success, on_error, thread_name):
            captured.update({"success": on_success, "error": on_error, "thread": thread_name})

    manager = _Manager()
    service, summary_getter, _ = _make_service(manager, summary={"mode": "chat"})

    payload = service.stop_stt_and_transcribe()

    assert payload["ok"] is True
    assert payload["spinner"] is True
    assert payload["status_tooltip"] == "Transcribing via openai"
    assert payload["status_summary"] == {"mode": "chat"}
    future = payload["transcription_future"]
    assert not future.done()

    captured["success"]("  hello  ")
    result = future.result()
    assert result["transcript"] == "hello"
    summary_getter.assert_called()


def test_stop_stt_and_transcribe_failure_returns_error_payload():
    class _Manager:
        def get_active_stt_provider(self):
            return "openai"

        def stop_and_transcribe_in_background(self, *args, **kwargs):
            raise RuntimeError("failure")

    manager = _Manager()
    logger = Mock()
    service, summary_getter, _ = _make_service(manager, summary={"mode": "chat"}, logger=logger)

    payload = service.stop_stt_and_transcribe()

    assert payload["ok"] is False
    assert payload["error"] == "Transcription failed: failure"
    assert payload["status_tooltip"] == "tooltip"
    future = payload["transcription_future"]
    result = future.result()
    assert result["error"] == "Transcription failed: failure"
    logger.error.assert_called_with(
        "Failed to schedule transcription with provider %s: %s",
        "openai",
        ANY,
        exc_info=True,
    )
    summary_getter.assert_called()


def test_maybe_text_to_speech_skips_when_disabled():
    manager = SimpleNamespace(
        get_tts_status=Mock(return_value=False),
        text_to_speech=AsyncMock(),
    )
    service, _, _ = _make_service(manager)

    asyncio.run(service.maybe_text_to_speech("hello"))

    manager.text_to_speech.assert_not_awaited()


def test_maybe_text_to_speech_skips_async_iterators():
    async def _stream():
        yield "chunk"

    manager = SimpleNamespace(
        get_tts_status=Mock(return_value=True),
        text_to_speech=AsyncMock(),
    )
    service, _, logger = _make_service(manager)

    asyncio.run(service.maybe_text_to_speech(_stream()))

    manager.text_to_speech.assert_not_awaited()
    logger.debug.assert_called_with(
        "Skipping text-to-speech for streaming async iterator response."
    )
