import asyncio
from collections.abc import AsyncIterator
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

if "yaml" not in sys.modules:
    sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = types.SimpleNamespace(
        load_dotenv=lambda *args, **kwargs: None,
        set_key=lambda *args, **kwargs: None,
        find_dotenv=lambda *args, **kwargs: "",
    )

from ATLAS.ATLAS import ATLAS


def test_generate_response_returns_with_conversation_id():
    atlas = ATLAS.__new__(ATLAS)
    atlas.logger = SimpleNamespace(error=Mock())
    atlas.current_persona = "Persona"

    provider_generate = AsyncMock(return_value="ok")
    provider_manager = SimpleNamespace(
        generate_response=provider_generate,
        set_current_conversation_id=Mock(),
    )
    atlas.provider_manager = provider_manager
    atlas.chat_session = SimpleNamespace(conversation_id="conversation-123")
    atlas._ensure_user_identity = Mock(return_value=("user", "User"))
    atlas.maybe_text_to_speech = AsyncMock()

    messages = [{"role": "user", "content": "Hello"}]
    result = asyncio.run(atlas.generate_response(messages))

    assert result == "ok"
    provider_generate.assert_awaited_once()
    await_args = provider_generate.await_args
    assert await_args.kwargs["conversation_id"] == "conversation-123"
    assert await_args.kwargs["messages"] is messages
    assert await_args.kwargs["current_persona"] == "Persona"
    assert await_args.kwargs["conversation_manager"] is atlas.chat_session
    atlas.maybe_text_to_speech.assert_awaited_once_with("ok")
    atlas.logger.error.assert_not_called()
    provider_manager.set_current_conversation_id.assert_called_with("conversation-123")


def test_generate_response_handles_tts_failure():
    atlas = ATLAS.__new__(ATLAS)
    atlas.logger = SimpleNamespace(error=Mock(), debug=Mock())
    atlas.current_persona = "Persona"

    provider_generate = AsyncMock(return_value="ok")
    provider_manager = SimpleNamespace(
        generate_response=provider_generate,
        set_current_conversation_id=Mock(),
    )
    atlas.provider_manager = provider_manager
    atlas.chat_session = SimpleNamespace(conversation_id="conversation-123")
    atlas._ensure_user_identity = Mock(return_value=("user", "User"))

    atlas.speech_manager = SimpleNamespace(
        get_tts_status=Mock(return_value=True),
        text_to_speech=AsyncMock(side_effect=RuntimeError("boom")),
    )

    messages = [{"role": "user", "content": "Hello"}]
    result = asyncio.run(atlas.generate_response(messages))

    assert result == "ok"
    provider_generate.assert_awaited_once()
    atlas.speech_manager.text_to_speech.assert_awaited_once_with("ok")
    assert atlas.logger.error.called
    error_message = atlas.logger.error.call_args[0][0]
    assert "Text-to-speech failed" in error_message


def test_generate_response_streaming_skips_tts_and_preserves_stream():
    atlas = ATLAS.__new__(ATLAS)
    atlas.logger = SimpleNamespace(error=Mock(), debug=Mock())
    atlas.current_persona = "Persona"

    async def stream_response():
        yield "Hello"
        yield " world"

    provider_generate = AsyncMock(return_value=stream_response())
    atlas.provider_manager = SimpleNamespace(
        generate_response=provider_generate,
        set_current_conversation_id=Mock(),
    )
    atlas.chat_session = SimpleNamespace(conversation_id="conversation-123")
    atlas._ensure_user_identity = Mock(return_value=("user", "User"))
    atlas.speech_manager = SimpleNamespace(
        get_tts_status=Mock(return_value=True),
        text_to_speech=AsyncMock(),
    )

    messages = [{"role": "user", "content": "Hello"}]

    async def run_test() -> str:
        response = await atlas.generate_response(messages)
        assert isinstance(response, AsyncIterator)
        chunks = []
        async for piece in response:
            chunks.append(piece)
        return "".join(chunks)

    collected = asyncio.run(run_test())

    assert collected == "Hello world"
    provider_generate.assert_awaited_once()
    atlas.speech_manager.text_to_speech.assert_not_called()
    atlas.logger.error.assert_not_called()
    atlas.logger.debug.assert_any_call(
        "Skipping text-to-speech for streaming async iterator response."
    )
