import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
