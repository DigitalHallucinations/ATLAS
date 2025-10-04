import logging

import pytest

from modules.Chat.chat_session import ChatSession


class DummyAtlas:
    def __init__(self):
        self._default_provider = "dummy-provider"
        self._default_model = "dummy-model"
        self.logger = logging.getLogger("ChatSessionPersonaSwitchTests")
        self.persona_manager = None
        self.provider_manager = None

    def get_default_provider(self):
        return self._default_provider

    def get_default_model(self):
        return self._default_model

    async def maybe_text_to_speech(self, response):  # pragma: no cover - simple stub
        return None


@pytest.mark.parametrize("missing_prompt", ["", None])
def test_switch_persona_without_prompt_removes_system_messages(missing_prompt):
    atlas = DummyAtlas()
    session = ChatSession(atlas)

    # Seed an active persona and a conversation history entry
    session.switch_persona("Active persona prompt")
    session.conversation_history.append({"role": "user", "content": "Hello"})

    session.switch_persona(missing_prompt)

    assert session.current_persona_prompt is None
    assert all(message["role"] != "system" for message in session.conversation_history)
    assert session.messages_since_last_reminder == 0
