import asyncio
import logging

from modules.Chat.chat_session import ChatSession


class _PersonaManagerStub:
    def __init__(self, prompt: str = "You are a helpful assistant."):
        self._prompt = prompt

    def get_current_persona_prompt(self):
        return self._prompt


class _ProviderManagerStub:
    def __init__(self, response_payload):
        self._response = response_payload
        self.last_conversation_id = None
        self.generate_calls = []

    def set_current_conversation_id(self, conversation_id: str) -> None:
        self.last_conversation_id = conversation_id

    async def generate_response(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self._response

    def get_default_model_for_provider(self, _provider: str) -> str:
        return "stub-model"


class _AtlasStub:
    def __init__(self, response_payload):
        self._default_provider = "stub-provider"
        self._default_model = "stub-model"
        self.persona_manager = _PersonaManagerStub()
        self.provider_manager = _ProviderManagerStub(response_payload)
        self.logger = logging.getLogger("ChatSessionThinkingTest")
        self.tts_inputs = []

    def get_default_provider(self):
        return self._default_provider

    def get_default_model(self):
        return self._default_model

    async def maybe_text_to_speech(self, payload):
        self.tts_inputs.append(payload)
        return None


def test_chat_session_records_thinking_text():
    response_payload = {"text": "Final answer", "thinking": "Deliberation"}
    atlas = _AtlasStub(response_payload)
    session = ChatSession(atlas)

    async def _send():
        return await session.send_message("Hello there")

    result = asyncio.run(_send())

    assert result == response_payload
    history = session.get_history()
    assert history[-1]["content"] == "Final answer"
    metadata = history[-1].get("metadata") or {}
    assert metadata.get("thinking") == "Deliberation"

    snapshot = list(session.iter_messages())
    assert snapshot[-1].get("metadata", {}).get("thinking") == "Deliberation"
    assert atlas.tts_inputs[-1] == response_payload
