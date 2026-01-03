import asyncio
import logging

from modules.Chat.chat_session import ChatSession


class DummyPersonaManager:
    def __init__(self, prompt="You are a helpful assistant."):
        self._prompt = prompt

    def get_current_persona_prompt(self):
        return self._prompt


class DummyProviderManager:
    async def generate_response(self, *args, **kwargs):
        return {"text": "Acknowledged."}


class DummyAtlas:
    def __init__(self):
        self._default_provider = "dummy-provider"
        self._default_model = "dummy-model"
        self.logger = logging.getLogger("ChatSessionReminderTests")
        self.persona_manager = DummyPersonaManager()
        self.provider_manager = DummyProviderManager()
        self.tts_calls = []

    def get_default_provider(self):
        return self._default_provider

    def get_default_model(self):
        return self._default_model

    async def maybe_text_to_speech(self, response):  # pragma: no cover - simple stub
        self.tts_calls.append(response)


def test_persona_reminder_counts_only_user_messages():
    atlas = DummyAtlas()
    session = ChatSession(atlas)
    session.reminder_interval = 2

    async def exercise():
        await session.send_message("Hello")

        reminders = [
            message
            for message in session.conversation_history
            if message["role"] == "system"
            and message["content"].startswith("Remember, you are acting as")
        ]
        assert reminders == []
        assert session.messages_since_last_reminder == 1

        await session.send_message("How are you?")

        reminders = [
            message
            for message in session.conversation_history
            if message["role"] == "system"
            and message["content"].startswith("Remember, you are acting as")
        ]
        assert len(reminders) == 1
        assert session.messages_since_last_reminder == 0

    asyncio.run(exercise())


def test_persona_reminder_reuses_existing_entry():
    atlas = DummyAtlas()
    session = ChatSession(atlas)
    session.reminder_interval = 2

    async def exercise():
        await session.send_message("Msg 1")
        await session.send_message("Msg 2")  # Triggers first reinforcement
        await session.send_message("Msg 3")
        await session.send_message("Msg 4")  # Triggers subsequent reinforcement

        reminders = [
            (idx, message)
            for idx, message in enumerate(session.conversation_history)
            if message["role"] == "system"
            and message["content"].startswith("Remember, you are acting as")
        ]

        assert len(reminders) == 1

        reminder_index, reminder_message = reminders[0]
        assert reminder_index == len(session.conversation_history) - 2
        assert reminder_message["content"].startswith("Remember, you are acting as")
        assert session.messages_since_last_reminder == 0

    asyncio.run(exercise())

