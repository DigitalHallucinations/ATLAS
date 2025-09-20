import logging
import tempfile
import unittest
from pathlib import Path

from modules.Chat.chat_session import (
    ChatExportResult,
    ChatHistoryExportError,
    ChatSession,
)


class DummyPersonaManager:
    def __init__(self, prompt="You are a helpful assistant."):
        self._prompt = prompt

    def get_current_persona_prompt(self):
        return self._prompt


class DummyProviderManager:
    def get_default_model_for_provider(self, provider):  # pragma: no cover - stub
        return "dummy-model"


class DummyAtlas:
    def __init__(self):
        self._default_provider = "dummy-provider"
        self._default_model = "dummy-model"
        self.logger = logging.getLogger("ChatSessionTests")
        self.persona_manager = DummyPersonaManager()
        self.provider_manager = DummyProviderManager()

    def get_default_provider(self):
        return self._default_provider

    def get_default_model(self):
        return self._default_model


class ChatSessionExportTests(unittest.TestCase):
    def setUp(self):
        self.atlas = DummyAtlas()
        self.session = ChatSession(self.atlas)

    def test_export_history_includes_system_and_metadata(self):
        self.session.conversation_history = [
            {
                "role": "system",
                "content": "Persona rules",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {"role": "user", "content": "Hello", "foo": "bar"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "chat.txt"
            result = self.session.export_history(export_path)

            self.assertIsInstance(result, ChatExportResult)
            self.assertEqual(result.message_count, 3)
            data = export_path.read_text(encoding="utf-8")

        self.assertIn("# Session information", data)
        self.assertIn("# Provider: dummy-provider", data)
        self.assertIn("# Model: dummy-model", data)
        self.assertIn("1. system: Persona rules", data)
        self.assertIn(
            '    metadata: {"timestamp": "2024-01-01T00:00:00Z"}',
            data,
        )
        self.assertIn("2. user: Hello", data)
        self.assertIn('    metadata: {"foo": "bar"}', data)
        self.assertIn("3. assistant: Hi there!", data)

    def test_export_history_raises_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "empty.txt"
            with self.assertRaises(ChatHistoryExportError):
                self.session.export_history(export_path)

    def test_iter_messages_returns_copies(self):
        original_message = {"role": "user", "content": "Test"}
        self.session.conversation_history = [original_message]

        messages = list(self.session.iter_messages())
        self.assertEqual(len(messages), 1)
        messages[0]["content"] = "Modified"

        self.assertEqual(self.session.conversation_history[0]["content"], "Test")


if __name__ == "__main__":
    unittest.main()
