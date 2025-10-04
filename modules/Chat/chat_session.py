"""Chat session management for the ATLAS application."""

from __future__ import annotations

import json
import os
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, Mapping, TypeVar, Union

from modules.background_tasks import run_async_in_thread


PathLike = Union[str, os.PathLike, Path]
T = TypeVar("T")


class ChatHistoryExportError(Exception):
    """Raised when exporting chat history fails."""


@dataclass(frozen=True)
class ChatExportResult:
    """Simple container describing the outcome of a chat history export."""

    path: Path
    message_count: int


class ChatSession:
    def __init__(self, atlas):
        self.ATLAS = atlas
        self.conversation_history = []
        self.current_model = None
        self.current_provider = None
        self.current_persona_prompt = None
        self.messages_since_last_reminder = 0
        self.reminder_interval = 10  # Remind of persona every 10 messages
        self.set_default_provider_and_model()

    def set_default_provider_and_model(self):
        self.current_provider = self.ATLAS.get_default_provider()
        self.current_model = self.ATLAS.get_default_model()
        self.ATLAS.logger.info(f"ChatSession initialized with provider: {self.current_provider} and model: {self.current_model}")

    async def send_message(self, message: str) -> Union[str, Dict[str, Any]]:
        new_persona_prompt = self.ATLAS.persona_manager.get_current_persona_prompt()
        # Check if persona has changed or if it's the first message
        if new_persona_prompt != self.current_persona_prompt or not self.conversation_history:
            self.switch_persona(new_persona_prompt)

        self.conversation_history.append({"role": "user", "content": message})
        self.messages_since_last_reminder += 1

        # Periodically reinforce the persona
        if self.messages_since_last_reminder >= self.reminder_interval:
            self.reinforce_persona()

        if not self.current_model or not self.current_provider:
            self.set_default_provider_and_model()

        try:
            response = await self.ATLAS.provider_manager.generate_response(
                messages=self.conversation_history,
                model=self.current_model,
                stream=False
            )
        except Exception as e:
            self.ATLAS.logger.error(f"Error generating response: {e}", exc_info=True)
            raise

        response_text: str
        audio_payload: Dict[str, Any] = {}

        if isinstance(response, dict):
            response_text = str(response.get("text") or "")
            audio_data = response.get("audio")
            if audio_data:
                audio_payload = {
                    "audio": audio_data,
                    "audio_format": response.get("audio_format"),
                    "audio_voice": response.get("audio_voice"),
                    "audio_id": response.get("audio_id"),
                }
        else:
            response_text = str(response or "")

        await self.ATLAS.maybe_text_to_speech(response)

        message_entry: Dict[str, Any] = {"role": "assistant", "content": response_text}
        if audio_payload:
            message_entry.update(audio_payload)

        self.conversation_history.append(message_entry)
        return response

    def run_in_background(
        self,
        coroutine_factory: Callable[[], Awaitable[T]],
        *,
        on_success: Callable[[T], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        thread_name: str | None = None,
    ) -> Future[T]:
        """Execute a chat coroutine on a worker thread and return a ``Future``."""

        return run_async_in_thread(
            coroutine_factory,
            on_success=on_success,
            on_error=on_error,
            logger=self.ATLAS.logger,
            thread_name=thread_name or "ChatSessionTask",
        )

    def switch_persona(self, new_persona_prompt: str):
        self.current_persona_prompt = new_persona_prompt
        # Remove any existing system messages
        self.conversation_history = [msg for msg in self.conversation_history if msg['role'] != 'system']
        # Insert the new persona's system prompt at the start of the conversation
        self.conversation_history.insert(0, {"role": "system", "content": new_persona_prompt})
        self.ATLAS.logger.info(f"Switched to new persona: {new_persona_prompt[:50]}...")  # Log first 50 chars
        self.messages_since_last_reminder = 0

    def reinforce_persona(self):
        if self.current_persona_prompt:
            reminder = {"role": "system", "content": f"Remember, you are acting as: {self.current_persona_prompt[:100]}..."}  # First 100 chars
            self.conversation_history.append(reminder)
            self.messages_since_last_reminder = 0
            self.ATLAS.logger.info("Reinforced persona in conversation")

    def reset_conversation(self):
        """
        Reset the conversation history.
        """
        self.conversation_history = []
        self.current_persona_prompt = None
        self.messages_since_last_reminder = 0
        self.set_default_provider_and_model()

    def set_model(self, model: str):
        """
        Set the current model for the chat session.

        Args:
            model (str): The model to use.
        """
        self.current_model = model
        self.ATLAS.logger.info(f"ChatSession model set to: {model}")

    def set_provider(self, provider: str):
        self.current_provider = provider
        self.ATLAS.logger.info(f"ChatSession provider set to: {provider}")
        # When changing the provider, we should also update the model to the default for that provider
        default_model = self.ATLAS.provider_manager.get_default_model_for_provider(provider)
        if default_model:
            self.set_model(default_model)

    def iter_messages(self) -> Iterator[Dict[str, object]]:
        """Yield shallow copies of the recorded conversation history.

        Returns:
            Iterator[Dict[str, object]]: Iterable of message dictionaries.
        """

        for message in self.conversation_history:
            # Return a copy to prevent callers from mutating internal state.
            yield dict(message)

    def export_history(self, path: PathLike) -> ChatExportResult:
        """Persist the recorded conversation history to ``path``.

        Args:
            path (PathLike): Target file location.

        Returns:
            ChatExportResult: Information about the written export.

        Raises:
            ChatHistoryExportError: If the conversation is empty or the file
                cannot be written.
        """

        try:
            target = Path(path)
        except TypeError as exc:
            message = f"Invalid export path: {exc}"
            self.ATLAS.logger.error(message)
            raise ChatHistoryExportError(message) from exc

        messages = list(self.iter_messages())
        if not messages:
            message = "No conversation history to export."
            self.ATLAS.logger.warning(message)
            raise ChatHistoryExportError(message)

        session_metadata = self._format_session_metadata()
        message_lines = self._format_messages(messages)
        export_text = "".join(session_metadata + message_lines)

        try:
            target.write_text(export_text, encoding="utf-8")
        except Exception as exc:
            message = f"Failed to export chat history to {target}: {exc}"
            self.ATLAS.logger.error(message, exc_info=True)
            raise ChatHistoryExportError(message) from exc

        result = ChatExportResult(path=target, message_count=len(messages))
        self.ATLAS.logger.info(
            "Exported %s messages to %s", result.message_count, result.path
        )
        return result

    def _format_session_metadata(self) -> list[str]:
        """Format session-level metadata for export."""

        metadata_lines: list[str] = []
        session_details: Dict[str, Union[str, None]] = {
            "Provider": self.current_provider,
            "Model": self.current_model,
        }

        recorded_details = [
            f"# {key}: {value}\n"
            for key, value in session_details.items()
            if value
        ]

        if recorded_details:
            metadata_lines.append("# Session information\n")
            metadata_lines.extend(recorded_details)
            metadata_lines.append("\n")

        return metadata_lines

    def _format_messages(self, messages: list[Mapping[str, object]]) -> list[str]:
        """Create human-readable representations of conversation messages."""

        formatted: list[str] = []
        for index, message in enumerate(messages, start=1):
            role = str(message.get("role", "unknown"))
            content = message.get("content", "")
            content_str = content if isinstance(content, str) else repr(content)
            formatted.append(f"{index}. {role}: {content_str}\n")

            metadata = {
                key: value
                for key, value in message.items()
                if key not in {"role", "content"}
            }
            if metadata:
                metadata_repr = self._serialize_metadata(metadata)
                formatted.append(f"    metadata: {metadata_repr}\n")

            formatted.append("\n")

        return formatted

    @staticmethod
    def _serialize_metadata(metadata: Mapping[str, object]) -> str:
        """Serialize metadata for inclusion in export output."""

        safe_metadata: Dict[str, object] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_metadata[key] = value
            else:
                safe_metadata[key] = repr(value)

        return json.dumps(safe_metadata, ensure_ascii=False, sort_keys=True)
