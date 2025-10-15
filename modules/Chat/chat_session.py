"""Chat session management for the ATLAS application."""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import Future
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, Mapping, Optional, TypeVar, Union

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
        self._conversation_id = self._generate_conversation_id()
        self._last_persona_reminder_index: int | None = None
        self.set_default_provider_and_model()

    def _generate_conversation_id(self) -> str:
        return uuid.uuid4().hex

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def get_conversation_id(self) -> str:
        """Return the current conversation identifier."""

        return self._conversation_id

    def set_default_provider_and_model(self):
        self.current_provider = self.ATLAS.get_default_provider()
        self.current_model = self.ATLAS.get_default_model()
        self.ATLAS.logger.info(f"ChatSession initialized with provider: {self.current_provider} and model: {self.current_model}")

    async def send_message(self, message: str) -> Union[str, Dict[str, Any]]:
        new_persona_prompt = self.ATLAS.persona_manager.get_current_persona_prompt()
        # Check if persona has changed or if it's the first message
        if new_persona_prompt != self.current_persona_prompt or not self.conversation_history:
            self.switch_persona(new_persona_prompt)

        ensure_identity = getattr(self.ATLAS, "_ensure_user_identity", None)
        active_user = None
        if callable(ensure_identity):
            try:
                active_user, _ = ensure_identity()
            except Exception:  # pragma: no cover - defensive fallback
                active_user = None

        self.add_message(
            user=str(active_user) if active_user is not None else None,
            conversation_id=self._conversation_id,
            role="user",
            content=message,
            metadata={"source": "user"},
        )
        self.messages_since_last_reminder += 1

        # Periodically reinforce the persona
        if self.messages_since_last_reminder >= self.reminder_interval:
            self.reinforce_persona()

        if not self.current_model or not self.current_provider:
            self.set_default_provider_and_model()

        try:
            provider_manager = self.ATLAS.provider_manager
            try:
                provider_manager.set_current_conversation_id(self._conversation_id)
            except AttributeError:  # pragma: no cover - defensive
                self.ATLAS.logger.debug(
                    "Provider manager does not support conversation ID updates during send_message."
                )

            response = await provider_manager.generate_response(
                messages=self.conversation_history,
                model=self.current_model,
                stream=False,
                conversation_id=self._conversation_id,
                conversation_manager=self,
                user=active_user,
            )
        except Exception as e:
            self.ATLAS.logger.error(f"Error generating response: {e}", exc_info=True)
            raise

        response_text: str
        thinking_text: Optional[str] = None
        audio_payload: Dict[str, Any] = {}

        if isinstance(response, dict):
            response_text = str(response.get("text") or "")
            if response.get("thinking") is not None:
                thinking_text = str(response.get("thinking") or "")
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

        metadata = {"source": "model"}
        if thinking_text:
            metadata["thinking"] = thinking_text

        self.add_message(
            user=str(active_user) if active_user is not None else None,
            conversation_id=self._conversation_id,
            role="assistant",
            content=response_text,
            metadata=metadata,
            **audio_payload,
        )
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

    def switch_persona(self, new_persona_prompt: str | None):
        prompt = new_persona_prompt or None
        self.current_persona_prompt = prompt
        # Remove any existing system messages
        self.conversation_history = [
            msg for msg in self.conversation_history if msg["role"] != "system"
        ]
        if prompt:
            # Insert the new persona's system prompt at the start of the conversation
            self.conversation_history.insert(0, {"role": "system", "content": prompt})
            self.ATLAS.logger.info(
                f"Switched to new persona: {prompt[:50]}..."
            )  # Log first 50 chars
        else:
            self.ATLAS.logger.info("Cleared persona system prompt; no persona active.")
        self.messages_since_last_reminder = 0
        self._last_persona_reminder_index = None

    def reinforce_persona(self):
        if not self.current_persona_prompt:
            return

        reminder = {
            "role": "system",
            "content": f"Remember, you are acting as: {self.current_persona_prompt[:100]}...",
        }  # First 100 chars

        existing_index = self._last_persona_reminder_index
        if existing_index is not None:
            if 0 <= existing_index < len(self.conversation_history):
                existing_message = self.conversation_history[existing_index]
                if self._is_persona_reminder(existing_message):
                    self.conversation_history.pop(existing_index)
            self._last_persona_reminder_index = None

        if self._last_persona_reminder_index is None:
            for idx in range(len(self.conversation_history) - 1, -1, -1):
                message = self.conversation_history[idx]
                if self._is_persona_reminder(message):
                    self.conversation_history.pop(idx)
                    break

        reminder_index = len(self.conversation_history)
        self.conversation_history.insert(reminder_index, reminder)
        self._last_persona_reminder_index = reminder_index
        self.messages_since_last_reminder = 0
        self.ATLAS.logger.info("Reinforced persona in conversation")

    @staticmethod
    def _is_persona_reminder(message: Mapping[str, Any]) -> bool:
        return (
            isinstance(message, Mapping)
            and message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("Remember, you are acting as")
        )

    def reset_conversation(self):
        """
        Reset the conversation history.
        """
        self.conversation_history = []
        self.current_persona_prompt = None
        self.messages_since_last_reminder = 0
        self._conversation_id = self._generate_conversation_id()
        self._last_persona_reminder_index = None
        self.set_default_provider_and_model()
        provider_manager = getattr(self.ATLAS, "provider_manager", None)
        if provider_manager is not None:
            try:
                provider_manager.set_current_conversation_id(self._conversation_id)
            except AttributeError:  # pragma: no cover - defensive
                self.ATLAS.logger.debug(
                    "Provider manager does not support conversation ID updates during reset."
                )

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

    @staticmethod
    def _current_timestamp() -> str:
        """Return a formatted timestamp for history entries."""

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _base_entry(
        self,
        *,
        role: str,
        content: Any,
        user: Any = None,
        conversation_id: str | None = None,
        timestamp: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"role": role, "content": content}
        entry["timestamp"] = timestamp or self._current_timestamp()

        combined_metadata: Dict[str, Any] = {}
        if conversation_id is not None:
            combined_metadata["conversation_id"] = conversation_id
        if user is not None:
            combined_metadata["user"] = user
        if metadata:
            combined_metadata.update(metadata)

        if combined_metadata:
            entry["metadata"] = combined_metadata

        return entry

    def add_message(
        self,
        user,
        conversation_id,
        role,
        content,
        timestamp=None,
        metadata: Dict[str, Any] | None = None,
        **extra_fields: Any,
    ) -> Dict[str, Any]:
        """Add a generic message to the conversation history.

        Mirrors the signature used within :mod:`ATLAS.ToolManager` so that
        other components can record messages without touching
        ``conversation_history`` directly.
        """

        entry = self._base_entry(
            role=role,
            content=content,
            user=user,
            conversation_id=conversation_id,
            timestamp=timestamp,
            metadata=metadata,
        )
        if extra_fields:
            entry.update(extra_fields)

        self.conversation_history.append(entry)
        return entry

    def _clone_tool_payload(self, value: Any) -> Any:
        """Return a JSON-friendly clone of ``value`` suitable for storage."""

        if isinstance(value, dict):
            return {key: self._clone_tool_payload(val) for key, val in value.items()}

        if isinstance(value, list):
            return [self._clone_tool_payload(item) for item in value]

        if isinstance(value, tuple):
            return [self._clone_tool_payload(item) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        try:
            return json.loads(json.dumps(value, ensure_ascii=False))
        except (TypeError, ValueError):
            return str(value)

    def _normalize_tool_response(
        self, response: Any, *, wrap_text: bool = True
    ) -> Any:
        """Normalize a tool response for structured storage."""

        if isinstance(response, dict):
            return self._clone_tool_payload(response)

        if isinstance(response, list):
            normalized_list = []
            for item in response:
                if isinstance(item, (dict, list, tuple)):
                    normalized_list.append(
                        self._normalize_tool_response(item, wrap_text=wrap_text)
                    )
                elif wrap_text:
                    normalized_list.append(
                        {
                            "type": "output_text",
                            "text": "" if item is None else str(item),
                        }
                    )
                else:
                    normalized_list.append("" if item is None else str(item))
            return normalized_list

        if isinstance(response, tuple):
            return self._normalize_tool_response(list(response), wrap_text=wrap_text)

        if wrap_text:
            text_value = "" if response is None else str(response)
            return {"type": "output_text", "text": text_value}

        return "" if response is None else str(response)

    def add_response(
        self,
        user,
        conversation_id,
        response,
        timestamp=None,
        *,
        tool_call_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
        wrap_text: bool = True,
    ) -> Dict[str, Any]:
        """Record a tool/function response in the conversation history."""

        base_metadata = {"name": "tool"}
        if metadata:
            base_metadata.update(metadata)

        normalized_content = self._normalize_tool_response(response, wrap_text=wrap_text)

        entry = self._base_entry(
            role="tool",
            content=normalized_content,
            user=user,
            conversation_id=conversation_id,
            timestamp=timestamp,
            metadata=base_metadata,
        )

        if tool_call_id is not None:
            entry["tool_call_id"] = tool_call_id
            entry_metadata = entry.setdefault("metadata", {})
            entry_metadata.setdefault("tool_call_id", tool_call_id)

        self.conversation_history.append(entry)
        return entry

    def get_history(self, user=None, conversation_id=None) -> list[Dict[str, Any]]:
        """Return a snapshot list of the current conversation history."""

        return list(self.iter_messages())

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
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            message = (
                f"Failed to create directory for chat history export at "
                f"{target.parent}: {exc}"
            )
            self.ATLAS.logger.error(message, exc_info=True)
            raise ChatHistoryExportError(message) from exc

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
