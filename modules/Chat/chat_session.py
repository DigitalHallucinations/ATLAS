"""Chat session management for the ATLAS application."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from concurrent.futures import Future
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Mapping, Optional, TypeVar, Union

from modules.background_tasks import run_async_in_thread
from modules.orchestration.consensus import (
    NegotiationError,
    NegotiationOutcome,
    NegotiationParticipant,
    Proposal,
    run_protocol,
)


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
        self.conversation_history: list[Dict[str, Any]] = []
        self.negotiation_history: list[Dict[str, Any]] = []
        self.current_model = None
        self.current_provider = None
        self.current_persona_prompt = None
        self.messages_since_last_reminder = 0
        self.reminder_interval = 10  # Remind of persona every 10 messages
        self._conversation_id = self._generate_conversation_id()
        self._last_persona_reminder_index: int | None = None
        self._conversation_repository = getattr(self.ATLAS, "conversation_repository", None)
        tenant_identifier = getattr(self.ATLAS, "tenant_id", None)
        self._tenant_id = str(tenant_identifier).strip() if tenant_identifier else "default"
        if self._conversation_repository is not None:
            try:
                self._conversation_repository.ensure_conversation(
                    self._conversation_id, tenant_id=self._tenant_id
                )
                retention = self.ATLAS.config_manager.get_conversation_retention_policies()
                history_limit = retention.get("history_message_limit") if retention else None
                stored_messages = self._conversation_repository.load_recent_messages(
                    self._conversation_id,
                    tenant_id=self._tenant_id,
                    limit=history_limit,
                )
                self.conversation_history.extend(stored_messages)
                self._apply_history_limit()
            except Exception as exc:  # pragma: no cover - persistence fallback
                self.ATLAS.logger.warning(
                    "Failed to load conversation history from store: %s", exc, exc_info=True
                )
                self._conversation_repository = None
        self.set_default_provider_and_model()

    def _generate_conversation_id(self) -> str:
        return str(uuid.uuid4())

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def get_conversation_id(self) -> str:
        """Return the current conversation identifier."""

        return self._conversation_id

    def set_default_provider_and_model(self):
        self.current_provider = self.ATLAS.get_default_provider()
        self.current_model = self.ATLAS.get_default_model()
        self.ATLAS.logger.debug(
            "ChatSession initialized with provider: %s and model: %s",
            self.current_provider,
            self.current_model,
        )

    async def send_message(self, message: str) -> Union[str, Dict[str, Any]]:
        new_persona_prompt = self.ATLAS.persona_manager.get_current_persona_prompt()
        # Check if persona has changed or if it's the first message
        if new_persona_prompt != self.current_persona_prompt or not self.conversation_history:
            self.switch_persona(new_persona_prompt)

        ensure_identity = getattr(self.ATLAS, "_ensure_user_identity", None)
        active_user = None
        active_display: str | None = None
        if callable(ensure_identity):
            try:
                identity = ensure_identity()
                if isinstance(identity, (list, tuple)) and identity:
                    active_user = identity[0]
                    if len(identity) > 1:
                        active_display = identity[1]
                else:
                    active_user = identity
            except Exception:  # pragma: no cover - defensive fallback
                active_user = None
                active_display = None

        self.add_message(
            user=str(active_user) if active_user is not None else None,
            conversation_id=self._conversation_id,
            role="user",
            content=message,
            metadata={"source": "user"},
            user_display_name=active_display,
            message_type="text",
            status="sent",
        )
        self.messages_since_last_reminder += 1

        # Periodically reinforce the persona
        if self.messages_since_last_reminder >= self.reminder_interval:
            self.reinforce_persona()

        if not self.current_model or not self.current_provider:
            self.set_default_provider_and_model()

        provider_manager = self.ATLAS.provider_manager
        try:
            provider_manager.set_current_conversation_id(self._conversation_id)
        except AttributeError:  # pragma: no cover - defensive
            self.ATLAS.logger.debug(
                "Provider manager does not support conversation ID updates during send_message."
            )

        negotiation_metadata: Dict[str, Any] | None = None
        response: Any = None
        negotiation_outcome: NegotiationOutcome | None = None

        collaboration_config = self._resolve_collaboration_config()
        if collaboration_config.get("enabled"):
            try:
                negotiation_outcome = await self._run_collaborative_exchange(
                    collaboration_config,
                    active_user=active_user,
                )
            except NegotiationError as exc:
                self.ATLAS.logger.error("Negotiation failed: %s", exc, exc_info=True)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.ATLAS.logger.error("Unexpected negotiation error: %s", exc, exc_info=True)

        if negotiation_outcome is not None and getattr(negotiation_outcome, "trace", None) is not None:
            trace_dict = negotiation_outcome.trace.to_dict()
            self.record_negotiation_trace(trace_dict)
            negotiation_metadata = {
                "trace_id": trace_dict.get("id"),
                "status": trace_dict.get("status"),
                "protocol": trace_dict.get("protocol"),
            }
            notes = trace_dict.get("notes")
            if notes:
                negotiation_metadata["notes"] = notes

            if negotiation_outcome.success and negotiation_outcome.selected_proposal is not None:
                selected = negotiation_outcome.selected_proposal
                response = (
                    selected.payload if selected.payload is not None else selected.content
                )
                negotiation_metadata["selected"] = {
                    "participant": selected.participant_id,
                    "score": selected.score,
                }

        if response is None:
            try:
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

        metadata: Dict[str, Any] = {"source": "model"}
        if thinking_text:
            metadata["thinking"] = thinking_text
        if negotiation_metadata:
            metadata["negotiation"] = negotiation_metadata

        self.add_message(
            user=str(active_user) if active_user is not None else None,
            conversation_id=self._conversation_id,
            role="assistant",
            content=response_text,
            metadata=metadata,
            message_type="text",
            status="sent",
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
            self.ATLAS.logger.debug(
                "Switched to new persona: %s...",
                prompt[:50],
            )  # Log first 50 chars
        else:
            self.ATLAS.logger.debug("Cleared persona system prompt; no persona active.")
        self.messages_since_last_reminder = 0
        self._last_persona_reminder_index = None
        self._sync_conversation_history()

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
        self.ATLAS.logger.debug("Reinforced persona in conversation")
        self._sync_conversation_history()

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
        old_conversation_id = self._conversation_id
        self._conversation_id = self._generate_conversation_id()
        self._last_persona_reminder_index = None
        self.negotiation_history = []
        self.set_default_provider_and_model()
        notifier = getattr(self.ATLAS, "notify_conversation_updated", None)
        if self._conversation_repository is not None:
            try:
                archived = False
                archive_metadata = {
                    "archived": True,
                    "archived_reason": "reset",
                }
                archive_fn = getattr(
                    self._conversation_repository, "archive_conversation", None
                )
                if callable(archive_fn):
                    archived = archive_fn(
                        old_conversation_id,
                        tenant_id=self._tenant_id,
                        metadata=archive_metadata,
                    )
                else:
                    self._conversation_repository.ensure_conversation(
                        old_conversation_id,
                        tenant_id=self._tenant_id,
                        metadata=archive_metadata,
                    )
                    archived = True
                if callable(notifier) and archived:
                    notifier(old_conversation_id, reason="archived")
                self._conversation_repository.ensure_conversation(
                    self._conversation_id, tenant_id=self._tenant_id
                )
                if callable(notifier):
                    notifier(self._conversation_id, reason="created")
            except Exception as exc:  # pragma: no cover - persistence fallback
                self.ATLAS.logger.warning(
                    "Failed to reset persistent conversation state: %s",
                    exc,
                    exc_info=True,
                )
        elif callable(notifier):
            notifier(self._conversation_id, reason="reset")
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
        self.ATLAS.logger.debug("ChatSession model set to: %s", model)

    def set_provider(self, provider: str):
        self.current_provider = provider
        self.ATLAS.logger.debug("ChatSession provider set to: %s", provider)
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

    def record_negotiation_trace(self, trace: Mapping[str, Any]) -> None:
        """Persist ``trace`` for later inspection via the UI."""

        try:
            payload = dict(trace)
        except Exception:  # pragma: no cover - defensive fallback
            payload = {"data": trace}
        self.negotiation_history.append(payload)

    def get_negotiation_history(self) -> list[Dict[str, Any]]:
        """Return copies of recorded negotiation traces."""

        return [dict(entry) for entry in self.negotiation_history]

    def _resolve_collaboration_config(self) -> Dict[str, Any]:
        persona_manager = getattr(self.ATLAS, "persona_manager", None)
        base_config: Dict[str, Any] = {}
        overrides: List[Mapping[str, Any]] = []

        if persona_manager is not None:
            persona_getter = getattr(persona_manager, "get_active_collaboration_profile", None)
            if callable(persona_getter):
                try:
                    candidate = persona_getter() or {}
                except Exception:  # pragma: no cover - defensive logging upstream
                    candidate = {}
                if isinstance(candidate, Mapping):
                    base_config = dict(candidate)

            override_getter = getattr(persona_manager, "get_skill_collaboration_overrides", None)
            if callable(override_getter):
                try:
                    skill_overrides = override_getter() or []
                except Exception:  # pragma: no cover - defensive logging upstream
                    skill_overrides = []
                for override in skill_overrides:
                    if isinstance(override, Mapping):
                        overrides.append(override)

        history_requests = self._extract_history_collaboration_requests()
        overrides.extend(history_requests)

        config = dict(base_config) if base_config else {}
        for override in overrides:
            config = self._merge_collaboration_configs(config, override)

        return config

    def _extract_history_collaboration_requests(self) -> List[Mapping[str, Any]]:
        requests: List[Mapping[str, Any]] = []
        for message in reversed(self.conversation_history):
            metadata = message.get("metadata") if isinstance(message, Mapping) else None
            if not isinstance(metadata, Mapping):
                continue
            collab = metadata.get("collaboration")
            if isinstance(collab, Mapping):
                requests.append(collab)
                break
        return requests

    def _merge_collaboration_configs(
        self, base: Mapping[str, Any], override: Mapping[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(base) if isinstance(base, Mapping) else {}
        if not isinstance(override, Mapping):
            return merged

        if "enabled" in override:
            merged["enabled"] = self._coerce_bool(override.get("enabled"))
        if override.get("protocol"):
            merged["protocol"] = str(override.get("protocol")).strip().lower()
        if "quorum" in override:
            merged["quorum"] = self._clamp_float(
                override.get("quorum"),
                default=float(merged.get("quorum", 0.5)),
                minimum=0.0,
                maximum=1.0,
            )
        if "timeout" in override:
            merged["timeout"] = self._clamp_float(
                override.get("timeout"),
                default=float(merged.get("timeout", 10.0)),
                minimum=0.0,
                maximum=None,
            )

        participants = list(merged.get("participants") or [])
        incoming = override.get("participants")
        if isinstance(incoming, list):
            for participant in incoming:
                if isinstance(participant, Mapping):
                    participants.append(dict(participant))
        if participants:
            merged["participants"] = participants

        return merged

    async def _run_collaborative_exchange(
        self, config: Mapping[str, Any], *, active_user: Optional[str]
    ) -> Optional[NegotiationOutcome]:
        provider_manager = getattr(self.ATLAS, "provider_manager", None)
        if provider_manager is None:
            return None

        participants = self._build_negotiation_participants(config)
        if len(participants) < 2:
            return None

        context = {
            "conversation_id": self._conversation_id,
            "user": active_user,
            "persona_prompt": self.current_persona_prompt,
        }

        desired_provider = self.current_provider
        try:
            outcome = await run_protocol(
                config.get("protocol", "vote"),
                participants,
                context=context,
                quorum_threshold=float(config.get("quorum", 0.5) or 0.0),
                timeout=float(config.get("timeout", 10.0) or 0.0),
            )
        finally:
            try:
                current_provider = getattr(provider_manager, "current_llm_provider", None)
            except Exception:  # pragma: no cover - defensive access
                current_provider = None

            if (
                desired_provider
                and current_provider
                and desired_provider != current_provider
            ):
                restorer = getattr(provider_manager, "switch_llm_provider", None)
                if callable(restorer):
                    try:
                        await restorer(desired_provider)
                    except Exception:  # pragma: no cover - defensive logging
                        self.ATLAS.logger.debug(
                            "Failed to restore provider after negotiation.",
                            exc_info=True,
                        )

        return outcome

    def _build_negotiation_participants(
        self, config: Mapping[str, Any]
    ) -> List[NegotiationParticipant]:
        provider_manager = getattr(self.ATLAS, "provider_manager", None)
        if provider_manager is None:
            return []

        participant_configs = config.get("participants")
        if not isinstance(participant_configs, list) or not participant_configs:
            participant_configs = [
                {
                    "id": "primary",
                    "provider": self.current_provider,
                    "model": self.current_model,
                }
            ]

        participants: List[NegotiationParticipant] = []

        for index, entry in enumerate(participant_configs):
            if not isinstance(entry, Mapping):
                continue

            agent_id = str(entry.get("id") or f"agent_{index + 1}").strip() or f"agent_{index + 1}"
            config_copy = dict(entry)

            async def _propose(
                context: Mapping[str, Any],
                *,
                agent_id: str = agent_id,
                config_copy: Mapping[str, Any] = config_copy,
            ) -> Proposal:
                messages = self._prepare_negotiation_messages(config_copy.get("system_prompt"))
                provider = config_copy.get("provider") or self.current_provider
                model = config_copy.get("model") or self.current_model
                payload = await provider_manager.generate_response(
                    messages=messages,
                    model=model,
                    provider=provider,
                    stream=False,
                    conversation_id=self._conversation_id,
                    conversation_manager=self,
                    user=context.get("user"),
                )
                return self._build_proposal(agent_id, payload, config_copy)

            participants.append(
                NegotiationParticipant(
                    participant_id=agent_id,
                    propose=_propose,
                )
            )

        return participants

    def _prepare_negotiation_messages(self, system_prompt: Any) -> List[Dict[str, Any]]:
        history = [dict(message) for message in self.conversation_history]
        if system_prompt:
            history.insert(0, {"role": "system", "content": str(system_prompt)})
        return history

    def _build_proposal(
        self,
        agent_id: str,
        payload: Any,
        config: Mapping[str, Any],
    ) -> Proposal:
        extra_metadata = config.get("metadata")
        metadata: Dict[str, Any] = dict(extra_metadata) if isinstance(extra_metadata, Mapping) else {}
        default_weight = config.get("weight")

        if isinstance(payload, Mapping):
            text = str(payload.get("text") or payload.get("content") or "")
            score = payload.get("score")
            rationale = payload.get("rationale")
            payload_metadata = payload.get("metadata")
            if isinstance(payload_metadata, Mapping):
                metadata.update(payload_metadata)
        else:
            text = str(payload or "")
            score = None
            rationale = None

        normalized_score = self._clamp_float(
            score,
            default=self._clamp_float(default_weight, default=1.0, minimum=None, maximum=None),
            minimum=None,
            maximum=None,
        )

        return Proposal(
            participant_id=agent_id,
            content=text,
            score=normalized_score,
            rationale=rationale,
            payload=payload,
            metadata=metadata,
        )

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on", "enabled"}:
                return True
            if lowered in {"false", "0", "no", "off", "disabled"}:
                return False
        return bool(value)

    @staticmethod
    def _clamp_float(
        value: Any,
        *,
        default: float,
        minimum: Optional[float],
        maximum: Optional[float],
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = float(default)

        if minimum is not None and result < minimum:
            result = minimum
        if maximum is not None and result > maximum:
            result = maximum

        return result

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

    def _persist_entry(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        if self._conversation_repository is None:
            return dict(entry)

        try:
            role = str(entry.get("role") or "assistant")
            content = entry.get("content")
            metadata = entry.get("metadata")
            if metadata is not None and not isinstance(metadata, Mapping):
                metadata = {"value": metadata}
            timestamp = entry.get("timestamp")
            message_id = entry.get("message_id")
            fallback_id = entry.get("id")
            message_type = entry.get("message_type")
            status = entry.get("status")
            extra = {
                key: value
                for key, value in entry.items()
                if key
                not in {
                    "id",
                    "conversation_id",
                    "role",
                    "content",
                    "metadata",
                    "timestamp",
                    "message_type",
                    "status",
                }
            }
            stored = self._conversation_repository.add_message(
                self._conversation_id,
                tenant_id=self._tenant_id,
                role=role,
                content=content,
                metadata=metadata,
                timestamp=timestamp,
                message_id=message_id or (str(fallback_id) if fallback_id else None),
                extra=extra,
                message_type=message_type,
                status=status,
            )
        except Exception as exc:  # pragma: no cover - defensive persistence
            self.ATLAS.logger.warning(
                "Failed to persist conversation entry: %s", exc, exc_info=True
            )
            return dict(entry)

        return stored

    def _sync_conversation_history(self) -> None:
        if self._conversation_repository is None:
            return

        snapshot = list(self.conversation_history)
        try:
            self._conversation_repository.ensure_conversation(
                self._conversation_id, tenant_id=self._tenant_id
            )
            self._conversation_repository.reset_conversation(
                self._conversation_id, tenant_id=self._tenant_id
            )
            refreshed: list[Dict[str, Any]] = []
            for entry in snapshot:
                refreshed.append(self._persist_entry(entry))
        except Exception as exc:  # pragma: no cover - defensive persistence
            self.ATLAS.logger.warning(
                "Failed to synchronize conversation history: %s", exc, exc_info=True
            )
        else:
            self.conversation_history = refreshed
            self._apply_history_limit()

    def _apply_history_limit(self) -> None:
        retention = self.ATLAS.config_manager.get_conversation_retention_policies()
        limit = retention.get("history_message_limit") if retention else None
        if not isinstance(limit, int) or limit <= 0:
            return

        if len(self.conversation_history) <= limit:
            return

        overflow = len(self.conversation_history) - limit
        removed_entries = self.conversation_history[:overflow]
        self.conversation_history = self.conversation_history[overflow:]

        if self._conversation_repository is None:
            return

        for entry in removed_entries:
            message_id = entry.get("id")
            if not message_id:
                continue
            try:
                self._conversation_repository.soft_delete_message(
                    self._conversation_id,
                    message_id,
                    tenant_id=self._tenant_id,
                    reason="history_retention",
                )
            except Exception:  # pragma: no cover - best-effort enforcement
                self.ATLAS.logger.debug(
                    "Failed to soft delete message %s during retention cleanup",
                    message_id,
                    exc_info=True,
                )

    def add_message(
        self,
        user,
        conversation_id,
        role,
        content,
        timestamp=None,
        metadata: Dict[str, Any] | None = None,
        user_display_name: str | None = None,
        user_metadata: Dict[str, Any] | None = None,
        session_identifier: str | None = None,
        session_metadata: Dict[str, Any] | None = None,
        message_type: str | None = None,
        status: str | None = None,
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
        entry["message_type"] = message_type or "text"
        entry["status"] = status or "sent"
        if user_display_name is not None:
            entry["user_display_name"] = user_display_name
        if user_metadata:
            entry["user_metadata"] = dict(user_metadata)
        if session_identifier is not None:
            entry["session_identifier"] = session_identifier
        if session_metadata:
            entry["session_metadata"] = dict(session_metadata)
        stored_entry: Dict[str, Any]
        if extra_fields:
            entry.update(extra_fields)

        if self._conversation_repository is not None:
            message_id = extra_fields.get("message_id") if extra_fields else None
            extra_payload = {
                key: value
                for key, value in entry.items()
                if key
                not in {
                    "role",
                    "content",
                    "metadata",
                    "timestamp",
                    "conversation_id",
                    "message_id",
                    "message_type",
                    "status",
                    "user_display_name",
                    "user_metadata",
                    "session_identifier",
                    "session_metadata",
                }
            }
            try:
                metadata_payload = entry.get("metadata")
                if metadata_payload is not None and not isinstance(metadata_payload, Mapping):
                    metadata_payload = {"value": metadata_payload}
                entry_metadata = metadata_payload
                user_value = None
                if isinstance(entry_metadata, Mapping):
                    user_value = entry_metadata.get("user")
                user_meta_payload = entry.get("user_metadata")
                if not isinstance(user_meta_payload, Mapping):
                    user_meta_payload = None
                session_meta_payload = entry.get("session_metadata")
                if not isinstance(session_meta_payload, Mapping):
                    session_meta_payload = None
                stored_entry = self._conversation_repository.add_message(
                    self._conversation_id,
                    tenant_id=self._tenant_id,
                    role=role,
                    content=content,
                    metadata=entry_metadata,
                    timestamp=entry.get("timestamp"),
                    message_id=message_id,
                    extra=extra_payload,
                    user=user_value,
                    user_display_name=entry.get("user_display_name"),
                    user_metadata=dict(user_meta_payload) if user_meta_payload else None,
                    session=entry.get("session_identifier"),
                    session_metadata=dict(session_meta_payload)
                    if session_meta_payload
                    else None,
                    message_type=entry.get("message_type"),
                    status=entry.get("status"),
                )
            except Exception as exc:  # pragma: no cover - persistence fallback
                self.ATLAS.logger.warning(
                    "Failed to persist message: %s", exc, exc_info=True
                )
                stored_entry = dict(entry)
            else:
                entry = stored_entry
        else:
            stored_entry = dict(entry)

        self.conversation_history.append(stored_entry)
        self._apply_history_limit()
        notifier = getattr(self.ATLAS, "notify_conversation_updated", None)
        if callable(notifier):
            try:
                notifier(self._conversation_id, reason="message")
            except Exception:  # pragma: no cover - defensive logging only
                self.ATLAS.logger.debug(
                    "Conversation update notification failed for %s", self._conversation_id, exc_info=True
                )
        return stored_entry

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

        stored_entry = self.add_message(
            user=user,
            conversation_id=conversation_id,
            role=entry.get("role", "tool"),
            content=entry.get("content"),
            timestamp=entry.get("timestamp"),
            metadata=entry.get("metadata"),
            message_type="tool",
            status="sent",
            **{
                key: value
                for key, value in entry.items()
                if key
                not in {"role", "content", "metadata", "timestamp", "id", "conversation_id"}
            },
        )
        return stored_entry

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
