"""Conversation service helpers for the ATLAS application."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing helper for optional dependency
    from modules.Chat.chat_session import ChatSession
    from modules.conversation_store import ConversationStoreRepository


class ConversationService:
    """Encapsulate conversation repository helpers and notifications."""

    def __init__(
        self,
        *,
        repository: "ConversationStoreRepository | None",
        logger: Any,
        tenant_id: str,
        chat_session_getter: Optional[Callable[[], "ChatSession"]] = None,
    ) -> None:
        self._repository = repository
        self._logger = logger
        self._tenant_id = str(tenant_id).strip() or "default"
        self._chat_session_getter = chat_session_getter
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

    # ------------------------------------------------------------------
    # Listener registration
    # ------------------------------------------------------------------
    def add_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback notified when the conversation list changes."""

        if not callable(listener):
            raise TypeError("listener must be callable")

        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously registered conversation listener."""

        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def notify_updated(self, conversation_id: Any, *, reason: str = "updated") -> None:
        """Notify UI components that a conversation has changed."""

        payload = {"conversation_id": str(conversation_id), "reason": reason}
        event = dict(payload)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as exc:  # pragma: no cover - defensive logging only
                if hasattr(self._logger, "debug"):
                    self._logger.debug(
                        "Conversation listener %s failed: %s", listener, exc, exc_info=True
                    )

    # ------------------------------------------------------------------
    # Tenant helpers
    # ------------------------------------------------------------------
    def _conversation_tenant(self) -> str:
        return self._tenant_id or "default"

    # ------------------------------------------------------------------
    # Repository helpers
    # ------------------------------------------------------------------
    def get_recent_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent conversations for the active tenant."""

        repository = self._repository
        if repository is None:
            return []

        try:
            window = max(int(limit), 1)
        except (TypeError, ValueError):
            window = 20

        try:
            return repository.list_conversations_for_tenant(
                self._conversation_tenant(),
                limit=window,
                order="desc",
            )
        except Exception as exc:  # pragma: no cover - database errors logged
            if hasattr(self._logger, "error"):
                self._logger.error(
                    "Failed to load recent conversations: %s", exc, exc_info=True
                )
            return []

    def list_all_conversations(self, *, order: str = "desc") -> List[Dict[str, Any]]:
        """Return all conversations for the tenant in the requested order."""

        repository = self._repository
        if repository is None:
            return []

        try:
            return repository.list_conversations_for_tenant(
                self._conversation_tenant(),
                order=order,
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            if hasattr(self._logger, "error"):
                self._logger.error(
                    "Failed to enumerate conversations: %s", exc, exc_info=True
                )
            return []

    def get_conversation_messages(
        self,
        conversation_id: Any,
        *,
        limit: Optional[int] = None,
        include_deleted: bool = True,
        batch_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return messages for ``conversation_id`` using the conversation store."""

        repository = self._repository
        if repository is None:
            return []

        if limit is None:
            remaining: Optional[int] = None
        else:
            try:
                remaining = max(int(limit), 0)
            except (TypeError, ValueError):
                remaining = None

        try:
            chunk_size = max(int(batch_size), 1)
        except (TypeError, ValueError):
            chunk_size = 200

        try:
            stream: Iterable[Dict[str, Any]] = repository.stream_conversation_messages(
                conversation_id,
                tenant_id=self._conversation_tenant(),
                batch_size=chunk_size,
                direction="forward",
                include_deleted=include_deleted,
            )

            messages: List[Dict[str, Any]] = []
            if remaining is None:
                messages.extend(stream)
            else:
                for message in stream:
                    if remaining <= 0:
                        break
                    messages.append(message)
                    remaining -= 1
            return messages
        except Exception as exc:  # pragma: no cover - persistence failures logged
            if hasattr(self._logger, "error"):
                self._logger.error(
                    "Failed to load messages for conversation %s: %s",
                    conversation_id,
                    exc,
                    exc_info=True,
                )
            return []

    def reset_conversation_messages(self, conversation_id: Any) -> Dict[str, Any]:
        """Remove stored messages for ``conversation_id`` while preserving the record."""

        repository = self._repository
        if repository is None:
            return {"success": False, "error": "Conversation store unavailable."}

        try:
            repository.reset_conversation(
                conversation_id,
                tenant_id=self._conversation_tenant(),
            )
        except Exception as exc:  # pragma: no cover - propagate details to UI payload
            if hasattr(self._logger, "error"):
                self._logger.error(
                    "Failed to reset conversation %s: %s",
                    conversation_id,
                    exc,
                    exc_info=True,
                )
            return {"success": False, "error": str(exc) or "Unable to reset conversation."}

        self.notify_updated(conversation_id, reason="reset")
        return {"success": True}

    def delete_conversation(self, conversation_id: Any) -> Dict[str, Any]:
        """Permanently delete ``conversation_id`` and associated assets."""

        repository = self._repository
        if repository is None:
            return {"success": False, "error": "Conversation store unavailable."}

        try:
            repository.hard_delete_conversation(
                conversation_id,
                tenant_id=self._conversation_tenant(),
            )
        except Exception as exc:  # pragma: no cover - persistence failure reported
            if hasattr(self._logger, "error"):
                self._logger.error(
                    "Failed to delete conversation %s: %s",
                    conversation_id,
                    exc,
                    exc_info=True,
                )
            return {"success": False, "error": str(exc) or "Unable to delete conversation."}

        self.notify_updated(conversation_id, reason="deleted")
        return {"success": True}

    # ------------------------------------------------------------------
    # Chat session helpers
    # ------------------------------------------------------------------
    def _require_chat_session(self) -> "ChatSession":
        if self._chat_session_getter is None:
            raise RuntimeError("Chat session is not initialized.")

        session = self._chat_session_getter()
        if session is None:
            raise RuntimeError("Chat session is not initialized.")
        return session

    def get_chat_history_snapshot(self) -> List[Dict[str, Any]]:
        """Return a safe copy of the current conversation history."""

        try:
            session = self._require_chat_session()
        except RuntimeError:
            return []

        getter = getattr(session, "get_history", None)
        if not callable(getter):
            return []

        try:
            history = getter()
        except Exception as exc:  # pragma: no cover - defensive logging only
            if hasattr(self._logger, "error"):
                self._logger.error(
                    "Failed to retrieve chat history snapshot: %s",
                    exc,
                    exc_info=True,
                )
            return []

        if isinstance(history, list):
            return list(history)

        return []

    def get_negotiation_log(self) -> List[Dict[str, Any]]:
        """Expose recorded negotiation traces to the UI layer."""

        try:
            session = self._require_chat_session()
        except RuntimeError:
            return []

        getter = getattr(session, "get_negotiation_history", None)
        if not callable(getter):
            return []

        try:
            history = getter()
        except Exception as exc:  # pragma: no cover - defensive logging only
            if hasattr(self._logger, "error"):
                self._logger.error(
                    "Failed to retrieve negotiation history: %s",
                    exc,
                    exc_info=True,
                )
            return []

        if isinstance(history, list):
            return [dict(entry) for entry in history]

        return []
