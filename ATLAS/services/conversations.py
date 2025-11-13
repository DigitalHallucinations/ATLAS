"""Conversation service helpers for the ATLAS application."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from modules.Server.conversation_routes import ConversationAuthorizationError

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
        server_getter: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._repository = repository
        self._logger = logger
        self._tenant_id = str(tenant_id).strip() or "default"
        self._chat_session_getter = chat_session_getter
        self._server_getter = server_getter
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
        page_size: Optional[int] = None,
        include_deleted: bool = True,
        batch_size: int = 200,
        cursor: Optional[str] = None,
        direction: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        message_types: Optional[Iterable[str]] = None,
        statuses: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Return messages for ``conversation_id`` using the best available pathway."""

        tenant_id = self._conversation_tenant()

        normalized_direction = str(direction or "forward").lower()
        if normalized_direction not in {"forward", "backward"}:
            normalized_direction = "forward"

        window: Optional[int]
        raw_page_size = page_size if page_size is not None else limit
        if raw_page_size is None:
            window = None
        else:
            try:
                window = max(int(raw_page_size), 1)
            except (TypeError, ValueError):
                window = None

        metadata_filter: Optional[Dict[str, Any]]
        if isinstance(metadata, Mapping):
            metadata_filter = {
                str(key): value for key, value in metadata.items() if str(key).strip()
            } or None
        else:
            metadata_filter = None

        type_filter: List[str] = []
        if isinstance(message_types, Iterable) and not isinstance(message_types, (str, bytes)):
            seen_types: Dict[str, None] = {}
            for value in message_types:
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                key = text
                if key not in seen_types:
                    seen_types[key] = None
            type_filter = list(seen_types.keys())

        status_filter: List[str] = []
        if isinstance(statuses, Iterable) and not isinstance(statuses, (str, bytes)):
            seen_statuses: Dict[str, None] = {}
            for value in statuses:
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                key = text
                if key not in seen_statuses:
                    seen_statuses[key] = None
            status_filter = list(seen_statuses.keys())

        use_server = bool(
            cursor
            or metadata_filter
            or type_filter
            or status_filter
            or window is not None
            or direction is not None
            or include_deleted is False
        )

        server = self._resolve_server()
        if use_server and server is not None:
            list_method = getattr(server, "list_messages", None)
            if callable(list_method):
                params: Dict[str, Any] = {"include_deleted": bool(include_deleted)}
                if window is not None:
                    params["page_size"] = window
                if cursor:
                    params["cursor"] = cursor
                if direction is not None or normalized_direction != "forward":
                    params["direction"] = normalized_direction
                if metadata_filter:
                    params["metadata"] = metadata_filter
                if type_filter:
                    params["message_types"] = type_filter
                if status_filter:
                    params["statuses"] = status_filter

                try:
                    response = list_method(
                        str(conversation_id),
                        params,
                        context={"tenant_id": tenant_id},
                    )
                except ConversationAuthorizationError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive logging only
                    if hasattr(self._logger, "warning"):
                        self._logger.warning(
                            "Server list_messages failed for conversation %s: %s",
                            conversation_id,
                            exc,
                            exc_info=True,
                        )
                else:
                    if isinstance(response, Mapping):
                        items = list(response.get("items") or [])
                        page_info = response.get("page")
                        if not isinstance(page_info, Mapping):
                            page_info = {}
                        return {
                            "items": items,
                            "page": {
                                "size": page_info.get("size", window or len(items)),
                                "direction": str(
                                    page_info.get("direction") or normalized_direction
                                ),
                                "next_cursor": page_info.get("next_cursor"),
                                "previous_cursor": page_info.get("previous_cursor"),
                            },
                        }

        repository = self._repository
        if repository is None:
            return {
                "items": [],
                "page": {
                    "size": 0,
                    "direction": normalized_direction,
                    "next_cursor": None,
                    "previous_cursor": None,
                },
            }

        try:
            chunk_size = max(int(batch_size), 1)
        except (TypeError, ValueError):
            chunk_size = 200

        try:
            stream: Iterable[Dict[str, Any]] = repository.stream_conversation_messages(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                batch_size=chunk_size,
                include_deleted=include_deleted,
            )
        except Exception as exc:  # pragma: no cover - persistence failures logged
            if hasattr(self._logger, "error"):
                self._logger.error(
                    "Failed to load messages for conversation %s: %s",
                    conversation_id,
                    exc,
                    exc_info=True,
                )
            return {
                "items": [],
                "page": {
                    "size": 0,
                    "direction": normalized_direction,
                    "next_cursor": None,
                    "previous_cursor": None,
                },
            }

        messages = list(stream)

        if metadata_filter:
            messages = [
                message
                for message in messages
                if _metadata_matches(message.get("metadata"), metadata_filter)
            ]

        if type_filter:
            messages = [
                message
                for message in messages
                if str(message.get("message_type") or "").strip() in type_filter
            ]

        if status_filter:
            messages = [
                message
                for message in messages
                if str(message.get("status") or "").strip() in status_filter
            ]

        if normalized_direction == "forward":
            messages.sort(key=_message_sort_key)
        else:
            messages.sort(key=_message_sort_key, reverse=True)

        if window is not None:
            messages = messages[:window]

        return {
            "items": messages,
            "page": {
                "size": window or len(messages),
                "direction": normalized_direction,
                "next_cursor": None,
                "previous_cursor": None,
            },
        }

    def _resolve_server(self) -> Any | None:
        if self._server_getter is None:
            return None
        try:
            return self._server_getter()
        except Exception as exc:  # pragma: no cover - defensive logging only
            if hasattr(self._logger, "debug"):
                self._logger.debug("Server getter failed: %s", exc, exc_info=True)
            return None

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

    def search_conversations(
        self,
        *,
        text: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        vector: Optional[Mapping[str, Any]] = None,
        conversation_ids: Optional[Iterable[Any]] = None,
        limit: int = 20,
        offset: int = 0,
        order: str = "desc",
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search conversation messages using text and/or vector filters."""

        repository = self._repository
        if repository is None:
            return {"count": 0, "items": []}

        try:
            limit_value = max(int(limit), 1)
        except (TypeError, ValueError):
            limit_value = 20

        try:
            offset_value = max(int(offset), 0)
        except (TypeError, ValueError):
            offset_value = 0

        order_value = str(order or "desc").lower()
        order_value = "asc" if order_value == "asc" else "desc"

        text_query = str(text or "").strip()
        metadata_filter = dict(metadata or {}) if isinstance(metadata, Mapping) else None

        vector_values: List[float] = []
        if isinstance(vector, Mapping):
            raw_values = vector.get("values")
            if isinstance(raw_values, Sequence):
                for component in raw_values:
                    try:
                        value = float(component)
                    except (TypeError, ValueError):
                        continue
                    vector_values.append(value)

        conversation_scope: List[Any]
        if conversation_ids:
            conversation_scope = [identifier for identifier in conversation_ids if identifier]
        else:
            try:
                records = repository.list_conversations_for_tenant(
                    self._conversation_tenant(),
                    order="desc",
                )
            except Exception as exc:  # pragma: no cover - logged for diagnostics
                if hasattr(self._logger, "error"):
                    self._logger.error(
                        "Failed to enumerate conversations for search: %s",
                        exc,
                        exc_info=True,
                    )
                records = []
            conversation_scope = [record.get("id") for record in records if record.get("id")]

        if not conversation_scope:
            return {"count": 0, "items": []}

        query_limit = max(offset_value + limit_value, 1)
        top_k_value: Optional[int]
        if top_k is not None:
            try:
                top_k_value = max(int(top_k), 1)
            except (TypeError, ValueError):
                top_k_value = None
        else:
            top_k_value = None
        if top_k_value is None:
            top_k_value = query_limit

        results: Dict[str, Dict[str, Any]] = {}

        if text_query or not vector_values:
            text_kwargs = {
                "conversation_ids": conversation_scope,
                "tenant_id": self._conversation_tenant(),
                "text": text_query,
                "metadata_filter": metadata_filter or None,
                "include_deleted": False,
                "order": order_value,
                "limit": query_limit,
            }
            try:
                iterator = repository.query_messages_by_text(**text_kwargs)
            except TypeError:
                text_kwargs.pop("text", None)
                text_kwargs["query_text"] = text_query
                iterator = repository.query_messages_by_text(**text_kwargs)
            try:
                for message in iterator:
                    identifier = str(message.get("id") or "")
                    if not identifier:
                        continue
                    record = results.setdefault(
                        identifier,
                        {
                            "conversation_id": message.get("conversation_id"),
                            "message": message,
                            "score": 0.0,
                        },
                    )
                    record["message"] = message
                    if text_query:
                        record["score"] = max(float(record["score"]), 1.0)
            except Exception as exc:  # pragma: no cover - repository iteration errors
                if hasattr(self._logger, "error"):
                    self._logger.error(
                        "Text conversation search failed: %s",
                        exc,
                        exc_info=True,
                    )

        if vector_values:
            vector_kwargs = {
                "conversation_ids": conversation_scope,
                "tenant_id": self._conversation_tenant(),
                "metadata_filter": metadata_filter or None,
                "include_deleted": False,
                "order": order_value,
                "offset": offset_value,
                "limit": query_limit,
                "top_k": top_k_value,
            }
            try:
                for message, vector_row in repository.query_message_vectors(**vector_kwargs):
                    identifier = str(message.get("id") or "")
                    if not identifier:
                        continue
                    embedding = vector_row.get("embedding") or []
                    try:
                        embedded_values = [float(component) for component in embedding]
                    except (TypeError, ValueError):
                        continue
                    similarity = self._cosine_similarity(vector_values, embedded_values)
                    if similarity <= 0:
                        continue
                    record = results.setdefault(
                        identifier,
                        {
                            "conversation_id": message.get("conversation_id"),
                            "message": message,
                            "score": 0.0,
                        },
                    )
                    record["message"] = message
                    record["score"] = max(float(record["score"]), float(similarity))
            except Exception as exc:  # pragma: no cover - repository iteration errors
                if hasattr(self._logger, "error"):
                    self._logger.error(
                        "Vector conversation search failed: %s",
                        exc,
                        exc_info=True,
                    )

        def _sort_key(entry: Mapping[str, Any]) -> tuple[float, float]:
            message = entry.get("message") or {}
            raw_created = message.get("created_at") or message.get("timestamp")
            created_ts = self._coerce_timestamp(raw_created)
            timestamp_key = created_ts if order_value == "asc" else -created_ts
            return (-float(entry.get("score", 0.0)), timestamp_key)

        filtered = [
            item
            for item in results.values()
            if item.get("score", 0.0) > 0 or (not text_query and not vector_values)
        ]

        ordered = sorted(filtered, key=_sort_key)
        windowed = ordered[offset_value : offset_value + limit_value]
        return {"count": len(windowed), "items": windowed}

    # ------------------------------------------------------------------
    # Retention helpers
    # ------------------------------------------------------------------
    def assess_retention_availability(
        self,
        *,
        runner: Optional[Callable[..., Mapping[str, Any]]],
        context: Optional[Mapping[str, Any]] = None,
    ) -> tuple[bool, Optional[str], Dict[str, Any]]:
        """Return whether retention can run along with a normalised context."""

        normalized_context, normalized_roles = self._normalize_retention_context(context)

        if not callable(runner):
            return (
                False,
                "Conversation retention endpoint is not available.",
                normalized_context,
            )

        if not normalized_roles.intersection({"admin", "system"}):
            return (
                False,
                "Administrative role required to trigger retention policies.",
                normalized_context,
            )

        return True, None, normalized_context

    def run_conversation_retention(
        self,
        *,
        runner: Optional[Callable[..., Mapping[str, Any]]],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke the server-backed retention route when permitted."""

        available, reason, normalized_context = self.assess_retention_availability(
            runner=runner,
            context=context,
        )

        if not available:
            return {"success": False, "error": reason or "Retention is unavailable."}

        try:
            result = runner(context=normalized_context)  # type: ignore[misc]
        except ConversationAuthorizationError as exc:
            message = str(exc).strip() or "Administrative role required to trigger retention policies."
            return {"success": False, "error": message}
        except Exception as exc:  # pragma: no cover - defensive logging only
            if hasattr(self._logger, "error"):
                self._logger.error(
                    "Conversation retention execution failed: %s",
                    exc,
                    exc_info=True,
                )
            return {"success": False, "error": "Conversation retention failed."}

        if isinstance(result, Mapping):
            return {"success": True, "stats": dict(result)}

        return {"success": True, "result": result}

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        try:
            dot = sum(a * b for a, b in zip(left, right))
            left_norm = math.sqrt(sum(a * a for a in left))
            right_norm = math.sqrt(sum(b * b for b in right))
        except Exception:
            return 0.0
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _coerce_timestamp(value: Any) -> float:
        if not value:
            return 0.0
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return 0.0
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()

    def _normalize_retention_context(
        self,
        context: Optional[Mapping[str, Any]],
    ) -> tuple[Dict[str, Any], set[str]]:
        payload: Dict[str, Any] = {"tenant_id": self._conversation_tenant()}
        roles_value: Any = ()

        if isinstance(context, Mapping):
            for key, value in context.items():
                if value is None:
                    continue
                if key == "roles":
                    roles_value = value
                    continue
                if key == "tenant_id":
                    token = str(value).strip()
                    if token:
                        payload["tenant_id"] = token
                    continue
                payload[key] = value

        roles_list = self._sanitize_roles(roles_value)
        if roles_list:
            payload["roles"] = tuple(roles_list)

        normalized_roles = {role.lower() for role in roles_list}
        return payload, normalized_roles

    @staticmethod
    def _sanitize_roles(raw_value: Any) -> List[str]:
        if raw_value is None:
            return []

        if isinstance(raw_value, str):
            candidates: Iterable[Any] = raw_value.replace(";", ",").split(",")
        elif isinstance(raw_value, Mapping):
            candidates = raw_value.values()
        elif isinstance(raw_value, Iterable):
            candidates = raw_value
        else:
            return []

        roles: List[str] = []
        for candidate in candidates:
            text = str(candidate).strip()
            if not text:
                continue
            if text not in roles:
                roles.append(text)
        return roles

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


def _metadata_matches(
    candidate: Any,
    expected: Mapping[str, Any],
) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    for key, value in expected.items():
        if candidate.get(key) != value:
            return False
    return True


def _message_sort_key(message: Mapping[str, Any]) -> Any:
    timestamp = message.get("created_at") or message.get("timestamp")
    if not timestamp:
        return (datetime.min.replace(tzinfo=timezone.utc), str(message.get("id")))
    text = str(timestamp)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return (datetime.min.replace(tzinfo=timezone.utc), str(message.get("id")))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return (parsed, str(message.get("id")))
