"""Async tool wrapper for episodic memory persistence in the conversation store."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from modules.conversation_store import ConversationStoreRepository
from modules.logging.logger import setup_logger

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from ATLAS.config import ConfigManager

__all__ = [
    "EpisodicMemoryTool",
    "store_episode",
    "query_episodes",
    "prune_episodes",
]


logger = setup_logger(__name__)


class EpisodicMemoryTool:
    """Expose episodic memory CRUD helpers through an async interface."""

    def __init__(
        self,
        *,
        config_manager: Optional["ConfigManager"] = None,
        repository: Optional[ConversationStoreRepository] = None,
    ) -> None:
        self._config_manager = config_manager
        self._repository = repository
        self._config_lock = threading.Lock()
        self._repository_lock = threading.Lock()

    def _get_config_manager(self) -> "ConfigManager":
        if self._config_manager is not None:
            return self._config_manager
        with self._config_lock:
            if self._config_manager is not None:
                return self._config_manager
            try:
                from ATLAS.config import ConfigManager as _ConfigManager
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.error("Failed to import ConfigManager for episodic memory: %s", exc)
                raise RuntimeError(
                    "ConfigManager is required to resolve the conversation store repository for episodic memory."
                ) from exc
            self._config_manager = _ConfigManager()
            return self._config_manager

    def _resolve_repository(self) -> ConversationStoreRepository:
        if self._repository is not None:
            return self._repository
        config_manager = self._get_config_manager()
        factory = config_manager.get_conversation_store_session_factory()
        if factory is None:
            raise RuntimeError("Conversation store session factory is not configured.")
        retention = config_manager.get_conversation_retention_policies()
        repository = ConversationStoreRepository(factory, retention=retention)
        with self._repository_lock:
            if self._repository is None:
                self._repository = repository
                try:
                    self._repository.create_schema()
                except Exception as exc:  # pragma: no cover - defensive logging only
                    logger.debug("Episodic memory schema initialization skipped: %s", exc)
        return self._repository

    async def store(
        self,
        *,
        tenant_id: Any,
        content: Any,
        tags: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        occurred_at: Any | None = None,
        expires_at: Any | None = None,
        conversation_id: Any | None = None,
        message_id: Any | None = None,
        user_id: Any | None = None,
        title: Optional[str] = None,
    ) -> Mapping[str, Any]:
        repository = self._resolve_repository()
        return await asyncio.to_thread(
            repository.append_episodic_memory,
            tenant_id=tenant_id,
            content=content,
            tags=tags,
            metadata=metadata,
            occurred_at=occurred_at,
            expires_at=expires_at,
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=user_id,
            title=title,
        )

    async def query(
        self,
        *,
        tenant_id: Any,
        tags_all: Optional[Sequence[Any]] = None,
        tags_any: Optional[Sequence[Any]] = None,
        from_time: Any | None = None,
        to_time: Any | None = None,
        limit: Optional[int] = None,
        offset: int = 0,
        conversation_id: Any | None = None,
        include_expired: bool = False,
        order: str = "desc",
    ) -> Mapping[str, Any]:
        repository = self._resolve_repository()
        episodes = await asyncio.to_thread(
            repository.query_episodic_memories,
            tenant_id=tenant_id,
            tags_all=tags_all,
            tags_any=tags_any,
            from_time=from_time,
            to_time=to_time,
            limit=limit,
            offset=offset,
            conversation_id=conversation_id,
            include_expired=include_expired,
            order=order,
        )
        return {"episodes": episodes, "count": len(episodes)}

    async def prune(
        self,
        *,
        tenant_id: Any,
        before: Any | None = None,
        expired_only: bool = False,
        limit: Optional[int] = None,
        conversation_id: Any | None = None,
    ) -> Mapping[str, Any]:
        repository = self._resolve_repository()
        removed = await asyncio.to_thread(
            repository.prune_episodic_memories,
            tenant_id=tenant_id,
            before=before,
            expired_only=expired_only,
            limit=limit,
            conversation_id=conversation_id,
        )
        return {"deleted": removed}


async def store_episode(
    *,
    tenant_id: Any,
    content: Any,
    tags: Optional[Sequence[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    occurred_at: Any | None = None,
    expires_at: Any | None = None,
    conversation_id: Any | None = None,
    message_id: Any | None = None,
    user_id: Any | None = None,
    title: Optional[str] = None,
    config_manager: Optional["ConfigManager"] = None,
) -> Mapping[str, Any]:
    tool = EpisodicMemoryTool(config_manager=config_manager)
    return await tool.store(
        tenant_id=tenant_id,
        content=content,
        tags=tags,
        metadata=metadata,
        occurred_at=occurred_at,
        expires_at=expires_at,
        conversation_id=conversation_id,
        message_id=message_id,
        user_id=user_id,
        title=title,
    )


async def query_episodes(
    *,
    tenant_id: Any,
    tags_all: Optional[Sequence[Any]] = None,
    tags_any: Optional[Sequence[Any]] = None,
    from_time: Any | None = None,
    to_time: Any | None = None,
    limit: Optional[int] = None,
    offset: int = 0,
    conversation_id: Any | None = None,
    include_expired: bool = False,
    order: str = "desc",
    config_manager: Optional["ConfigManager"] = None,
) -> Mapping[str, Any]:
    tool = EpisodicMemoryTool(config_manager=config_manager)
    return await tool.query(
        tenant_id=tenant_id,
        tags_all=tags_all,
        tags_any=tags_any,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        offset=offset,
        conversation_id=conversation_id,
        include_expired=include_expired,
        order=order,
    )


async def prune_episodes(
    *,
    tenant_id: Any,
    before: Any | None = None,
    expired_only: bool = False,
    limit: Optional[int] = None,
    conversation_id: Any | None = None,
    config_manager: Optional["ConfigManager"] = None,
) -> Mapping[str, Any]:
    tool = EpisodicMemoryTool(config_manager=config_manager)
    return await tool.prune(
        tenant_id=tenant_id,
        before=before,
        expired_only=expired_only,
        limit=limit,
        conversation_id=conversation_id,
    )
