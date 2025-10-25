"""Async helpers exposing tenant-scoped memory graph operations."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Mapping, Optional, Sequence

from modules.conversation_store import ConversationStoreRepository
from modules.logging.logger import setup_logger

try:  # pragma: no cover - ConfigManager may be unavailable in tests
    from ATLAS.config import ConfigManager
except Exception:  # pragma: no cover - fallback when ConfigManager import fails
    ConfigManager = None  # type: ignore

__all__ = [
    "MemoryGraphTool",
    "run_memory_graph",
    "upsert_graph_nodes",
    "upsert_graph_edges",
    "query_graph",
    "remove_graph_entries",
]

logger = setup_logger(__name__)


def _ensure_sequence(items: Optional[Sequence[Mapping[str, Any]]]) -> Sequence[Mapping[str, Any]]:
    if items is None:
        return []
    if isinstance(items, Sequence):
        return items
    raise TypeError("Expected a sequence of mapping values.")


class MemoryGraphTool:
    """Expose conversation store graph helpers through an async API."""

    def __init__(
        self,
        *,
        config_manager: Optional[ConfigManager] = None,
        repository: Optional[ConversationStoreRepository] = None,
    ) -> None:
        if repository is not None:
            self._repository = repository
        else:
            self._repository = None
        if repository is None:
            if config_manager is not None:
                self._config_manager = config_manager
            elif ConfigManager is not None:
                self._config_manager = ConfigManager()
            else:  # pragma: no cover - fallback when config unavailable
                self._config_manager = None
        else:
            self._config_manager = config_manager
        self._lock = threading.Lock()

    def _resolve_repository(self) -> ConversationStoreRepository:
        if self._repository is not None:
            return self._repository
        if self._config_manager is None:
            raise RuntimeError(
                "ConfigManager is required to resolve the memory graph conversation store repository."
            )
        factory = self._config_manager.get_conversation_store_session_factory()
        if factory is None:
            raise RuntimeError("Conversation store session factory is not configured.")
        retention = self._config_manager.get_conversation_retention_policies()
        repository = ConversationStoreRepository(factory, retention=retention)
        with self._lock:
            if self._repository is None:
                self._repository = repository
                try:
                    self._repository.create_schema()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Memory graph schema initialisation skipped: %s", exc)
        return self._repository

    async def upsert_nodes(
        self,
        *,
        tenant_id: Any,
        nodes: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        repository = self._resolve_repository()
        stored = await asyncio.to_thread(
            repository.upsert_graph_nodes,
            tenant_id=tenant_id,
            nodes=list(nodes),
        )
        return {"nodes": stored}

    async def upsert_edges(
        self,
        *,
        tenant_id: Any,
        edges: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        repository = self._resolve_repository()
        stored = await asyncio.to_thread(
            repository.upsert_graph_edges,
            tenant_id=tenant_id,
            edges=list(edges),
        )
        return {"edges": stored}

    async def query(
        self,
        *,
        tenant_id: Any,
        node_keys: Optional[Sequence[Any]] = None,
        edge_types: Optional[Sequence[Any]] = None,
    ) -> Mapping[str, Any]:
        repository = self._resolve_repository()
        return await asyncio.to_thread(
            repository.query_graph,
            tenant_id=tenant_id,
            node_keys=list(node_keys) if node_keys is not None else None,
            edge_types=list(edge_types) if edge_types is not None else None,
        )

    async def remove(
        self,
        *,
        tenant_id: Any,
        node_keys: Optional[Sequence[Any]] = None,
        edge_keys: Optional[Sequence[Any]] = None,
        edge_ids: Optional[Sequence[Any]] = None,
    ) -> Mapping[str, int]:
        repository = self._resolve_repository()
        return await asyncio.to_thread(
            repository.delete_graph_entries,
            tenant_id=tenant_id,
            node_keys=list(node_keys) if node_keys is not None else None,
            edge_keys=list(edge_keys) if edge_keys is not None else None,
            edge_ids=list(edge_ids) if edge_ids is not None else None,
        )

    async def run(
        self,
        *,
        operation: str,
        tenant_id: Any,
        nodes: Optional[Sequence[Mapping[str, Any]]] = None,
        edges: Optional[Sequence[Mapping[str, Any]]] = None,
        node_keys: Optional[Sequence[Any]] = None,
        edge_keys: Optional[Sequence[Any]] = None,
        edge_ids: Optional[Sequence[Any]] = None,
        edge_types: Optional[Sequence[Any]] = None,
    ) -> Mapping[str, Any]:
        action = str(operation or "").strip().lower()
        if not action:
            raise ValueError("Operation must be provided for memory.graph tool calls.")

        if action == "upsert_nodes":
            sequence = _ensure_sequence(nodes)
            if not sequence:
                raise ValueError("Nodes must be provided when using the upsert_nodes operation.")
            result = await self.upsert_nodes(tenant_id=tenant_id, nodes=sequence)
            return {"operation": action, **result}

        if action == "upsert_edges":
            sequence = _ensure_sequence(edges)
            if not sequence:
                raise ValueError("Edges must be provided when using the upsert_edges operation.")
            result = await self.upsert_edges(tenant_id=tenant_id, edges=sequence)
            return {"operation": action, **result}

        if action == "query":
            result = await self.query(
                tenant_id=tenant_id,
                node_keys=node_keys,
                edge_types=edge_types,
            )
            return {"operation": action, **result}

        if action == "remove":
            result = await self.remove(
                tenant_id=tenant_id,
                node_keys=node_keys,
                edge_keys=edge_keys,
                edge_ids=edge_ids,
            )
            return {"operation": action, **result}

        raise ValueError(f"Unsupported memory.graph operation '{operation}'.")


async def run_memory_graph(
    *,
    operation: str,
    tenant_id: Any,
    nodes: Optional[Sequence[Mapping[str, Any]]] = None,
    edges: Optional[Sequence[Mapping[str, Any]]] = None,
    node_keys: Optional[Sequence[Any]] = None,
    edge_keys: Optional[Sequence[Any]] = None,
    edge_ids: Optional[Sequence[Any]] = None,
    edge_types: Optional[Sequence[Any]] = None,
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    tool = MemoryGraphTool(config_manager=config_manager)
    return await tool.run(
        operation=operation,
        tenant_id=tenant_id,
        nodes=nodes,
        edges=edges,
        node_keys=node_keys,
        edge_keys=edge_keys,
        edge_ids=edge_ids,
        edge_types=edge_types,
    )


async def upsert_graph_nodes(
    *,
    tenant_id: Any,
    nodes: Sequence[Mapping[str, Any]],
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    tool = MemoryGraphTool(config_manager=config_manager)
    return await tool.upsert_nodes(tenant_id=tenant_id, nodes=nodes)


async def upsert_graph_edges(
    *,
    tenant_id: Any,
    edges: Sequence[Mapping[str, Any]],
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    tool = MemoryGraphTool(config_manager=config_manager)
    return await tool.upsert_edges(tenant_id=tenant_id, edges=edges)


async def query_graph(
    *,
    tenant_id: Any,
    node_keys: Optional[Sequence[Any]] = None,
    edge_types: Optional[Sequence[Any]] = None,
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    tool = MemoryGraphTool(config_manager=config_manager)
    return await tool.query(
        tenant_id=tenant_id,
        node_keys=node_keys,
        edge_types=edge_types,
    )


async def remove_graph_entries(
    *,
    tenant_id: Any,
    node_keys: Optional[Sequence[Any]] = None,
    edge_keys: Optional[Sequence[Any]] = None,
    edge_ids: Optional[Sequence[Any]] = None,
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, int]:
    tool = MemoryGraphTool(config_manager=config_manager)
    return await tool.remove(
        tenant_id=tenant_id,
        node_keys=node_keys,
        edge_keys=edge_keys,
        edge_ids=edge_ids,
    )
