"""Graph helpers for the conversation store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, List, Mapping, Optional, Sequence, Set

from ._compat import Session, and_, delete, or_, select
from ._shared import (
    _dt_to_iso,
    _normalize_edge_key,
    _normalize_edge_type,
    _normalize_json_like,
    _normalize_node_key,
    _normalize_tenant_id,
)
from .models import GraphEdge, GraphNode


@dataclass
class _GraphNodePayload:
    key: str
    label: Optional[str]
    node_type: Optional[str]
    metadata: Optional[Dict[str, Any]]
    metadata_supplied: bool
    label_supplied: bool
    node_type_supplied: bool


@dataclass
class _GraphEdgePayload:
    edge_key: Optional[str]
    source_key: str
    target_key: str
    edge_type: Optional[str]
    weight: Optional[float]
    metadata: Optional[Dict[str, Any]]
    metadata_supplied: bool
    edge_type_supplied: bool
    weight_supplied: bool
    edge_key_supplied: bool


class GraphStore:
    """CRUD helpers for the episodic memory graph."""

    def __init__(self, session_scope: Callable[[], ContextManager[Session]]) -> None:
        self._session_scope = session_scope

    # ------------------------------------------------------------------
    # Normalisation helpers

    @staticmethod
    def _normalize_graph_node_payload(node: Mapping[str, Any]) -> _GraphNodePayload:
        if not isinstance(node, Mapping):
            raise TypeError("Graph node definitions must be mappings")

        raw_key = (
            node.get("key")
            or node.get("node_key")
            or node.get("node_id")
            or node.get("id")
        )
        key = _normalize_node_key(raw_key)

        label_supplied = "label" in node
        label = node.get("label")
        if label is not None:
            label = str(label).strip() or None

        node_type_supplied = "node_type" in node or "type" in node
        node_type = node.get("type") or node.get("node_type")
        if node_type is not None:
            node_type = str(node_type).strip() or None

        metadata_supplied = "metadata" in node
        metadata_payload: Optional[Dict[str, Any]]
        if metadata_supplied:
            metadata_payload = dict(_normalize_json_like(node.get("metadata") or {}))
        else:
            metadata_payload = None

        return _GraphNodePayload(
            key=key,
            label=label,
            node_type=node_type,
            metadata=metadata_payload,
            metadata_supplied=metadata_supplied,
            label_supplied=label_supplied,
            node_type_supplied=node_type_supplied,
        )

    @staticmethod
    def _normalize_graph_edge_payload(edge: Mapping[str, Any]) -> _GraphEdgePayload:
        if not isinstance(edge, Mapping):
            raise TypeError("Graph edge definitions must be mappings")

        source = (
            edge.get("source")
            or edge.get("from")
            or edge.get("source_key")
            or edge.get("source_id")
        )
        target = (
            edge.get("target")
            or edge.get("to")
            or edge.get("target_key")
            or edge.get("target_id")
        )
        source_key = _normalize_node_key(source)
        target_key = _normalize_node_key(target)

        edge_key_supplied = "edge_key" in edge or "key" in edge
        raw_edge_key = edge.get("edge_key") or edge.get("key")
        edge_key = _normalize_edge_key(raw_edge_key)

        edge_type_supplied = "edge_type" in edge or "type" in edge
        raw_type = edge.get("edge_type") or edge.get("type")
        edge_type = _normalize_edge_type(raw_type)

        weight_supplied = "weight" in edge
        weight_value = edge.get("weight")
        weight = float(weight_value) if weight_value is not None else None

        metadata_supplied = "metadata" in edge
        metadata_payload: Optional[Dict[str, Any]]
        if metadata_supplied:
            metadata_payload = dict(_normalize_json_like(edge.get("metadata") or {}))
        else:
            metadata_payload = None

        return _GraphEdgePayload(
            edge_key=edge_key,
            source_key=source_key,
            target_key=target_key,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata_payload,
            metadata_supplied=metadata_supplied,
            edge_type_supplied=edge_type_supplied,
            weight_supplied=weight_supplied,
            edge_key_supplied=edge_key_supplied,
        )

    # ------------------------------------------------------------------
    # Serialisation helpers

    @staticmethod
    def _serialize_node(node: GraphNode) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": str(node.id),
            "tenant_id": node.tenant_id,
            "key": node.node_key,
            "label": node.label,
            "type": node.node_type,
            "metadata": dict(node.meta or {}),
            "created_at": _dt_to_iso(node.created_at),
            "updated_at": _dt_to_iso(node.updated_at),
        }
        return payload

    @staticmethod
    def _serialize_edge(
        edge: GraphEdge, node_index: Optional[Mapping[uuid.UUID, GraphNode]] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": str(edge.id),
            "tenant_id": edge.tenant_id,
            "edge_key": edge.edge_key,
            "source_id": str(edge.source_id),
            "target_id": str(edge.target_id),
            "edge_type": edge.edge_type,
            "weight": float(edge.weight) if edge.weight is not None else None,
            "metadata": dict(edge.meta or {}),
            "created_at": _dt_to_iso(edge.created_at),
            "updated_at": _dt_to_iso(edge.updated_at),
        }
        if node_index is not None:
            source_node = node_index.get(edge.source_id)
            target_node = node_index.get(edge.target_id)
            if source_node is not None:
                payload.setdefault("source_key", source_node.node_key)
            if target_node is not None:
                payload.setdefault("target_key", target_node.node_key)
        return payload

    # ------------------------------------------------------------------
    # CRUD operations

    def upsert_graph_nodes(
        self,
        *,
        tenant_id: Any,
        nodes: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not nodes:
            return []

        tenant_key = _normalize_tenant_id(tenant_id)
        payloads = [self._normalize_graph_node_payload(node) for node in nodes]

        with self._session_scope() as session:
            stored: List[Dict[str, Any]] = []
            for payload in payloads:
                record = session.execute(
                    select(GraphNode)
                    .where(GraphNode.tenant_id == tenant_key)
                    .where(GraphNode.node_key == payload.key)
                ).scalar_one_or_none()

                if record is None:
                    record = GraphNode(
                        tenant_id=tenant_key,
                        node_key=payload.key,
                        label=payload.label,
                        node_type=payload.node_type,
                        meta=payload.metadata or {},
                    )
                    session.add(record)
                    session.flush()
                else:
                    updated = False
                    if payload.label_supplied and record.label != payload.label:
                        record.label = payload.label
                        updated = True
                    if payload.node_type_supplied and record.node_type != payload.node_type:
                        record.node_type = payload.node_type
                        updated = True
                    if payload.metadata_supplied:
                        record.meta = payload.metadata or {}
                        updated = True
                    if updated:
                        session.flush()

                stored.append(self._serialize_node(record))
            return stored

    def upsert_graph_edges(
        self,
        *,
        tenant_id: Any,
        edges: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not edges:
            return []

        tenant_key = _normalize_tenant_id(tenant_id)
        payloads = [self._normalize_graph_edge_payload(edge) for edge in edges]

        required_keys: Set[str] = set()
        for payload in payloads:
            required_keys.add(payload.source_key)
            required_keys.add(payload.target_key)

        with self._session_scope() as session:
            rows = session.execute(
                select(GraphNode)
                .where(GraphNode.tenant_id == tenant_key)
                .where(GraphNode.node_key.in_(sorted(required_keys)))
            ).scalars().all()
            nodes_by_key = {record.node_key: record for record in rows}
            missing = sorted(required_keys - set(nodes_by_key))
            if missing:
                raise ValueError("Unknown graph node keys for edge upsert: " + ", ".join(missing))

            stored_edges: List[Dict[str, Any]] = []
            for payload in payloads:
                source_node = nodes_by_key[payload.source_key]
                target_node = nodes_by_key[payload.target_key]

                record = session.execute(
                    select(GraphEdge)
                    .where(GraphEdge.tenant_id == tenant_key)
                    .where(GraphEdge.source_id == source_node.id)
                    .where(GraphEdge.target_id == target_node.id)
                ).scalar_one_or_none()

                if record is None:
                    record = GraphEdge(
                        tenant_id=tenant_key,
                        edge_key=payload.edge_key if payload.edge_key_supplied else None,
                        source=source_node,
                        target=target_node,
                        edge_type=payload.edge_type if payload.edge_type_supplied else None,
                        weight=payload.weight if payload.weight_supplied else None,
                        meta=payload.metadata or {},
                    )
                    session.add(record)
                    session.flush()
                else:
                    if payload.edge_key_supplied and record.edge_key != payload.edge_key:
                        record.edge_key = payload.edge_key
                    if record.source_id != source_node.id:
                        record.source = source_node
                    if record.target_id != target_node.id:
                        record.target = target_node
                    if payload.edge_type_supplied and record.edge_type != payload.edge_type:
                        record.edge_type = payload.edge_type
                    if payload.weight_supplied and record.weight != payload.weight:
                        record.weight = payload.weight
                    if payload.metadata_supplied:
                        record.meta = payload.metadata or {}
                    session.flush()

                stored_edges.append(
                    self._serialize_edge(
                        record,
                        node_index={
                            source_node.id: source_node,
                            target_node.id: target_node,
                        },
                    )
                )

            return stored_edges

    def query_graph(
        self,
        *,
        tenant_id: Any,
        node_keys: Optional[Sequence[Any]] = None,
        edge_types: Optional[Sequence[Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        tenant_key = _normalize_tenant_id(tenant_id)
        requested_keys = {
            _normalize_node_key(item) for item in (node_keys or []) if item is not None
        }
        edge_type_filters = {
            value
            for value in (
                _normalize_edge_type(item)
                for item in (edge_types or [])
            )
            if value is not None
        }

        with self._session_scope() as session:
            node_stmt = select(GraphNode).where(GraphNode.tenant_id == tenant_key)
            if requested_keys:
                node_stmt = node_stmt.where(GraphNode.node_key.in_(sorted(requested_keys)))
            node_stmt = node_stmt.order_by(GraphNode.node_key.asc())
            seed_nodes = session.execute(node_stmt).scalars().all()

            nodes_by_id: Dict[uuid.UUID, GraphNode] = {record.id: record for record in seed_nodes}

            edges: List[GraphEdge]
            if requested_keys and not nodes_by_id:
                edges = []
            else:
                edge_stmt = select(GraphEdge).where(GraphEdge.tenant_id == tenant_key)
                if requested_keys:
                    node_ids = list(nodes_by_id.keys())
                    if node_ids:
                        edge_stmt = edge_stmt.where(
                            or_(
                                GraphEdge.source_id.in_(node_ids),
                                GraphEdge.target_id.in_(node_ids),
                            )
                        )
                if edge_type_filters:
                    edge_stmt = edge_stmt.where(
                        GraphEdge.edge_type.in_(sorted(edge_type_filters))
                    )
                edge_stmt = edge_stmt.order_by(
                    GraphEdge.created_at.asc(), GraphEdge.id.asc()
                )
                edges = session.execute(edge_stmt).scalars().all()

            missing_ids: Set[uuid.UUID] = set()
            for edge in edges:
                if edge.source_id not in nodes_by_id:
                    missing_ids.add(edge.source_id)
                if edge.target_id not in nodes_by_id:
                    missing_ids.add(edge.target_id)

            if missing_ids:
                extra_nodes = session.execute(
                    select(GraphNode)
                    .where(GraphNode.tenant_id == tenant_key)
                    .where(GraphNode.id.in_(list(missing_ids)))
                ).scalars().all()
                for record in extra_nodes:
                    nodes_by_id.setdefault(record.id, record)

            serialized_nodes = [
                self._serialize_node(record)
                for record in sorted(nodes_by_id.values(), key=lambda item: item.node_key)
            ]
            serialized_edges = [
                self._serialize_edge(edge, node_index=nodes_by_id) for edge in edges
            ]

            return {"nodes": serialized_nodes, "edges": serialized_edges}

    def delete_graph_entries(
        self,
        *,
        tenant_id: Any,
        node_keys: Optional[Sequence[Any]] = None,
        edge_keys: Optional[Sequence[Any]] = None,
        edge_ids: Optional[Sequence[Any]] = None,
    ) -> Dict[str, int]:
        tenant_key = _normalize_tenant_id(tenant_id)

        normalized_node_keys = {
            _normalize_node_key(item) for item in (node_keys or []) if item is not None
        }
        normalized_edge_keys = {
            _normalize_edge_key(item) for item in (edge_keys or []) if item is not None
        }
        normalized_edge_ids = [
            uuid.UUID(str(item)) for item in (edge_ids or []) if item is not None
        ]

        with self._session_scope() as session:
            stats = {"nodes": 0, "edges": 0}
            if normalized_node_keys:
                result = session.execute(
                    delete(GraphNode)
                    .where(GraphNode.tenant_id == tenant_key)
                    .where(GraphNode.node_key.in_(sorted(normalized_node_keys)))
                )
                stats["nodes"] += int(result.rowcount or 0)

            if normalized_edge_keys or normalized_edge_ids:
                stmt = delete(GraphEdge).where(GraphEdge.tenant_id == tenant_key)
                if normalized_edge_keys:
                    stmt = stmt.where(GraphEdge.edge_key.in_(sorted(normalized_edge_keys)))
                if normalized_edge_ids:
                    stmt = stmt.where(GraphEdge.id.in_(normalized_edge_ids))
                result = session.execute(stmt)
                stats["edges"] += int(result.rowcount or 0)

            return stats


__all__ = ["GraphStore"]
