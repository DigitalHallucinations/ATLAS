"""Human-in-the-loop approval workflow helper."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from ATLAS.config import ConfigManager
from modules.Tools.tool_event_system import publish_bus_event
from modules.task_store import TaskService, TaskStatus

__all__ = ["HITLApprovalError", "HITLApprovalTool", "hitl_approval"]


class HITLApprovalError(RuntimeError):
    """Raised when a HITL approval request cannot be processed."""


_VALID_OPERATIONS = {"request", "status", "resolve"}
_TERMINAL_STATUS_ALIASES = {
    "done": TaskStatus.DONE,
    "complete": TaskStatus.DONE,
    "completed": TaskStatus.DONE,
    "approved": TaskStatus.DONE,
    "cancelled": TaskStatus.CANCELLED,
    "canceled": TaskStatus.CANCELLED,
    "rejected": TaskStatus.CANCELLED,
    "declined": TaskStatus.CANCELLED,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_operation(operation: Any) -> str:
    if not isinstance(operation, str):
        raise HITLApprovalError("operation must be a string")
    normalized = operation.strip().lower()
    if normalized not in _VALID_OPERATIONS:
        raise HITLApprovalError(
            f"Unsupported operation '{operation}'. Expected one of: {', '.join(sorted(_VALID_OPERATIONS))}"
        )
    return normalized


def _ensure_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HITLApprovalError(f"{field} must be provided")
    return value.strip()


def _normalize_reason(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HITLApprovalError("reason must be a non-empty string when requesting approval")
    return value.strip()


def _normalize_context(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HITLApprovalError("context must be a mapping when provided")
    return {str(key): item for key, item in value.items()}


def _normalize_assignees(entries: Any) -> list[str]:
    if entries is None:
        return []
    if isinstance(entries, (str, bytes)):
        raise HITLApprovalError("assignees must be a sequence of strings")
    if not isinstance(entries, Sequence):
        raise HITLApprovalError("assignees must be a sequence of strings")
    normalized: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        candidate = entry.strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _normalize_escalation(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HITLApprovalError("escalation_policy must be a mapping when provided")
    return {str(key): item for key, item in value.items()}


def _normalize_resolution(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HITLApprovalError("resolution must be a mapping when provided")
    return {str(key): item for key, item in value.items()}


def _normalize_target_status(value: Any) -> TaskStatus:
    if value is None:
        return TaskStatus.DONE
    if isinstance(value, TaskStatus):
        if value not in {TaskStatus.DONE, TaskStatus.CANCELLED}:
            raise HITLApprovalError("target_status must be one of: done, cancelled")
        return value
    text = str(value).strip().lower()
    if not text:
        return TaskStatus.DONE
    if text not in _TERMINAL_STATUS_ALIASES:
        raise HITLApprovalError("target_status must be one of: done, cancelled, approved, rejected, declined")
    return _TERMINAL_STATUS_ALIASES[text]


def _build_title(reason: str) -> str:
    prefix = "HITL Approval"
    if not reason:
        return prefix
    summary = reason.strip()
    if len(summary) > 80:
        summary = f"{summary[:77]}..."
    return f"{prefix}: {summary}"


def _serialize_task(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    metadata_copy = dict(metadata) if isinstance(metadata, Mapping) else {}
    return {
        "task_id": record.get("id"),
        "status": record.get("status"),
        "title": record.get("title"),
        "description": record.get("description"),
        "metadata": metadata_copy,
        "updated_at": record.get("updated_at"),
        "created_at": record.get("created_at"),
    }


class HITLApprovalTool:
    """Tool that files and monitors human approval tasks via :class:`TaskService`."""

    def __init__(
        self,
        *,
        config_manager: Optional[ConfigManager] = None,
        task_service: Optional[TaskService] = None,
    ) -> None:
        self._config_manager = config_manager
        self._task_service = task_service

    def _resolve_task_service(self) -> TaskService:
        if self._task_service is not None:
            return self._task_service

        manager = self._config_manager
        if manager is None:
            manager = ConfigManager()
            self._config_manager = manager

        getter = getattr(manager, "get_task_service", None)
        if not callable(getter):
            raise HITLApprovalError("Task service is not configured")

        service = getter()
        if service is None:
            raise HITLApprovalError("Task service is not configured")

        self._task_service = service
        return service

    async def run(
        self,
        *,
        operation: str,
        tenant_id: str,
        conversation_id: Optional[str] = None,
        reason: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        assignees: Optional[Sequence[str]] = None,
        escalation_policy: Optional[Mapping[str, Any]] = None,
        task_id: Optional[str] = None,
        resolution: Optional[Mapping[str, Any]] = None,
        target_status: Optional[Any] = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        op = _normalize_operation(operation)
        tenant = _ensure_identifier(tenant_id, field="tenant_id")
        service = self._resolve_task_service()

        if op == "request":
            conversation = _ensure_identifier(conversation_id, field="conversation_id")
            summary = _normalize_reason(reason)
            context_payload = _normalize_context(context)
            assignee_payload = _normalize_assignees(assignees)
            escalation_payload = _normalize_escalation(escalation_policy)

            hitl_metadata: MutableMapping[str, Any] = {
                "reason": summary,
                "context": context_payload,
                "assignees": assignee_payload,
                "escalation": escalation_payload,
                "requested_at": _now_iso(),
                "status": "pending",
            }

            metadata: MutableMapping[str, Any] = {
                "hitl": hitl_metadata,
                "tags": ["hitl", "approval"],
            }

            record = service.create_task(
                _build_title(summary),
                tenant_id=tenant,
                description=summary,
                status=TaskStatus.REVIEW,
                conversation_id=conversation,
                metadata=metadata,
            )

            publish_bus_event(
                "hitl.approval.requested",
                {
                    "task_id": record.get("id"),
                    "tenant_id": tenant,
                    "conversation_id": conversation,
                    "reason": summary,
                    "assignees": assignee_payload,
                },
            )

            payload = _serialize_task(record)
            payload["operation"] = op
            return payload

        if op == "status":
            identifier = _ensure_identifier(task_id, field="task_id")
            record = service.get_task(identifier, tenant_id=tenant)
            publish_bus_event(
                "hitl.approval.status",
                {
                    "task_id": identifier,
                    "tenant_id": tenant,
                    "status": record.get("status"),
                },
            )
            payload = _serialize_task(record)
            payload["operation"] = op
            return payload

        identifier = _ensure_identifier(task_id, field="task_id")
        resolution_payload = _normalize_resolution(resolution)
        desired_status = _normalize_target_status(target_status)

        snapshot = service.get_task(identifier, tenant_id=tenant)
        metadata = dict(snapshot.get("metadata") or {})
        hitl_metadata: MutableMapping[str, Any] = dict(metadata.get("hitl") or {})
        resolution_block: MutableMapping[str, Any] = dict(hitl_metadata.get("resolution") or {})
        resolution_block.update(resolution_payload)
        resolution_block.setdefault("resolved_at", _now_iso())
        hitl_metadata["resolution"] = resolution_block
        hitl_metadata["status"] = "completed" if desired_status == TaskStatus.DONE else "cancelled"
        metadata["hitl"] = hitl_metadata

        service.update_task(
            identifier,
            tenant_id=tenant,
            changes={"metadata": metadata},
        )

        record = service.transition_task(
            identifier,
            tenant_id=tenant,
            target_status=desired_status,
        )

        publish_bus_event(
            "hitl.approval.resolved",
            {
                "task_id": identifier,
                "tenant_id": tenant,
                "status": record.get("status"),
                "resolution": hitl_metadata.get("resolution"),
            },
        )

        payload = _serialize_task(record)
        payload["operation"] = op
        return payload


async def hitl_approval(**kwargs: Any) -> Mapping[str, Any]:
    tool = HITLApprovalTool()
    return await tool.run(**kwargs)
