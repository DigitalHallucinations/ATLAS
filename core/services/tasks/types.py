"""
Task service types and domain events.

Defines DTOs for service operations and domain events for
integration with the ATLAS messaging system.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import IntEnum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence
from uuid import UUID

# Re-export core task models for convenience
from modules.task_store.models import (
    Task,
    TaskStatus,
    TaskAssignment,
    TaskAssignmentStatus,
    TaskDependency,
    TaskEvent,
    TaskEventType,
)


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


# =============================================================================
# Priority Levels (SOTA Enhancement)
# =============================================================================


class TaskPriority(IntEnum):
    """
    Task priority levels with semantic aliases.
    
    Uses a 1-100 numeric scale for fine-grained ordering,
    with named constants for common use cases.
    """
    
    BACKGROUND = 10   # Low priority background tasks
    LOW = 25          # Low priority
    MEDIUM = 50       # Default/normal priority
    HIGH = 75         # High priority
    CRITICAL = 100    # Highest priority, urgent
    
    @classmethod
    def from_value(cls, value: int) -> "TaskPriority":
        """Get closest priority level from numeric value."""
        if value >= 90:
            return cls.CRITICAL
        elif value >= 60:
            return cls.HIGH
        elif value >= 40:
            return cls.MEDIUM
        elif value >= 20:
            return cls.LOW
        return cls.BACKGROUND
    
    @classmethod
    def from_name(cls, name: str) -> "TaskPriority":
        """Get priority level from name string."""
        name_upper = name.upper()
        if hasattr(cls, name_upper):
            return getattr(cls, name_upper)
        raise ValueError(f"Unknown priority name: {name}")


# =============================================================================
# Dependency Types (SOTA Enhancement)
# =============================================================================


class DependencyType(IntEnum):
    """Types of task dependencies."""
    
    FINISH_TO_START = 1   # Default: predecessor must finish before successor starts
    START_TO_START = 2    # Successor can start when predecessor starts
    FINISH_TO_FINISH = 3  # Successor finishes when predecessor finishes
    SOFT = 4              # Non-blocking dependency (advisory only)


# =============================================================================
# Domain Events
# =============================================================================


@dataclass(frozen=True)
class TaskCreated:
    """Emitted when a task is created."""
    
    task_id: str
    title: str
    tenant_id: str
    actor_id: str
    status: str = "draft"
    priority: int = 50
    owner_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    conversation_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    actor_type: str = "user"
    event_type: str = "task.created"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status,
            "priority": self.priority,
            "owner_id": self.owner_id,
            "assigned_agent": self.assigned_agent,
            "conversation_id": self.conversation_id,
            "parent_task_id": self.parent_task_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TaskUpdated:
    """Emitted when a task is updated."""
    
    task_id: str
    title: str
    changed_fields: tuple[str, ...]
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "task.updated"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "task_id": self.task_id,
            "title": self.title,
            "changed_fields": list(self.changed_fields),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TaskDeleted:
    """Emitted when a task is deleted."""
    
    task_id: str
    title: str
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "task.deleted"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "task_id": self.task_id,
            "title": self.title,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TaskStatusChanged:
    """Emitted when a task's status changes."""
    
    task_id: str
    title: str
    from_status: str
    to_status: str
    tenant_id: str
    actor_id: str
    reason: Optional[str] = None
    actor_type: str = "user"
    event_type: str = "task.status_changed"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "task_id": self.task_id,
            "title": self.title,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TaskAssigned:
    """Emitted when a task is assigned to a user or agent."""
    
    task_id: str
    title: str
    assignee_id: str
    assignee_type: Literal["user", "agent"]
    tenant_id: str
    actor_id: str
    previous_assignee_id: Optional[str] = None
    role: str = "owner"
    actor_type: str = "user"
    event_type: str = "task.assigned"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "task_id": self.task_id,
            "title": self.title,
            "assignee_id": self.assignee_id,
            "assignee_type": self.assignee_type,
            "previous_assignee_id": self.previous_assignee_id,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TaskCompleted:
    """Emitted when a task is completed."""
    
    task_id: str
    title: str
    tenant_id: str
    actor_id: str
    actual_cost: Optional[Decimal] = None
    result_summary: Optional[str] = None
    actor_type: str = "user"
    event_type: str = "task.completed"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "task_id": self.task_id,
            "title": self.title,
            "actual_cost": str(self.actual_cost) if self.actual_cost else None,
            "result_summary": self.result_summary,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TaskCancelled:
    """Emitted when a task is cancelled."""
    
    task_id: str
    title: str
    reason: str
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "task.cancelled"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "task_id": self.task_id,
            "title": self.title,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TaskAgentAssigned:
    """Emitted when an agent is assigned to a task (SOTA)."""
    
    task_id: str
    title: str
    assigned_agent: str
    previous_agent: Optional[str]
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "task.agent_assigned"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "task_id": self.task_id,
            "title": self.title,
            "assigned_agent": self.assigned_agent,
            "previous_agent": self.previous_agent,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class SubtaskCreated:
    """Emitted when a subtask is created."""
    
    task_id: str
    parent_task_id: str
    title: str
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "task.subtask_created"
    entity_id: str = field(default_factory=_generate_uuid)
    timestamp: datetime = field(default_factory=_now_utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "title": self.title,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# DTOs for Service Operations
# =============================================================================


@dataclass
class TaskCreate:
    """DTO for creating a new task."""
    
    title: str
    tenant_id: str
    conversation_id: Optional[str] = None
    description: Optional[str] = None
    priority: int = TaskPriority.MEDIUM
    owner_id: Optional[str] = None
    session_id: Optional[str] = None
    due_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    
    # SOTA fields
    assigned_agent: Optional[str] = None
    estimated_cost: Optional[Decimal] = None
    actual_cost: Optional[Decimal] = None
    timeout_seconds: Optional[int] = None
    execution_context: Optional[Dict[str, Any]] = None
    plan_id: Optional[str] = None
    plan_step_index: Optional[int] = None
    parent_task_id: Optional[str] = None


@dataclass
class SubtaskCreate:
    """DTO for creating a subtask under a parent task."""
    
    title: str
    tenant_id: str
    description: Optional[str] = None
    priority: int = TaskPriority.MEDIUM
    owner_id: Optional[str] = None
    session_id: Optional[str] = None
    due_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    assigned_agent: Optional[str] = None
    estimated_cost: Optional[Decimal] = None
    timeout_seconds: Optional[int] = None
    execution_context: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None


@dataclass
class TaskUpdate:
    """DTO for updating an existing task."""
    
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[int] = None
    owner_id: Optional[str] = None
    due_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # SOTA fields
    assigned_agent: Optional[str] = None
    estimated_cost: Optional[Decimal] = None
    actual_cost: Optional[Decimal] = None
    timeout_seconds: Optional[int] = None
    execution_context: Optional[Dict[str, Any]] = None
    plan_id: Optional[str] = None
    plan_step_index: Optional[int] = None
    parent_task_id: Optional[str] = None
    
    def changed_fields(self) -> tuple[str, ...]:
        """Return list of fields that are set (not None)."""
        return tuple(
            field_name 
            for field_name in [
                'title', 'description', 'priority', 'owner_id', 'due_at', 'metadata',
                'assigned_agent', 'estimated_cost', 'actual_cost',
                'timeout_seconds', 'execution_context', 'plan_id', 'plan_step_index',
                'parent_task_id'
            ]
            if getattr(self, field_name) is not None
        )


@dataclass
class TaskFilters:
    """Filters for listing tasks."""
    
    status: Optional[str | Sequence[str]] = None
    owner_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    conversation_id: Optional[str] = None
    job_id: Optional[str] = None
    plan_id: Optional[str] = None
    priority_min: Optional[int] = None
    priority_max: Optional[int] = None
    due_before: Optional[datetime] = None
    due_after: Optional[datetime] = None
    cursor: Any = None  # Type depends on repository implementation
    limit: int = 50


@dataclass
class TaskResult:
    """DTO for task completion result."""
    
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    result_summary: Optional[str] = None
    actual_cost: Optional[Decimal] = None
    
    # Execution metrics
    execution_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    tool_calls: Optional[int] = None


@dataclass
class DependencyCreate:
    """DTO for creating a task dependency."""
    
    depends_on_id: str
    dependency_type: DependencyType = DependencyType.FINISH_TO_START
    relationship_type: str = "blocks"


# =============================================================================
# Response DTOs
# =============================================================================


@dataclass
class TaskResponse:
    """Response DTO for task operations."""
    
    id: str
    title: str
    description: Optional[str]
    status: str
    priority: int
    owner_id: Optional[str]
    tenant_id: str
    conversation_id: Optional[str]
    session_id: Optional[str]
    due_at: Optional[datetime]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    # SOTA fields
    assigned_agent: Optional[str] = None
    estimated_cost: Optional[Decimal] = None
    actual_cost: Optional[Decimal] = None
    timeout_seconds: Optional[int] = None
    execution_context: Optional[Dict[str, Any]] = None
    plan_id: Optional[str] = None
    plan_step_index: Optional[int] = None
    parent_task_id: Optional[str] = None
    
    # Related entities (optionally loaded)
    dependencies: Optional[List[Dict[str, Any]]] = None
    subtasks: Optional[List["TaskResponse"]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResponse":
        """Create from repository dict."""
        metadata = data.get("metadata", {})
        return cls(
            id=str(data["id"]),
            title=data["title"],
            description=data.get("description"),
            status=data.get("status", "draft"),
            priority=data.get("priority", 50),
            owner_id=str(data["owner_id"]) if data.get("owner_id") else None,
            tenant_id=data.get("tenant_id", ""),
            conversation_id=str(data["conversation_id"]) if data.get("conversation_id") else None,
            session_id=str(data["session_id"]) if data.get("session_id") else None,
            due_at=data.get("due_at"),
            metadata=metadata,
            created_at=data.get("created_at", _now_utc()),
            updated_at=data.get("updated_at", _now_utc()),
            assigned_agent=metadata.get("assigned_agent"),
            estimated_cost=Decimal(metadata["estimated_cost"]) if metadata.get("estimated_cost") else None,
            actual_cost=Decimal(metadata["actual_cost"]) if metadata.get("actual_cost") else None,
            timeout_seconds=metadata.get("timeout_seconds"),
            execution_context=metadata.get("execution_context"),
            plan_id=metadata.get("plan_id"),
            plan_step_index=metadata.get("plan_step_index"),
            parent_task_id=metadata.get("parent_task_id"),
            dependencies=data.get("dependencies"),
            subtasks=None,
        )


@dataclass
class TaskListResponse:
    """Response DTO for paginated task list."""
    
    tasks: List[TaskResponse]
    next_cursor: Optional[tuple[datetime, str]] = None
    total_count: Optional[int] = None


# =============================================================================
# Export all types
# =============================================================================


__all__ = [
    # Re-exported models
    "Task",
    "TaskStatus",
    "TaskAssignment",
    "TaskAssignmentStatus",
    "TaskDependency",
    "TaskEvent",
    "TaskEventType",
    
    # SOTA enums
    "TaskPriority",
    "DependencyType",
    
    # Domain events
    "TaskCreated",
    "TaskUpdated",
    "TaskDeleted",
    "TaskStatusChanged",
    "TaskAssigned",
    "TaskCompleted",
    "TaskCancelled",
    "TaskAgentAssigned",
    "SubtaskCreated",
    
    # DTOs
    "TaskCreate",
    "TaskUpdate",
    "TaskFilters",
    "TaskResult",
    "DependencyCreate",
    
    # Response DTOs
    "TaskResponse",
    "TaskListResponse",
]
