"""
Job service types and domain events.

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

# Re-export core job models for convenience
from modules.job_store.models import (
    Job,
    JobStatus,
    JobRun,
    JobRunStatus,
    JobTaskLink,
    JobAssignment,
    JobAssignmentStatus,
    JobSchedule,
    JobEvent,
    JobEventType,
)


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


# =============================================================================
# Cancellation Modes (SOTA Enhancement)
# =============================================================================


class CancellationMode(IntEnum):
    """Cancellation semantics for jobs."""
    
    GRACEFUL = 1  # Let current step finish, then cancel
    HARD = 2      # Abort immediately


# =============================================================================
# Dependency Types (SOTA Enhancement)
# =============================================================================


class DependencyType(IntEnum):
    """Types of task dependencies within a job."""
    
    FINISH_TO_START = 1   # Default: predecessor must finish before successor starts
    START_TO_START = 2    # Successor can start when predecessor starts
    FINISH_TO_FINISH = 3  # Successor finishes when predecessor finishes
    SOFT = 4              # Non-blocking dependency (advisory only)


# =============================================================================
# Domain Events
# =============================================================================


@dataclass(frozen=True)
class JobCreated:
    """Emitted when a job is created."""
    
    job_id: str
    job_name: str
    tenant_id: str
    actor_id: str
    status: str = "draft"
    owner_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    estimated_cost: Optional[Decimal] = None
    actor_type: str = "user"
    event_type: str = "job.created"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "status": self.status,
            "owner_id": self.owner_id,
            "assigned_agent": self.assigned_agent,
            "estimated_cost": str(self.estimated_cost) if self.estimated_cost else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobUpdated:
    """Emitted when a job is updated."""
    
    job_id: str
    job_name: str
    changed_fields: tuple[str, ...]
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "job.updated"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "changed_fields": list(self.changed_fields),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobDeleted:
    """Emitted when a job is deleted."""
    
    job_id: str
    job_name: str
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "job.deleted"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobStatusChanged:
    """Emitted when a job's status changes."""
    
    job_id: str
    job_name: str
    from_status: str
    to_status: str
    tenant_id: str
    actor_id: str
    reason: Optional[str] = None
    actor_type: str = "user"
    event_type: str = "job.status_changed"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobScheduled:
    """Emitted when a job is scheduled."""
    
    job_id: str
    job_name: str
    schedule_type: str
    expression: str
    next_run_at: Optional[datetime]
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "job.scheduled"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "schedule_type": self.schedule_type,
            "expression": self.expression,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobStarted:
    """Emitted when a job starts execution."""
    
    job_id: str
    job_name: str
    run_number: int
    tenant_id: str
    actor_id: str
    assigned_agent: Optional[str] = None
    actor_type: str = "user"
    event_type: str = "job.started"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "run_number": self.run_number,
            "assigned_agent": self.assigned_agent,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobSucceeded:
    """Emitted when a job completes successfully."""
    
    job_id: str
    job_name: str
    run_number: int
    tenant_id: str
    actor_id: str
    actual_cost: Optional[Decimal] = None
    result_summary: Optional[str] = None
    actor_type: str = "user"
    event_type: str = "job.succeeded"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "run_number": self.run_number,
            "actual_cost": str(self.actual_cost) if self.actual_cost else None,
            "result_summary": self.result_summary,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobFailed:
    """Emitted when a job fails."""
    
    job_id: str
    job_name: str
    run_number: int
    error: str
    tenant_id: str
    actor_id: str
    error_code: Optional[str] = None
    retryable: bool = True
    actor_type: str = "user"
    event_type: str = "job.failed"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "run_number": self.run_number,
            "error": self.error,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobCancelled:
    """Emitted when a job is cancelled."""
    
    job_id: str
    job_name: str
    reason: str
    cancellation_mode: str
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "job.cancelled"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "reason": self.reason,
            "cancellation_mode": self.cancellation_mode,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobCheckpointed:
    """Emitted when a job checkpoint is saved (SOTA)."""
    
    job_id: str
    job_name: str
    checkpoint_step: int
    tenant_id: str
    actor_id: str
    actor_type: str = "system"
    event_type: str = "job.checkpointed"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "checkpoint_step": self.checkpoint_step,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class JobAgentAssigned:
    """Emitted when an agent is assigned to a job (SOTA)."""
    
    job_id: str
    job_name: str
    assigned_agent: str
    previous_agent: Optional[str]
    tenant_id: str
    actor_id: str
    actor_type: str = "user"
    event_type: str = "job.agent_assigned"
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
            "job_id": self.job_id,
            "job_name": self.job_name,
            "assigned_agent": self.assigned_agent,
            "previous_agent": self.previous_agent,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# DTOs for Service Operations
# =============================================================================


@dataclass
class JobCreate:
    """DTO for creating a new job."""
    
    name: str
    tenant_id: str
    description: Optional[str] = None
    owner_id: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # SOTA fields
    assigned_agent: Optional[str] = None
    estimated_cost: Optional[Decimal] = None
    actual_cost: Optional[Decimal] = None
    timeout_seconds: Optional[int] = None
    execution_context: Optional[Dict[str, Any]] = None
    checkpoint_data: Optional[Dict[str, Any]] = None
    plan_id: Optional[str] = None
    plan_step_index: Optional[int] = None


@dataclass
class JobUpdate:
    """DTO for updating an existing job."""
    
    name: Optional[str] = None
    description: Optional[str] = None
    owner_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # SOTA fields
    assigned_agent: Optional[str] = None
    estimated_cost: Optional[Decimal] = None
    actual_cost: Optional[Decimal] = None
    timeout_seconds: Optional[int] = None
    execution_context: Optional[Dict[str, Any]] = None
    checkpoint_data: Optional[Dict[str, Any]] = None
    plan_id: Optional[str] = None
    plan_step_index: Optional[int] = None
    
    def changed_fields(self) -> tuple[str, ...]:
        """Return list of fields that are set (not None)."""
        return tuple(
            field_name 
            for field_name in [
                'name', 'description', 'owner_id', 'metadata',
                'assigned_agent', 'estimated_cost', 'actual_cost',
                'timeout_seconds', 'execution_context', 'checkpoint_data',
                'plan_id', 'plan_step_index'
            ]
            if getattr(self, field_name) is not None
        )


@dataclass
class JobFilters:
    """Filters for listing jobs."""
    
    status: Optional[str | Sequence[str]] = None
    owner_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    plan_id: Optional[str] = None
    cursor: Optional[str] = None
    limit: int = 50


@dataclass
class ScheduleCreate:
    """DTO for creating a job schedule."""
    
    schedule_type: str = "cron"
    expression: str = ""
    timezone: str = "UTC"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class JobResult:
    """DTO for job completion result."""
    
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    result_summary: Optional[str] = None
    actual_cost: Optional[Decimal] = None
    
    # Execution metrics
    execution_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    tool_calls: Optional[int] = None


@dataclass
class TaskLinkCreate:
    """DTO for linking a task to a job."""
    
    task_id: str
    relationship_type: str = "relates_to"
    dependency_type: DependencyType = DependencyType.FINISH_TO_START
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class JobCheckpoint:
    """DTO for job checkpoint data (SOTA)."""
    
    step_index: int
    step_name: str
    state: Dict[str, Any]
    execution_context: Dict[str, Any]
    created_at: datetime = field(default_factory=_now_utc)


# =============================================================================
# Response DTOs
# =============================================================================


@dataclass
class JobResponse:
    """Response DTO for job operations."""
    
    id: str
    name: str
    description: Optional[str]
    status: str
    owner_id: Optional[str]
    tenant_id: str
    conversation_id: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    # SOTA fields
    assigned_agent: Optional[str] = None
    estimated_cost: Optional[Decimal] = None
    actual_cost: Optional[Decimal] = None
    timeout_seconds: Optional[int] = None
    execution_context: Optional[Dict[str, Any]] = None
    checkpoint_data: Optional[Dict[str, Any]] = None
    plan_id: Optional[str] = None
    plan_step_index: Optional[int] = None
    
    # Related entities (optionally loaded)
    schedule: Optional[Dict[str, Any]] = None
    runs: Optional[List[Dict[str, Any]]] = None
    tasks: Optional[List[Dict[str, Any]]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobResponse":
        """Create from repository dict."""
        return cls(
            id=str(data["id"]),
            name=data["name"],
            description=data.get("description"),
            status=data.get("status", "draft"),
            owner_id=str(data["owner_id"]) if data.get("owner_id") else None,
            tenant_id=data["tenant_id"],
            conversation_id=str(data["conversation_id"]) if data.get("conversation_id") else None,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", _now_utc()),
            updated_at=data.get("updated_at", _now_utc()),
            assigned_agent=data.get("metadata", {}).get("assigned_agent"),
            estimated_cost=Decimal(data["metadata"]["estimated_cost"]) if data.get("metadata", {}).get("estimated_cost") else None,
            actual_cost=Decimal(data["metadata"]["actual_cost"]) if data.get("metadata", {}).get("actual_cost") else None,
            timeout_seconds=data.get("metadata", {}).get("timeout_seconds"),
            execution_context=data.get("metadata", {}).get("execution_context"),
            checkpoint_data=data.get("metadata", {}).get("checkpoint_data"),
            plan_id=data.get("metadata", {}).get("plan_id"),
            plan_step_index=data.get("metadata", {}).get("plan_step_index"),
            schedule=data.get("schedule"),
            runs=data.get("runs"),
            tasks=data.get("tasks"),
        )


@dataclass
class JobRunResponse:
    """Response DTO for job run."""
    
    id: str
    job_id: str
    run_number: int
    status: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class JobListResponse:
    """Response DTO for paginated job list."""
    
    jobs: List[JobResponse]
    next_cursor: Optional[str] = None
    total_count: Optional[int] = None


# =============================================================================
# Export all types
# =============================================================================


__all__ = [
    # Re-exported models
    "Job",
    "JobStatus",
    "JobRun",
    "JobRunStatus",
    "JobTaskLink",
    "JobAssignment",
    "JobAssignmentStatus",
    "JobSchedule",
    "JobEvent",
    "JobEventType",
    
    # SOTA enums
    "CancellationMode",
    "DependencyType",
    
    # Domain events
    "JobCreated",
    "JobUpdated",
    "JobDeleted",
    "JobStatusChanged",
    "JobScheduled",
    "JobStarted",
    "JobSucceeded",
    "JobFailed",
    "JobCancelled",
    "JobCheckpointed",
    "JobAgentAssigned",
    
    # DTOs
    "JobCreate",
    "JobUpdate",
    "JobFilters",
    "ScheduleCreate",
    "JobResult",
    "TaskLinkCreate",
    "JobCheckpoint",
    
    # Response DTOs
    "JobResponse",
    "JobRunResponse",
    "JobListResponse",
]
