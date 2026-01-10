---
audience: contributors, integrators
status: in_review
last_verified: 2026-01-10
source_of_truth: core/services/jobs/service.py; core/services/jobs/types.py
---

# Job Service Layer

The `core.services.jobs.JobService` provides a high-level service facade for job lifecycle management with permission checks, MessageBus events, and consistent `OperationResult` returns. This service follows the ATLAS services layer pattern established by `BudgetPolicyService` and `CalendarEventService`.

## Architecture Overview

The job service layer consists of four modules:

| Module | Purpose |
| --- | --- |
| `types.py` | Domain enums, events, DTOs, and response types |
| `permissions.py` | `JobPermissionChecker` for tenant-isolated access control |
| `exceptions.py` | Job-specific exception hierarchy |
| `service.py` | `JobService` with CRUD, lifecycle, and SOTA operations |

## Permission Scopes

| Scope | Grants |
| --- | --- |
| `job:admin` | Full access, implies all other scopes |
| `job:write` | Create, update, delete jobs (implies read) |
| `job:read` | Read job data and list jobs |
| `job:execute` | Start, complete, fail, cancel jobs (implies read) |

### Tenant Isolation

All operations enforce tenant isolation:

- Actors can only access jobs within their own tenant
- System actors (type `system`) bypass tenant restrictions
- Cross-tenant access is denied with `PERMISSION_DENIED` error

## Status Lifecycle

Jobs progress through the following states:

```Text
draft → scheduled → running → succeeded
                          ↘ failed
                          ↘ cancelled
```

### Allowed Transitions

| From | To |
| --- | --- |
| `draft` | `scheduled`, `running`, `cancelled` |
| `scheduled` | `running`, `cancelled` |
| `running` | `succeeded`, `failed`, `cancelled` |
| `failed` | `running` (retry) |

Terminal states (`succeeded`, `cancelled`) allow no further transitions except through explicit retry operations.

## SOTA Enhancement Fields

The service supports state-of-the-art agentic workflow features:

| Field | Purpose |
| --- | --- |
| `assigned_agent` | Persona name assigned to execute the job |
| `execution_context` | Scratchpad for intermediate state during execution |
| `checkpoint_data` | Resumable state for long-running jobs |
| `estimated_cost` | Budget forecast for the job |
| `actual_cost` | Recorded spend after completion |
| `timeout_seconds` | Maximum execution time |
| `plan_id` | Links job to a higher-level orchestration plan |
| `plan_step_index` | Position within the plan sequence |
| `delegation_chain` | Tracks persona handoffs during execution |

## Core Operations

### Create Job

```python
from core.services.jobs import JobService, JobCreate

job_data = JobCreate(
    name="Data Analysis Job",
    tenant_id="tenant_1",
    description="Analyze Q4 sales data",
    assigned_agent="analyst-persona",
    estimated_cost=Decimal("15.00"),
)

result = job_service.create_job(actor, job_data)
if result.success:
    job = result.data  # JobResponse
else:
    print(result.error_code, result.error_message)
```

### Lifecycle Transitions

```python
# Schedule a job
result = job_service.schedule_job(actor, job_id, tenant_id, schedule)

# Start execution
result = job_service.start_job(actor, job_id, tenant_id)

# Complete with result
from core.services.jobs.types import JobResult
result = job_service.complete_job(actor, job_id, tenant_id, JobResult(
    success=True,
    result_summary="Analysis complete",
    actual_cost=Decimal("12.50"),
))

# Or fail
result = job_service.fail_job(actor, job_id, tenant_id, 
    error_message="Data source unavailable",
    error_code="DATA_SOURCE_ERROR",
)

# Or cancel
result = job_service.cancel_job(actor, job_id, tenant_id,
    reason="User requested cancellation",
)
```

### Checkpoints

```python
from core.services.jobs import JobCheckpoint

checkpoint = JobCheckpoint(
    step_index=3,
    step_name="Processing batch 3",
    state={"processed": 300, "remaining": 200},
    execution_context={"current_file": "batch_003.csv"},
)

result = job_service.save_checkpoint(actor, job_id, tenant_id, checkpoint)
```

### Agent Assignment

```python
result = job_service.assign_agent(actor, job_id, tenant_id,
    agent_name="researcher-persona",
)
```

## Domain Events

The service publishes events via MessageBus:

| Event | Topic | Trigger |
| --- | --- | --- |
| `JobCreated` | `job.created` | Job created |
| `JobUpdated` | `job.updated` | Job metadata updated |
| `JobDeleted` | `job.deleted` | Job deleted |
| `JobStatusChanged` | `job.status_changed` | Status transition |
| `JobScheduled` | `job.scheduled` | Job scheduled |
| `JobStarted` | `job.started` | Execution began |
| `JobSucceeded` | `job.succeeded` | Job completed successfully |
| `JobFailed` | `job.failed` | Job failed |
| `JobCancelled` | `job.cancelled` | Job cancelled |
| `JobCheckpointed` | `job.checkpointed` | Checkpoint saved |
| `JobAgentAssigned` | `job.agent_assigned` | Agent assigned |

## Error Handling

The service uses a typed exception hierarchy:

| Exception | Error Code | Description |
| --- | --- | --- |
| `JobNotFoundError` | `JOB_NOT_FOUND` | Job does not exist |
| `JobTransitionError` | `INVALID_TRANSITION` | Invalid status transition |
| `JobDependencyError` | `DEPENDENCY_ERROR` | Linked task not complete |
| `JobConcurrencyError` | `CONCURRENCY_ERROR` | Optimistic lock failed |
| `JobScheduleError` | `SCHEDULE_ERROR` | Schedule configuration invalid |
| `JobValidationError` | `VALIDATION_ERROR` | Input validation failed |

All operations return `OperationResult[T]` with:

- `success: bool` - Operation outcome
- `data: Optional[T]` - Result payload on success
- `error_code: Optional[str]` - Error identifier on failure
- `error_message: Optional[str]` - Human-readable description

## Integration with Tasks

Jobs can be linked to tasks through `link_task` and `unlink_task`:

```python
result = job_service.link_task(actor, job_id, tenant_id, task_id,
    relationship_type="step",
    metadata={"order": 1},
)

# Get linked tasks
result = job_service.list_linked_tasks(actor, job_id, tenant_id)

# Unlink
result = job_service.unlink_task(actor, job_id, tenant_id, task_id)
```

See the [Task Service Layer](../tasks/service.md) for task-side operations.

## See Also

- [Job Lifecycle](lifecycle.md) - State machine and analytics
- [Job APIs](api.md) - REST route mappings
- [Job Scheduling](scheduling.md) - Recurrence and manifests
- [Task Service](../tasks/service.md) - Companion task service
