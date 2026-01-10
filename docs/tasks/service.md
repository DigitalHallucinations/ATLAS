---
audience: contributors, integrators
status: in_review
last_verified: 2026-01-10
source_of_truth: core/services/tasks/service.py; core/services/tasks/types.py
---

# Task Service Layer

The `core.services.tasks.TaskService` provides a high-level service facade for task lifecycle management with permission checks, MessageBus events, and consistent `OperationResult` returns. This service follows the ATLAS services layer pattern established by `BudgetPolicyService` and `CalendarEventService`.

## Architecture Overview

The task service layer consists of four modules:

| Module | Purpose |
| --- | --- |
| `types.py` | Domain enums, events, DTOs, and response types |
| `permissions.py` | `TaskPermissionChecker` for tenant-isolated access control |
| `exceptions.py` | Task-specific exception hierarchy |
| `service.py` | `TaskService` with CRUD, lifecycle, subtasks, and dependencies |

## Permission Scopes

| Scope | Grants |
| --- | --- |
| `task:admin` | Full access, implies all other scopes |
| `task:write` | Create, update, delete, lifecycle operations (implies read) |
| `task:read` | Read task data and list tasks |

### Tenant Isolation

All operations enforce tenant isolation:

- Actors can only access tasks within their own tenant
- System actors (type `system`) bypass tenant restrictions
- Task owners have elevated access to their own tasks
- Cross-tenant access is denied with `PERMISSION_DENIED` error

## Status Lifecycle

Tasks progress through the following states:

```Text
pending → in_progress → completed
                    ↘ cancelled
blocked → pending (when dependencies resolve)
```

### Allowed Transitions

| From | To |
| --- | --- |
| `pending` | `in_progress`, `blocked`, `cancelled` |
| `blocked` | `pending`, `cancelled` |
| `in_progress` | `completed`, `cancelled`, `pending` (rollback) |

Terminal states (`completed`, `cancelled`) allow no further transitions.

## Priority System

Tasks use a 1-100 priority scale:

| Range | Meaning |
| --- | --- |
| 1-20 | Low priority |
| 21-40 | Below normal |
| 41-60 | Normal (default: 50) |
| 61-80 | Above normal |
| 81-100 | High priority / Urgent |

Priority affects scheduling order and can be used by the orchestrator to determine execution sequence.

## SOTA Enhancement Fields

The service supports state-of-the-art agentic workflow features:

| Field | Purpose |
| --- | --- |
| `assigned_agent` | Persona name assigned to execute the task |
| `assignee_id` | User or agent ID for assignment |
| `assignee_type` | `"user"` or `"agent"` |
| `execution_context` | Scratchpad for intermediate state during execution |
| `estimated_cost` | Budget forecast for the task |
| `actual_cost` | Recorded spend after completion |
| `timeout_seconds` | Maximum execution time |
| `parent_id` | Links subtask to parent task |
| `job_id` | Links task to a job |
| `delegation_chain` | Tracks persona handoffs during execution |

## Core Operations

### Create Task

```python
from core.services.tasks import TaskService, TaskCreate

task_data = TaskCreate(
    title="Research competitor pricing",
    tenant_id="tenant_1",
    description="Gather Q4 pricing data from top 5 competitors",
    priority=70,  # Above normal priority
    assigned_agent="researcher-persona",
    job_id="job_123",  # Link to parent job
)

result = task_service.create_task(actor, task_data)
if result.success:
    task = result.data  # TaskResponse
else:
    print(result.error_code, result.error_message)
```

### Lifecycle Transitions

```python
# Start a task
result = task_service.start_task(actor, task_id, tenant_id)

# Complete a task
result = task_service.complete_task(actor, task_id, tenant_id,
    completion_notes="Data gathered for all 5 competitors",
    actual_cost=Decimal("3.50"),
)

# Cancel a task
result = task_service.cancel_task(actor, task_id, tenant_id,
    reason="No longer needed",
)
```

### Subtasks

Tasks can have nested subtasks:

```python
from core.services.tasks import SubtaskCreate

subtask_data = SubtaskCreate(
    title="Research Competitor A",
    tenant_id="tenant_1",
    priority=75,
)

result = task_service.create_subtask(actor, parent_task_id, tenant_id, subtask_data)

# Get all subtasks
result = task_service.get_subtasks(actor, parent_task_id, tenant_id)
```

### Dependencies

Tasks can depend on other tasks:

```python
from core.services.tasks import DependencyCreate

dependency = DependencyCreate(
    depends_on_id="task_prerequisite",
    dependency_type="blocks",  # or "requires", "follows"
)

result = task_service.add_dependency(actor, task_id, tenant_id, dependency)

# Check if dependencies are satisfied
result = task_service.dependencies_complete(actor, task_id, tenant_id)

# Get all dependencies
result = task_service.get_dependencies(actor, task_id, tenant_id)

# Remove dependency
result = task_service.remove_dependency(actor, task_id, tenant_id, depends_on_id)
```

#### Circular Dependency Detection

The service detects and prevents circular dependencies:

```python
# If task_a depends on task_b, and task_b already depends on task_a:
result = task_service.add_dependency(actor, task_a_id, tenant_id, 
    DependencyCreate(depends_on_id=task_b_id))
# Returns error_code="CIRCULAR_DEPENDENCY"
```

### Assignment

```python
# Assign to an agent
result = task_service.assign_task(actor, task_id, tenant_id,
    assignee_id="analyst-persona",
    assignee_type="agent",
)

# Assign to a user
result = task_service.assign_task(actor, task_id, tenant_id,
    assignee_id="user_123",
    assignee_type="user",
)
```

### Execution Context

```python
# Update execution context (scratchpad)
result = task_service.update_execution_context(actor, task_id, tenant_id,
    context_updates={"current_step": "data_collection", "items_processed": 42},
    merge=True,  # Merge with existing context
)
```

### Priority Updates

```python
result = task_service.set_priority(actor, task_id, tenant_id, priority=95)
```

## Domain Events

The service publishes events via MessageBus:

| Event | Topic | Trigger |
| --- | --- | --- |
| `TaskCreated` | `task.created` | Task created |
| `TaskUpdated` | `task.updated` | Task metadata updated |
| `TaskDeleted` | `task.deleted` | Task deleted |
| `TaskStatusChanged` | `task.status_changed` | Status transition |
| `TaskCompleted` | `task.completed` | Task completed |
| `TaskCancelled` | `task.cancelled` | Task cancelled |
| `TaskAssigned` | `task.assigned` | Task assigned to user/agent |
| `SubtaskCreated` | `task.subtask_created` | Subtask created |
| `DependencyAdded` | `task.dependency_added` | Dependency added |

## Error Handling

The service uses a typed exception hierarchy:

| Exception | Error Code | Description |
| --- | --- | --- |
| `TaskNotFoundError` | `TASK_NOT_FOUND` | Task does not exist |
| `TaskTransitionError` | `INVALID_TRANSITION` | Invalid status transition |
| `TaskDependencyError` | `DEPENDENCY_ERROR` | Dependency issue |
| `TaskCircularDependencyError` | `CIRCULAR_DEPENDENCY` | Circular dependency detected |
| `TaskSubtaskError` | `SUBTASK_ERROR` | Subtask operation failed |
| `TaskConcurrencyError` | `CONCURRENCY_ERROR` | Optimistic lock failed |
| `TaskValidationError` | `VALIDATION_ERROR` | Input validation failed |

All operations return `OperationResult[T]` with:

- `success: bool` - Operation outcome
- `data: Optional[T]` - Result payload on success
- `error_code: Optional[str]` - Error identifier on failure
- `error_message: Optional[str]` - Human-readable description

## Querying Tasks

### List Tasks

```python
from core.services.tasks import TaskFilters

filters = TaskFilters(
    status="in_progress",
    job_id="job_123",
    assignee_id="researcher-persona",
    priority_min=70,
)

result = task_service.list_tasks(actor, tenant_id, filters, limit=50)
```

### Get Tasks by Job

```python
result = task_service.get_tasks_by_job(actor, job_id, tenant_id)
```

### Get Pending/Running Tasks

```python
result = task_service.get_pending_tasks(actor, tenant_id, limit=20)
result = task_service.get_in_progress_tasks(actor, tenant_id, limit=20)
```

## Integration with Jobs

Tasks can be linked to jobs via the `job_id` field:

```python
task_data = TaskCreate(
    title="Job Step 1",
    tenant_id="tenant_1",
    job_id="job_123",
)
```

See the [Job Service Layer](../jobs/service.md) for job-side operations including `link_task` and `unlink_task`.

## See Also

- [Task Overview](overview.md) - High-level task concepts
- [Job Service](../jobs/service.md) - Companion job service
- [Job Lifecycle](../jobs/lifecycle.md) - How tasks relate to job completion
