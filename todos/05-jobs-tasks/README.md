# Job & Task Service Migration

> **Status**: 📋 Planning  
> **Priority**: High  
> **Complexity**: Medium  
> **Effort**: 3-5 days  
> **Created**: 2026-01-07

---

## Overview

Migrate job and task services from `modules/` to `core/services/` with enhanced patterns:

- `modules/job_store/service.py` → `core/services/jobs/`
- `modules/task_store/service.py` → `core/services/tasks/`

Both services will gain standardized patterns, MessageBus events, and permission checks.

---

## Phases

### Phase 1: JobService Migration

- [ ] **1.1** Create `core/services/jobs/` package:
  - `__init__.py`
  - `types.py` - Job-specific types, events
  - `permissions.py` - Job permission checker
  - `service.py` - Migrate from modules/job_store/service.py
- [ ] **1.2** Add `OperationResult` return types
- [ ] **1.3** Add MessageBus events:
  - `job.created`, `job.updated`, `job.deleted`
  - `job.status_changed`, `job.scheduled`, `job.started`
  - `job.succeeded`, `job.failed`, `job.cancelled`
- [ ] **1.4** Add permission checks (actor parameter)
- [ ] **1.5** Update all callers to use new location
- [ ] **1.6** Keep `modules/job_store/` for repository only
- [ ] **1.7** Deprecate old import path with warning
- [ ] **1.8** Write unit tests

### Phase 2: TaskService Migration

- [ ] **2.1** Create `core/services/tasks/` package:
  - `__init__.py`
  - `types.py` - Task-specific types, events
  - `permissions.py` - Task permission checker
  - `service.py` - Migrate from modules/task_store/service.py
- [ ] **2.2** Add `OperationResult` return types
- [ ] **2.3** Add MessageBus events:
  - `task.created`, `task.updated`, `task.deleted`
  - `task.status_changed`, `task.assigned`
  - `task.completed`, `task.cancelled`
- [ ] **2.4** Add permission checks
- [ ] **2.5** Update all callers to use new location
- [ ] **2.6** Keep `modules/task_store/` for repository only
- [ ] **2.7** Write unit tests

### Phase 3: Cross-Service Integration

- [ ] **3.1** Job-Task relationship methods
- [ ] **3.2** Subscribe each service to the other's events
- [ ] **3.3** Integration tests

---

## Service Methods

### JobService

```python
class JobService:
    # CRUD
    def create_job(self, actor: Actor, job: JobCreate) -> OperationResult[Job]: ...
    def get_job(self, actor: Actor, job_id: UUID) -> OperationResult[Job]: ...
    def update_job(self, actor: Actor, job_id: UUID, updates: JobUpdate) -> OperationResult[Job]: ...
    def delete_job(self, actor: Actor, job_id: UUID) -> OperationResult[None]: ...
    def list_jobs(self, actor: Actor, filters: JobFilters) -> OperationResult[list[Job]]: ...
    
    # Lifecycle
    def schedule_job(self, actor: Actor, job_id: UUID, schedule: Schedule) -> OperationResult[Job]: ...
    def start_job(self, actor: Actor, job_id: UUID) -> OperationResult[Job]: ...
    def cancel_job(self, actor: Actor, job_id: UUID, reason: str) -> OperationResult[Job]: ...
    def complete_job(self, actor: Actor, job_id: UUID, result: JobResult) -> OperationResult[Job]: ...
    def fail_job(self, actor: Actor, job_id: UUID, error: str) -> OperationResult[Job]: ...
    
    # Queries
    def get_pending_jobs(self, actor: Actor) -> OperationResult[list[Job]]: ...
    def get_running_jobs(self, actor: Actor) -> OperationResult[list[Job]]: ...
    def get_job_history(self, actor: Actor, job_id: UUID) -> OperationResult[list[JobRun]]: ...
```

### TaskService

```python
class TaskService:
    # CRUD
    def create_task(self, actor: Actor, task: TaskCreate) -> OperationResult[Task]: ...
    def get_task(self, actor: Actor, task_id: UUID) -> OperationResult[Task]: ...
    def update_task(self, actor: Actor, task_id: UUID, updates: TaskUpdate) -> OperationResult[Task]: ...
    def delete_task(self, actor: Actor, task_id: UUID) -> OperationResult[None]: ...
    def list_tasks(self, actor: Actor, filters: TaskFilters) -> OperationResult[list[Task]]: ...
    
    # Lifecycle
    def assign_task(self, actor: Actor, task_id: UUID, assignee: str) -> OperationResult[Task]: ...
    def complete_task(self, actor: Actor, task_id: UUID) -> OperationResult[Task]: ...
    def cancel_task(self, actor: Actor, task_id: UUID, reason: str) -> OperationResult[Task]: ...
    def reopen_task(self, actor: Actor, task_id: UUID) -> OperationResult[Task]: ...
    
    # Hierarchy
    def create_subtask(self, actor: Actor, parent_id: UUID, task: TaskCreate) -> OperationResult[Task]: ...
    def get_subtasks(self, actor: Actor, parent_id: UUID) -> OperationResult[list[Task]]: ...
```

---

## MessageBus Events

### Job Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `job.created` | `JobEvent` | JobService |
| `job.updated` | `JobEvent` | JobService |
| `job.deleted` | `JobEvent` | JobService |
| `job.status_changed` | `JobStatusEvent` | JobService |
| `job.scheduled` | `JobScheduleEvent` | JobService |
| `job.started` | `JobEvent` | JobService |
| `job.succeeded` | `JobResultEvent` | JobService |
| `job.failed` | `JobErrorEvent` | JobService |
| `job.cancelled` | `JobEvent` | JobService |

### Task Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `task.created` | `TaskEvent` | TaskService |
| `task.updated` | `TaskEvent` | TaskService |
| `task.deleted` | `TaskEvent` | TaskService |
| `task.status_changed` | `TaskStatusEvent` | TaskService |
| `task.assigned` | `TaskAssignEvent` | TaskService |
| `task.completed` | `TaskEvent` | TaskService |
| `task.cancelled` | `TaskEvent` | TaskService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/jobs/__init__.py` | Package exports |
| `core/services/jobs/types.py` | Dataclasses, events |
| `core/services/jobs/permissions.py` | JobPermissionChecker |
| `core/services/jobs/service.py` | JobService |
| `core/services/tasks/__init__.py` | Package exports |
| `core/services/tasks/types.py` | Dataclasses, events |
| `core/services/tasks/permissions.py` | TaskPermissionChecker |
| `core/services/tasks/service.py` | TaskService |
| `tests/services/jobs/` | Job service tests |
| `tests/services/tasks/` | Task service tests |

---

## Files to Modify

| File | Changes |
|------|---------|
| `core/services/__init__.py` | Export job/task services |
| `modules/job_store/service.py` | Deprecate, redirect |
| `modules/task_store/service.py` | Deprecate, redirect |
| `GTKUI/Job_manager/*.py` | Use new JobService |
| `GTKUI/Task_manager/*.py` | Use new TaskService |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/job_store/` - Repository layer (keep)
- `modules/task_store/` - Repository layer (keep)
- `core/messaging/` - MessageBus for events

---

## Success Criteria

1. Both services migrated to core/services/
2. All operations use OperationResult
3. MessageBus events firing on all state changes
4. Permission checks on all operations
5. Repository layers remain in modules/
6. Old import paths deprecated
7. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Should job execution be in JobService or separate ExecutorService? | Same service / Separate | TBD |
| Task-Job linking strategy? | Many-to-many / Task belongs to Job | TBD |
| Job retry logic location? | In service / In executor | TBD |
| Task priority levels? | Low/Medium/High / Custom scale | TBD |
