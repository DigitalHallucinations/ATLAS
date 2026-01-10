# Job & Task Service Migration

> **Status**: ✅ Complete  
> **Priority**: High  
> **Complexity**: Medium  
> **Effort**: 5-7 days  
> **Created**: 2026-01-07  
> **Updated**: 2026-01-10  
> **Completed**: 2026-01-10

---

## Overview

Migrate job and task services from `modules/` to `core/services/` with enhanced patterns and SOTA agent capabilities:

- `modules/job_store/service.py` → `core/services/jobs/`
- `modules/task_store/service.py` → `core/services/tasks/`

Both services will gain standardized patterns, MessageBus events, and permission checks.

---

## Phases

### Phase 1: JobService Migration

- [x] **1.1** Create `core/services/jobs/` package:
  - `__init__.py`
  - `types.py` - Job-specific types, events
  - `permissions.py` - Job permission checker
  - `service.py` - Migrate from modules/job_store/service.py
- [x] **1.2** Add `OperationResult` return types
- [x] **1.3** Add MessageBus events:
  - `job.created`, `job.updated`, `job.deleted`
  - `job.status_changed`, `job.scheduled`, `job.started`
  - `job.succeeded`, `job.failed`, `job.cancelled`
- [x] **1.4** Add permission checks (actor parameter)
- [x] **1.5** Update all callers to use new location
- [x] **1.6** Keep `modules/job_store/` for repository only
- [x] **1.7** Deprecate old import path with warning
- [x] **1.8** Write unit tests

### Phase 2: TaskService Migration

- [x] **2.1** Create `core/services/tasks/` package:
  - `__init__.py`
  - `types.py` - Task-specific types, events
  - `permissions.py` - Task permission checker
  - `service.py` - Migrate from modules/task_store/service.py
- [x] **2.2** Add `OperationResult` return types
- [x] **2.3** Add MessageBus events:
  - `task.created`, `task.updated`, `task.deleted`
  - `task.status_changed`, `task.assigned`
  - `task.completed`, `task.cancelled`
- [x] **2.4** Add permission checks
- [x] **2.5** Update all callers to use new location
- [x] **2.6** Keep `modules/task_store/` for repository only
- [x] **2.7** Write unit tests

### Phase 3: Cross-Service Integration

- [x] **3.1** Job-Task relationship methods (many-to-many with `JobTask` join table)
- [x] **3.2** Subscribe each service to the other's events
- [x] **3.3** Integration tests

### Phase 4: SOTA Enhancements

- [x] **4.1** Add `execution_context` JSON field for scratchpad/working memory
- [x] **4.2** Add `estimated_cost` / `actual_cost` fields with budget integration
- [x] **4.3** Add `assigned_agent` field for persona/agent assignment
- [x] **4.4** Add `checkpoint_data` for job/task resumability
- [x] **4.5** Add `timeout_seconds` and timeout handling
- [x] **4.6** Enhanced cancellation semantics (graceful/hard modes)
- [x] **4.7** Add `plan_id` linkage for planning integration
- [x] **4.8** Add dependency types (finish_to_start, start_to_start, finish_to_finish)

### Phase 5: ExecutorService (Deferred)

> **Note**: ExecutorService is deferred to a future iteration. The current implementation provides the foundation but full distributed execution strategies require additional infrastructure work.

- [ ] **5.1** Create `core/services/execution/` package
- [ ] **5.2** Define execution strategies (sync, async, distributed)
- [ ] **5.3** Retry logic with configurable policies (max retries, backoff)
- [ ] **5.4** Checkpoint save/restore for long-running jobs
- [ ] **5.5** Timeout enforcement and graceful shutdown
- [ ] **5.6** Integration with JobService lifecycle events

---

## Completion Summary

### Files Created

| File | Purpose |
|------|---------|
| `core/services/jobs/__init__.py` | Package exports |
| `core/services/jobs/types.py` | Dataclasses, events, SOTA fields |
| `core/services/jobs/permissions.py` | JobPermissionChecker with tenant isolation |
| `core/services/jobs/service.py` | JobService with CRUD, lifecycle, SOTA |
| `core/services/jobs/exceptions.py` | Job-specific exception hierarchy |
| `core/services/tasks/__init__.py` | Package exports |
| `core/services/tasks/types.py` | Dataclasses, events, SOTA fields |
| `core/services/tasks/permissions.py` | TaskPermissionChecker with tenant isolation |
| `core/services/tasks/service.py` | TaskService with CRUD, lifecycle, subtasks, dependencies |
| `core/services/tasks/exceptions.py` | Task-specific exception hierarchy |
| `tests/services/jobs/test_job_service.py` | JobService unit tests (23 tests) |
| `tests/services/jobs/test_permissions.py` | JobPermissionChecker tests (21 tests) |
| `tests/services/tasks/test_task_service.py` | TaskService unit tests (32 tests) |
| `tests/services/tasks/test_permissions.py` | TaskPermissionChecker tests (18 tests) |
| `docs/jobs/service.md` | Job service layer documentation |
| `docs/tasks/service.md` | Task service layer documentation |

### Test Results

- **JobService tests**: 23 passed
- **TaskService tests**: 32 passed
- **Job permissions tests**: 21 passed
- **Task permissions tests**: 18 passed
- **Total jobs/tasks tests**: 105 passed
- **All service tests**: 342 passed

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
| Should job execution be in JobService or separate ExecutorService? | Same service / Separate | **Separate** - Clean separation aligns with SOTA Orchestrator-Worker patterns |
| Task-Job linking strategy? | Many-to-many / Task belongs to Job | **Many-to-many** with `JobTask` join table and `role` field (primary, supporting, prerequisite) |
| Job retry logic location? | In service / In executor | **Split** - Policy in service, execution in ExecutorService |
| Task priority levels? | Low/Medium/High / Custom scale | **Numeric (1-100)** with semantic aliases (CRITICAL=100, HIGH=75, MEDIUM=50, LOW=25, BACKGROUND=10) |

---

## SOTA Considerations

### Execution Context / Working Memory
- Jobs and tasks need `execution_context` JSON for agent scratchpad during execution
- Supports context window management for what the agent "sees" while working

### Observability Hooks
- Emit structured telemetry (latency, token usage, tool calls)
- Add spans/traces for distributed tracing (integrates with 30-observability)

### Checkpointing & Resumability
- Long-running jobs support checkpoint save/restore
- Failure recovery from last successful step
- `checkpoint_data` and `last_completed_step` fields

### Cost/Budget Association
- Jobs track `estimated_cost` vs `actual_cost`
- Budget gates before execution starts (integrates with 02-budget)

### Agent Assignment
- `assigned_agent` field for persona/agent working the task
- `delegation_chain` for handoff history (integrates with 21-multi-agent)

### Plan Linkage
- `plan_id` reference to originating plan
- `plan_step_index` for position in plan (integrates with 23-planning)
