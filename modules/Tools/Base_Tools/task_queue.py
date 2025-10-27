"""Durable task queue built on top of APScheduler.

This module exposes a thin convenience layer for scheduling work items backed by
APScheduler job stores.  Jobs can be enqueued for one-off execution with an
optional delay, registered on cron-style recurring schedules, cancelled, and
inspected for their most recent execution details.  A retry policy with
configurable backoff is applied for one-off jobs and can be observed through
monitoring callbacks.

The task queue **requires** a PostgreSQL job store.  The default configuration
derives its DSN from :class:`ATLAS.config.ConfigManager`, sharing the
conversation store connection unless a dedicated ``task_queue.jobstore_url`` (or
``job_scheduling.job_store_url``) is defined.
"""

from __future__ import annotations

import atexit
import datetime as _dt
import random
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

try:  # pragma: no cover - exercised implicitly via import side effects
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED, JobEvent
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.jobstores.base import ConflictingIdError, JobLookupError
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - exercised when dependency missing
    _APSCHEDULER_IMPORT_ERROR = exc

    def _raise_missing_apscheduler(symbol: str | None = None) -> None:
        """Raise a consistent error when APScheduler is unavailable."""

        detail = (
            "APScheduler is required for the task queue tool. Install it with `pip install APScheduler` "
            "or disable the `task_queue_default` tool provider in your configuration."
        )
        if symbol:
            detail = f"{detail} (Attempted to access `{symbol}`.)"
        if _APSCHEDULER_IMPORT_ERROR is not None:
            detail = f"{detail} Original error: {_APSCHEDULER_IMPORT_ERROR}."
        raise ModuleNotFoundError(detail) from _APSCHEDULER_IMPORT_ERROR

    class _MissingAPSchedulerProxy:
        """Proxy that raises a clear error when APScheduler objects are accessed."""

        def __init__(self, symbol: str) -> None:
            self._symbol = symbol

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            _raise_missing_apscheduler(self._symbol)

        def __getattr__(self, attr: str) -> Any:
            _raise_missing_apscheduler(f"{self._symbol}.{attr}")

    EVENT_JOB_ERROR = EVENT_JOB_EXECUTED = EVENT_JOB_MISSED = 0
    JobEvent = Any  # type: ignore

    class ConflictingIdError(RuntimeError):
        """Fallback error used when APScheduler is not installed."""

    class JobLookupError(RuntimeError):
        """Fallback error used when APScheduler is not installed."""

    ThreadPoolExecutor = _MissingAPSchedulerProxy("ThreadPoolExecutor")  # type: ignore
    SQLAlchemyJobStore = _MissingAPSchedulerProxy("SQLAlchemyJobStore")  # type: ignore
    BackgroundScheduler = _MissingAPSchedulerProxy("BackgroundScheduler")  # type: ignore
    CronTrigger = _MissingAPSchedulerProxy("CronTrigger")  # type: ignore
    DateTrigger = _MissingAPSchedulerProxy("DateTrigger")  # type: ignore

from modules.logging.logger import setup_logger

if TYPE_CHECKING:  # pragma: no cover - import solely for type checking
    from ATLAS.config import ConfigManager


__all__ = [
    "TaskQueueError",
    "JobNotFoundError",
    "DuplicateTaskError",
    "RetryPolicy",
    "TaskEvent",
    "TaskQueueService",
    "build_task_queue_service",
    "get_default_task_queue_service",
    "enqueue_task",
    "schedule_cron_task",
    "cancel_task",
    "get_task_status",
]


logger = setup_logger(__name__)


def _resolve_config_manager(config_manager: "ConfigManager | None") -> "ConfigManager | None":
    """Return a ConfigManager instance when available."""

    if config_manager is not None:
        return config_manager

    try:  # pragma: no cover - exercised indirectly via startup/import paths
        from ATLAS.config import ConfigManager as _ConfigManager
    except ModuleNotFoundError:  # pragma: no cover - missing configuration subsystem
        logger.debug("ConfigManager module not found; proceeding without configuration", exc_info=True)
        return None
    except ImportError:  # pragma: no cover - circular/partial imports during startup
        logger.debug(
            "ConfigManager import failed during task queue initialization; proceeding without configuration",
            exc_info=True,
        )
        return None

    return _ConfigManager()


class TaskQueueError(RuntimeError):
    """Base class for queue-related errors."""


class JobNotFoundError(TaskQueueError):
    """Raised when a job lookup fails."""


class DuplicateTaskError(TaskQueueError):
    """Raised when an idempotent enqueue attempts to schedule a duplicate job."""


@dataclass(frozen=True)
class RetryPolicy:
    """Simple retry/backoff configuration."""

    max_attempts: int = 3
    backoff_seconds: float = 30.0
    jitter_seconds: float = 5.0
    backoff_multiplier: float = 2.0

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "backoff_seconds": self.backoff_seconds,
            "jitter_seconds": self.jitter_seconds,
            "backoff_multiplier": self.backoff_multiplier,
        }


@dataclass(frozen=True)
class TaskEvent:
    """Event emitted when job execution state changes."""

    job_id: str
    state: str
    attempt: int
    timestamp: _dt.datetime
    payload: Mapping[str, Any]
    error: Optional[str] = None
    next_run_time: Optional[_dt.datetime] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


TaskMonitor = Callable[[TaskEvent], None]
TaskExecutor = Callable[[Mapping[str, Any]], None]


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def _ensure_tzaware(value: _dt.datetime, tz: _dt.tzinfo) -> _dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=tz)
    return value.astimezone(tz)


def _default_executor(_: Mapping[str, Any]) -> None:
    """Fallback executor used when no callable is provided."""

    logger.debug("Task executed with default no-op executor")


class TaskQueueService:
    """High-level interface around APScheduler for durable task scheduling."""

    def __init__(
        self,
        *,
        config_manager: Optional[ConfigManager] = None,
        jobstore_url: Optional[str] = None,
        timezone: Optional[_dt.tzinfo | str] = None,
        executor: Optional[TaskExecutor] = None,
        max_workers: Optional[int] = None,
        misfire_grace_time: Optional[float] = None,
        coalesce: Optional[bool] = None,
        max_instances: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        start_paused: bool = False,
    ) -> None:
        self._config_manager = _resolve_config_manager(config_manager)
        settings = self._load_settings(
            jobstore_url=jobstore_url,
            timezone=timezone,
            max_workers=max_workers,
            misfire_grace_time=misfire_grace_time,
            coalesce=coalesce,
            max_instances=max_instances,
            retry_policy=retry_policy,
        )

        self._retry_policy = settings["retry_policy"]
        self._timezone = settings["timezone"]
        self._executor = executor or _default_executor
        self._monitor_callbacks: list[TaskMonitor] = []
        self._monitor_lock = threading.RLock()
        self._lock = threading.RLock()
        self._job_metadata: Dict[str, MutableMapping[str, Any]] = {}
        self._service_id = uuid.uuid4().hex

        job_defaults = {
            "coalesce": settings["coalesce"],
            "misfire_grace_time": settings["misfire_grace_time"],
            "max_instances": settings["max_instances"],
        }

        self._scheduler = BackgroundScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=settings["jobstore_url"])},
            executors={"default": ThreadPoolExecutor(max_workers=settings["max_workers"])},
            job_defaults=job_defaults,
            timezone=self._timezone,
        )
        if hasattr(self._scheduler, "add_listener"):
            self._scheduler.add_listener(
                self._handle_event,
                EVENT_JOB_ERROR | EVENT_JOB_EXECUTED | EVENT_JOB_MISSED,
            )
        else:  # pragma: no cover - exercised only with test doubles
            logger.debug(
                "BackgroundScheduler stub missing 'add_listener'; task queue events will not be emitted."
            )

        if start_paused:
            self._scheduler.start(paused=True)
        else:
            self._scheduler.start()

        with _REGISTRY_LOCK:
            _SERVICE_REGISTRY[self._service_id] = self

        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _load_settings(
        self,
        *,
        jobstore_url: Optional[str],
        timezone: Optional[_dt.tzinfo | str],
        max_workers: Optional[int],
        misfire_grace_time: Optional[float],
        coalesce: Optional[bool],
        max_instances: Optional[int],
        retry_policy: Optional[RetryPolicy],
    ) -> Dict[str, Any]:
        config_block: Mapping[str, Any] = {}
        env_overrides: Dict[str, Any] = {}

        if self._config_manager is not None:
            candidate = self._config_manager.get_config("task_queue", {})
            if isinstance(candidate, Mapping):
                config_block = candidate
            env_url = self._config_manager.get_config("TASK_QUEUE_JOBSTORE_URL")
            if isinstance(env_url, str) and env_url.strip():
                env_overrides["jobstore_url"] = env_url.strip()
            env_workers = self._config_manager.get_config("TASK_QUEUE_MAX_WORKERS")
            if isinstance(env_workers, int):
                env_overrides["max_workers"] = env_workers

        resolved: Dict[str, Any] = dict(config_block)
        resolved.update(env_overrides)

        if jobstore_url is not None:
            resolved["jobstore_url"] = jobstore_url
        if timezone is not None:
            resolved["timezone"] = timezone
        if max_workers is not None:
            resolved["max_workers"] = max_workers
        if misfire_grace_time is not None:
            resolved["misfire_grace_time"] = misfire_grace_time
        if coalesce is not None:
            resolved["coalesce"] = coalesce
        if max_instances is not None:
            resolved["max_instances"] = max_instances
        if retry_policy is not None:
            resolved["retry_policy"] = retry_policy

        tz = resolved.get("timezone")
        if isinstance(tz, str):
            if tz.upper() == "UTC":
                tzinfo: _dt.tzinfo = _dt.timezone.utc
            else:
                try:
                    import zoneinfo

                    tzinfo = zoneinfo.ZoneInfo(tz)
                except Exception:
                    logger.warning("Unknown timezone '%s', defaulting to UTC", tz)
                    tzinfo = _dt.timezone.utc
        elif isinstance(tz, _dt.tzinfo):
            tzinfo = tz
        else:
            tzinfo = _dt.timezone.utc

        default_retry = resolved.get("retry_policy")
        if isinstance(default_retry, RetryPolicy):
            retry = default_retry
        elif isinstance(default_retry, Mapping):
            retry = RetryPolicy(
                max_attempts=int(default_retry.get("max_attempts", 3)),
                backoff_seconds=float(default_retry.get("backoff_seconds", 30.0)),
                jitter_seconds=float(default_retry.get("jitter_seconds", 5.0)),
                backoff_multiplier=float(default_retry.get("backoff_multiplier", 2.0)),
            )
        else:
            retry = RetryPolicy()

        workers = int(resolved.get("max_workers", 4) or 4)
        misfire = float(resolved.get("misfire_grace_time", 60.0) or 60.0)
        coalesce_value = bool(resolved.get("coalesce", False))
        max_instances_value = int(resolved.get("max_instances", 1) or 1)

        jobstore = resolved.get("jobstore_url")
        if isinstance(jobstore, str) and jobstore.strip():
            jobstore_url = jobstore.strip()
        else:
            jobstore_url = self._resolve_default_jobstore_url()

        self._ensure_postgresql_jobstore(jobstore_url)

        return {
            "jobstore_url": jobstore_url,
            "timezone": tzinfo,
            "max_workers": workers,
            "misfire_grace_time": misfire,
            "coalesce": coalesce_value,
            "max_instances": max_instances_value,
            "retry_policy": retry,
        }

    def _resolve_default_jobstore_url(self) -> str:
        if self._config_manager is None:
            raise TaskQueueError(
                "Task queue requires a PostgreSQL job store URL. Provide one via the "
                "`jobstore_url` argument or configure ConfigManager with a "
                "`task_queue.jobstore_url`/`job_scheduling.job_store_url`."
            )

        getter = getattr(self._config_manager, "get_job_store_url", None)
        if getter is None:
            raise TaskQueueError(
                "ConfigManager does not expose `get_job_store_url`; update the "
                "configuration subsystem or pass a job store URL explicitly."
            )

        jobstore_url = getter()
        if not isinstance(jobstore_url, str) or not jobstore_url.strip():
            raise TaskQueueError(
                "Task queue requires a configured PostgreSQL job store URL. "
                "Set `job_scheduling.job_store_url` in your configuration or "
                "provide a `task_queue.jobstore_url`."
            )

        return jobstore_url.strip()

    @staticmethod
    def _ensure_postgresql_jobstore(url: str) -> None:
        if not isinstance(url, str) or not url.strip():
            raise TaskQueueError("Job store URL must be a non-empty string.")

        normalized = url.strip()
        scheme = normalized.split(":", 1)[0].lower()
        if not scheme.startswith("postgresql"):
            raise TaskQueueError(
                "The task queue only supports PostgreSQL-backed job stores. "
                f"Received URL with scheme '{scheme}'."
            )

    # ------------------------------------------------------------------
    # Monitoring hooks
    # ------------------------------------------------------------------
    def add_monitor(self, callback: TaskMonitor) -> None:
        """Register a callback invoked on task state changes."""

        with self._monitor_lock:
            self._monitor_callbacks.append(callback)

    def set_executor(self, executor: TaskExecutor) -> None:
        """Replace the callable invoked when jobs are executed."""

        if not callable(executor):
            raise TypeError("Executor must be callable")

        with self._lock:
            self._executor = executor

    def remove_monitor(self, callback: TaskMonitor) -> None:
        with self._monitor_lock:
            try:
                self._monitor_callbacks.remove(callback)
            except ValueError:
                pass

    def get_retry_policy(self) -> RetryPolicy:
        return self._retry_policy

    # ------------------------------------------------------------------
    # Job orchestration
    # ------------------------------------------------------------------
    def enqueue_task(
        self,
        *,
        name: str,
        payload: Mapping[str, Any],
        delay_seconds: Optional[float] = None,
        run_at: Optional[_dt.datetime] = None,
        idempotency_key: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> Mapping[str, Any]:
        policy = retry_policy or self._retry_policy
        run_time = self._resolve_run_time(delay_seconds=delay_seconds, run_at=run_at)
        job_id = self._resolve_job_id(name=name, idempotency_key=idempotency_key)

        with self._lock:
            metadata = self._job_metadata.get(job_id)
            existing_job = self._scheduler.get_job(job_id)
            if idempotency_key and (existing_job or metadata):
                if existing_job:
                    next_run = existing_job.next_run_time
                else:
                    next_run = metadata.get("next_run_time") if metadata else None
                return self._build_status(job_id, override={"state": metadata.get("state", "scheduled"), "next_run_time": next_run})

            self._schedule_job(
                job_id=job_id,
                name=name,
                trigger=DateTrigger(run_date=run_time, timezone=self._timezone),
                payload=dict(payload),
                policy=policy,
                attempt=1,
                recurring=False,
            )
            return self._build_status(job_id)

    def schedule_cron(
        self,
        *,
        name: str,
        cron_schedule: str | Mapping[str, Any] | None = None,
        cron_fields: Optional[Mapping[str, Any]] = None,
        payload: Mapping[str, Any],
        idempotency_key: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> Mapping[str, Any]:
        policy = retry_policy or self._retry_policy
        job_id = self._resolve_job_id(name=name, idempotency_key=idempotency_key)
        spec: str | Mapping[str, Any] | None = cron_fields if cron_fields is not None else cron_schedule
        if spec is None:
            raise ValueError("Either cron_schedule or cron_fields must be provided")
        trigger = self._build_cron_trigger(spec)

        with self._lock:
            existing_job = self._scheduler.get_job(job_id)
            if idempotency_key and existing_job is not None:
                return self._build_status(job_id)

            self._schedule_job(
                job_id=job_id,
                name=name,
                trigger=trigger,
                payload=dict(payload),
                policy=policy,
                attempt=1,
                recurring=True,
            )
            return self._build_status(job_id)

    def cancel(self, job_id: str) -> Mapping[str, Any]:
        try:
            self._scheduler.remove_job(job_id)
        except JobLookupError as exc:  # pragma: no cover - defensive guard
            raise JobNotFoundError(str(exc)) from exc

        with self._lock:
            metadata = self._job_metadata.setdefault(job_id, {})
            metadata.update({
                "state": "cancelled",
                "cancelled_at": _utc_now(),
            })
            return self._build_status(job_id)

    def get_status(self, job_id: str) -> Mapping[str, Any]:
        with self._lock:
            return self._build_status(job_id)

    def list_jobs(self) -> Iterable[Mapping[str, Any]]:
        with self._lock:
            ids = {job.id for job in self._scheduler.get_jobs()}
            ids.update(self._job_metadata.keys())
            return [self._build_status(job_id) for job_id in sorted(ids)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_job_id(self, *, name: str, idempotency_key: Optional[str]) -> str:
        if idempotency_key:
            key = idempotency_key.strip()
            if not key:
                raise ValueError("Idempotency key must be a non-empty string")
            return f"idemp-{uuid.uuid5(uuid.NAMESPACE_URL, key)}"
        return f"task-{uuid.uuid4().hex}-{name.strip() or 'job'}"

    def _resolve_run_time(
        self,
        *,
        delay_seconds: Optional[float],
        run_at: Optional[_dt.datetime],
    ) -> _dt.datetime:
        tz = self._timezone
        if run_at is not None:
            return _ensure_tzaware(run_at, tz)

        delay = float(delay_seconds or 0.0)
        if delay < 0:
            raise ValueError("delay_seconds cannot be negative")
        return _utc_now().astimezone(tz) + _dt.timedelta(seconds=delay)

    def _build_cron_trigger(self, cron_schedule: str | Mapping[str, Any]) -> CronTrigger:
        tz = self._timezone
        if isinstance(cron_schedule, str):
            try:
                return CronTrigger.from_crontab(cron_schedule, timezone=tz)
            except ValueError as exc:
                raise ValueError(f"Invalid cron expression: {exc}") from exc
        if isinstance(cron_schedule, Mapping):
            return CronTrigger(timezone=tz, **cron_schedule)
        raise TypeError("cron_schedule must be a string or mapping")

    def _schedule_job(
        self,
        *,
        job_id: str,
        name: str,
        trigger,
        payload: Mapping[str, Any],
        policy: RetryPolicy,
        attempt: int,
        recurring: bool,
    ) -> None:
        metadata = self._job_metadata.setdefault(job_id, {})
        metadata.update(
            {
                "name": name,
                "payload": dict(payload),
                "policy": policy,
                "attempts": attempt - 1,
                "state": "scheduled",
                "recurring": recurring,
            }
        )

        kwargs = {
            "service_id": self._service_id,
            "job_id": job_id,
            "name": name,
            "payload": dict(payload),
            "attempt": attempt,
            "policy": policy,
            "recurring": recurring,
        }

        try:
            self._scheduler.add_job(
                func=_dispatch_job,
                trigger=trigger,
                id=job_id,
                replace_existing=True,
                kwargs=kwargs,
                name=name,
            )
        except ConflictingIdError as exc:  # pragma: no cover - defensive guard
            raise DuplicateTaskError(str(exc)) from exc

        metadata["next_run_time"] = trigger.get_next_fire_time(None, _utc_now())

    def _calculate_backoff(self, *, policy: RetryPolicy, attempt: int) -> float:
        base = policy.backoff_seconds * (policy.backoff_multiplier ** max(0, attempt - 1))
        jitter = random.uniform(0, policy.jitter_seconds) if policy.jitter_seconds > 0 else 0.0
        return base + jitter

    def _execute_job(
        self,
        *,
        job_id: str,
        name: str,
        payload: Mapping[str, Any],
        attempt: int,
        policy: RetryPolicy,
        recurring: bool,
    ) -> None:
        self._update_metadata(job_id, state="running", attempt=attempt)
        event_payload = dict(payload)
        try:
            self._executor(
                {
                    "job_id": job_id,
                    "name": name,
                    "payload": event_payload,
                    "attempt": attempt,
                    "recurring": recurring,
                }
            )
        except Exception as exc:  # pragma: no cover - exceptions exercised in tests
            self._handle_failure(job_id=job_id, name=name, payload=payload, attempt=attempt, policy=policy, recurring=recurring, error=exc)
            raise
        else:
            self._handle_success(job_id=job_id, payload=payload, attempt=attempt, recurring=recurring)

    def _handle_success(self, *, job_id: str, payload: Mapping[str, Any], attempt: int, recurring: bool) -> None:
        with self._lock:
            metadata = self._job_metadata.setdefault(job_id, {})
            metadata["attempts"] = attempt
            metadata["last_success_at"] = _utc_now()
            if recurring:
                metadata["state"] = "scheduled"
                metadata["last_run_status"] = "succeeded"
                job = self._scheduler.get_job(job_id)
                metadata["next_run_time"] = job.next_run_time if job else None
            else:
                metadata["state"] = "succeeded"
                metadata["next_run_time"] = None

        self._emit_event(
            TaskEvent(
                job_id=job_id,
                state="succeeded",
                attempt=attempt,
                timestamp=_utc_now(),
                payload=dict(payload),
                next_run_time=self._job_metadata[job_id].get("next_run_time"),
                metadata=dict(self._job_metadata[job_id]),
            )
        )

    def _handle_failure(
        self,
        *,
        job_id: str,
        name: str,
        payload: Mapping[str, Any],
        attempt: int,
        policy: RetryPolicy,
        recurring: bool,
        error: Exception,
    ) -> None:
        with self._lock:
            metadata = self._job_metadata.setdefault(job_id, {})
            metadata["attempts"] = attempt
            metadata["last_error"] = str(error)

        remaining = policy.max_attempts - attempt
        if remaining > 0 and not recurring:
            delay = self._calculate_backoff(policy=policy, attempt=attempt)
            next_run = _utc_now().astimezone(self._timezone) + _dt.timedelta(seconds=delay)
            trigger = DateTrigger(run_date=next_run, timezone=self._timezone)
            self._schedule_job(
                job_id=job_id,
                name=name,
                trigger=trigger,
                payload=payload,
                policy=policy,
                attempt=attempt + 1,
                recurring=False,
            )
            state = "retrying"
        else:
            with self._lock:
                metadata = self._job_metadata.setdefault(job_id, {})
                metadata["state"] = "failed"
                metadata["next_run_time"] = None
            state = "failed"

        self._emit_event(
            TaskEvent(
                job_id=job_id,
                state=state,
                attempt=attempt,
                timestamp=_utc_now(),
                payload=dict(payload),
                error=str(error),
                next_run_time=self._job_metadata[job_id].get("next_run_time"),
                metadata=dict(self._job_metadata[job_id]),
            )
        )

    def _update_metadata(self, job_id: str, *, state: str, attempt: int) -> None:
        with self._lock:
            metadata = self._job_metadata.setdefault(job_id, {})
            metadata["state"] = state
            metadata["attempts"] = max(attempt, metadata.get("attempts", 0))
            metadata["last_run_at"] = _utc_now()

    def _build_status(self, job_id: str, *, override: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        metadata = self._job_metadata.setdefault(job_id, {})
        job = self._scheduler.get_job(job_id)
        next_run_time = None
        if job is not None:
            next_run_time = job.next_run_time
            metadata.setdefault("name", job.name)
            metadata.setdefault("payload", job.kwargs.get("payload", {}))
            metadata["next_run_time"] = next_run_time
            metadata.setdefault("state", "scheduled")
        if override is not None:
            metadata.update({k: v for k, v in override.items() if v is not None})

        status = {
            "job_id": job_id,
            "name": metadata.get("name"),
            "state": metadata.get("state", "unknown"),
            "attempts": metadata.get("attempts", 0),
            "next_run_time": metadata.get("next_run_time"),
            "recurring": metadata.get("recurring", False),
            "retry_policy": metadata.get("policy", self._retry_policy).to_dict(),
        }

        if "last_error" in metadata:
            status["last_error"] = metadata["last_error"]
        if "last_success_at" in metadata:
            status["last_success_at"] = metadata["last_success_at"]
        if "last_run_at" in metadata:
            status["last_run_at"] = metadata["last_run_at"]
        if "last_run_status" in metadata:
            status["last_run_status"] = metadata["last_run_status"]
        return status

    def _handle_event(self, event: JobEvent) -> None:
        if event.code == EVENT_JOB_MISSED:
            with self._lock:
                metadata = self._job_metadata.setdefault(event.job_id, {})
                metadata["state"] = "missed"
                metadata["last_error"] = "Job missed scheduled run"
            self._emit_event(
                TaskEvent(
                    job_id=event.job_id,
                    state="missed",
                    attempt=self._job_metadata[event.job_id].get("attempts", 0),
                    timestamp=_utc_now(),
                    payload=dict(self._job_metadata[event.job_id].get("payload", {})),
                    error="Job missed scheduled run",
                    next_run_time=self._job_metadata[event.job_id].get("next_run_time"),
                    metadata=dict(self._job_metadata[event.job_id]),
                )
            )

    def _emit_event(self, event: TaskEvent) -> None:
        with self._monitor_lock:
            callbacks = list(self._monitor_callbacks)
        for callback in callbacks:
            try:
                callback(event)
            except Exception:  # pragma: no cover - monitoring hooks must not fail queue
                logger.exception("Task monitor callback raised an exception")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self, wait: bool = False) -> None:
        if getattr(self, "_scheduler", None) is None:
            return
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
        with _REGISTRY_LOCK:
            _SERVICE_REGISTRY.pop(self._service_id, None)


_DEFAULT_SERVICE: Optional[TaskQueueService] = None
_DEFAULT_LOCK = threading.Lock()
_REGISTRY_LOCK = threading.Lock()
_SERVICE_REGISTRY: Dict[str, "TaskQueueService"] = {}


def _dispatch_job(*, service_id: str, **kwargs: Any) -> None:
    service = _SERVICE_REGISTRY.get(service_id)
    if service is None:
        logger.warning("No task queue service found for job %s", kwargs.get("job_id"))
        return
    job_kwargs = dict(kwargs)
    job_kwargs.pop("service_id", None)
    service._execute_job(**job_kwargs)


def build_task_queue_service(
    *,
    config_manager: Optional[ConfigManager] = None,
    **settings: Any,
) -> TaskQueueService:
    """Instantiate a :class:`TaskQueueService` with optional overrides."""

    return TaskQueueService(config_manager=config_manager, **settings)


def get_default_task_queue_service(*, config_manager: Optional[ConfigManager] = None) -> TaskQueueService:
    global _DEFAULT_SERVICE
    with _DEFAULT_LOCK:
        if _DEFAULT_SERVICE is None:
            _DEFAULT_SERVICE = TaskQueueService(config_manager=config_manager)
        return _DEFAULT_SERVICE


def enqueue_task(
    *,
    name: str,
    payload: Mapping[str, Any],
    delay_seconds: Optional[float] = None,
    run_at: Optional[_dt.datetime] = None,
    idempotency_key: Optional[str] = None,
    retry_policy: Optional[RetryPolicy] = None,
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    service = get_default_task_queue_service(config_manager=config_manager)
    return service.enqueue_task(
        name=name,
        payload=payload,
        delay_seconds=delay_seconds,
        run_at=run_at,
        idempotency_key=idempotency_key,
        retry_policy=retry_policy,
    )


def schedule_cron_task(
    *,
    name: str,
    cron_schedule: str | Mapping[str, Any] | None = None,
    cron_fields: Optional[Mapping[str, Any]] = None,
    payload: Mapping[str, Any],
    idempotency_key: Optional[str] = None,
    retry_policy: Optional[RetryPolicy] = None,
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    service = get_default_task_queue_service(config_manager=config_manager)
    return service.schedule_cron(
        name=name,
        cron_schedule=cron_schedule,
        cron_fields=cron_fields,
        payload=payload,
        idempotency_key=idempotency_key,
        retry_policy=retry_policy,
    )


def cancel_task(job_id: str, *, config_manager: Optional[ConfigManager] = None) -> Mapping[str, Any]:
    service = get_default_task_queue_service(config_manager=config_manager)
    return service.cancel(job_id)


def get_task_status(job_id: str, *, config_manager: Optional[ConfigManager] = None) -> Mapping[str, Any]:
    service = get_default_task_queue_service(config_manager=config_manager)
    return service.get_status(job_id)

