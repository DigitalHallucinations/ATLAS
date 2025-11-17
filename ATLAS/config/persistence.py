"""Persistence-focused configuration helpers for :mod:`ATLAS.config`."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Optional, Sequence, Tuple
from collections.abc import Mapping
from urllib.parse import urlparse

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

from modules.conversation_store.mongo_repository import MongoConversationStoreRepository
from modules.job_store import JobService, MongoJobStoreRepository
from modules.job_store.repository import JobStoreRepository
from modules.task_store import TaskService, TaskStoreRepository
from modules.Tools.Base_Tools.task_queue import (
    TaskQueueService,
    get_default_task_queue_service,
)

from .core import (
    ConversationStoreBackendOption,
    _DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND,
    _UNSET,
    default_conversation_store_backend_name,
    get_default_conversation_store_backends,
    infer_conversation_store_backend,
)


KV_STORE_UNSET = object()


def coerce_bool_flag(value: Any, default: bool) -> bool:
    """Return a truthy flag respecting textual representations."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


@dataclass(kw_only=True)
class PersistenceConfigSection:
    """Aggregate persistence related configuration helpers."""

    config: MutableMapping[str, Any]
    yaml_config: MutableMapping[str, Any]
    env_config: Mapping[str, Any]
    logger: Any
    normalize_job_store_url: Callable[[Any, str], str]
    write_yaml_callback: Callable[[], None]
    create_engine: Callable[..., Any]
    inspect_engine: Callable[..., Any]
    make_url: Callable[..., Any]
    sessionmaker_factory: Callable[..., Any]
    conversation_required_tables: Callable[[], set[str]]
    default_conversation_dsn_map: Mapping[str, str] | None = None
    default_conversation_dsn: str | None = None
    conversation_backend_options: Sequence[ConversationStoreBackendOption] | None = None

    kv_engine_cache: Dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backends = (
            tuple(self.conversation_backend_options)
            if self.conversation_backend_options
            else get_default_conversation_store_backends()
        )
        dsn_map = dict(self.default_conversation_dsn_map or {})
        default_backend = default_conversation_store_backend_name()
        if self.default_conversation_dsn and default_backend not in dsn_map:
            dsn_map[default_backend] = self.default_conversation_dsn
        for option in backends:
            dsn_map.setdefault(option.name, option.dsn)

        self.conversation = ConversationStoreConfigSection(
            config=self.config,
            yaml_config=self.yaml_config,
            env_config=self.env_config,
            logger=self.logger,
            write_yaml_callback=self.write_yaml_callback,
            default_dsn_by_backend=dsn_map,
            backend_options=backends,
            create_engine=self.create_engine,
            inspect_engine=self.inspect_engine,
            make_url=self.make_url,
            sessionmaker_factory=self.sessionmaker_factory,
            conversation_required_tables=self.conversation_required_tables,
        )
        self.kv_store = KVStoreConfigSection(
            config=self.config,
            yaml_config=self.yaml_config,
            env_config=self.env_config,
            logger=self.logger,
            normalize_job_store_url=self.normalize_job_store_url,
            write_yaml_callback=self.write_yaml_callback,
            engine_factory=self.create_engine,
            make_url=self.make_url,
            conversation_engine_getter=self.conversation.get_engine,
            kv_engine_cache=self.kv_engine_cache,
        )

    def apply(self) -> None:
        """Populate the configuration dictionary with persistence defaults."""

        self.kv_store.apply()
        self.conversation.apply()


class PersistenceConfigMixin:
    """Mixin exposing persistence and task-queue helpers for ConfigManager."""

    def get_kv_store_settings(self) -> Dict[str, Any]:
        """Return normalized configuration for the key-value store adapter."""

        return copy.deepcopy(self.persistence.kv_store.get_settings())

    def set_kv_store_settings(
        self,
        *,
        url: Any = _UNSET,
        reuse_conversation_store: Optional[bool] = None,
        namespace_quota_bytes: Any = _UNSET,
        global_quota_bytes: Any = _UNSET,
        pool: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist configuration for the PostgreSQL-backed key-value store."""

        updated = self.persistence.kv_store.set_settings(
            url=url if url is not _UNSET else KV_STORE_UNSET,
            reuse_conversation_store=reuse_conversation_store,
            namespace_quota_bytes=(
                namespace_quota_bytes if namespace_quota_bytes is not _UNSET else KV_STORE_UNSET
            ),
            global_quota_bytes=(
                global_quota_bytes if global_quota_bytes is not _UNSET else KV_STORE_UNSET
            ),
            pool=pool,
        )
        return copy.deepcopy(updated)

    def get_kv_store_engine(
        self,
        *,
        adapter_config: Optional[Mapping[str, Any]] = None,
    ) -> Engine | None:
        """Return a SQLAlchemy engine configured for the KV store."""

        return self.persistence.kv_store.get_engine(adapter_config=adapter_config)

    def get_conversation_database_config(self) -> Dict[str, Any]:
        """Return the merged configuration block for the conversation database."""

        return self.persistence.conversation.get_config()

    def get_conversation_backend(self) -> Optional[str]:
        """Return the selected conversation store backend name, if configured."""

        return self.persistence.conversation.get_backend()

    def get_conversation_store_backends(self) -> Tuple[ConversationStoreBackendOption, ...]:
        """Return the available conversation store backend options."""

        return self.persistence.conversation.available_backends()

    def ensure_postgres_conversation_store(self) -> str:
        """Verify the configured conversation store without provisioning it."""

        return self.persistence.conversation.ensure_postgres_store()

    def _conversation_store_required_tables(self) -> set[str]:
        """Return the set of tables that must exist in the conversation database."""

        from modules.conversation_store.models import Base as ConversationBase

        return {table.name for table in ConversationBase.metadata.tables.values()}

    def is_conversation_store_verified(self) -> bool:
        """Return ``True`` if the conversation store connection has been verified."""

        return self.persistence.conversation.is_verified()

    def get_conversation_retention_policies(self) -> Dict[str, Any]:
        """Return configured retention policies for the conversation store."""

        return self.persistence.conversation.get_retention_policies()

    def get_retention_worker_settings(self) -> Dict[str, Any]:
        """Return configuration overrides for the retention worker scheduler."""

        return copy.deepcopy(self.persistence.conversation.get_retention_worker_settings())

    def set_conversation_retention(
        self,
        *,
        days: Optional[int] = None,
        history_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Persist conversation retention settings under the conversation block."""

        return self.persistence.conversation.set_retention(
            days=days,
            history_limit=history_limit,
        )

    def get_conversation_store_engine(self) -> Any | None:
        """Return a configured engine or client for the conversation store."""

        return self.persistence.conversation.get_engine()

    def get_conversation_store_session_factory(
        self,
    ) -> sessionmaker | MongoConversationStoreRepository | None:
        """Return a configured session factory or repository for the conversation store."""

        return self.persistence.conversation.get_session_factory()

    def get_task_store_session_factory(self) -> sessionmaker | None:
        """Return a session factory for task persistence that shares the conversation engine."""

        if self._task_session_factory is not None:
            return self._task_session_factory

        conversation_factory = self.get_conversation_store_session_factory()
        if conversation_factory is None:
            return None
        if isinstance(conversation_factory, MongoConversationStoreRepository):
            return None

        self._task_session_factory = conversation_factory
        return conversation_factory

    def get_task_repository(self) -> TaskStoreRepository | None:
        """Return a configured repository for task persistence."""

        if self._task_repository is not None:
            return self._task_repository

        factory = self.get_task_store_session_factory()
        if factory is None:
            return None

        repository = TaskStoreRepository(factory)
        try:
            repository.create_schema()
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.logger.warning("Failed to initialize task store schema: %s", exc)
        self._task_repository = repository
        return repository

    def get_task_service(self) -> TaskService | None:
        """Return the task lifecycle service backed by the repository."""

        if self._task_service is not None:
            return self._task_service

        repository = self.get_task_repository()
        if repository is None:
            return None

        service = TaskService(repository)
        self._task_service = service
        return service

    def get_job_repository(self) -> JobStoreRepository | MongoJobStoreRepository | None:
        """Return a configured repository for scheduled job persistence."""

        if self._job_repository is not None:
            return self._job_repository

        repository = self._build_job_repository()
        if repository is None:
            return None

        self._job_repository = repository
        return repository

    def _build_job_repository(self) -> JobStoreRepository | MongoJobStoreRepository | None:
        settings = self.get_job_scheduling_settings()
        job_store_url = settings.get("job_store_url")
        if isinstance(job_store_url, str) and self._is_mongo_jobstore(job_store_url):
            repository = self._build_mongo_job_repository(job_store_url)
            if repository is not None:
                return repository

        factory = self.get_task_store_session_factory()
        if factory is None:
            return None

        previous = getattr(self, "_job_mongo_client", None)
        if previous is not None:
            close = getattr(previous, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
            self._job_mongo_client = None

        repository = JobStoreRepository(factory)
        try:
            repository.create_schema()
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.logger.warning("Failed to initialize job store schema: %s", exc)
        return repository

    @staticmethod
    def _is_mongo_jobstore(url: Any) -> bool:
        if not isinstance(url, str):
            return False
        candidate = url.strip().lower()
        return candidate.startswith("mongodb://") or candidate.startswith("mongodb+srv://")

    def _build_mongo_job_repository(self, url: str) -> MongoJobStoreRepository | None:
        normalized = url.strip()
        if not normalized:
            return None

        try:  # pragma: no cover - optional dependency path
            from pymongo import MongoClient
        except Exception as exc:
            self.logger.warning("PyMongo driver unavailable for job store: %s", exc)
            return None

        try:
            client = MongoClient(normalized)
        except Exception as exc:
            self.logger.error("Failed to connect to Mongo job store", exc_info=True)
            return None

        parsed = urlparse(normalized)
        database_name = parsed.path.lstrip("/").split("?", 1)[0] or "atlas_jobs"
        try:
            database = client.get_database(database_name)
        except Exception as exc:
            self.logger.error("Mongo job store configuration invalid", exc_info=True)
            client.close()
            return None

        repository = MongoJobStoreRepository.from_database(database, client=client)
        previous = getattr(self, "_job_mongo_client", None)
        if previous is not None and previous is not client:
            close = getattr(previous, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
        self._job_mongo_client = client
        return repository

    def get_job_service(self) -> JobService | None:
        """Return the job orchestration service backed by the repository."""

        if self._job_service is not None:
            return self._job_service

        repository = self.get_job_repository()
        if repository is None:
            return None

        service = JobService(repository)
        self._job_service = service
        return service

    def get_job_manager(self) -> "JobManager" | None:
        """Return the active job manager registered with the configuration."""

        return self._job_manager

    def set_job_manager(self, manager: "JobManager" | None) -> None:
        """Record the active job manager for downstream consumers."""

        self._job_manager = manager

    def get_job_scheduler(self) -> "JobScheduler" | None:
        """Return the active job scheduler registered with the configuration."""

        return self._job_scheduler

    def set_job_scheduler(self, scheduler: "JobScheduler" | None) -> None:
        """Record the active job scheduler for downstream consumers."""

        self._job_scheduler = scheduler

    def get_default_task_queue_service(self) -> TaskQueueService | None:
        """Return the shared task queue service used for scheduled jobs."""

        if self._task_queue_service is not None:
            return self._task_queue_service

        if not self.is_job_scheduling_enabled():
            self.logger.debug("Job scheduling disabled; skipping task queue initialisation")
            return None

        try:
            service = get_default_task_queue_service(config_manager=self)
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.logger.warning("Failed to initialize task queue service: %s", exc)
            return None

        self._task_queue_service = service
        return service

    def _build_conversation_store_session_factory(
        self,
    ) -> Tuple[Engine | None, sessionmaker | None]:
        ensured_url = self.ensure_postgres_conversation_store()

        config = self.get_conversation_database_config()
        url = ensured_url

        try:
            parsed_url = make_url(url)
        except Exception as exc:
            message = f"Invalid conversation database URL {url!r}: {exc}"
            self.logger.error(message)
            raise RuntimeError(message) from exc

        drivername = (parsed_url.drivername or "").lower()
        dialect = drivername.split("+", 1)[0]
        if dialect != "postgresql":
            message = (
                "Conversation database URL must use the 'postgresql' dialect; "
                f"received '{parsed_url.drivername}'."
            )
            self.logger.error(message)
            raise RuntimeError(message)

        pool_config = config.get("pool") or {}
        engine_kwargs: Dict[str, Any] = {}
        if isinstance(pool_config, Mapping):
            size = pool_config.get("size")
            if size is not None:
                engine_kwargs["pool_size"] = int(size)
            max_overflow = pool_config.get("max_overflow")
            if max_overflow is not None:
                engine_kwargs["max_overflow"] = int(max_overflow)
            timeout = pool_config.get("timeout")
            if timeout is not None:
                engine_kwargs["pool_timeout"] = int(timeout)

        engine = create_engine(url, future=True, **engine_kwargs)
        factory = sessionmaker(bind=engine, future=True)
        return engine, factory

    def get_job_scheduling_settings(self) -> Dict[str, Any]:
        """Return the merged configuration for job scheduling defaults."""

        settings: Dict[str, Any] = {}
        stored = self.config.get('job_scheduling')
        if isinstance(stored, Mapping):
            settings.update(stored)

        job_store_value = settings.get('job_store_url')
        if isinstance(job_store_value, str) and job_store_value.strip():
            try:
                settings['job_store_url'] = self._normalize_job_store_url(
                    job_store_value,
                    'job_scheduling.job_store_url',
                )
            except ValueError as exc:
                self.logger.warning("Ignoring invalid job scheduling job store URL: %s", exc)
                settings.pop('job_store_url', None)

        queue_block = self.config.get('task_queue')
        if isinstance(queue_block, Mapping):
            if 'job_store_url' not in settings and queue_block.get('jobstore_url'):
                settings['job_store_url'] = queue_block.get('jobstore_url')
            if 'max_workers' not in settings and queue_block.get('max_workers') is not None:
                settings['max_workers'] = queue_block.get('max_workers')
            if 'timezone' not in settings and queue_block.get('timezone'):
                settings['timezone'] = queue_block.get('timezone')
            if 'queue_size' not in settings and queue_block.get('queue_size') is not None:
                settings['queue_size'] = queue_block.get('queue_size')
            if 'retry_policy' not in settings and queue_block.get('retry_policy'):
                settings['retry_policy'] = queue_block.get('retry_policy')

        if 'job_store_url' not in settings or not settings['job_store_url']:
            try:
                default_job_store = self.get_job_store_url(require=False)
            except RuntimeError:
                default_job_store = ''
            if default_job_store:
                settings['job_store_url'] = default_job_store

        settings['enabled'] = bool(settings.get('enabled'))

        if settings.get('max_workers') is not None:
            try:
                settings['max_workers'] = int(settings['max_workers'])
            except (TypeError, ValueError):
                settings.pop('max_workers', None)

        if settings.get('queue_size') is not None:
            try:
                settings['queue_size'] = int(settings['queue_size'])
            except (TypeError, ValueError):
                settings.pop('queue_size', None)

        retry = settings.get('retry_policy')
        if isinstance(retry, Mapping):
            settings['retry_policy'] = {
                'max_attempts': int(retry.get('max_attempts', 3) or 3),
                'backoff_seconds': float(retry.get('backoff_seconds', 30.0) or 30.0),
                'jitter_seconds': float(retry.get('jitter_seconds', 5.0) or 5.0),
                'backoff_multiplier': float(retry.get('backoff_multiplier', 2.0) or 2.0),
            }
        else:
            settings['retry_policy'] = {
                'max_attempts': 3,
                'backoff_seconds': 30.0,
                'jitter_seconds': 5.0,
                'backoff_multiplier': 2.0,
            }

        return settings

    def set_job_scheduling_settings(
        self,
        *,
        enabled: Optional[bool] = None,
        job_store_url: Any = _UNSET,
        max_workers: Any = _UNSET,
        retry_policy: Optional[Mapping[str, Any]] = None,
        timezone: Any = _UNSET,
        queue_size: Any = _UNSET,
    ) -> Dict[str, Any]:
        """Persist job scheduling defaults and mirror them to task queue settings."""

        existing = self.get_job_scheduling_settings()
        updated = dict(existing)

        if enabled is not None:
            updated['enabled'] = bool(enabled)

        if job_store_url is not _UNSET:
            text = str(job_store_url).strip() if isinstance(job_store_url, str) else str(job_store_url or '')
            if text:
                normalized = self._normalize_job_store_url(text, 'job_scheduling.job_store_url')
                updated['job_store_url'] = normalized
            else:
                updated.pop('job_store_url', None)

        if max_workers is not _UNSET:
            if max_workers in (None, ''):
                updated.pop('max_workers', None)
            else:
                updated['max_workers'] = int(max_workers)

        if timezone is not _UNSET:
            if timezone in (None, ''):
                updated.pop('timezone', None)
            else:
                updated['timezone'] = str(timezone).strip()

        if queue_size is not _UNSET:
            if queue_size in (None, ''):
                updated.pop('queue_size', None)
            else:
                normalized_queue = int(queue_size)
                if normalized_queue <= 0:
                    raise ValueError("Scheduler queue size must be a positive integer")
                updated['queue_size'] = normalized_queue

        if retry_policy is not None:
            normalized_retry = {}
            normalized_retry['max_attempts'] = int(retry_policy.get('max_attempts', 3) or 3)
            normalized_retry['backoff_seconds'] = float(retry_policy.get('backoff_seconds', 30.0) or 30.0)
            normalized_retry['jitter_seconds'] = float(retry_policy.get('jitter_seconds', 5.0) or 5.0)
            normalized_retry['backoff_multiplier'] = float(retry_policy.get('backoff_multiplier', 2.0) or 2.0)
            updated['retry_policy'] = normalized_retry

        self.yaml_config['job_scheduling'] = dict(updated)
        self.config['job_scheduling'] = dict(updated)

        queue_block = dict(self.yaml_config.get('task_queue', {}))
        if updated.get('job_store_url'):
            queue_block['jobstore_url'] = updated['job_store_url']
        else:
            queue_block.pop('jobstore_url', None)

        if updated.get('max_workers') is not None:
            queue_block['max_workers'] = updated['max_workers']
        else:
            queue_block.pop('max_workers', None)

        if updated.get('timezone'):
            queue_block['timezone'] = updated['timezone']
        else:
            queue_block.pop('timezone', None)

        if updated.get('queue_size') is not None:
            queue_block['queue_size'] = updated['queue_size']
        else:
            queue_block.pop('queue_size', None)

        if updated.get('retry_policy'):
            queue_block['retry_policy'] = dict(updated['retry_policy'])
        else:
            queue_block.pop('retry_policy', None)

        if queue_block:
            self.yaml_config['task_queue'] = dict(queue_block)
            self.config['task_queue'] = dict(queue_block)
        else:
            self.yaml_config.pop('task_queue', None)
            self.config.pop('task_queue', None)

        self._write_yaml_config()
        return dict(updated)

    def is_job_scheduling_enabled(self) -> bool:
        """Return True when job scheduling should be initialised."""

        settings = self.get_job_scheduling_settings()
        return bool(settings.get('enabled'))

    def get_job_store_url(self, require: bool = True) -> str:
        """Return the configured PostgreSQL job store URL for the task queue."""

        candidates: list[tuple[str, Any]] = []

        stored = self.config.get('job_scheduling')
        if isinstance(stored, Mapping):
            candidates.append(('job_scheduling.job_store_url', stored.get('job_store_url')))

        queue_block = self.config.get('task_queue')
        if isinstance(queue_block, Mapping):
            candidates.append(('task_queue.jobstore_url', queue_block.get('jobstore_url')))

        env_url = self.env_config.get('TASK_QUEUE_JOBSTORE_URL')
        if env_url is None:
            env_url = self.config.get('TASK_QUEUE_JOBSTORE_URL')
        candidates.append(('TASK_QUEUE_JOBSTORE_URL', env_url))

        for source, value in candidates:
            if value is None:
                continue
            try:
                return self._normalize_job_store_url(value, source)
            except ValueError as exc:
                self.logger.debug("Skipping %s for job store URL: %s", source, exc)

        conversation_config = self.get_conversation_database_config()
        conversation_url = conversation_config.get('url')
        if conversation_url:
            try:
                return self._normalize_job_store_url(conversation_url, 'conversation_database.url')
            except ValueError as exc:
                self.logger.debug("Conversation store URL unsuitable for job store: %s", exc)

        fallback_dsn = _DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND.get("postgresql")
        if fallback_dsn is None:
            default_backend = default_conversation_store_backend_name()
            fallback_dsn = _DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND.get(default_backend)
        if fallback_dsn is None and _DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND:
            fallback_dsn = next(iter(_DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND.values()))

        if not fallback_dsn:
            if require:
                raise RuntimeError(
                    "Task queue requires a PostgreSQL job store URL. "
                    "Set `job_scheduling.job_store_url` or `task_queue.jobstore_url` "
                    "to a PostgreSQL DSN."
                )
            return ''

        try:
            return self._normalize_job_store_url(
                fallback_dsn,
                'default conversation store DSN',
            )
        except ValueError:
            if require:
                raise RuntimeError(
                    "Task queue requires a PostgreSQL job store URL. "
                    "Set `job_scheduling.job_store_url` or `task_queue.jobstore_url` "
                    "to a PostgreSQL DSN."
                )
            return ''

    @staticmethod
    def _normalize_job_store_url(url: Any, source: str) -> str:
        """Validate and normalize PostgreSQL job store URLs."""

        if not isinstance(url, str):
            raise ValueError(f"{source} must be a string")

        candidate = url.strip()
        if not candidate:
            raise ValueError(f"{source} must not be empty")

        parsed = urlparse(candidate)
        scheme = (parsed.scheme or '').lower()
        if scheme.startswith('postgresql'):
            return candidate
        if scheme.startswith('mongodb'):
            return candidate

        raise ValueError(
            f"{source} must use the 'postgresql' or 'mongodb' scheme; received '{parsed.scheme or 'missing'}'"
        )


@dataclass(kw_only=True)
class KVStoreConfigSection:
    """Manage key-value store configuration and helpers."""

    config: MutableMapping[str, Any]
    yaml_config: MutableMapping[str, Any]
    env_config: Mapping[str, Any]
    logger: Any
    normalize_job_store_url: Callable[[Any, str], str]
    write_yaml_callback: Callable[[], None]
    engine_factory: Callable[..., Any]
    make_url: Callable[..., Any]
    conversation_engine_getter: Callable[[], Any | None]
    kv_engine_cache: Dict[tuple[Any, ...], Any]

    def apply(self) -> None:
        tools_block = self.config.get("tools")
        if not isinstance(tools_block, Mapping):
            tools_block = {}
        else:
            tools_block = dict(tools_block)

        kv_block = tools_block.get("kv_store")
        if not isinstance(kv_block, Mapping):
            kv_block = {}
        else:
            kv_block = dict(kv_block)

        default_adapter = kv_block.get("default_adapter")
        if isinstance(default_adapter, str) and default_adapter.strip():
            kv_block["default_adapter"] = default_adapter.strip().lower()
        else:
            kv_block["default_adapter"] = "postgres"

        adapters_block = kv_block.get("adapters")
        if not isinstance(adapters_block, Mapping):
            adapters_block = {}
        else:
            adapters_block = dict(adapters_block)

        postgres_block = adapters_block.get("postgres")
        if not isinstance(postgres_block, Mapping):
            postgres_block = {}
        else:
            postgres_block = dict(postgres_block)

        env_kv_url = self.env_config.get("ATLAS_KV_STORE_URL")
        if env_kv_url and not postgres_block.get("url"):
            postgres_block["url"] = env_kv_url

        namespace_quota_value = postgres_block.get("namespace_quota_bytes")
        if namespace_quota_value is None:
            env_namespace_quota = self.env_config.get("ATLAS_KV_NAMESPACE_QUOTA_BYTES")
            if env_namespace_quota is not None:
                try:
                    postgres_block["namespace_quota_bytes"] = int(env_namespace_quota)
                except (TypeError, ValueError):
                    self.logger.warning(
                        "Invalid ATLAS_KV_NAMESPACE_QUOTA_BYTES value %r; expected integer",
                        env_namespace_quota,
                    )
        else:
            try:
                postgres_block["namespace_quota_bytes"] = int(namespace_quota_value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid namespace_quota_bytes value %r; expected integer",
                    namespace_quota_value,
                )
                postgres_block.pop("namespace_quota_bytes", None)

        if "namespace_quota_bytes" not in postgres_block:
            postgres_block["namespace_quota_bytes"] = 1_048_576

        global_quota_value = postgres_block.get("global_quota_bytes")
        if global_quota_value not in (None, ""):
            try:
                postgres_block["global_quota_bytes"] = int(global_quota_value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid global_quota_bytes value %r; expected integer",
                    global_quota_value,
                )
                postgres_block.pop("global_quota_bytes", None)
        else:
            env_global_quota = self.env_config.get("ATLAS_KV_GLOBAL_QUOTA_BYTES")
            if env_global_quota not in (None, ""):
                try:
                    postgres_block["global_quota_bytes"] = int(env_global_quota)
                except (TypeError, ValueError):
                    self.logger.warning(
                        "Invalid ATLAS_KV_GLOBAL_QUOTA_BYTES value %r; expected integer",
                        env_global_quota,
                    )
                else:
                    if postgres_block["global_quota_bytes"] <= 0:
                        postgres_block.pop("global_quota_bytes", None)

        reuse_value = postgres_block.get("reuse_conversation_store")
        if reuse_value is None:
            env_reuse = self.env_config.get("ATLAS_KV_REUSE_CONVERSATION")
            if env_reuse is not None:
                postgres_block["reuse_conversation_store"] = str(env_reuse).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            else:
                postgres_block["reuse_conversation_store"] = True
        else:
            if isinstance(reuse_value, str):
                normalized_reuse = reuse_value.strip().lower()
                if normalized_reuse in {"1", "true", "yes", "on"}:
                    postgres_block["reuse_conversation_store"] = True
                elif normalized_reuse in {"0", "false", "no", "off"}:
                    postgres_block["reuse_conversation_store"] = False
                else:
                    postgres_block["reuse_conversation_store"] = bool(reuse_value)
            else:
                postgres_block["reuse_conversation_store"] = bool(reuse_value)

        pool_block = postgres_block.get("pool")
        if not isinstance(pool_block, Mapping):
            pool_block = {}
        else:
            pool_block = dict(pool_block)

        for env_key, setting_key in (
            ("ATLAS_KV_POOL_SIZE", "size"),
            ("ATLAS_KV_MAX_OVERFLOW", "max_overflow"),
            ("ATLAS_KV_POOL_TIMEOUT", "timeout"),
        ):
            value = self.env_config.get(env_key)
            if value is None or pool_block.get(setting_key) not in (None, ""):
                continue
            try:
                if setting_key == "timeout":
                    pool_block[setting_key] = float(value)
                else:
                    pool_block[setting_key] = int(value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid %s value %r; expected numeric", env_key, value
                )

        postgres_block["pool"] = pool_block
        adapters_block["postgres"] = postgres_block

        sqlite_block = adapters_block.get("sqlite")
        if not isinstance(sqlite_block, Mapping):
            sqlite_block = {}
        else:
            sqlite_block = dict(sqlite_block)

        url_value = sqlite_block.get("url")
        if isinstance(url_value, str) and url_value.strip():
            sqlite_block["url"] = url_value.strip()
        else:
            sqlite_block["url"] = "sqlite:///atlas_kv.sqlite"

        namespace_quota_value = sqlite_block.get("namespace_quota_bytes")
        if namespace_quota_value not in (None, ""):
            try:
                sqlite_block["namespace_quota_bytes"] = int(namespace_quota_value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid namespace_quota_bytes value %r; expected integer",
                    namespace_quota_value,
                )
                sqlite_block.pop("namespace_quota_bytes", None)
        if "namespace_quota_bytes" not in sqlite_block:
            sqlite_block["namespace_quota_bytes"] = postgres_block.get(
                "namespace_quota_bytes",
                1_048_576,
            )

        global_quota_value = sqlite_block.get("global_quota_bytes")
        if global_quota_value in (None, ""):
            sqlite_block["global_quota_bytes"] = None
        else:
            try:
                sqlite_block["global_quota_bytes"] = int(global_quota_value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid global_quota_bytes value %r; expected integer",
                    global_quota_value,
                )
                sqlite_block["global_quota_bytes"] = None

        reuse_sqlite = sqlite_block.get("reuse_conversation_store")
        if reuse_sqlite is None:
            sqlite_block["reuse_conversation_store"] = False
        else:
            sqlite_block["reuse_conversation_store"] = coerce_bool_flag(
                reuse_sqlite,
                False,
            )

        adapters_block["sqlite"] = sqlite_block
        kv_block["adapters"] = adapters_block
        tools_block["kv_store"] = kv_block

        self.config["tools"] = tools_block

    # Exposed helpers --------------------------------------------------
    def get_settings(self) -> Dict[str, Any]:
        tools_block = self.config.get("tools", {})
        normalized: Dict[str, Any] = {"default_adapter": "postgres", "adapters": {}}

        if isinstance(tools_block, Mapping):
            kv_block = tools_block.get("kv_store")
            if isinstance(kv_block, Mapping):
                default_adapter = kv_block.get("default_adapter")
                if isinstance(default_adapter, str) and default_adapter.strip():
                    normalized["default_adapter"] = default_adapter.strip().lower()
                adapters = kv_block.get("adapters")
                if isinstance(adapters, Mapping):
                    postgres = adapters.get("postgres")
                    if isinstance(postgres, Mapping):
                        normalized["adapters"]["postgres"] = self._normalize_postgres_settings(
                            postgres
                        )
                    sqlite = adapters.get("sqlite")
                    if isinstance(sqlite, Mapping):
                        normalized["adapters"]["sqlite"] = self._normalize_sqlite_settings(sqlite)

        if "postgres" not in normalized["adapters"]:
            normalized["adapters"]["postgres"] = self._normalize_postgres_settings({})
        if "sqlite" not in normalized["adapters"]:
            normalized["adapters"]["sqlite"] = self._normalize_sqlite_settings({})

        return normalized

    def set_settings(
        self,
        *,
        url: Any = KV_STORE_UNSET,
        reuse_conversation_store: Optional[bool] = None,
        namespace_quota_bytes: Any = KV_STORE_UNSET,
        global_quota_bytes: Any = KV_STORE_UNSET,
        pool: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        settings = self.get_settings()
        postgres = dict(settings["adapters"].get("postgres", {}))

        if url is not KV_STORE_UNSET:
            if url in (None, ""):
                postgres.pop("url", None)
            else:
                postgres["url"] = self._normalize_kv_store_url(
                    url, "tools.kv_store.adapters.postgres.url"
                )

        if reuse_conversation_store is not None:
            postgres["reuse_conversation_store"] = bool(reuse_conversation_store)

        if namespace_quota_bytes is not KV_STORE_UNSET:
            if namespace_quota_bytes in (None, ""):
                postgres.pop("namespace_quota_bytes", None)
            else:
                postgres["namespace_quota_bytes"] = int(namespace_quota_bytes)

        if global_quota_bytes is not KV_STORE_UNSET:
            if global_quota_bytes in (None, ""):
                postgres["global_quota_bytes"] = None
            else:
                value = int(global_quota_bytes)
                postgres["global_quota_bytes"] = value if value > 0 else None

        if pool is not None:
            normalized_pool: Dict[str, Any] = {}
            if isinstance(pool, Mapping):
                size_value = pool.get("size")
                if size_value not in (None, ""):
                    normalized_pool["size"] = int(size_value)
                overflow_value = pool.get("max_overflow")
                if overflow_value not in (None, ""):
                    normalized_pool["max_overflow"] = int(overflow_value)
                timeout_value = pool.get("timeout")
                if timeout_value not in (None, ""):
                    normalized_pool["timeout"] = float(timeout_value)
            if normalized_pool:
                postgres["pool"] = normalized_pool
            else:
                postgres.pop("pool", None)

        adapters = dict(settings.get("adapters", {}))
        adapters["postgres"] = postgres
        updated = {"default_adapter": settings.get("default_adapter", "postgres"), "adapters": adapters}

        tools_yaml = self.yaml_config.get("tools")
        if isinstance(tools_yaml, Mapping):
            new_tools_yaml = dict(tools_yaml)
        else:
            new_tools_yaml = {}
        new_tools_yaml["kv_store"] = dict(updated)
        self.yaml_config["tools"] = new_tools_yaml

        tools_config = self.config.get("tools")
        if isinstance(tools_config, Mapping):
            new_tools_config = dict(tools_config)
        else:
            new_tools_config = {}
        new_tools_config["kv_store"] = dict(updated)
        self.config["tools"] = new_tools_config

        self.kv_engine_cache.clear()
        self.write_yaml_callback()
        return dict(updated)

    def get_engine(self, *, adapter_config: Optional[Mapping[str, Any]] = None) -> Any | None:
        override_config: Dict[str, Any] = {}
        if isinstance(adapter_config, Mapping):
            override_config = dict(adapter_config)

        settings = self.get_settings()
        postgres = settings["adapters"].get("postgres", {})

        reuse = coerce_bool_flag(
            override_config.get("reuse_conversation_store"),
            postgres.get("reuse_conversation_store", True),
        )

        url_override = override_config.get("url")
        if reuse and not url_override:
            engine = self.conversation_engine_getter()
            if engine is None:
                return None
            return engine

        pool_settings: Dict[str, Any] = {}
        base_pool = postgres.get("pool")
        if isinstance(base_pool, Mapping):
            pool_settings.update(base_pool)
        override_pool = override_config.get("pool")
        if isinstance(override_pool, Mapping):
            for key, value in override_pool.items():
                if value in (None, ""):
                    continue
                pool_settings[key] = value

        url_value = url_override or postgres.get("url")
        if not url_value:
            raise RuntimeError("KV store PostgreSQL URL is not configured")

        normalized_url = self._normalize_kv_store_url(
            url_value, "tools.kv_store.adapters.postgres.url"
        )
        cache_key = (
            normalized_url,
            pool_settings.get("size"),
            pool_settings.get("max_overflow"),
            pool_settings.get("timeout"),
        )

        engine = self.kv_engine_cache.get(cache_key)
        if engine is not None:
            return engine

        engine_kwargs: Dict[str, Any] = {"future": True}
        if pool_settings.get("size") is not None:
            engine_kwargs["pool_size"] = int(pool_settings["size"])
        if pool_settings.get("max_overflow") is not None:
            engine_kwargs["max_overflow"] = int(pool_settings["max_overflow"])
        if pool_settings.get("timeout") is not None:
            engine_kwargs["pool_timeout"] = float(pool_settings["timeout"])

        engine = self.engine_factory(normalized_url, **engine_kwargs)
        self.kv_engine_cache[cache_key] = engine
        return engine

    # Internal helpers -------------------------------------------------
    def _normalize_kv_store_url(self, url: Any, source: str) -> str:
        normalized = self.normalize_job_store_url(url, source)
        try:
            parsed = self.make_url(normalized)
        except Exception:
            return normalized
        if getattr(parsed, "drivername", "") == "postgresql":
            parsed = parsed.set(drivername="postgresql+psycopg")
        try:
            return parsed.render_as_string(hide_password=False)
        except AttributeError:
            return str(parsed)

    def _normalize_postgres_settings(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        settings: Dict[str, Any] = {
            "reuse_conversation_store": True,
            "namespace_quota_bytes": 1_048_576,
            "global_quota_bytes": None,
            "pool": {},
        }

        url_value = raw.get("url")
        if isinstance(url_value, str) and url_value.strip():
            settings["url"] = url_value.strip()

        settings["reuse_conversation_store"] = coerce_bool_flag(
            raw.get("reuse_conversation_store"),
            True,
        )

        namespace_value = raw.get("namespace_quota_bytes")
        if namespace_value not in (None, ""):
            try:
                settings["namespace_quota_bytes"] = int(namespace_value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid namespace_quota_bytes value %r; expected integer",
                    namespace_value,
                )

        global_value = raw.get("global_quota_bytes")
        if global_value in (None, ""):
            settings["global_quota_bytes"] = None
        else:
            try:
                parsed_global = int(global_value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid global_quota_bytes value %r; expected integer",
                    global_value,
                )
                settings["global_quota_bytes"] = None
            else:
                settings["global_quota_bytes"] = parsed_global if parsed_global > 0 else None

        pool_raw = raw.get("pool")
        normalized_pool: Dict[str, Any] = {}
        if isinstance(pool_raw, Mapping):
            size_value = pool_raw.get("size")
            if size_value not in (None, ""):
                try:
                    normalized_pool["size"] = int(size_value)
                except (TypeError, ValueError):
                    self.logger.warning(
                        "Invalid pool size value %r; expected integer",
                        size_value,
                    )
            overflow_value = pool_raw.get("max_overflow")
            if overflow_value not in (None, ""):
                try:
                    normalized_pool["max_overflow"] = int(overflow_value)
                except (TypeError, ValueError):
                    self.logger.warning(
                        "Invalid pool max_overflow value %r; expected integer",
                        overflow_value,
                    )
            timeout_value = pool_raw.get("timeout")
        if timeout_value not in (None, ""):
            try:
                normalized_pool["timeout"] = float(timeout_value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid pool timeout value %r; expected numeric",
                    timeout_value,
                )

        settings["pool"] = normalized_pool
        return settings

    def _normalize_sqlite_settings(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
        settings: Dict[str, Any] = {
            "reuse_conversation_store": False,
            "namespace_quota_bytes": 1_048_576,
            "global_quota_bytes": None,
        }

        url_value = raw.get("url")
        if isinstance(url_value, str) and url_value.strip():
            settings["url"] = url_value.strip()
        else:
            settings["url"] = "sqlite:///atlas_kv.sqlite"

        settings["reuse_conversation_store"] = coerce_bool_flag(
            raw.get("reuse_conversation_store"),
            False,
        )

        namespace_value = raw.get("namespace_quota_bytes")
        if namespace_value not in (None, ""):
            try:
                settings["namespace_quota_bytes"] = int(namespace_value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid namespace_quota_bytes value %r; expected integer",
                    namespace_value,
                )

        global_value = raw.get("global_quota_bytes")
        if global_value in (None, ""):
            settings["global_quota_bytes"] = None
        else:
            try:
                parsed_global = int(global_value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid global_quota_bytes value %r; expected integer",
                    global_value,
                )
                settings["global_quota_bytes"] = None
            else:
                settings["global_quota_bytes"] = parsed_global if parsed_global > 0 else None

        return settings

@dataclass(kw_only=True)
class ConversationStoreConfigSection:
    """Manage conversation store configuration and lifecycle."""

    config: MutableMapping[str, Any]
    yaml_config: MutableMapping[str, Any]
    env_config: Mapping[str, Any]
    logger: Any
    write_yaml_callback: Callable[[], None]
    default_dsn_by_backend: Mapping[str, str]
    backend_options: Sequence[ConversationStoreBackendOption]
    default_backend_name: str = field(default_factory=default_conversation_store_backend_name)
    create_engine: Callable[..., Any]
    inspect_engine: Callable[..., Any]
    make_url: Callable[..., Any]
    sessionmaker_factory: Callable[..., Any]
    conversation_required_tables: Callable[[], set[str]]

    _conversation_store_verified: bool = False
    _conversation_engine: Any | None = None
    _conversation_session_factory: Any | None = None
    _backend_by_name: Dict[str, ConversationStoreBackendOption] = field(init=False, repr=False)
    _default_dsn_map: Dict[str, str] = field(init=False, repr=False)
    _default_url_generated: bool = False

    def __post_init__(self) -> None:
        if not self.backend_options:
            raise ValueError("At least one conversation backend option must be provided")
        self._backend_by_name = {option.name: option for option in self.backend_options}
        self._default_dsn_map = dict(self.default_dsn_by_backend or {})
        for option in self.backend_options:
            self._default_dsn_map.setdefault(option.name, option.dsn)
        if self.default_backend_name not in self._backend_by_name:
            self.default_backend_name = self.backend_options[0].name
        self._default_url_generated = False

    RETENTION_WORKER_DEFAULTS: ClassVar[Mapping[str, Any]] = {
        "interval_seconds": 3600.0,
        "min_interval_seconds": 60.0,
        "max_interval_seconds": 7200.0,
        "backlog_low_water": 0,
        "backlog_high_water": 50,
        "catchup_multiplier": 0.5,
        "recovery_growth": 1.5,
        "slow_run_threshold": 0.8,
        "fast_run_threshold": 0.3,
        "jitter_seconds": 30.0,
        "error_backoff_factor": 2.0,
        "error_backoff_max_seconds": 21600.0,
    }

    def apply(self) -> None:
        conversation_store_block = self.config.get("conversation_database")
        if not isinstance(conversation_store_block, Mapping):
            conversation_store_block = {}
        else:
            conversation_store_block = dict(conversation_store_block)

        env_backend = self.env_config.get("CONVERSATION_DATABASE_BACKEND")
        backend_value: Any
        if env_backend is not None:
            backend_value = env_backend
        else:
            backend_value = conversation_store_block.get("backend")
        normalized_backend = self._normalize_backend(backend_value)
        if normalized_backend is None:
            normalized_backend = self.default_backend_name
        conversation_store_block["backend"] = normalized_backend

        env_db_url = self.env_config.get("CONVERSATION_DATABASE_URL")
        if env_db_url and not conversation_store_block.get("url"):
            conversation_store_block["url"] = env_db_url

        if not conversation_store_block.get("url"):
            default_dsn = self._default_dsn_for_backend(normalized_backend)
            if default_dsn:
                conversation_store_block["url"] = default_dsn
                self._default_url_generated = True

        pool_block = conversation_store_block.get("pool")
        if not isinstance(pool_block, Mapping):
            pool_block = {}
        else:
            pool_block = dict(pool_block)

        for env_key, setting_key in (
            ("CONVERSATION_DATABASE_POOL_SIZE", "size"),
            ("CONVERSATION_DATABASE_MAX_OVERFLOW", "max_overflow"),
            ("CONVERSATION_DATABASE_POOL_TIMEOUT", "timeout"),
        ):
            value = self.env_config.get(env_key)
            if value is None:
                continue
            try:
                pool_block[setting_key] = int(value)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid %s value %r; expected integer", env_key, value
                )

        conversation_store_block["pool"] = pool_block

        retention_block = conversation_store_block.get("retention")
        if not isinstance(retention_block, Mapping):
            retention_block = {}
        else:
            retention_block = dict(retention_block)

        env_retention_days = self.env_config.get("CONVERSATION_DATABASE_RETENTION_DAYS")
        if env_retention_days is not None:
            try:
                retention_block["days"] = int(env_retention_days)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid CONVERSATION_DATABASE_RETENTION_DAYS value %r",
                    env_retention_days,
                )

        retention_block.setdefault("history_message_limit", 500)
        conversation_store_block["retention"] = retention_block

        worker_block = conversation_store_block.get("retention_worker")
        if not isinstance(worker_block, Mapping):
            worker_block = {}
        else:
            worker_block = dict(worker_block)

        for key, value in self.RETENTION_WORKER_DEFAULTS.items():
            worker_block.setdefault(key, value)

        conversation_store_block["retention_worker"] = worker_block
        self.config["conversation_database"] = conversation_store_block

    # Exposed helpers --------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        block = self.config.get("conversation_database")
        if not isinstance(block, Mapping):
            return {}
        return dict(block)

    def get_backend(self) -> Optional[str]:
        config = self.get_config()
        backend_value = self._normalize_backend(config.get("backend"))
        if backend_value:
            return backend_value

        url_value = config.get("url")
        if isinstance(url_value, str) and url_value.strip():
            inferred = self._normalize_backend(infer_conversation_store_backend(url_value))
            if inferred:
                return inferred

        return None

    def available_backends(self) -> Tuple[ConversationStoreBackendOption, ...]:
        """Return the configured conversation backend defaults."""

        return tuple(self.backend_options)

    def ensure_postgres_store(self) -> str:
        self._conversation_store_verified = False

        config = self.get_config()
        backend_config = self._normalize_backend(config.get("backend"))
        url_value = config.get("url")
        if isinstance(url_value, str):
            url = url_value.strip()
        else:
            url = url_value or ""

        if self._is_mongo_url(url) or backend_config == "mongodb":
            if url:
                self._persist_url(url)
            self._persist_backend("mongodb")
            self._conversation_store_verified = True
            return url

        generated_default = False
        if not url:
            backend_config = backend_config or self.default_backend_name
            url = self._default_dsn_for_backend(backend_config)
            generated_default = True
            self.logger.info(
                "No conversation database URL configured; defaulting to %s",
                url,
            )

        try:
            parsed_url = self.make_url(url)
        except Exception as exc:
            message = f"Invalid conversation database URL {url!r}: {exc}"
            self.logger.error(message)
            raise RuntimeError(message) from exc

        drivername = (parsed_url.drivername or "").lower()
        dialect = drivername.split("+", 1)[0]
        inferred_backend = self._normalize_backend(backend_config)
        if inferred_backend is None:
            inferred_backend = self._normalize_backend(infer_conversation_store_backend(drivername))
        if inferred_backend is None:
            inferred_backend = self._normalize_backend(infer_conversation_store_backend(url))

        backend_option = self._backend_by_name.get(inferred_backend or "")
        if backend_option is not None and dialect != backend_option.dialect:
            message = (
                "Conversation database URL does not match the configured backend; "
                f"expected '{backend_option.dialect}' but received '{parsed_url.drivername}'."
            )
            self.logger.error(message)
            raise RuntimeError(message)

        try:
            engine = self.create_engine(url, future=True)
        except Exception as exc:
            message = (
                "Conversation store verification failed: unable to initialise the SQLAlchemy engine. "
                "Run the standalone setup utility to provision the database."
            )
            self.logger.error(message, exc_info=True)
            raise RuntimeError(message) from exc

        try:
            inspector = self.inspect_engine(engine)
            existing_tables = {name for name in inspector.get_table_names()}
        except Exception as exc:
            engine.dispose()
            message = (
                "Conversation store verification failed: unable to connect to the configured database. "
                "Run the standalone setup utility to provision the database."
            )
            self.logger.error(message, exc_info=True)
            raise RuntimeError(f"{message} (original error: {exc})") from exc

        required_tables = self.conversation_required_tables()
        missing_tables = required_tables.difference(existing_tables)
        engine.dispose()

        if missing_tables:
            missing_list = ", ".join(sorted(missing_tables)) or "unknown"
            message = (
                "Conversation store verification failed: missing required tables "
                f"[{missing_list}]. Run the standalone setup utility to provision the database."
            )
            self.logger.error(message)
            raise RuntimeError(message)

        persisted_backend = inferred_backend or self.default_backend_name
        self._persist_backend(persisted_backend)
        self._persist_url(url)

        if generated_default or self._default_url_generated:
            self.write_yaml_callback()
            self._default_url_generated = False

        self._conversation_store_verified = True
        return url

    def is_verified(self) -> bool:
        return bool(self._conversation_store_verified)

    def get_retention_policies(self) -> Dict[str, Any]:
        config = self.get_config()
        retention = config.get("retention") or {}
        if isinstance(retention, Mapping):
            return dict(retention)
        return {}

    def get_retention_worker_settings(self) -> Dict[str, Any]:
        config = self.get_config()
        block = config.get("retention_worker")
        if not isinstance(block, Mapping):
            block = {}

        settings: Dict[str, Any] = {}
        defaults = dict(self.RETENTION_WORKER_DEFAULTS)

        def _as_float(key: str, value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(defaults[key])

        def _as_int(key: str, value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(defaults[key])

        settings["interval_seconds"] = _as_float("interval_seconds", block.get("interval_seconds"))
        settings["min_interval_seconds"] = _as_float("min_interval_seconds", block.get("min_interval_seconds"))
        settings["max_interval_seconds"] = _as_float("max_interval_seconds", block.get("max_interval_seconds"))
        settings["backlog_low_water"] = _as_int("backlog_low_water", block.get("backlog_low_water"))
        settings["backlog_high_water"] = _as_int("backlog_high_water", block.get("backlog_high_water"))
        settings["catchup_multiplier"] = _as_float("catchup_multiplier", block.get("catchup_multiplier"))
        settings["recovery_growth"] = _as_float("recovery_growth", block.get("recovery_growth"))
        settings["slow_run_threshold"] = _as_float("slow_run_threshold", block.get("slow_run_threshold"))
        settings["fast_run_threshold"] = _as_float("fast_run_threshold", block.get("fast_run_threshold"))
        settings["jitter_seconds"] = _as_float("jitter_seconds", block.get("jitter_seconds"))
        settings["error_backoff_factor"] = _as_float("error_backoff_factor", block.get("error_backoff_factor"))
        settings["error_backoff_max_seconds"] = _as_float(
            "error_backoff_max_seconds",
            block.get("error_backoff_max_seconds"),
        )

        return settings

    def set_retention(
        self,
        *,
        days: Optional[int] = None,
        history_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        block = dict(self.get_config())
        retention = dict(block.get("retention") or {})

        if days is None:
            retention.pop("days", None)
            retention.pop("max_days", None)
        else:
            if int(days) <= 0:
                raise ValueError("Retention days must be a positive integer")
            retention["days"] = int(days)

        if history_limit is None:
            retention.pop("history_message_limit", None)
        else:
            if int(history_limit) <= 0:
                raise ValueError("Conversation history limit must be a positive integer")
            retention["history_message_limit"] = int(history_limit)

        if retention:
            block["retention"] = dict(retention)
        else:
            block.pop("retention", None)

        conversation_block = dict(block)
        self.yaml_config.setdefault("conversation_database", {})
        yaml_block = dict(self.yaml_config["conversation_database"])
        yaml_block.update(conversation_block)
        self.yaml_config["conversation_database"] = yaml_block
        self.config["conversation_database"] = dict(yaml_block)
        self.write_yaml_callback()
        return dict(yaml_block.get("retention", {}))

    def get_engine(self) -> Any | None:
        if self._conversation_engine is None:
            self.get_session_factory()
        return self._conversation_engine

    def get_session_factory(self) -> Any | None:
        if self._conversation_session_factory is not None:
            return self._conversation_session_factory

        engine, factory = self._build_session_factory()
        self._conversation_engine = engine
        self._conversation_session_factory = factory
        return factory

    # Internal helpers -------------------------------------------------
    def _persist_url(self, url: str) -> None:
        if not isinstance(url, str) or not url:
            return

        config_block = self.config.get("conversation_database")
        yaml_block = self.yaml_config.get("conversation_database")

        if isinstance(config_block, Mapping):
            updated_config_block = dict(config_block)
        elif isinstance(yaml_block, Mapping):
            updated_config_block = dict(yaml_block)
        else:
            updated_config_block = {}
        updated_config_block["url"] = url
        self.config["conversation_database"] = updated_config_block

        if isinstance(yaml_block, Mapping):
            updated_yaml_block = dict(yaml_block)
        elif isinstance(config_block, Mapping):
            updated_yaml_block = dict(config_block)
        else:
            updated_yaml_block = {}
        updated_yaml_block["url"] = url
        self.yaml_config["conversation_database"] = updated_yaml_block

    def _persist_backend(self, backend: str | None) -> None:
        if not backend:
            return

        config_block = self.config.get("conversation_database")
        yaml_block = self.yaml_config.get("conversation_database")

        if isinstance(config_block, Mapping):
            updated_config_block = dict(config_block)
        elif isinstance(yaml_block, Mapping):
            updated_config_block = dict(yaml_block)
        else:
            updated_config_block = {}
        updated_config_block["backend"] = backend
        self.config["conversation_database"] = updated_config_block

        if isinstance(yaml_block, Mapping):
            updated_yaml_block = dict(yaml_block)
        elif isinstance(config_block, Mapping):
            updated_yaml_block = dict(config_block)
        else:
            updated_yaml_block = {}
        updated_yaml_block["backend"] = backend
        self.yaml_config["conversation_database"] = updated_yaml_block

    def _default_dsn_for_backend(self, backend: str | None) -> str:
        if backend and backend in self._backend_by_name:
            mapped = self._default_dsn_map.get(backend)
            if mapped:
                return mapped
            return self._backend_by_name[backend].dsn
        fallback_backend = self.default_backend_name
        mapped = self._default_dsn_map.get(fallback_backend)
        if mapped:
            return mapped
        if fallback_backend in self._backend_by_name:
            return self._backend_by_name[fallback_backend].dsn
        if self._default_dsn_map:
            return next(iter(self._default_dsn_map.values()))
        first_option = next(iter(self.backend_options), None)
        return first_option.dsn if first_option is not None else ""

    def _normalize_backend(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            candidate = value.strip().lower()
            if not candidate:
                return None
            if candidate in self._backend_by_name:
                return candidate
            inferred = infer_conversation_store_backend(candidate)
            if inferred and inferred in self._backend_by_name:
                return inferred
            return candidate
        return None

    @staticmethod
    def _is_mongo_url(value: str | None) -> bool:
        if not value:
            return False
        normalized = value.strip().lower()
        return normalized.startswith("mongodb://") or normalized.startswith("mongodb+srv://")

    def _build_session_factory(self) -> tuple[Any | None, Any | None]:
        config = self.get_config()
        backend = self._normalize_backend(config.get("backend"))
        url_value = config.get("url")
        if isinstance(url_value, str):
            url = url_value.strip()
        else:
            url = url_value or ""

        if not url:
            backend = backend or self.default_backend_name
            url = self._default_dsn_for_backend(backend)

        if self._is_mongo_url(url) or backend == "mongodb":
            return self._build_mongo_backend(url)

        ensured_url = self.ensure_postgres_store()
        url = ensured_url

        try:
            parsed_url = self.make_url(url)
        except Exception as exc:
            message = f"Invalid conversation database URL {url!r}: {exc}"
            self.logger.error(message)
            raise RuntimeError(message) from exc

        drivername = (parsed_url.drivername or "").lower()
        dialect = drivername.split("+", 1)[0]
        backend = self._normalize_backend(config.get("backend"))
        if backend is None:
            backend = self._normalize_backend(infer_conversation_store_backend(drivername))

        backend_option = self._backend_by_name.get(backend or "")
        if backend_option is not None and dialect != backend_option.dialect:
            message = (
                "Conversation database URL does not match the configured backend; "
                f"expected '{backend_option.dialect}' but received '{parsed_url.drivername}'."
            )
            self.logger.error(message)
            raise RuntimeError(message)

        pool_config = config.get("pool") or {}
        engine_kwargs: Dict[str, Any] = {}
        if backend_option is not None and backend_option.dialect == "postgresql":
            if isinstance(pool_config, Mapping):
                size = pool_config.get("size")
                if size is not None:
                    engine_kwargs["pool_size"] = int(size)
                max_overflow = pool_config.get("max_overflow")
                if max_overflow is not None:
                    engine_kwargs["max_overflow"] = int(max_overflow)
                timeout = pool_config.get("timeout")
                if timeout is not None:
                    engine_kwargs["pool_timeout"] = int(timeout)

        engine = self.create_engine(url, future=True, **engine_kwargs)
        factory = self.sessionmaker_factory(bind=engine, future=True)
        return engine, factory

    def _build_mongo_backend(self, url: str) -> tuple[Any | None, MongoConversationStoreRepository | None]:
        normalized_url = url.strip()
        if not normalized_url:
            normalized_url = self._default_dsn_for_backend("mongodb")

        if not normalized_url:
            self.logger.error("MongoDB backend requires a configured URL")
            return None, None

        try:  # pragma: no cover - optional dependency path
            from pymongo import MongoClient
        except Exception as exc:
            self.logger.warning("PyMongo driver unavailable: %s", exc)
            return None, None

        try:
            client = MongoClient(normalized_url)
        except Exception as exc:
            message = (
                "Conversation store verification failed: unable to connect to the configured MongoDB database."
            )
            self.logger.error(message, exc_info=True)
            raise RuntimeError(message) from exc

        database_name = urlparse(normalized_url).path.lstrip("/").split("?", 1)[0] or "atlas"
        database = client.get_database(database_name)
        repository = MongoConversationStoreRepository.from_database(database, client=client)
        try:
            repository.ensure_indexes()
        except RuntimeError:
            client.close()
            raise
        except Exception as exc:
            self.logger.warning("Failed to ensure MongoDB indexes: %s", exc)

        self._persist_backend("mongodb")
        self._persist_url(normalized_url)
        self._conversation_store_verified = True
        return client, repository
