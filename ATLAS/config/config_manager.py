# ATLAS/config.py

from __future__ import annotations

import copy
import json
import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional dependency handling for test environments
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.engine import Engine
    from sqlalchemy.engine.url import make_url
    from sqlalchemy.orm import sessionmaker
except Exception:  # pragma: no cover - lightweight fallbacks when SQLAlchemy is absent
    class Engine:  # type: ignore[assignment]
        pass

    def create_engine(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy create_engine is unavailable in this environment")

    def inspect(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy inspect is unavailable in this environment")

    def make_url(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy make_url is unavailable in this environment")

    class _Sessionmaker:  # pragma: no cover - placeholder mirroring SQLAlchemy API
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("SQLAlchemy sessionmaker is unavailable in this environment")

    sessionmaker = _Sessionmaker  # type: ignore[assignment]
else:  # pragma: no cover - sanitize stubbed implementations
    if not isinstance(sessionmaker, type):  # pragma: no cover - compatibility with stubs
        class _Sessionmaker:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("SQLAlchemy sessionmaker is unavailable in this environment")

        sessionmaker = _Sessionmaker  # type: ignore[assignment]

from .messaging import MessagingConfigSection
from .persistence import KV_STORE_UNSET, PersistenceConfigSection
from .providers import ProviderConfigSections
from .tooling import ToolingConfigSection
from modules.orchestration.message_bus import (
    InMemoryQueueBackend,
    MessageBus,
    RedisStreamBackend,
    configure_message_bus,
)
from modules.job_store import JobService
from modules.job_store.repository import JobStoreRepository
from modules.task_store import TaskService, TaskStoreRepository
from modules.Tools.Base_Tools.task_queue import (
    TaskQueueService,
    get_default_task_queue_service,
)
from urllib.parse import urlparse

_UNSET = object()

_DEFAULT_CONVERSATION_STORE_DSN = (
    "postgresql+psycopg://atlas:atlas@localhost:5432/atlas"
)
from modules.logging.logger import setup_logger
from dotenv import load_dotenv, set_key, find_dotenv
import yaml


if TYPE_CHECKING:
    from modules.orchestration.job_manager import JobManager
    from modules.orchestration.job_scheduler import JobScheduler

class ConfigManager:
    UNSET = _UNSET
    """
    Manages configuration settings for the application, including loading
    environment variables and handling API keys for various providers.
    """

    _DEFAULT_HUGGINGFACE_GENERATION_SETTINGS: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 50,
        "max_tokens": 100,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "early_stopping": False,
        "do_sample": False,
    }

    def __init__(self):
        """
        Initializes the ConfigManager by loading environment variables and loading configuration settings.

        Notes:
            If the default provider credential is missing the manager records a warning instead of
            raising an exception so the application can continue initializing.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Setup logger early to log any issues
        self.logger = setup_logger(__name__)
        
        # Load configurations from .env and config.yaml
        self.env_config = self._load_env_config()
        self._yaml_path = self._compute_yaml_path()
        self.yaml_config = self._load_yaml_config()
        
        # Merge configurations, with YAML config overriding env config if there's overlap
        self.config = {**self.env_config, **self.yaml_config}

        # Normalize any persisted provider model cache to ensure predictable structure.
        self._model_cache: Dict[str, List[str]] = self._normalize_model_cache(
            self.yaml_config.get("MODEL_CACHE")
        )
        self.config["MODEL_CACHE"] = copy.deepcopy(self._model_cache)
        if "MODEL_CACHE" not in self.yaml_config:
            self.yaml_config["MODEL_CACHE"] = copy.deepcopy(self._model_cache)

        # --- Tooling defaults & sandbox configuration -----------------
        self.tooling = ToolingConfigSection(
            config=self.config,
            yaml_config=self.yaml_config,
            env_config=self.env_config,
            logger=self.logger,
        )
        self.tooling.apply()

        # --- Persistence (KV + conversation stores) -------------------
        self.persistence = PersistenceConfigSection(
            config=self.config,
            yaml_config=self.yaml_config,
            env_config=self.env_config,
            logger=self.logger,
            normalize_job_store_url=self._normalize_job_store_url,
            write_yaml_callback=self._write_yaml_config,
            create_engine=create_engine,
            inspect_engine=inspect,
            make_url=make_url,
            sessionmaker_factory=sessionmaker,
            conversation_required_tables=self._conversation_store_required_tables,
            default_conversation_dsn=_DEFAULT_CONVERSATION_STORE_DSN,
        )
        self.persistence.apply()

        # --- Task queue defaults --------------------------------------
        queue_block = self.config.get('task_queue')
        if not isinstance(queue_block, Mapping):
            queue_block = {}
        else:
            queue_block = dict(queue_block)

        job_store_candidate = queue_block.get('jobstore_url')
        if isinstance(job_store_candidate, str) and job_store_candidate.strip():
            try:
                queue_block['jobstore_url'] = self._normalize_job_store_url(
                    job_store_candidate,
                    'task_queue.jobstore_url',
                )
            except ValueError as exc:
                self.logger.warning("Ignoring invalid task queue job store URL: %s", exc)
                queue_block.pop('jobstore_url', None)

        if 'jobstore_url' not in queue_block:
            try:
                default_job_store = self.get_job_store_url(require=False)
            except RuntimeError:
                default_job_store = ''
            if default_job_store:
                queue_block['jobstore_url'] = default_job_store

        if queue_block:
            self.config['task_queue'] = queue_block
        else:
            self.config.pop('task_queue', None)

        # --- Messaging backend defaults -------------------------------
        self.messaging = MessagingConfigSection(
            config=self.config,
            yaml_config=self.yaml_config,
            env_config=self.env_config,
            logger=self.logger,
            write_yaml_callback=self._write_yaml_config,
        )
        self.messaging.apply()

        # Provider-specific sections manage their own persistence concerns
        self.providers = ProviderConfigSections(manager=self)
        self.providers.apply()
        huggingface_block = self.yaml_config.get("HUGGINGFACE")
        if isinstance(huggingface_block, Mapping):
            self.config["HUGGINGFACE"] = dict(huggingface_block)

        # Derive other paths from APP_ROOT
        app_root = self.config.get('APP_ROOT', '.')
        self.config['MODEL_CACHE_DIR'] = os.path.join(
            app_root,
            'modules',
            'Providers',
            'HuggingFace',
            'model_cache'
        )
        self.config.setdefault(
            'SPEECH_CACHE_DIR',
            os.path.join(app_root, 'assets', 'SCOUT', 'tts_mp3')
        )
        # Ensure the model_cache directory exists
        os.makedirs(self.config['MODEL_CACHE_DIR'], exist_ok=True)

        self.providers.initialize_pending_warnings()

        self._message_bus: Optional[MessageBus] = None
        self._task_session_factory: sessionmaker | None = None
        self._task_repository: TaskStoreRepository | None = None
        self._task_service: TaskService | None = None
        self._job_repository: JobStoreRepository | None = None
        self._job_service: JobService | None = None
        self._task_queue_service: TaskQueueService | None = None
        self._job_manager: "JobManager" | None = None
        self._job_scheduler: "JobScheduler" | None = None

    def _normalize_network_allowlist(self, value):
        """Return a sanitized allowlist for sandboxed tool networking."""

        if value is None or value is False:
            return None

        if isinstance(value, str):
            candidate = value.strip()
            return [candidate] if candidate else None

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            normalized = []
            for item in value:
                host = str(item).strip()
                if host:
                    normalized.append(host)
            return normalized or None

        return None

    def _normalize_model_cache(self, value: Any) -> Dict[str, List[str]]:
        """Normalize persisted provider model caches into a predictable mapping."""

        normalized: Dict[str, List[str]] = {}

        if isinstance(value, Mapping):
            items = value.items()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            # Support legacy list-of-tuples structures if encountered.
            items = []
            for entry in value:
                if isinstance(entry, Sequence) and len(entry) == 2:
                    items.append((entry[0], entry[1]))
        else:
            items = []

        for provider, models in items:
            if not isinstance(provider, str):
                try:
                    provider_key = str(provider)
                except Exception:
                    continue
            else:
                provider_key = provider

            provider_key = provider_key.strip()
            if not provider_key:
                continue

            seen: set[str] = set()
            ordered: List[str] = []

            candidate_iterable: Iterable[Any]
            if isinstance(models, Mapping):
                candidate_iterable = models.values()
            elif isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
                candidate_iterable = models
            else:
                candidate_iterable = []

            for entry in candidate_iterable:
                if isinstance(entry, str):
                    candidate = entry.strip()
                else:
                    candidate = str(entry).strip() if entry is not None else ""

                if not candidate or candidate in seen:
                    continue

                ordered.append(candidate)
                seen.add(candidate)

            normalized[provider_key] = ordered

        return normalized

    def get_conversation_database_config(self) -> Dict[str, Any]:
        """Return the merged configuration block for the conversation database."""

        return self.persistence.conversation.get_config()

    def ensure_postgres_conversation_store(self) -> str:
        """Verify the configured PostgreSQL conversation store without provisioning it."""

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

    # --- Service factory helpers -------------------------------------
    def get_conversation_store_engine(self) -> Engine | None:
        """Return a configured SQLAlchemy engine for the conversation store."""

        return self.persistence.conversation.get_engine()

    def get_conversation_store_session_factory(self) -> sessionmaker | None:
        """Return a configured session factory for the conversation store."""

        return self.persistence.conversation.get_session_factory()

    def get_task_store_session_factory(self) -> sessionmaker | None:
        """Return a session factory for task persistence that shares the conversation engine."""

        if self._task_session_factory is not None:
            return self._task_session_factory

        conversation_factory = self.get_conversation_store_session_factory()
        if conversation_factory is None:
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

    def get_job_repository(self) -> JobStoreRepository | None:
        """Return a configured repository for scheduled job persistence."""

        if self._job_repository is not None:
            return self._job_repository

        factory = self.get_task_store_session_factory()
        if factory is None:
            return None

        repository = JobStoreRepository(factory)
        try:
            repository.create_schema()
        except Exception as exc:  # pragma: no cover - defensive logging only
            self.logger.warning("Failed to initialize job store schema: %s", exc)
        self._job_repository = repository
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

    # --- Provider fallback configuration -----------------------------
    def get_llm_fallback_config(self) -> Dict[str, Any]:
        """Return the configured fallback provider settings with sensible defaults."""

        return self.providers.get_llm_fallback_config()

    def get_messaging_settings(self) -> Dict[str, Any]:
        """Return the configured messaging backend settings."""

        return self.messaging.get_settings()

    def set_messaging_settings(
        self,
        *,
        backend: str,
        redis_url: Optional[str] = None,
        stream_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist messaging backend preferences and clear cached bus instances."""

        block = self.messaging.set_settings(
            backend=backend,
            redis_url=redis_url,
            stream_prefix=stream_prefix,
        )
        self._message_bus = None
        return dict(block)

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



    def get_kv_store_engine(self, *, adapter_config: Optional[Mapping[str, Any]] = None) -> Engine | None:
        """Return a SQLAlchemy engine configured for the KV store."""

        return self.persistence.kv_store.get_engine(adapter_config=adapter_config)

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
                updated['queue_size'] = int(queue_size)

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

        try:
            return self._normalize_job_store_url(
                _DEFAULT_CONVERSATION_STORE_DSN,
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

    def configure_message_bus(self) -> MessageBus:
        """Instantiate and configure the global message bus."""

        if self._message_bus is not None:
            return self._message_bus

        settings = self.get_messaging_settings()
        backend_name = str(settings.get('backend', 'in_memory')).lower()
        backend: Optional[Any]
        if backend_name == 'redis':
            redis_url = settings.get('redis_url')
            stream_prefix = settings.get('stream_prefix', 'atlas_bus')
            try:
                backend = RedisStreamBackend(str(redis_url), stream_prefix=str(stream_prefix))
            except Exception as exc:  # pragma: no cover - redis optional dependency
                self.logger.warning(
                    "Falling back to in-memory message bus backend due to Redis configuration error: %s",
                    exc,
                )
                backend = InMemoryQueueBackend()
        else:
            backend = InMemoryQueueBackend()

        self._message_bus = configure_message_bus(backend)
        return self._message_bus

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
        if not scheme.startswith('postgresql'):
            raise ValueError(
                f"{source} must use the 'postgresql' dialect; received '{parsed.scheme or 'missing'}'"
            )

        return candidate



    def _get_provider_env_keys(self) -> Dict[str, str]:
        """Return the mapping between provider display names and environment keys."""
        return self.providers.get_env_keys()

    def _compute_yaml_path(self) -> str:
        """Return the absolute path to the persistent YAML configuration file."""

        return os.path.join(
            self.env_config.get('APP_ROOT', '.'),
            'ATLAS',
            'config',
            'atlas_config.yaml',
        )

    def _load_env_config(self) -> Dict[str, Any]:
        """
        Loads environment variables into the configuration dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing all loaded environment configuration settings.
        """
        config = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'DEFAULT_PROVIDER': os.getenv('DEFAULT_PROVIDER', 'OpenAI'),
            'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'gpt-4o'),
            'MISTRAL_API_KEY': os.getenv('MISTRAL_API_KEY'),
            'MISTRAL_BASE_URL': os.getenv('MISTRAL_BASE_URL'),
            'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'GROK_API_KEY': os.getenv('GROK_API_KEY'),
            'XI_API_KEY': os.getenv('XI_API_KEY'),
            'OPENAI_BASE_URL': os.getenv('OPENAI_BASE_URL'),
            'OPENAI_ORGANIZATION': os.getenv('OPENAI_ORGANIZATION'),
            'JAVASCRIPT_EXECUTOR_BIN': os.getenv('JAVASCRIPT_EXECUTOR_BIN'),
            'JAVASCRIPT_EXECUTOR_ARGS': os.getenv('JAVASCRIPT_EXECUTOR_ARGS'),
            'APP_ROOT': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        }
        self.logger.debug("APP_ROOT is set to: %s", config["APP_ROOT"])
        return config

    def _load_yaml_config(self) -> Dict[str, Any]:
        """
        Loads configuration settings from the config.yaml file.

        Returns:
            Dict[str, Any]: A dictionary containing all loaded YAML configuration settings.
        """
        yaml_path = getattr(self, '_yaml_path', None) or self._compute_yaml_path()

        if not os.path.exists(yaml_path):
            self.logger.error(f"Configuration file not found: {yaml_path}")
            return {}
        
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file) or {}
                self.logger.debug("Loaded configuration from %s", yaml_path)
                return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {yaml_path}: {e}")
            return {}

    def _persist_env_value(self, env_key: str, value: Optional[str]):
        """Persist an environment-backed configuration value."""

        env_path = find_dotenv()
        if not env_path:
            app_root = None
            if isinstance(getattr(self, "env_config", None), Mapping):
                app_root = self.env_config.get("APP_ROOT")
            if not app_root:
                app_root = os.getenv("APP_ROOT")
            if not app_root:
                app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            env_path = os.path.abspath(os.path.join(app_root, ".env"))

            os.makedirs(os.path.dirname(env_path), exist_ok=True)
            if not os.path.exists(env_path):
                with open(env_path, "a", encoding="utf-8"):
                    pass

        # Persist the value to the .env file and refresh the loaded environment.
        set_key(env_path, env_key, value or "")
        load_dotenv(env_path, override=True)

        # Synchronize in-memory state and environment variables.
        if value is None or value == "":
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = value

        self.env_config[env_key] = value
        if value is None:
            self.config.pop(env_key, None)
        else:
            self.config[env_key] = value

        providers = getattr(self, "providers", None)
        if providers is not None:
            providers.sync_provider_warning(env_key, value)

    def set_google_credentials(self, credentials_path: str):
        """Persist Google application credentials and refresh process state."""

        self.providers.google.set_credentials(credentials_path)

    def get_pending_provider_warnings(self) -> Dict[str, str]:
        """Return provider credential warnings that should be surfaced to operators."""
        return self.providers.get_pending_provider_warnings()

    def is_default_provider_ready(self) -> bool:
        """Return True when the configured default provider has a usable credential."""
        return self.providers.is_default_provider_ready()

    def get_google_speech_settings(self) -> Dict[str, Any]:
        """Return persisted Google speech preferences when available."""
        return self.providers.google.get_speech_settings()

    def set_google_speech_settings(
        self,
        *,
        tts_voice: Optional[str] = None,
        stt_language: Optional[str] = None,
        auto_punctuation: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Persist Google speech preferences to the YAML configuration."""
        return self.providers.google.set_speech_settings(
            tts_voice=tts_voice,
            stt_language=stt_language,
            auto_punctuation=auto_punctuation,
        )

    def set_hf_token(self, token: str):
        """Persist the Hugging Face access token."""

        if not token:
            raise ValueError("Hugging Face token cannot be empty.")

        self._persist_env_value("HUGGINGFACE_API_KEY", token)
        self.logger.info("Hugging Face token updated.")

    def set_elevenlabs_api_key(self, api_key: str):
        """Persist the ElevenLabs access token."""

        if not api_key:
            raise ValueError("ElevenLabs API key cannot be empty.")

        self._persist_env_value("XI_API_KEY", api_key)
        self.logger.info("ElevenLabs API key updated.")

    def set_openai_speech_config(
        self,
        *,
        api_key: Optional[str] = None,
        stt_provider: Optional[str] = None,
        tts_provider: Optional[str] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ):
        """Persist OpenAI speech configuration values."""
        self.providers.openai.set_speech_config(
            api_key=api_key,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
        )

    def set_openai_llm_settings(
        self,
        *,
        model: Optional[str],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[str] = None,
        enable_code_interpreter: Optional[bool] = None,
        enable_file_search: Optional[bool] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        json_mode: Optional[Any] = None,
        json_schema: Optional[Any] = None,
        audio_enabled: Optional[bool] = None,
        audio_voice: Optional[str] = None,
        audio_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist OpenAI chat-completion defaults and related metadata."""
        return self.providers.openai.set_llm_settings(
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            stream=stream,
            function_calling=function_calling,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            enable_code_interpreter=enable_code_interpreter,
            enable_file_search=enable_file_search,
            base_url=base_url,
            organization=organization,
            reasoning_effort=reasoning_effort,
            json_mode=json_mode,
            json_schema=json_schema,
            audio_enabled=audio_enabled,
            audio_voice=audio_voice,
            audio_format=audio_format,
        )


    def get_google_llm_settings(self) -> Dict[str, Any]:
        """Return the persisted Google LLM defaults, if configured."""

        return self.providers.google.get_llm_settings()


    def set_google_llm_settings(
        self,
        *,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[Any] = None,
        candidate_count: Optional[Any] = None,
        max_output_tokens: Optional[Any] = None,
        stop_sequences: Optional[Any] = None,
        safety_settings: Optional[Any] = None,
        response_mime_type: Optional[str] = None,
        system_instruction: Optional[str] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        function_call_mode: Optional[str] = None,
        allowed_function_names: Optional[Any] = None,
        response_schema: Optional[Any] = None,
        cached_allowed_function_names: Any = _UNSET,
        seed: Optional[Any] = None,
        response_logprobs: Optional[bool] = None,
        base_url: Any = _UNSET,
    ) -> Dict[str, Any]:
        """Persist default configuration for the Google Gemini provider.

        Args:
            model: Default Gemini model identifier.
        """

        return self.providers.google.set_llm_settings(
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
            safety_settings=safety_settings,
            response_mime_type=response_mime_type,
            system_instruction=system_instruction,
            stream=stream,
            function_calling=function_calling,
            function_call_mode=function_call_mode,
            allowed_function_names=allowed_function_names,
            response_schema=response_schema,
            cached_allowed_function_names=cached_allowed_function_names,
            seed=seed,
            response_logprobs=response_logprobs,
            base_url=base_url,
        )

    def get_config(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by key with an optional default."""

        return self.config.get(key, default)

    def get_default_provider(self) -> Optional[str]:
        """Return the configured default provider name, if set."""

        value = self.get_config("DEFAULT_PROVIDER")
        if isinstance(value, str):
            token = value.strip()
            return token or None
        return None

    def get_default_model(self) -> Optional[str]:
        """Return the configured default LLM model identifier, if set."""

        value = self.get_config("DEFAULT_MODEL")
        if isinstance(value, str):
            token = value.strip()
            return token or None
        return None

    def set_default_provider(self, provider: Optional[str]) -> Optional[str]:
        """Persist the default provider across configuration stores."""

        normalized = provider.strip() if isinstance(provider, str) else None
        if not normalized:
            normalized = None

        if normalized is None:
            self.yaml_config.pop("DEFAULT_PROVIDER", None)
            self.config.pop("DEFAULT_PROVIDER", None)
        else:
            self.yaml_config["DEFAULT_PROVIDER"] = normalized
            self.config["DEFAULT_PROVIDER"] = normalized

        self._persist_env_value("DEFAULT_PROVIDER", normalized)
        self.env_config["DEFAULT_PROVIDER"] = normalized
        self._write_yaml_config()
        return normalized

    def set_default_model(self, model: Optional[str]) -> Optional[str]:
        """Persist the default model selection across configuration stores."""

        normalized = model.strip() if isinstance(model, str) else None
        if not normalized:
            normalized = None

        if normalized is None:
            self.yaml_config.pop("DEFAULT_MODEL", None)
            self.config.pop("DEFAULT_MODEL", None)
        else:
            self.yaml_config["DEFAULT_MODEL"] = normalized
            self.config["DEFAULT_MODEL"] = normalized

        self._persist_env_value("DEFAULT_MODEL", normalized)
        self.env_config["DEFAULT_MODEL"] = normalized
        self._write_yaml_config()
        return normalized


    def get_mistral_llm_settings(self) -> Dict[str, Any]:
        """Return persisted defaults for the Mistral chat provider."""

        return self.providers.mistral.get_llm_settings()


    def set_mistral_llm_settings(
        self,
        *,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        safe_prompt: Optional[bool] = None,
        stream: Optional[bool] = None,
        random_seed: Optional[Any] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        stop_sequences: Any = _UNSET,
        json_mode: Optional[Any] = None,
        json_schema: Optional[Any] = None,
        max_retries: Optional[int] = None,
        retry_min_seconds: Optional[int] = None,
        retry_max_seconds: Optional[int] = None,
        base_url: Any = _UNSET,
        prompt_mode: Any = _UNSET,
    ) -> Dict[str, Any]:
        """Persist default configuration for the Mistral chat provider."""

        return self.providers.mistral.set_llm_settings(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            safe_prompt=safe_prompt,
            stream=stream,
            random_seed=random_seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stop_sequences=stop_sequences,
            json_mode=json_mode,
            json_schema=json_schema,
            max_retries=max_retries,
            retry_min_seconds=retry_min_seconds,
            retry_max_seconds=retry_max_seconds,
            base_url=base_url,
            prompt_mode=prompt_mode,
        )


    def get_huggingface_generation_settings(self) -> Dict[str, Any]:
        """Return persisted Hugging Face generation defaults."""

        defaults = copy.deepcopy(self._DEFAULT_HUGGINGFACE_GENERATION_SETTINGS)
        block = self.config.get("HUGGINGFACE")
        stored: Optional[Mapping[str, Any]]
        if isinstance(block, Mapping):
            stored = block.get("generation_settings")  # type: ignore[assignment]
        else:
            stored = None

        if isinstance(stored, Mapping):
            for key, value in stored.items():
                if key in defaults:
                    defaults[key] = value

        return defaults

    def set_huggingface_generation_settings(self, settings: Mapping[str, Any]) -> Dict[str, Any]:
        """Persist Hugging Face generation defaults with validation."""

        if not isinstance(settings, Mapping):
            raise ValueError("Settings must be provided as a mapping")

        normalized = copy.deepcopy(self.get_huggingface_generation_settings())

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized_str = value.strip().lower()
                if normalized_str in {"1", "true", "yes", "on"}:
                    return True
                if normalized_str in {"0", "false", "no", "off"}:
                    return False
            return bool(value)

        def _normalize_value(key: str, value: Any) -> Any:
            if value is None:
                raise ValueError(f"{key.replace('_', ' ').title()} cannot be None")

            if key == "temperature":
                numeric = float(value)
                if not 0.0 <= numeric <= 2.0:
                    raise ValueError("Temperature must be between 0.0 and 2.0")
                return numeric
            if key == "top_p":
                numeric = float(value)
                if not 0.0 <= numeric <= 1.0:
                    raise ValueError("Top-p must be between 0.0 and 1.0")
                return numeric
            if key == "top_k":
                integer = int(value)
                if integer < 0:
                    raise ValueError("Top-k must be greater than or equal to 0")
                return integer
            if key == "max_tokens":
                integer = int(value)
                if integer <= 0:
                    raise ValueError("Max tokens must be a positive integer")
                return integer
            if key in {"presence_penalty", "frequency_penalty"}:
                numeric = float(value)
                if not -2.0 <= numeric <= 2.0:
                    raise ValueError(
                        f"{key.replace('_', ' ').title()} must be between -2.0 and 2.0"
                    )
                return numeric
            if key == "repetition_penalty":
                numeric = float(value)
                if numeric <= 0:
                    raise ValueError("Repetition penalty must be greater than 0")
                return numeric
            if key == "length_penalty":
                numeric = float(value)
                if numeric < 0:
                    raise ValueError("Length penalty must be non-negative")
                return numeric
            if key in {"early_stopping", "do_sample"}:
                return _coerce_bool(value)

            raise ValueError(f"Unsupported Hugging Face setting '{key}'")

        for key in self._DEFAULT_HUGGINGFACE_GENERATION_SETTINGS:
            if key not in settings:
                continue
            normalized[key] = _normalize_value(key, settings[key])

        block = self.yaml_config.get("HUGGINGFACE")
        if isinstance(block, Mapping):
            block_dict: Dict[str, Any] = dict(block)
        else:
            block_dict = {}

        block_dict["generation_settings"] = copy.deepcopy(normalized)
        self.yaml_config["HUGGINGFACE"] = copy.deepcopy(block_dict)
        self.config["HUGGINGFACE"] = copy.deepcopy(block_dict)

        self._write_yaml_config()

        return copy.deepcopy(normalized)


    def set_anthropic_settings(
        self,
        *,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Any = _UNSET,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None,
        stop_sequences: Any = _UNSET,
        tool_choice: Any = _UNSET,
        tool_choice_name: Any = _UNSET,
        metadata: Any = _UNSET,
        thinking: Optional[bool] = None,
        thinking_budget: Any = _UNSET,
    ) -> Dict[str, Any]:
        """Persist Anthropic defaults while validating incoming payloads."""

        defaults = {
            'model': 'claude-3-opus-20240229',
            'stream': True,
            'function_calling': False,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': None,
            'max_output_tokens': None,
            'timeout': 60,
            'max_retries': 3,
            'retry_delay': 5,
            'stop_sequences': [],
            'tool_choice': 'auto',
            'tool_choice_name': None,
            'metadata': {},
            'thinking': False,
            'thinking_budget': None,
        }

        settings_block = dict(defaults)
        existing = self.yaml_config.get('ANTHROPIC_LLM')
        if isinstance(existing, dict):
            for key in settings_block.keys():
                if key in existing and existing[key] is not None:
                    try:
                        if key == 'top_k':
                            settings_block[key] = self._coerce_optional_bounded_int(
                                existing[key],
                                field='Top-k',
                                minimum=1,
                                maximum=500,
                            )
                        elif key == 'stop_sequences':
                            settings_block[key] = self._coerce_stop_sequences(existing[key])
                        elif key == 'metadata':
                            settings_block[key] = self._coerce_metadata(existing[key])
                        else:
                            settings_block[key] = existing[key]
                    except ValueError as exc:  # pragma: no cover - defensive logging
                        self.logger.warning(
                            "Ignoring persisted Anthropic %s override: %s",
                            key,
                            exc,
                        )

        def _normalize_model(value: Optional[str], previous: Optional[str]) -> str:
            if value is None:
                return previous or defaults['model']
            if not isinstance(value, str):
                raise ValueError("Model must be provided as a string.")
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("Model cannot be empty.")
            return cleaned

        def _normalize_bool(value: Optional[bool], previous: bool) -> bool:
            if value is None:
                return bool(previous)
            return bool(value)

        def _normalize_float(
            value: Optional[Any],
            previous: float,
            *,
            field: str,
            minimum: float,
            maximum: float,
        ) -> float:
            candidate: Optional[float]
            if value is None:
                candidate = float(previous)
            else:
                try:
                    candidate = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{field} must be a number.") from exc
            if candidate is None or not minimum <= candidate <= maximum:
                raise ValueError(
                    f"{field} must be between {minimum} and {maximum}."
                )
            return candidate

        def _normalize_optional_int(
            value: Optional[Any],
            previous: Optional[int],
            *,
            field: str,
            minimum: int,
        ) -> Optional[int]:
            if value is None:
                candidate = previous
            elif value == "":
                candidate = None
            else:
                try:
                    candidate = int(value)  # type: ignore[assignment]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{field} must be an integer or left blank.") from exc
            if candidate is None:
                return None
            if candidate < minimum:
                raise ValueError(f"{field} must be >= {minimum}.")
            return candidate

        def _normalize_int(
            value: Optional[Any],
            previous: int,
            *,
            field: str,
            minimum: int,
        ) -> int:
            if value is None:
                return int(previous)
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be an integer.") from exc
            if parsed < minimum:
                raise ValueError(f"{field} must be >= {minimum}.")
            return parsed

        settings_block['model'] = _normalize_model(model, settings_block.get('model'))
        settings_block['stream'] = _normalize_bool(stream, settings_block.get('stream', True))
        settings_block['function_calling'] = _normalize_bool(
            function_calling,
            settings_block.get('function_calling', False),
        )
        settings_block['temperature'] = _normalize_float(
            temperature,
            float(settings_block.get('temperature', defaults['temperature'])),
            field='Temperature',
            minimum=0.0,
            maximum=1.0,
        )
        settings_block['top_p'] = _normalize_float(
            top_p,
            float(settings_block.get('top_p', defaults['top_p'])),
            field='Top-p',
            minimum=0.0,
            maximum=1.0,
        )
        if top_k is not _UNSET:
            settings_block['top_k'] = self._coerce_optional_bounded_int(
                top_k,
                field='Top-k',
                minimum=1,
                maximum=500,
            )
        settings_block['max_output_tokens'] = _normalize_optional_int(
            max_output_tokens,
            settings_block.get('max_output_tokens', defaults['max_output_tokens']),
            field='Max output tokens',
            minimum=1,
        )
        settings_block['timeout'] = _normalize_int(
            timeout,
            settings_block.get('timeout', defaults['timeout']),
            field='Timeout',
            minimum=1,
        )
        settings_block['max_retries'] = _normalize_int(
            max_retries,
            settings_block.get('max_retries', defaults['max_retries']),
            field='Additional retries (after first attempt)',
            minimum=0,
        )
        settings_block['retry_delay'] = _normalize_int(
            retry_delay,
            settings_block.get('retry_delay', defaults['retry_delay']),
            field='Retry delay',
            minimum=0,
        )
        if stop_sequences is not _UNSET:
            settings_block['stop_sequences'] = self._coerce_stop_sequences(stop_sequences)

        if tool_choice is not _UNSET:
            choice, choice_name = self._normalise_tool_choice(
                tool_choice,
                tool_choice_name if tool_choice_name is not _UNSET else settings_block.get('tool_choice_name'),
                previous_choice=settings_block.get('tool_choice'),
                previous_name=settings_block.get('tool_choice_name'),
            )
            settings_block['tool_choice'] = choice
            settings_block['tool_choice_name'] = choice_name

        if metadata is not _UNSET:
            settings_block['metadata'] = self._coerce_metadata(metadata)

        if thinking is not None:
            settings_block['thinking'] = bool(thinking)

        if thinking_budget is not _UNSET:
            settings_block['thinking_budget'] = _normalize_optional_int(
                thinking_budget,
                settings_block.get('thinking_budget'),
                field='Thinking budget tokens',
                minimum=1,
            )

        self.yaml_config['ANTHROPIC_LLM'] = dict(settings_block)
        self.config['ANTHROPIC_LLM'] = dict(settings_block)

        self._write_yaml_config()

        return dict(settings_block)

    def get_anthropic_settings(self) -> Dict[str, Any]:
        """Return Anthropic defaults merged with persisted overrides."""

        defaults = {
            'model': 'claude-3-opus-20240229',
            'stream': True,
            'function_calling': False,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': None,
            'max_output_tokens': None,
            'timeout': 60,
            'max_retries': 3,
            'retry_delay': 5,
            'stop_sequences': [],
            'tool_choice': 'auto',
            'tool_choice_name': None,
            'metadata': {},
            'thinking': False,
            'thinking_budget': None,
        }

        stored = self.get_config('ANTHROPIC_LLM')
        if isinstance(stored, dict):
            for key in defaults.keys():
                if key in stored and stored[key] is not None:
                    try:
                        if key == 'top_k':
                            defaults[key] = self._coerce_optional_bounded_int(
                                stored[key],
                                field='Top-k',
                                minimum=1,
                                maximum=500,
                            )
                        elif key == 'stop_sequences':
                            defaults[key] = self._coerce_stop_sequences(stored[key])
                        elif key == 'metadata':
                            defaults[key] = self._coerce_metadata(stored[key])
                        else:
                            defaults[key] = stored[key]
                    except ValueError as exc:  # pragma: no cover - defensive logging
                        self.logger.warning(
                            "Ignoring persisted Anthropic %s override while loading: %s",
                            key,
                            exc,
                        )

        return defaults

    @staticmethod
    def _coerce_optional_bounded_int(
        value: Any,
        *,
        field: str,
        minimum: int,
        maximum: int,
    ) -> Optional[int]:
        """Validate an optional integer ensuring it falls within provided bounds."""

        if value in {None, ""}:
            return None

        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer or left blank.") from exc

        if parsed < minimum or parsed > maximum:
            raise ValueError(
                f"{field} must be between {minimum} and {maximum}."
            )

        return parsed

    @staticmethod
    def _coerce_stop_sequences(value: Any) -> List[str]:
        """Coerce stop sequences supplied as a CSV string or list of strings."""

        if value is None or value == "":
            return []

        tokens: List[str] = []
        if isinstance(value, str):
            tokens = [part.strip() for part in value.split(',') if part.strip()]
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                if item in {None, ""}:
                    continue
                if not isinstance(item, str):
                    raise ValueError("Stop sequences must be strings.")
                cleaned = item.strip()
                if cleaned:
                    tokens.append(cleaned)
        else:
            raise ValueError(
                "Stop sequences must be provided as a comma-separated string or list of strings."
            )

        if len(tokens) > 4:
            raise ValueError("Stop sequences cannot contain more than 4 entries.")

        return tokens

    @staticmethod
    def _coerce_function_call_mode(value: Any, *, default: str) -> str:
        """Validate Gemini function call mode values against supported options."""

        valid_modes = {"auto", "any", "none", "require"}

        if value in {None, ""}:
            return default

        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in valid_modes:
                return candidate
        raise ValueError("Function call mode must be one of: auto, any, none, require.")

    @staticmethod
    def _coerce_allowed_function_names(value: Any) -> List[str]:
        """Normalise allowed function names as a list of distinct strings."""

        if value in (None, "", []):
            return []

        names: List[str] = []
        if isinstance(value, str):
            tokens = [part.strip() for part in value.split(',') if part and part.strip()]
            names.extend(tokens)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            for item in value:
                if item in {None, ""}:
                    continue
                if not isinstance(item, str):
                    raise ValueError("Allowed function names must be strings.")
                cleaned = item.strip()
                if cleaned:
                    names.append(cleaned)
        else:
            raise ValueError(
                "Allowed function names must be provided as a comma-separated string or sequence of strings."
            )

        seen = set()
        deduped: List[str] = []
        for name in names:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped

    def _normalise_tool_choice(
        self,
        value: Any,
        provided_name: Any,
        *,
        previous_choice: Optional[str],
        previous_name: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        """Validate tool choice inputs and map to Anthropic API expectations."""

        alias_map = {
            "required": "any",
        }

        choice_value: Optional[str]
        name_value: Optional[str] = None

        if isinstance(value, Mapping):
            raw_type = str(value.get("type", "")).strip().lower()
            choice_value = alias_map.get(raw_type, raw_type)
            if value.get("name") is not None:
                provided_name = value.get("name")
        elif isinstance(value, str):
            cleaned = value.strip().lower()
            choice_value = alias_map.get(cleaned, cleaned)
        elif value in {None, ""}:
            choice_value = None
        else:
            choice_value = None

        if choice_value not in {"auto", "any", "none", "tool"}:
            choice_value = previous_choice or "auto"

        if choice_value == "tool":
            name_candidate: Optional[str]
            if provided_name in {None, ""}:
                name_candidate = previous_name
            else:
                name_candidate = str(provided_name).strip()

            if not name_candidate:
                self.logger.warning(
                    "Specific Anthropic tool choice ignored because no tool name was provided.",
                )
                return "auto", None

            name_value = name_candidate
        else:
            name_value = None

        return choice_value or "auto", name_value

    @staticmethod
    def _coerce_metadata(value: Any) -> Dict[str, str]:
        """Normalise metadata payloads to a mapping of string keys and values."""

        if value is None or value == "" or (isinstance(value, Mapping) and not value):
            return {}

        items: List[Tuple[Any, Any]] = []

        def _append_from_mapping(mapping: Mapping[Any, Any]) -> None:
            for key, val in mapping.items():
                items.append((key, val))

        if isinstance(value, Mapping):
            _append_from_mapping(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            parsed: Any
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, Mapping):
                _append_from_mapping(parsed)
            elif isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
                for entry in parsed:
                    if isinstance(entry, Mapping):
                        _append_from_mapping(entry)
                    elif isinstance(entry, Sequence) and len(entry) == 2:
                        items.append((entry[0], entry[1]))
                    else:
                        raise ValueError(
                            "Metadata entries supplied as a list must be key/value pairs.",
                        )
            else:
                segments = [segment.strip() for segment in text.replace("\n", ",").split(",")]
                for segment in segments:
                    if not segment:
                        continue
                    if "=" not in segment:
                        raise ValueError(
                            "Metadata text must use key=value syntax or valid JSON.",
                        )
                    key, val = segment.split("=", 1)
                    items.append((key, val))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for entry in value:
                if isinstance(entry, Mapping):
                    _append_from_mapping(entry)
                elif isinstance(entry, Sequence) and len(entry) == 2:
                    items.append((entry[0], entry[1]))
                else:
                    raise ValueError(
                        "Metadata entries supplied as a list must be key/value pairs.",
                    )
        else:
            raise ValueError(
                "Metadata must be a mapping, JSON string, or iterable of key/value pairs.",
            )

        metadata: Dict[str, str] = {}
        for key, val in items:
            if key in {None, ""}:
                raise ValueError("Metadata keys must be non-empty strings.")
            cleaned_key = str(key).strip()
            if not cleaned_key:
                raise ValueError("Metadata keys must be non-empty strings.")
            cleaned_val = "" if val is None else str(val).strip()
            metadata[cleaned_key] = cleaned_val

        if len(metadata) > 16:
            raise ValueError("Metadata cannot contain more than 16 entries.")

        return metadata


    def get_openai_api_key(self) -> str:
        """
        Retrieves the OpenAI API key from the configuration.

        Returns:
            str: The OpenAI API key.
        """
        return self.get_config('OPENAI_API_KEY')

    def get_mistral_api_key(self) -> str:
        """
        Retrieves the Mistral API key from the configuration.

        Returns:
            str: The Mistral API key.
        """
        return self.get_config('MISTRAL_API_KEY')

    def get_huggingface_api_key(self) -> str:
        """
        Retrieves the HuggingFace API key from the configuration.

        Returns:
            str: The HuggingFace API key.
        """
        return self.get_config('HUGGINGFACE_API_KEY')

    def get_google_api_key(self) -> str:
        """
        Retrieves the Google API key from the configuration.

        Returns:
            str: The Google API key.
        """
        return self.get_config('GOOGLE_API_KEY')

    def get_anthropic_api_key(self) -> str:
        """
        Retrieves the Anthropic API key from the configuration.

        Returns:
            str: The Anthropic API key.
        """
        return self.get_config('ANTHROPIC_API_KEY')

    def get_grok_api_key(self) -> str:
        """
        Retrieves the Grok API key from the configuration.

        Returns:
            str: The Grok API key.
        """
        return self.get_config('GROK_API_KEY')

    def get_app_root(self) -> str:
        """
        Retrieves the application's root directory path.

        Returns:
            str: The path to the application's root directory.
        """
        return self.get_config('APP_ROOT')

    def set_tenant_id(self, tenant_id: Optional[str]) -> Optional[str]:
        """Persist the active tenant identifier."""

        normalized = tenant_id.strip() if isinstance(tenant_id, str) else None
        if not normalized:
            normalized = None

        if normalized is None:
            self.yaml_config.pop('tenant_id', None)
            self.config.pop('tenant_id', None)
        else:
            self.yaml_config['tenant_id'] = normalized
            self.config['tenant_id'] = normalized

        self._write_yaml_config()
        return normalized

    def set_http_server_autostart(self, enabled: bool) -> bool:
        """Persist whether the embedded HTTP server should start automatically."""

        block = dict(self.yaml_config.get('http_server', {}))
        block['auto_start'] = bool(enabled)
        self.yaml_config['http_server'] = dict(block)
        self.config['http_server'] = dict(block)
        self._write_yaml_config()
        return bool(enabled)

    def update_api_key(self, provider_name: str, new_api_key: str):
        """
        Updates the API key for a specified provider in the .env file and reloads
        the environment variables to reflect the changes immediately.

        Args:
            provider_name (str): The name of the provider whose API key is to be updated.
            new_api_key (str): The new API key to set for the provider.

        Raises:
            FileNotFoundError: If the .env file is not found.
            ValueError: If the provider name does not have a corresponding API key mapping.
        """

        self.providers.update_api_key(provider_name, new_api_key)


    def has_provider_api_key(self, provider_name: str) -> bool:
        """
        Determine whether an API key is configured for the given provider.

        Args:
            provider_name (str): The name of the provider to check.

        Returns:
            bool: True if an API key exists for the provider, False otherwise.
        """

        return self.providers.has_provider_api_key(provider_name)


    def _is_api_key_set(self, provider_name: str) -> bool:
        """
        Checks if the API key for a specified provider is set.

        Args:
            provider_name (str): The name of the provider.

        Returns:
            bool: True if the API key is set, False otherwise.
        """

        return self.providers.is_api_key_set(provider_name)


    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves metadata for available providers without exposing raw secrets.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are provider names and values contain
            availability metadata such as whether the credential is set, a masked hint, and the
            stored length.
        """

        return self.providers.get_available_providers()


    @staticmethod
    def _mask_secret_preview(secret: str) -> str:
        """Return a masked preview of a secret without revealing its contents."""

        if not secret:
            return ""

        visible_count = min(len(secret), 8)
        return "•" * visible_count

    @staticmethod
    def _sanitize_tool_value(value: Any) -> Any:
        """Return a JSON-serializable representation for persisted tool settings."""

        if isinstance(value, Mapping):
            return {
                str(key): ConfigManager._sanitize_tool_value(subvalue)
                for key, subvalue in value.items()
            }

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [ConfigManager._sanitize_tool_value(item) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    @staticmethod
    def _sanitize_tool_settings_block(value: Any) -> Dict[str, Any]:
        """Normalize persisted tool settings into a dictionary."""

        if not isinstance(value, Mapping):
            return {}

        return {
            str(key): ConfigManager._sanitize_tool_value(subvalue)
            for key, subvalue in value.items()
        }

    @staticmethod
    def _extract_auth_env_definitions(
        auth_block: Optional[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Return environment definitions declared in an authentication manifest block."""

        definitions: Dict[str, Dict[str, Any]] = {}

        if not isinstance(auth_block, Mapping):
            return definitions

        def _register_env(env_name: Any, metadata: Optional[Mapping[str, Any]] = None) -> None:
            if not isinstance(env_name, str):
                env_candidate = str(env_name) if env_name is not None else ""
            else:
                env_candidate = env_name

            token = env_candidate.strip()
            if not token:
                return

            merged: Dict[str, Any] = dict(definitions.get(token, {}))
            if isinstance(metadata, Mapping):
                for key, value in metadata.items():
                    merged[key] = value
            definitions[token] = merged

        env_value = auth_block.get("env")
        env_required: Optional[bool] = None
        if isinstance(auth_block.get("required"), bool):
            env_required = bool(auth_block["required"])

        if isinstance(env_value, str):
            metadata: Dict[str, Any] = {}
            if env_required is not None:
                metadata["required"] = env_required
            _register_env(env_value, metadata)
        elif isinstance(env_value, Sequence) and not isinstance(env_value, (str, bytes, bytearray)):
            for entry in env_value:
                if isinstance(entry, str):
                    metadata: Dict[str, Any] = {}
                    if env_required is not None:
                        metadata["required"] = env_required
                    _register_env(entry, metadata)

        envs_value = auth_block.get("envs")
        if isinstance(envs_value, Mapping):
            for key, value in envs_value.items():
                metadata: Dict[str, Any] = {}
                env_name: Optional[str] = None

                if isinstance(value, str):
                    env_name = value
                elif isinstance(value, Mapping):
                    candidate = value.get("env")
                    if isinstance(candidate, str):
                        env_name = candidate
                    optional_flag = value.get("optional")
                    if isinstance(optional_flag, bool):
                        metadata["optional"] = optional_flag
                        metadata.setdefault("required", not optional_flag)
                    required_flag = value.get("required")
                    if isinstance(required_flag, bool):
                        metadata["required"] = required_flag
                        if "optional" not in metadata:
                            metadata["optional"] = not required_flag
                elif value is None:
                    env_name = None

                if env_name is None and isinstance(key, str):
                    env_name = key

                if env_name is not None:
                    _register_env(env_name, metadata)
        elif isinstance(envs_value, Sequence) and not isinstance(envs_value, (str, bytes, bytearray)):
            for value in envs_value:
                metadata = {}
                env_name = None
                if isinstance(value, str):
                    env_name = value
                elif isinstance(value, Mapping):
                    candidate = value.get("env")
                    if isinstance(candidate, str):
                        env_name = candidate
                    optional_flag = value.get("optional")
                    if isinstance(optional_flag, bool):
                        metadata["optional"] = optional_flag
                        metadata.setdefault("required", not optional_flag)
                    required_flag = value.get("required")
                    if isinstance(required_flag, bool):
                        metadata["required"] = required_flag
                        if "optional" not in metadata:
                            metadata["optional"] = not required_flag
                if env_name is not None:
                    _register_env(env_name, metadata)

        return definitions

    @staticmethod
    def _extract_auth_env_keys(auth_block: Optional[Mapping[str, Any]]) -> List[str]:
        """Return normalized environment variable keys declared in a manifest auth block."""

        definitions = ConfigManager._extract_auth_env_definitions(auth_block)
        normalized: List[str] = []
        for candidate in definitions.keys():
            token = candidate.strip()
            if token and token not in normalized:
                normalized.append(token)
        return normalized

    def _collect_credential_status(self, env_keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Return masked credential metadata for the provided environment keys."""

        status: Dict[str, Dict[str, Any]] = {}

        for env_key in env_keys:
            value = self.get_config(env_key)
            if value is None:
                secret = ""
            elif isinstance(value, str):
                secret = value
            else:
                secret = str(value)

            configured = bool(secret)
            status[env_key] = {
                "configured": configured,
                "hint": self._mask_secret_preview(secret) if configured else "",
                "source": "env",
            }

        return status

    def get_tool_config_snapshot(
        self,
        *,
        manifest_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
        tool_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return sanitized tool configuration data for UI consumption."""

        config_tools = self.config.get("tools", {})
        yaml_tools = self.yaml_config.get("tools", {})

        candidates: List[str] = []
        if isinstance(tool_names, Iterable):
            for name in tool_names:
                if isinstance(name, str):
                    token = name.strip()
                    if token and token not in candidates:
                        candidates.append(token)

        if isinstance(config_tools, Mapping):
            for name in config_tools.keys():
                token = str(name)
                if token and token not in candidates:
                    candidates.append(token)

        if isinstance(yaml_tools, Mapping):
            for name in yaml_tools.keys():
                token = str(name)
                if token and token not in candidates:
                    candidates.append(token)

        if isinstance(manifest_lookup, Mapping):
            for name in manifest_lookup.keys():
                token = str(name)
                if token and token not in candidates:
                    candidates.append(token)

        snapshot: Dict[str, Dict[str, Any]] = {}

        for name in candidates:
            settings_block: Dict[str, Any] = {}

            if isinstance(config_tools, Mapping) and name in config_tools:
                settings_block = self._sanitize_tool_settings_block(config_tools[name])
            elif isinstance(yaml_tools, Mapping) and name in yaml_tools:
                settings_block = self._sanitize_tool_settings_block(yaml_tools[name])

            auth_block: Optional[Mapping[str, Any]] = None
            if isinstance(manifest_lookup, Mapping):
                manifest_entry = manifest_lookup.get(name)
                if isinstance(manifest_entry, Mapping):
                    auth_block = manifest_entry.get("auth")
                    if not isinstance(auth_block, Mapping):
                        auth_block = None

            env_definitions = self._extract_auth_env_definitions(auth_block)
            env_keys = list(env_definitions.keys())
            credentials = self._collect_credential_status(env_keys)

            for env_key, metadata in env_definitions.items():
                if not metadata:
                    continue
                entry = credentials.get(env_key)
                if entry is None:
                    entry = {"configured": False, "hint": "", "source": "env"}
                    credentials[env_key] = entry
                for meta_key, meta_value in metadata.items():
                    if meta_key in {"configured", "hint", "source"}:
                        continue
                    entry[meta_key] = meta_value

            snapshot[name] = {
                "settings": copy.deepcopy(settings_block),
                "credentials": credentials,
            }

        return snapshot

    def set_tool_settings(self, tool_name: str, settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Persist sanitized tool settings to the YAML configuration."""

        normalized_name = str(tool_name or "").strip()
        if not normalized_name:
            raise ValueError("Tool name is required when updating settings.")

        sanitized_settings: Dict[str, Any] = {}
        if settings is not None:
            if not isinstance(settings, Mapping):
                raise TypeError("Tool settings must be a mapping when provided.")
            sanitized_settings = self._sanitize_tool_settings_block(settings)

        yaml_tools_block: Dict[str, Any] = {}
        existing_yaml_tools = self.yaml_config.get("tools")
        if isinstance(existing_yaml_tools, Mapping):
            yaml_tools_block = copy.deepcopy(existing_yaml_tools)

        config_tools_block: Dict[str, Any] = {}
        existing_config_tools = self.config.get("tools")
        if isinstance(existing_config_tools, Mapping):
            config_tools_block = copy.deepcopy(existing_config_tools)

        if sanitized_settings:
            yaml_tools_block[normalized_name] = copy.deepcopy(sanitized_settings)
            config_tools_block[normalized_name] = copy.deepcopy(sanitized_settings)
        else:
            yaml_tools_block.pop(normalized_name, None)
            config_tools_block.pop(normalized_name, None)

        if yaml_tools_block:
            self.yaml_config["tools"] = yaml_tools_block
        else:
            self.yaml_config.pop("tools", None)

        self.config["tools"] = config_tools_block

        self._write_yaml_config()
        return copy.deepcopy(sanitized_settings)

    def set_tool_credentials(
        self,
        tool_name: str,
        credentials: Mapping[str, Any],
        *,
        manifest_auth: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Persist tool credentials according to manifest-defined environment keys."""

        if not isinstance(credentials, Mapping):
            raise TypeError("Tool credentials payload must be a mapping.")

        normalized_name = str(tool_name or "").strip()
        if not normalized_name:
            raise ValueError("Tool name is required when updating credentials.")

        env_keys = set(self._extract_auth_env_keys(manifest_auth))
        if not env_keys:
            for candidate in credentials.keys():
                if isinstance(candidate, str) and candidate.strip():
                    env_keys.add(candidate.strip())

        for raw_key in credentials.keys():
            if not isinstance(raw_key, str):
                continue
            token = raw_key.strip()
            if not token:
                continue
            if token not in env_keys:
                env_keys.add(token)

        for env_key in sorted(env_keys):
            value = credentials.get(env_key, ConfigManager.UNSET)
            if value is ConfigManager.UNSET:
                continue

            if value is None:
                sanitized = None
            else:
                text = str(value)
                sanitized = text.strip()
                if not sanitized:
                    sanitized = None

            self._persist_env_value(env_key, sanitized)

        return self._collect_credential_status(sorted(env_keys))

    def get_skill_config_snapshot(
        self,
        *,
        manifest_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
        skill_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return sanitized skill configuration data for UI consumption."""

        config_skills = self.config.get("skills", {})
        yaml_skills = self.yaml_config.get("skills", {})

        candidates: List[str] = []
        if isinstance(skill_names, Iterable):
            for name in skill_names:
                if isinstance(name, str):
                    token = name.strip()
                    if token and token not in candidates:
                        candidates.append(token)

        if isinstance(config_skills, Mapping):
            for name in config_skills.keys():
                token = str(name)
                if token and token not in candidates:
                    candidates.append(token)

        if isinstance(yaml_skills, Mapping):
            for name in yaml_skills.keys():
                token = str(name)
                if token and token not in candidates:
                    candidates.append(token)

        if isinstance(manifest_lookup, Mapping):
            for name in manifest_lookup.keys():
                token = str(name)
                if token and token not in candidates:
                    candidates.append(token)

        snapshot: Dict[str, Dict[str, Any]] = {}

        for name in candidates:
            settings_block: Dict[str, Any] = {}

            if isinstance(config_skills, Mapping) and name in config_skills:
                settings_block = self._sanitize_tool_settings_block(config_skills[name])
            elif isinstance(yaml_skills, Mapping) and name in yaml_skills:
                settings_block = self._sanitize_tool_settings_block(yaml_skills[name])

            auth_block: Optional[Mapping[str, Any]] = None
            if isinstance(manifest_lookup, Mapping):
                manifest_entry = manifest_lookup.get(name)
                if isinstance(manifest_entry, Mapping):
                    auth_candidate = manifest_entry.get("auth")
                    if isinstance(auth_candidate, Mapping):
                        auth_block = auth_candidate

            env_definitions = self._extract_auth_env_definitions(auth_block)
            env_keys = list(env_definitions.keys())
            credentials = self._collect_credential_status(env_keys)

            for env_key, metadata in env_definitions.items():
                if not metadata:
                    continue
                entry = credentials.get(env_key)
                if entry is None:
                    entry = {"configured": False, "hint": "", "source": "env"}
                    credentials[env_key] = entry
                for meta_key, meta_value in metadata.items():
                    if meta_key in {"configured", "hint", "source"}:
                        continue
                    entry[meta_key] = meta_value

            snapshot[name] = {
                "settings": copy.deepcopy(settings_block),
                "credentials": credentials,
            }

        return snapshot

    def set_skill_settings(self, skill_name: str, settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Persist sanitized skill settings to the YAML configuration."""

        normalized_name = str(skill_name or "").strip()
        if not normalized_name:
            raise ValueError("Skill name is required when updating settings.")

        sanitized_settings: Dict[str, Any] = {}
        if settings is not None:
            if not isinstance(settings, Mapping):
                raise TypeError("Skill settings must be a mapping when provided.")
            sanitized_settings = self._sanitize_tool_settings_block(settings)

        yaml_skills_block: Dict[str, Any] = {}
        existing_yaml_skills = self.yaml_config.get("skills")
        if isinstance(existing_yaml_skills, Mapping):
            yaml_skills_block = copy.deepcopy(existing_yaml_skills)

        config_skills_block: Dict[str, Any] = {}
        existing_config_skills = self.config.get("skills")
        if isinstance(existing_config_skills, Mapping):
            config_skills_block = copy.deepcopy(existing_config_skills)

        if sanitized_settings:
            yaml_skills_block[normalized_name] = copy.deepcopy(sanitized_settings)
            config_skills_block[normalized_name] = copy.deepcopy(sanitized_settings)
        else:
            yaml_skills_block.pop(normalized_name, None)
            config_skills_block.pop(normalized_name, None)

        if yaml_skills_block:
            self.yaml_config["skills"] = yaml_skills_block
        else:
            self.yaml_config.pop("skills", None)

        self.config["skills"] = config_skills_block

        self._write_yaml_config()
        return copy.deepcopy(sanitized_settings)

    def set_skill_credentials(
        self,
        skill_name: str,
        credentials: Mapping[str, Any],
        *,
        manifest_auth: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Persist skill credentials according to manifest-defined environment keys."""

        if not isinstance(credentials, Mapping):
            raise TypeError("Skill credentials payload must be a mapping.")

        normalized_name = str(skill_name or "").strip()
        if not normalized_name:
            raise ValueError("Skill name is required when updating credentials.")

        env_keys = set(self._extract_auth_env_keys(manifest_auth))
        if not env_keys:
            for candidate in credentials.keys():
                if isinstance(candidate, str) and candidate.strip():
                    env_keys.add(candidate.strip())

        for raw_key in credentials.keys():
            if not isinstance(raw_key, str):
                continue
            token = raw_key.strip()
            if not token:
                continue
            if token not in env_keys:
                env_keys.add(token)

        for env_key in sorted(env_keys):
            value = credentials.get(env_key, ConfigManager.UNSET)
            if value is ConfigManager.UNSET:
                continue

            if value is None:
                sanitized = None
            else:
                text = str(value)
                sanitized = text.strip()
                if not sanitized:
                    sanitized = None

            self._persist_env_value(env_key, sanitized)

        return self._collect_credential_status(sorted(env_keys))

    def get_skill_metadata(self, skill_name: str) -> Dict[str, Optional[str]]:
        """Return persisted review metadata for ``skill_name`` if available."""

        normalized_name = str(skill_name or "").strip()
        if not normalized_name:
            raise ValueError("Skill name is required when reading metadata.")

        review_status: Optional[str] = None
        tester_notes: Optional[str] = None

        config_block = self.config.get("skill_metadata")
        yaml_block = self.yaml_config.get("skill_metadata")

        stored_entry: Mapping[str, Any] | None = None
        if isinstance(config_block, Mapping):
            candidate = config_block.get(normalized_name)
            if isinstance(candidate, Mapping):
                stored_entry = candidate
        if stored_entry is None and isinstance(yaml_block, Mapping):
            candidate = yaml_block.get(normalized_name)
            if isinstance(candidate, Mapping):
                stored_entry = candidate

        if stored_entry is not None:
            status_value = stored_entry.get("review_status")
            if isinstance(status_value, str):
                review_status = status_value.strip() or None
            tester_value = stored_entry.get("tester_notes")
            if isinstance(tester_value, str):
                tester_notes = tester_value.strip() or None
            elif tester_value is None:
                tester_notes = None

        return {
            "review_status": review_status,
            "tester_notes": tester_notes,
        }

    def set_skill_metadata(
        self,
        skill_name: str,
        metadata: Optional[Mapping[str, Any]],
    ) -> Dict[str, Optional[str]]:
        """Persist review metadata such as tester notes for ``skill_name``."""

        normalized_name = str(skill_name or "").strip()
        if not normalized_name:
            raise ValueError("Skill name is required when updating metadata.")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("Skill metadata payload must be a mapping when provided.")

        existing_config = self.config.get("skill_metadata")
        existing_yaml = self.yaml_config.get("skill_metadata")

        existing_entry: Mapping[str, Any] | None = None
        if isinstance(existing_config, Mapping):
            candidate = existing_config.get(normalized_name)
            if isinstance(candidate, Mapping):
                existing_entry = candidate
        if existing_entry is None and isinstance(existing_yaml, Mapping):
            candidate = existing_yaml.get(normalized_name)
            if isinstance(candidate, Mapping):
                existing_entry = candidate

        def _sanitize_text(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        existing_status = _sanitize_text(existing_entry.get("review_status")) if existing_entry else None
        existing_notes = _sanitize_text(existing_entry.get("tester_notes")) if existing_entry else None

        if metadata is None:
            sanitized_status = None
            sanitized_notes = None
        else:
            if "review_status" in metadata:
                sanitized_status = _sanitize_text(metadata.get("review_status"))
            else:
                sanitized_status = existing_status
            if "tester_notes" in metadata:
                sanitized_notes = _sanitize_text(metadata.get("tester_notes"))
            else:
                sanitized_notes = existing_notes

        record: Dict[str, Optional[str]] = {}
        if sanitized_status is not None:
            record["review_status"] = sanitized_status
        if sanitized_notes is not None:
            record["tester_notes"] = sanitized_notes

        yaml_block: Dict[str, Any] = {}
        if isinstance(existing_yaml, Mapping):
            yaml_block = copy.deepcopy(existing_yaml)

        config_block: Dict[str, Any] = {}
        if isinstance(existing_config, Mapping):
            config_block = copy.deepcopy(existing_config)

        if record:
            yaml_block[normalized_name] = copy.deepcopy(record)
            config_block[normalized_name] = copy.deepcopy(record)
        else:
            yaml_block.pop(normalized_name, None)
            config_block.pop(normalized_name, None)

        if yaml_block:
            self.yaml_config["skill_metadata"] = yaml_block
        else:
            self.yaml_config.pop("skill_metadata", None)

        self.config["skill_metadata"] = config_block

        self._write_yaml_config()

        return {
            "review_status": sanitized_status,
            "tester_notes": sanitized_notes,
        }

    # Additional methods to handle TTS_ENABLED from config.yaml
    def get_tts_enabled(self) -> bool:
        """
        Retrieves the TTS enabled status from the configuration.

        Returns:
            bool: True if TTS is enabled, False otherwise.
        """
        return self.get_config('TTS_ENABLED', False)

    def set_tts_enabled(self, value: bool):
        """
        Sets the TTS enabled status in the configuration.

        Args:
            value (bool): True to enable TTS, False to disable.
        """
        self.yaml_config['TTS_ENABLED'] = value
        self.config['TTS_ENABLED'] = value
        self.logger.debug("TTS_ENABLED set to %s", value)
        # Optionally, write back to config.yaml if persistence is required
        self._write_yaml_config()

    def set_default_tts_provider(self, provider: Optional[str]) -> Optional[str]:
        """Persist the default TTS provider selection."""

        normalized = provider.strip() if isinstance(provider, str) else None
        if not normalized:
            normalized = None

        if normalized is None:
            self.yaml_config.pop('DEFAULT_TTS_PROVIDER', None)
            self.config.pop('DEFAULT_TTS_PROVIDER', None)
        else:
            self.yaml_config['DEFAULT_TTS_PROVIDER'] = normalized
            self.config['DEFAULT_TTS_PROVIDER'] = normalized

        self._write_yaml_config()
        return normalized

    def set_default_stt_provider(self, provider: Optional[str]) -> Optional[str]:
        """Persist the default STT provider selection."""

        normalized = provider.strip() if isinstance(provider, str) else None
        if not normalized:
            normalized = None

        if normalized is None:
            self.yaml_config.pop('DEFAULT_STT_PROVIDER', None)
            self.config.pop('DEFAULT_STT_PROVIDER', None)
        else:
            self.yaml_config['DEFAULT_STT_PROVIDER'] = normalized
            self.config['DEFAULT_STT_PROVIDER'] = normalized

        self._write_yaml_config()
        return normalized

    def set_default_speech_providers(
        self,
        *,
        tts_provider: Optional[str] = None,
        stt_provider: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Persist TTS/STT provider defaults in a single operation."""

        result = {
            'tts_provider': self.set_default_tts_provider(tts_provider),
            'stt_provider': self.set_default_stt_provider(stt_provider),
        }
        return result

    def set_ui_debug_log_max_lines(self, max_lines: Optional[int]) -> Optional[int]:
        """Persist the maximum number of UI debug log lines retained."""

        normalized: Optional[int]
        if max_lines is None:
            normalized = None
        else:
            try:
                normalized = int(max_lines)
            except (TypeError, ValueError):
                normalized = None

        if normalized is not None:
            # Enforce a practical lower bound so the UI handler can operate safely.
            normalized = max(100, normalized)
            self.yaml_config['UI_DEBUG_LOG_MAX_LINES'] = normalized
            self.config['UI_DEBUG_LOG_MAX_LINES'] = normalized
        else:
            self.yaml_config.pop('UI_DEBUG_LOG_MAX_LINES', None)
            self.config.pop('UI_DEBUG_LOG_MAX_LINES', None)

        self._write_yaml_config()
        return normalized

    def set_ui_debug_logger_names(self, logger_names: Optional[Sequence[str]]) -> List[str]:
        """Persist the list of logger names mirrored in the UI debug console."""

        normalized: List[str] = []
        if logger_names is not None:
            for entry in logger_names:
                sanitized = str(entry).strip()
                if sanitized:
                    normalized.append(sanitized)

        if normalized:
            self.yaml_config['UI_DEBUG_LOGGERS'] = list(normalized)
            self.config['UI_DEBUG_LOGGERS'] = list(normalized)
        else:
            self.yaml_config.pop('UI_DEBUG_LOGGERS', None)
            self.config.pop('UI_DEBUG_LOGGERS', None)

        self._write_yaml_config()
        return list(normalized)

    def get_ui_terminal_wrap_enabled(self) -> bool:
        """Return whether terminal sections in the UI should wrap lines."""

        value = self.get_config('UI_TERMINAL_WRAP_ENABLED', self.UNSET)

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'true', '1', 'yes', 'on'}:
                return True
            if normalized in {'false', '0', 'no', 'off'}:
                return False

        return True

    def set_ui_terminal_wrap_enabled(self, enabled: bool) -> bool:
        """Persist the line wrapping preference for UI terminal sections."""

        normalized = bool(enabled)
        self.yaml_config['UI_TERMINAL_WRAP_ENABLED'] = normalized
        self.config['UI_TERMINAL_WRAP_ENABLED'] = normalized
        self._write_yaml_config()
        return normalized

    def export_yaml_config(self, destination: str | os.PathLike[str] | Path) -> str:
        """Write the current YAML configuration to ``destination``.

        The exported file mirrors the structure persisted in ``config.yaml``. The
        returned string is the absolute path written to disk.
        """

        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.yaml_config, handle, sort_keys=False)
        return str(path)

    def import_yaml_config(self, source: str | os.PathLike[str] | Path) -> Dict[str, Any]:
        """Load configuration data from ``source`` and persist it as the active YAML.

        The parsed mapping becomes the in-memory ``yaml_config`` and is written to
        the default ``config.yaml`` path. A dictionary copy of the loaded
        configuration is returned.
        """

        path = Path(source).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            try:
                loaded = yaml.safe_load(handle) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(loaded, Mapping):
            raise ValueError(f"Configuration file {path} must contain a mapping")

        self.yaml_config = copy.deepcopy(dict(loaded))
        self.config = {**self.env_config, **self.yaml_config}
        self._model_cache = self._normalize_model_cache(
            self.yaml_config.get("MODEL_CACHE")
        )
        self.config["MODEL_CACHE"] = copy.deepcopy(self._model_cache)
        if "MODEL_CACHE" not in self.yaml_config:
            self.yaml_config["MODEL_CACHE"] = copy.deepcopy(self._model_cache)

        self._write_yaml_config()
        return copy.deepcopy(self.yaml_config)

    def _write_yaml_config(self):
        """
        Writes the current YAML configuration back to the config.yaml file.
        """
        yaml_path = getattr(self, '_yaml_path', None) or self._compute_yaml_path()
        try:
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.yaml_config, file)
                if file.tell() == 0 and self.yaml_config:
                    file.write(json.dumps(self.yaml_config))
            self.logger.debug("Configuration written to %s", yaml_path)
        except Exception as e:
            self.logger.error(f"Failed to write configuration to {yaml_path}: {e}")
