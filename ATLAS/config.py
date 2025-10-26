# ATLAS/config.py

from __future__ import annotations

import copy
import json
import os
import shlex
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional dependency handling for test environments
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
    from sqlalchemy.engine.url import make_url
    from sqlalchemy.orm import sessionmaker
except Exception:  # pragma: no cover - lightweight fallbacks when SQLAlchemy is absent
    class Engine:  # type: ignore[assignment]
        pass

    def create_engine(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy create_engine is unavailable in this environment")

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

_SUPPORTED_MISTRAL_PROMPT_MODES = {"reasoning"}
from modules.Providers.Google.settings_resolver import GoogleSettingsResolver
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

        # Ensure configurable tool timeout/budget defaults are present
        tool_defaults = self.config.get('tool_defaults')
        if not isinstance(tool_defaults, Mapping):
            tool_defaults = {}
        else:
            tool_defaults = dict(tool_defaults)
        tool_defaults.setdefault('timeout_seconds', 30)
        tool_defaults.setdefault('max_cost_per_session', None)
        self.config['tool_defaults'] = tool_defaults

        conversation_block = self.config.get('conversation')
        if not isinstance(conversation_block, Mapping):
            conversation_block = {}
        else:
            conversation_block = dict(conversation_block)
        conversation_block.setdefault('max_tool_duration_ms', 120000)
        self.config['conversation'] = conversation_block

        tool_logging_block = self.config.get('tool_logging')
        if not isinstance(tool_logging_block, Mapping):
            tool_logging_block = {}
        else:
            tool_logging_block = dict(tool_logging_block)
        tool_logging_block.setdefault('log_full_payloads', False)
        tool_logging_block.setdefault('payload_summary_length', 256)
        self.config['tool_logging'] = tool_logging_block

        tools_block = self.config.get('tools')
        if not isinstance(tools_block, Mapping):
            tools_block = {}
        else:
            tools_block = dict(tools_block)

        js_block = tools_block.get('javascript_executor')
        if not isinstance(js_block, Mapping):
            js_block = {}
        else:
            js_block = dict(js_block)

        env_executable = self.env_config.get('JAVASCRIPT_EXECUTOR_BIN')
        if env_executable and not js_block.get('executable'):
            js_block['executable'] = env_executable

        env_args = self.env_config.get('JAVASCRIPT_EXECUTOR_ARGS')
        if env_args and not js_block.get('args'):
            try:
                js_block['args'] = shlex.split(env_args)
            except ValueError:
                js_block['args'] = env_args

        js_block.setdefault('default_timeout', 5.0)
        js_block.setdefault('cpu_time_limit', 2.0)
        js_block.setdefault('memory_limit_bytes', 256 * 1024 * 1024)
        js_block.setdefault('max_output_bytes', 64 * 1024)
        js_block.setdefault('max_file_bytes', 128 * 1024)
        js_block.setdefault('max_files', 32)

        tools_block['javascript_executor'] = js_block
        self.config['tools'] = tools_block

        tool_safety_block = self.config.get('tool_safety')
        if not isinstance(tool_safety_block, Mapping):
            tool_safety_block = {}
        else:
            tool_safety_block = dict(tool_safety_block)
        normalized_allowlist = self._normalize_network_allowlist(
            tool_safety_block.get('network_allowlist')
        )
        tool_safety_block['network_allowlist'] = normalized_allowlist
        self.config['tool_safety'] = tool_safety_block

        # Ensure any persisted OpenAI speech preferences are reflected in the active config
        self._synchronize_openai_speech_block()

        conversation_store_block = self.config.get("conversation_database")
        if not isinstance(conversation_store_block, Mapping):
            conversation_store_block = {}
        else:
            conversation_store_block = dict(conversation_store_block)

        env_db_url = self.env_config.get("CONVERSATION_DATABASE_URL")
        if env_db_url and not conversation_store_block.get("url"):
            conversation_store_block["url"] = env_db_url

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
                    "Invalid CONVERSATION_DATABASE_RETENTION_DAYS value %r", env_retention_days
                )

        retention_block.setdefault("history_message_limit", 500)
        conversation_store_block["retention"] = retention_block
        self.config["conversation_database"] = conversation_store_block

        huggingface_block = self.yaml_config.get("HUGGINGFACE")
        if isinstance(huggingface_block, Mapping):
            self.config["HUGGINGFACE"] = dict(huggingface_block)

        messaging_block = self.config.get('messaging')
        if not isinstance(messaging_block, Mapping):
            messaging_block = {}
        else:
            messaging_block = dict(messaging_block)
        backend_name = str(messaging_block.get('backend') or 'in_memory').lower()
        messaging_block['backend'] = backend_name
        if backend_name == 'redis':
            default_url = self.env_config.get('REDIS_URL', 'redis://localhost:6379/0')
            messaging_block.setdefault('redis_url', default_url)
            messaging_block.setdefault('stream_prefix', 'atlas_bus')
        self.config['messaging'] = messaging_block

        # Track provider/environment key relationships for faster lookups
        self._provider_env_lookup = {
            env_key: provider
            for provider, env_key in self._get_provider_env_keys().items()
        }

        # Maintain any deferred credential warnings so onboarding flows can surface them
        self._pending_provider_warnings: Dict[str, str] = {}

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

        # Record a warning instead of aborting when the default provider is missing credentials
        default_provider = self.config.get('DEFAULT_PROVIDER', 'OpenAI')
        if not self._is_api_key_set(default_provider):
            warning_message = (
                f"API key for provider '{default_provider}' is not configured. "
                "Protected features will remain unavailable until a key is provided."
            )
            self.logger.warning(warning_message)
            self._pending_provider_warnings[default_provider] = warning_message

        self._message_bus: Optional[MessageBus] = None
        self._conversation_engine: Engine | None = None
        self._conversation_session_factory: sessionmaker | None = None
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

        block = self.config.get("conversation_database")
        if not isinstance(block, Mapping):
            return {}
        return dict(block)

    def ensure_postgres_conversation_store(self) -> str:
        """Ensure the configured PostgreSQL conversation store is initialised."""

        config = self.get_conversation_database_config()
        url = config.get("url")
        if not url:
            message = (
                "Conversation database URL is required. Set CONVERSATION_DATABASE_URL "
                "or configure conversation_database.url with a PostgreSQL DSN."
            )
            self.logger.error(message)
            raise RuntimeError(message)

        from modules.conversation_store.bootstrap import (
            BootstrapError,
            bootstrap_conversation_store,
        )

        try:
            ensured_url = bootstrap_conversation_store(url)
        except BootstrapError as exc:
            message = f"Failed to bootstrap conversation store: {exc}"
            self.logger.error(message)
            raise RuntimeError(message) from exc

        if ensured_url != url:
            conversation_block = self.config.setdefault("conversation_database", {})
            conversation_block["url"] = ensured_url

            yaml_block = self.yaml_config.get("conversation_database")
            if isinstance(yaml_block, Mapping):
                updated_block = dict(yaml_block)
                updated_block["url"] = ensured_url
            else:
                updated_block = {"url": ensured_url}
            self.yaml_config["conversation_database"] = updated_block

        return ensured_url

    def get_conversation_retention_policies(self) -> Dict[str, Any]:
        """Return configured retention policies for the conversation store."""

        config = self.get_conversation_database_config()
        retention = config.get("retention") or {}
        if isinstance(retention, Mapping):
            return dict(retention)
        return {}

    def get_conversation_store_engine(self) -> Engine | None:
        """Return a configured SQLAlchemy engine for the conversation store."""

        if self._conversation_engine is None:
            self.get_conversation_store_session_factory()
        return self._conversation_engine

    def get_conversation_store_session_factory(self) -> sessionmaker | None:
        """Return a configured session factory for the conversation store."""

        if self._conversation_session_factory is not None:
            return self._conversation_session_factory

        engine, factory = self._build_conversation_store_session_factory()
        self._conversation_engine = engine
        self._conversation_session_factory = factory
        return factory

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

    def get_llm_fallback_config(self) -> Dict[str, Any]:
        """Return the configured fallback provider settings with sensible defaults."""

        fallback_block: Dict[str, Any] = {}
        stored = self.get_config('LLM_FALLBACK')
        if isinstance(stored, Mapping):
            fallback_block.update(stored)

        env_provider = self.get_config('LLM_FALLBACK_PROVIDER')
        if isinstance(env_provider, str) and env_provider.strip():
            fallback_block['provider'] = env_provider.strip()

        env_model = self.get_config('LLM_FALLBACK_MODEL')
        if isinstance(env_model, str) and env_model.strip():
            fallback_block['model'] = env_model.strip()

        provider = fallback_block.get('provider') or self.get_default_provider() or 'OpenAI'
        fallback_block['provider'] = provider

        provider_defaults: Dict[str, Any] = {}
        defaults_lookup = {
            'OpenAI': self.get_openai_llm_settings,
            'Mistral': self.get_mistral_llm_settings,
            'Google': self.get_google_llm_settings,
            'Anthropic': self.get_anthropic_settings,
        }

        getter = defaults_lookup.get(provider)
        if callable(getter):
            provider_defaults = getter()
        else:
            provider_defaults = {}

        merged: Dict[str, Any] = copy.deepcopy(provider_defaults)
        merged.update(fallback_block)

        if not merged.get('model'):
            default_model = provider_defaults.get('model') if isinstance(provider_defaults, Mapping) else None
            if not default_model:
                default_model = self.get_default_model()
            merged['model'] = default_model

        if 'stream' not in merged and isinstance(provider_defaults, Mapping):
            merged['stream'] = provider_defaults.get('stream', True)

        if 'max_tokens' not in merged and isinstance(provider_defaults, Mapping):
            merged['max_tokens'] = provider_defaults.get('max_tokens')

        return merged

    def get_messaging_settings(self) -> Dict[str, Any]:
        """Return the configured messaging backend settings."""

        configured = self.config.get('messaging')
        if isinstance(configured, Mapping):
            return dict(configured)
        return {'backend': 'in_memory'}

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

    def _get_provider_env_keys(self) -> Dict[str, str]:
        """Return the mapping between provider display names and environment keys."""

        return {
            "OpenAI": "OPENAI_API_KEY",
            "Mistral": "MISTRAL_API_KEY",
            "Google": "GOOGLE_API_KEY",
            "HuggingFace": "HUGGINGFACE_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Grok": "GROK_API_KEY",
            "ElevenLabs": "XI_API_KEY",
        }

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

        self._sync_provider_warning(env_key, value)

    def set_google_credentials(self, credentials_path: str):
        """Persist Google application credentials and refresh process state."""

        if not credentials_path:
            raise ValueError("Google credentials path cannot be empty.")

        self._persist_env_value("GOOGLE_APPLICATION_CREDENTIALS", credentials_path)
        self.logger.info("Google credentials path updated.")

    def _sync_provider_warning(self, env_key: str, value: Optional[str]) -> None:
        """Refresh pending provider warnings when credential values change."""

        provider_name = None
        lookup = getattr(self, "_provider_env_lookup", None)
        if isinstance(lookup, Mapping):
            provider_name = lookup.get(env_key)
        else:
            for provider, candidate in self._get_provider_env_keys().items():
                if candidate == env_key:
                    provider_name = provider
                    break

        if not provider_name:
            return

        if value:
            self._pending_provider_warnings.pop(provider_name, None)
        else:
            warning_message = (
                f"API key for provider '{provider_name}' is not configured. "
                "Protected features will remain unavailable until a key is provided."
            )
            self._pending_provider_warnings[provider_name] = warning_message

    def get_pending_provider_warnings(self) -> Dict[str, str]:
        """Return provider credential warnings that should be surfaced to operators."""

        return dict(self._pending_provider_warnings)

    def is_default_provider_ready(self) -> bool:
        """Return True when the configured default provider has a usable credential."""

        default_provider = self.get_default_provider()
        if not default_provider:
            return True
        return default_provider not in self._pending_provider_warnings

    def get_google_speech_settings(self) -> Dict[str, Any]:
        """Return persisted Google speech preferences when available."""

        block = self.yaml_config.get("GOOGLE_SPEECH")
        if not isinstance(block, dict):
            block = {}

        settings = {
            "tts_voice": block.get("tts_voice"),
            "stt_language": block.get("stt_language"),
            "auto_punctuation": block.get("auto_punctuation"),
        }

        if settings["auto_punctuation"] is not None:
            settings["auto_punctuation"] = bool(settings["auto_punctuation"])

        return settings

    def set_google_speech_settings(
        self,
        *,
        tts_voice: Optional[str] = None,
        stt_language: Optional[str] = None,
        auto_punctuation: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Persist Google speech preferences to the YAML configuration."""

        block = {}
        existing = self.yaml_config.get("GOOGLE_SPEECH")
        if isinstance(existing, dict):
            block.update(existing)

        if tts_voice is None:
            block.pop("tts_voice", None)
        else:
            block["tts_voice"] = tts_voice

        if stt_language is None:
            block.pop("stt_language", None)
        else:
            block["stt_language"] = stt_language

        if auto_punctuation is None:
            block.pop("auto_punctuation", None)
        else:
            block["auto_punctuation"] = bool(auto_punctuation)

        if block:
            self.yaml_config['GOOGLE_SPEECH'] = block
            self.config['GOOGLE_SPEECH'] = dict(block)
        else:
            self.yaml_config.pop('GOOGLE_SPEECH', None)
            self.config.pop('GOOGLE_SPEECH', None)

        self._write_yaml_config()
        return self.get_google_speech_settings()

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

        if api_key is not None:
            if not api_key:
                raise ValueError("OpenAI API key cannot be empty.")
            self._persist_env_value("OPENAI_API_KEY", api_key)
            self.logger.info("OpenAI API key updated for speech services.")

        config_updates = {
            "OPENAI_STT_PROVIDER": stt_provider,
            "OPENAI_TTS_PROVIDER": tts_provider,
            "OPENAI_LANGUAGE": language,
            "OPENAI_TASK": task,
            "OPENAI_INITIAL_PROMPT": initial_prompt,
        }

        for key, value in config_updates.items():
            if value is not None:
                self.config[key] = value
            elif key in self.config:
                self.config[key] = None

        block = {}
        existing = self.yaml_config.get("OPENAI_SPEECH")
        if isinstance(existing, Mapping):
            block.update(existing)

        block_updates = {
            "stt_provider": stt_provider,
            "tts_provider": tts_provider,
            "language": language,
            "task": task,
            "initial_prompt": initial_prompt,
        }

        for block_key, value in block_updates.items():
            if value is None:
                block.pop(block_key, None)
            else:
                block[block_key] = value

        if block:
            self.yaml_config["OPENAI_SPEECH"] = block
            self.config["OPENAI_SPEECH"] = dict(block)
        else:
            self.yaml_config.pop("OPENAI_SPEECH", None)
            self.config.pop("OPENAI_SPEECH", None)

        self._write_yaml_config()

        # Re-apply the speech block to ensure top-level config stays in sync
        self._synchronize_openai_speech_block()

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

        if not model:
            raise ValueError("A default OpenAI model must be provided.")

        normalized_temperature = 0.0 if temperature is None else float(temperature)
        if not 0.0 <= normalized_temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0.")

        normalized_top_p = 1.0 if top_p is None else float(top_p)
        if not 0.0 <= normalized_top_p <= 1.0:
            raise ValueError("Top-p must be between 0.0 and 1.0.")

        normalized_frequency_penalty = 0.0 if frequency_penalty is None else float(frequency_penalty)
        if not -2.0 <= normalized_frequency_penalty <= 2.0:
            raise ValueError("Frequency penalty must be between -2.0 and 2.0.")

        normalized_presence_penalty = 0.0 if presence_penalty is None else float(presence_penalty)
        if not -2.0 <= normalized_presence_penalty <= 2.0:
            raise ValueError("Presence penalty must be between -2.0 and 2.0.")

        normalized_max_tokens = 4000 if max_tokens is None else int(max_tokens)
        if normalized_max_tokens <= 0:
            raise ValueError("Max tokens must be a positive integer.")

        normalized_max_output_tokens = (
            None if max_output_tokens is None else int(max_output_tokens)
        )
        if normalized_max_output_tokens is not None and normalized_max_output_tokens <= 0:
            raise ValueError("Max output tokens must be a positive integer when provided.")

        normalized_stream = True if stream is None else bool(stream)
        normalized_function_calling = True if function_calling is None else bool(function_calling)

        allowed_effort = {"low", "medium", "high"}
        if reasoning_effort is None:
            normalized_reasoning_effort = "medium"
        else:
            normalized_reasoning_effort = str(reasoning_effort).lower()
            if normalized_reasoning_effort not in allowed_effort:
                raise ValueError(
                    "Reasoning effort must be one of: low, medium, high."
                )

        def _normalize_json_mode(value: Any, existing: bool) -> bool:
            if value is None:
                return existing
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if not normalized:
                    return existing
                if normalized in {"1", "true", "yes", "on", "json", "json_object"}:
                    return True
                if normalized in {"0", "false", "no", "off", "text", "none"}:
                    return False
                return existing
            try:
                return bool(value)
            except Exception:
                return existing

        def _normalize_json_schema(
            value: Any, existing: Optional[Dict[str, Any]]
        ) -> Optional[Dict[str, Any]]:
            if value is None:
                return existing

            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")

            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON schema must be valid JSON: {exc.msg}"
                    ) from exc

            if value is False:
                return None

            if not isinstance(value, dict):
                raise ValueError("JSON schema must be provided as an object or JSON string.")

            if not value:
                return None

            schema_payload = value.get("schema") if isinstance(value, dict) else None
            schema_name = value.get("name") if isinstance(value, dict) else None

            if schema_payload is None:
                schema_payload = value
                schema_like_keys = {
                    "$schema",
                    "$ref",
                    "type",
                    "properties",
                    "items",
                    "oneOf",
                    "anyOf",
                    "allOf",
                    "definitions",
                    "patternProperties",
                }
                if isinstance(schema_payload, dict) and not (
                    schema_like_keys & set(schema_payload.keys())
                ):
                    raise ValueError(
                        "JSON schema must include a 'schema' object or a valid schema definition."
                    )

            if not isinstance(schema_payload, dict):
                raise ValueError("The 'schema' entry for the JSON schema must be an object.")

            if schema_name is None:
                if isinstance(existing, dict):
                    schema_name = existing.get("name")
                if not schema_name:
                    schema_name = "atlas_response"

            normalized: Dict[str, Any] = {
                "name": str(schema_name).strip() or "atlas_response",
                "schema": schema_payload,
            }

            strict_value = value.get("strict") if isinstance(value, dict) else None
            if strict_value is None and isinstance(existing, dict):
                strict_value = existing.get("strict")

            if strict_value is not None:
                normalized["strict"] = bool(strict_value)

            # Deep-copy via JSON round-trip to avoid mutating caller-owned structures.
            try:
                return json.loads(json.dumps(normalized))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"JSON schema contains non-serializable content: {exc}") from exc

        sanitized_base_url = (base_url or "").strip() or None
        sanitized_org = (organization or "").strip() or None

        settings_block = {}
        existing = self.yaml_config.get('OPENAI_LLM')
        if isinstance(existing, dict):
            settings_block.update(existing)

        previous_json_mode = bool(settings_block.get('json_mode', False))
        normalized_json_mode = _normalize_json_mode(json_mode, previous_json_mode)

        previous_schema = settings_block.get('json_schema')
        if not isinstance(previous_schema, dict):
            previous_schema = None
        normalized_json_schema = _normalize_json_schema(json_schema, previous_schema)

        previous_parallel_tool_calls = bool(settings_block.get('parallel_tool_calls', True))
        if parallel_tool_calls is None:
            normalized_parallel_tool_calls = previous_parallel_tool_calls
        else:
            normalized_parallel_tool_calls = bool(parallel_tool_calls)

        previous_code_interpreter = bool(settings_block.get('enable_code_interpreter', False))
        if enable_code_interpreter is None:
            normalized_code_interpreter = previous_code_interpreter
        else:
            normalized_code_interpreter = bool(enable_code_interpreter)

        previous_file_search = bool(settings_block.get('enable_file_search', False))
        if enable_file_search is None:
            normalized_file_search = previous_file_search
        else:
            normalized_file_search = bool(enable_file_search)

        previous_audio_enabled = bool(settings_block.get("audio_enabled", False))
        if audio_enabled is None:
            normalized_audio_enabled = previous_audio_enabled
        else:
            normalized_audio_enabled = bool(audio_enabled)

        def _normalize_audio_string(value: Optional[str], existing: Optional[str]) -> Optional[str]:
            if value is None:
                return existing

            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return None
                return cleaned

            return existing

        previous_voice = settings_block.get("audio_voice")
        previous_format = settings_block.get("audio_format")

        normalized_audio_voice = _normalize_audio_string(audio_voice, previous_voice)

        normalized_audio_format = _normalize_audio_string(audio_format, previous_format)
        if isinstance(normalized_audio_format, str):
            normalized_audio_format = normalized_audio_format.lower()

        def _normalize_tool_choice(value: Optional[str], existing_value: Optional[str]) -> Optional[str]:
            if value is None:
                return existing_value

            if isinstance(value, str):
                normalized_value = value.strip().lower()
                if not normalized_value:
                    return None
                if normalized_value in {"auto", "none", "required"}:
                    return normalized_value

            return existing_value

        normalized_tool_choice = _normalize_tool_choice(tool_choice, settings_block.get('tool_choice'))

        if not normalized_function_calling:
            normalized_code_interpreter = False
            normalized_file_search = False

        settings_block.update(
            {
                'model': model,
                'temperature': normalized_temperature,
                'top_p': normalized_top_p,
                'frequency_penalty': normalized_frequency_penalty,
                'presence_penalty': normalized_presence_penalty,
                'max_tokens': normalized_max_tokens,
                'max_output_tokens': normalized_max_output_tokens,
                'stream': normalized_stream,
                'function_calling': normalized_function_calling,
                'parallel_tool_calls': normalized_parallel_tool_calls,
                'tool_choice': normalized_tool_choice,
                'reasoning_effort': normalized_reasoning_effort,
                'base_url': sanitized_base_url,
                'organization': sanitized_org,
                'json_mode': normalized_json_mode,
                'json_schema': normalized_json_schema,
                'enable_code_interpreter': normalized_code_interpreter,
                'enable_file_search': normalized_file_search,
                'audio_enabled': normalized_audio_enabled,
                'audio_voice': normalized_audio_voice,
                'audio_format': normalized_audio_format,
            }
        )

        self.yaml_config['OPENAI_LLM'] = settings_block
        self.config['OPENAI_LLM'] = dict(settings_block)

        # Persist environment-backed values.
        self._persist_env_value('DEFAULT_MODEL', model)
        self.config['DEFAULT_MODEL'] = model

        self._persist_env_value('OPENAI_BASE_URL', sanitized_base_url)
        self._persist_env_value('OPENAI_ORGANIZATION', sanitized_org)

        # Synchronize cached environment map for convenience.
        self.env_config['DEFAULT_MODEL'] = model
        self.env_config['OPENAI_BASE_URL'] = sanitized_base_url
        self.env_config['OPENAI_ORGANIZATION'] = sanitized_org

        self._write_yaml_config()

        return dict(settings_block)


    def get_google_llm_settings(self) -> Dict[str, Any]:
        """Return the persisted Google LLM defaults, if configured."""

        defaults: Dict[str, Any] = {
            'stream': True,
            'function_calling': True,
            'function_call_mode': 'auto',
            'allowed_function_names': [],
            'cached_allowed_function_names': [],
            'response_schema': {},
            'seed': None,
            'response_logprobs': False,
        }

        settings = self.yaml_config.get('GOOGLE_LLM')
        if isinstance(settings, dict):
            normalized = copy.deepcopy(settings)
        else:
            normalized = {}

        merged: Dict[str, Any] = copy.deepcopy(defaults)
        merged.update(normalized)
        resolver = GoogleSettingsResolver(stored=normalized, defaults=defaults)
        merged['stream'] = resolver.resolve_bool(
            'stream',
            None,
            default=True,
        )
        merged['function_calling'] = resolver.resolve_bool(
            'function_calling',
            None,
            default=True,
        )
        try:
            merged['seed'] = resolver.resolve_seed(
                None,
                allow_invalid_stored=True,
            )
        except ValueError as exc:
            self.logger.warning(
                "Ignoring persisted Google seed due to validation error: %s",
                exc,
            )
            merged['seed'] = None
        merged['response_logprobs'] = resolver.resolve_bool(
            'response_logprobs',
            None,
            default=False,
        )
        try:
            merged['function_call_mode'] = self._coerce_function_call_mode(
                merged.get('function_call_mode'),
                default='auto',
            )
        except ValueError as exc:
            self.logger.warning(
                "Ignoring persisted Google function call mode due to validation error: %s",
                exc,
            )
            merged['function_call_mode'] = 'auto'

        try:
            merged['allowed_function_names'] = self._coerce_allowed_function_names(
                merged.get('allowed_function_names')
            )
        except ValueError as exc:
            self.logger.warning(
                "Ignoring persisted Google allowed function names due to validation error: %s",
                exc,
            )
            merged['allowed_function_names'] = []

        try:
            merged['cached_allowed_function_names'] = self._coerce_allowed_function_names(
                merged.get('cached_allowed_function_names')
            )
        except ValueError as exc:
            self.logger.warning(
                "Ignoring persisted Google cached allowed names due to validation error: %s",
                exc,
            )
            merged['cached_allowed_function_names'] = []

        try:
            merged['response_schema'] = resolver.resolve_response_schema(
                None,
                allow_invalid_stored=True,
            )
        except ValueError as exc:
            self.logger.warning(
                "Ignoring persisted Google response schema due to validation error: %s",
                exc,
            )
            merged['response_schema'] = {}

        return merged

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
    ) -> Dict[str, Any]:
        """Persist default configuration for the Google Gemini provider.

        Args:
            model: Default Gemini model identifier.
            temperature: Sampling temperature between 0 and 2.
            top_p: Nucleus sampling threshold between 0 and 1.
            top_k: Optional integer limiting candidate tokens considered.
            candidate_count: Optional integer specifying number of returned candidates.
            max_output_tokens: Optional integer limiting response length. ``None`` clears
                the limit while ``32000`` is used by default when not overridden.
            stop_sequences: Optional collection of strings used to halt generation.
            safety_settings: Optional safety filter configuration.
            response_mime_type: Optional MIME type hint for responses.
            system_instruction: Optional default system instruction prompt.
            stream: Optional flag to toggle streaming responses by default.
            function_calling: Optional flag that enables Gemini tool calling by default.
            response_schema: Optional JSON schema applied to structured responses.
            cached_allowed_function_names: Optional sequence preserving the allowlist
                when tool calling is temporarily disabled.
            seed: Optional deterministic seed applied to Gemini generations.
            response_logprobs: Optional flag requesting token log probabilities.

        Returns:
            dict: Persisted Google defaults.
        """

        if not isinstance(model, str) or not model.strip():
            raise ValueError("A default Google model must be provided.")

        defaults = {
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': None,
            'candidate_count': 1,
            'max_output_tokens': 32000,
            'stop_sequences': [],
            'safety_settings': [],
            'response_mime_type': None,
            'system_instruction': None,
            'stream': True,
            'function_calling': True,
            'function_call_mode': 'auto',
            'allowed_function_names': [],
            'cached_allowed_function_names': [],
            'response_schema': {},
            'seed': None,
            'response_logprobs': False,
        }

        existing_settings = {}
        stored_block = self.yaml_config.get('GOOGLE_LLM')
        if isinstance(stored_block, dict):
            existing_settings = copy.deepcopy(stored_block)

        settings_block: Dict[str, Any] = copy.deepcopy(defaults)
        settings_block.update(existing_settings)
        settings_block['model'] = model.strip()

        def _normalize_stop_sequences(
            value: Optional[Any],
            previous: Any,
        ) -> List[str]:
            if value is None:
                if isinstance(previous, list):
                    return list(previous)
                if previous in {None, ""}:
                    return []
                return self._coerce_stop_sequences(previous)
            return self._coerce_stop_sequences(value)

        def _coerce_safety_settings(
            value: Optional[Any],
            previous: Any,
        ) -> List[Dict[str, str]]:
            if value is None:
                return copy.deepcopy(previous) if isinstance(previous, list) else []

            if value in ({}, []):
                return []

            normalized: List[Dict[str, str]] = []

            if isinstance(value, Mapping):
                for category, threshold in value.items():
                    cleaned_category = str(category).strip()
                    if not cleaned_category:
                        continue
                    if threshold in {None, ""}:
                        raise ValueError("Safety setting threshold cannot be empty.")
                    normalized.append(
                        {
                            'category': cleaned_category,
                            'threshold': str(threshold).strip(),
                        }
                    )
                return normalized

            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for entry in value:
                    if entry is None:
                        continue
                    if isinstance(entry, str) and not entry.strip():
                        continue
                    if not isinstance(entry, Mapping):
                        raise ValueError(
                            "Safety settings must be provided as mappings or mapping sequences."
                        )
                    category = (
                        entry.get('category')
                        or entry.get('harmCategory')
                        or entry.get('name')
                    )
                    threshold = (
                        entry.get('threshold')
                        or entry.get('thresholdValue')
                        or entry.get('harmBlockThreshold')
                    )
                    if not category:
                        raise ValueError(
                            "Safety setting entries must include a category value."
                        )
                    if threshold in {None, ""}:
                        raise ValueError(
                            "Safety setting entries must include a threshold value."
                        )
                    normalized.append(
                        {
                            'category': str(category).strip(),
                            'threshold': str(threshold).strip(),
                        }
                    )
                return normalized

            raise ValueError(
                "Safety settings must be provided as a mapping or list of mappings."
            )

        def _normalize_optional_text(
            value: Optional[Any],
            previous: Optional[str],
        ) -> Optional[str]:
            if value is None:
                return previous
            cleaned = str(value).strip()
            return cleaned or None

        resolver = GoogleSettingsResolver(
            stored=existing_settings,
            defaults=defaults,
        )

        settings_block['temperature'] = resolver.resolve_float(
            'temperature',
            temperature,
            field='Temperature',
            minimum=0.0,
            maximum=2.0,
        )
        settings_block['top_p'] = resolver.resolve_float(
            'top_p',
            top_p,
            field='Top-p',
            minimum=0.0,
            maximum=1.0,
        )
        settings_block['top_k'] = resolver.resolve_optional_int(
            'top_k',
            top_k,
            field='Top-k',
            minimum=1,
        )
        settings_block['candidate_count'] = resolver.resolve_int(
            'candidate_count',
            candidate_count,
            field='Candidate count',
            minimum=1,
        )

        settings_block['stop_sequences'] = _normalize_stop_sequences(
            stop_sequences,
            settings_block.get('stop_sequences', defaults['stop_sequences']),
        )
        settings_block['safety_settings'] = _coerce_safety_settings(
            safety_settings,
            settings_block.get('safety_settings', defaults['safety_settings']),
        )
        settings_block['response_mime_type'] = _normalize_optional_text(
            response_mime_type,
            settings_block.get('response_mime_type', defaults['response_mime_type']),
        )
        settings_block['system_instruction'] = _normalize_optional_text(
            system_instruction,
            settings_block.get('system_instruction', defaults['system_instruction']),
        )
        settings_block['stream'] = resolver.resolve_bool(
            'stream',
            stream,
            default=defaults['stream'],
        )
        settings_block['function_calling'] = resolver.resolve_bool(
            'function_calling',
            function_calling,
            default=defaults['function_calling'],
        )
        settings_block['seed'] = resolver.resolve_seed(
            seed,
            allow_invalid_stored=True,
        )
        settings_block['response_logprobs'] = resolver.resolve_bool(
            'response_logprobs',
            response_logprobs,
            default=defaults['response_logprobs'],
        )

        candidate_mode: Any
        if function_call_mode is not None:
            candidate_mode = function_call_mode
        else:
            candidate_mode = existing_settings.get(
                'function_call_mode', defaults['function_call_mode']
            )

        if candidate_mode in {None, ""}:
            candidate_mode = defaults['function_call_mode']

        if isinstance(candidate_mode, str):
            try:
                settings_block['function_call_mode'] = self._coerce_function_call_mode(
                    candidate_mode,
                    default=defaults['function_call_mode'],
                )
            except ValueError as exc:
                raise ValueError(
                    "Function call mode must be one of: auto, any, none, require."
                ) from exc
        else:
            raise ValueError(
                "Function call mode must be a string value."
            )

        def _normalize_allowed_function_names(
            value: Optional[Any],
            previous: Any,
        ) -> List[str]:
            if value is None:
                source = previous
            else:
                source = value
            try:
                names = self._coerce_allowed_function_names(source)
            except ValueError as exc:
                raise ValueError(
                    "Allowed function names must be a sequence of non-empty strings."
                ) from exc
            return names

        previous_allowed = settings_block.get('allowed_function_names', defaults['allowed_function_names'])
        if not previous_allowed:
            previous_allowed = settings_block.get(
                'cached_allowed_function_names', defaults['cached_allowed_function_names']
            )
        normalized_allowed = _normalize_allowed_function_names(
            allowed_function_names,
            previous_allowed,
        )

        try:
            existing_cache = self._coerce_allowed_function_names(
                settings_block.get('cached_allowed_function_names', defaults['cached_allowed_function_names'])
            )
        except ValueError:
            existing_cache = []

        if cached_allowed_function_names is _UNSET:
            cache_source: Any = normalized_allowed if normalized_allowed else existing_cache
        else:
            cache_source = cached_allowed_function_names

        try:
            normalized_cache = self._coerce_allowed_function_names(cache_source)
        except ValueError as exc:
            raise ValueError(
                "Cached allowed function names must be a sequence of non-empty strings."
            ) from exc

        settings_block['cached_allowed_function_names'] = normalized_cache

        if settings_block['function_calling'] is False:
            settings_block['function_call_mode'] = 'none'

        if settings_block['function_call_mode'] == 'none':
            settings_block['allowed_function_names'] = []
        else:
            if (
                not normalized_allowed
                and normalized_cache
                and cached_allowed_function_names is _UNSET
                and allowed_function_names is None
            ):
                settings_block['allowed_function_names'] = list(normalized_cache)
            else:
                settings_block['allowed_function_names'] = normalized_allowed
        settings_block['max_output_tokens'] = resolver.resolve_max_output_tokens(
            max_output_tokens,
        )

        schema = resolver.resolve_response_schema(
            response_schema,
            allow_invalid_stored=True,
        )

        if schema:
            mime_value = settings_block.get('response_mime_type') or ''
            normalized_mime = str(mime_value).strip().lower()
            if not normalized_mime:
                settings_block['response_mime_type'] = 'application/json'
            elif normalized_mime != 'application/json':
                raise ValueError(
                    "Response MIME type must be 'application/json' when a response schema is provided."
                )
            else:
                settings_block['response_mime_type'] = 'application/json'

        settings_block['response_schema'] = schema

        self.yaml_config['GOOGLE_LLM'] = copy.deepcopy(settings_block)
        self.config['GOOGLE_LLM'] = copy.deepcopy(settings_block)

        self._write_yaml_config()

        return copy.deepcopy(settings_block)


    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value by its key.

        Args:
            key (str): The configuration key to retrieve.
            default (Any, optional): The default value to return if the key is not found.

        Returns:
            Any: The value associated with the key, or the default value if key is absent.
        """
        return self.config.get(key, default)

    def get_javascript_executor_settings(self) -> Dict[str, Any]:
        """Return a sanitized settings block for the JavaScript executor."""

        tools_block = self.get_config('tools', {})
        settings: Dict[str, Any] = {}
        if isinstance(tools_block, Mapping):
            candidate = tools_block.get('javascript_executor')
            if isinstance(candidate, Mapping):
                settings.update(candidate)

        args_value = settings.get('args')
        if isinstance(args_value, str):
            try:
                settings['args'] = shlex.split(args_value)
            except ValueError:
                settings['args'] = [args_value]
        elif isinstance(args_value, Sequence) and not isinstance(args_value, (str, bytes)):
            settings['args'] = list(args_value)

        return settings

    def get_model_cache_dir(self) -> str:
        """
        Retrieves the directory path where models are cached.

        Returns:
            str: The path to the model cache directory.
        """
        return self.get_config('MODEL_CACHE_DIR')

    def get_cached_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """Return cached provider models persisted via the configuration backend."""

        cache = copy.deepcopy(self._model_cache)

        if provider is not None:
            key = provider.strip() if isinstance(provider, str) else str(provider)
            return {key: cache.get(key, [])}

        return cache

    def set_cached_models(self, provider: str, models: Iterable[Any]) -> List[str]:
        """Persist the cached model list for a provider to the YAML configuration."""

        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Provider name must be a non-empty string.")

        provider_key = provider.strip()

        seen: set[str] = set()
        ordered: List[str] = []
        for entry in models:
            if isinstance(entry, str):
                candidate = entry.strip()
            else:
                candidate = str(entry).strip() if entry is not None else ""

            if not candidate or candidate in seen:
                continue

            ordered.append(candidate)
            seen.add(candidate)

        self._model_cache[provider_key] = list(ordered)
        self.config["MODEL_CACHE"] = copy.deepcopy(self._model_cache)
        self.yaml_config["MODEL_CACHE"] = copy.deepcopy(self._model_cache)
        self._write_yaml_config()

        return list(ordered)

    def get_speech_cache_dir(self) -> str:
        """Return the directory path used to cache generated speech files."""

        cache_dir = self.get_config('SPEECH_CACHE_DIR')
        if cache_dir:
            return cache_dir

        app_root = self.get_app_root() or '.'
        cache_dir = os.path.join(app_root, 'assets', 'SCOUT', 'tts_mp3')
        self.config['SPEECH_CACHE_DIR'] = cache_dir
        return cache_dir

    def get_default_provider(self) -> str:
        """
        Retrieves the default provider name from the configuration.

        Returns:
            str: The name of the default provider.
        """
        return self.get_config('DEFAULT_PROVIDER')

    def get_default_model(self) -> str:
        """
        Retrieves the default model name from the configuration.

        Returns:
            str: The name of the default model.
        """
        return self.get_config('DEFAULT_MODEL')

    def get_active_user(self) -> Optional[str]:
        """Return the username persisted as the active account."""

        value = self.get_config('ACTIVE_USER')
        if isinstance(value, str):
            sanitized = value.strip()
            if sanitized:
                return sanitized
        return None

    def set_active_user(self, username: Optional[str]) -> Optional[str]:
        """Persist the active user to the YAML configuration."""

        sanitized: Optional[str]
        if isinstance(username, str):
            sanitized = username.strip() or None
        else:
            sanitized = None

        if sanitized is None:
            self.yaml_config.pop('ACTIVE_USER', None)
            self.config.pop('ACTIVE_USER', None)
            self.logger.debug("Cleared active user from configuration")
        else:
            self.yaml_config['ACTIVE_USER'] = sanitized
            self.config['ACTIVE_USER'] = sanitized
            self.logger.info("Persisted active user '%s'", sanitized)

        self._write_yaml_config()
        return sanitized

    def get_openai_llm_settings(self) -> Dict[str, Any]:
        """Return persisted OpenAI LLM defaults merged with environment values."""

        defaults = {
            'model': self.get_config('DEFAULT_MODEL', 'gpt-4o'),
            'temperature': 0.0,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'max_tokens': 4000,
            'max_output_tokens': None,
            'stream': True,
            'function_calling': True,
            'parallel_tool_calls': True,
            'tool_choice': None,
            'reasoning_effort': 'medium',
            'base_url': self.get_config('OPENAI_BASE_URL'),
            'organization': self.get_config('OPENAI_ORGANIZATION'),
            'json_mode': False,
            'json_schema': None,
            'enable_code_interpreter': False,
            'enable_file_search': False,
            'audio_enabled': False,
            'audio_voice': 'alloy',
            'audio_format': 'wav',
        }

        stored = self.get_config('OPENAI_LLM')
        if isinstance(stored, dict):
            defaults.update({k: stored.get(k, defaults.get(k)) for k in defaults.keys()})

        return defaults


    def get_mistral_llm_settings(self) -> Dict[str, Any]:
        """Return persisted defaults for the Mistral chat provider."""

        defaults: Dict[str, Any] = {
            'model': 'mistral-large-latest',
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': None,
            'safe_prompt': False,
            'stream': True,
            'random_seed': None,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'tool_choice': 'auto',
            'parallel_tool_calls': True,
            'stop_sequences': [],
            'json_mode': False,
            'json_schema': None,
            'max_retries': 3,
            'retry_min_seconds': 4,
            'retry_max_seconds': 10,
            'base_url': self.get_config('MISTRAL_BASE_URL'),
            'prompt_mode': None,
        }

        stored = self.get_config('MISTRAL_LLM')
        if isinstance(stored, dict):
            merged = dict(defaults)

            model = stored.get('model')
            if isinstance(model, str) and model.strip():
                merged['model'] = model.strip()

            def _coerce_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
                if value is None:
                    return default
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return default
                if number < minimum or number > maximum:
                    return default
                return number

            def _coerce_int(value: Any, *, allow_zero: bool = False) -> Optional[int]:
                if value is None or value == "":
                    return None
                try:
                    candidate = int(value)
                except (TypeError, ValueError):
                    return None
                if candidate < 0:
                    return None
                if candidate == 0 and not allow_zero:
                    return None
                return candidate

            def _coerce_bool(value: Any, default: bool) -> bool:
                if value is None:
                    return default
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if normalized in {"1", "true", "yes", "on"}:
                        return True
                    if normalized in {"0", "false", "no", "off"}:
                        return False
                    return default
                return bool(value)

            def _coerce_json_schema(value: Any) -> Optional[Dict[str, Any]]:
                if value is None or value == "":
                    return None

                if isinstance(value, (bytes, bytearray)):
                    value = value.decode("utf-8")

                if isinstance(value, str):
                    text = value.strip()
                    if not text:
                        return None
                    try:
                        value = json.loads(text)
                    except json.JSONDecodeError:
                        return None

                if not isinstance(value, dict):
                    return None

                try:
                    return json.loads(json.dumps(value))
                except (TypeError, ValueError):
                    return None

            merged['temperature'] = _coerce_float(
                stored.get('temperature'),
                default=float(defaults['temperature']),
                minimum=0.0,
                maximum=2.0,
            )
            merged['top_p'] = _coerce_float(
                stored.get('top_p'),
                default=float(defaults['top_p']),
                minimum=0.0,
                maximum=1.0,
            )
            max_tokens = _coerce_int(stored.get('max_tokens'))
            merged['max_tokens'] = max_tokens

            merged['safe_prompt'] = _coerce_bool(
                stored.get('safe_prompt'),
                default=bool(defaults['safe_prompt']),
            )
            merged['stream'] = _coerce_bool(
                stored.get('stream'),
                default=bool(defaults['stream']),
            )
            merged['parallel_tool_calls'] = _coerce_bool(
                stored.get('parallel_tool_calls'),
                default=bool(defaults['parallel_tool_calls']),
            )

            random_seed = _coerce_int(stored.get('random_seed'), allow_zero=True)
            merged['random_seed'] = random_seed

            merged['frequency_penalty'] = _coerce_float(
                stored.get('frequency_penalty'),
                default=float(defaults['frequency_penalty']),
                minimum=-2.0,
                maximum=2.0,
            )
            merged['presence_penalty'] = _coerce_float(
                stored.get('presence_penalty'),
                default=float(defaults['presence_penalty']),
                minimum=-2.0,
                maximum=2.0,
            )

            tool_choice = stored.get('tool_choice')
            if isinstance(tool_choice, Mapping):
                merged['tool_choice'] = dict(tool_choice)
            elif isinstance(tool_choice, str):
                merged['tool_choice'] = tool_choice.strip() or defaults['tool_choice']
            elif tool_choice is None:
                merged['tool_choice'] = defaults['tool_choice']
            else:
                merged['tool_choice'] = defaults['tool_choice']

            try:
                merged['stop_sequences'] = self._coerce_stop_sequences(
                    stored.get('stop_sequences')
                )
            except ValueError:
                merged['stop_sequences'] = list(defaults['stop_sequences'])

            merged['json_mode'] = _coerce_bool(
                stored.get('json_mode'),
                default=bool(defaults['json_mode']),
            )
            merged['json_schema'] = _coerce_json_schema(stored.get('json_schema'))

            def _coerce_prompt_mode(value: Any) -> Optional[str]:
                if value in {None, ""}:
                    return None
                if isinstance(value, (bytes, bytearray)):
                    try:
                        value = value.decode("utf-8")
                    except Exception:
                        return None
                if isinstance(value, str):
                    normalized = value.strip().lower()
                    if not normalized:
                        return None
                    if normalized in _SUPPORTED_MISTRAL_PROMPT_MODES:
                        return normalized
                    return None
                return None

            merged['prompt_mode'] = _coerce_prompt_mode(stored.get('prompt_mode'))

            def _coerce_base_url(value: Any) -> Optional[str]:
                if value in {None, ""}:
                    return None
                if isinstance(value, (bytes, bytearray)):
                    try:
                        value = value.decode('utf-8')
                    except Exception:
                        return None
                if isinstance(value, str):
                    candidate = value.strip()
                    if not candidate:
                        return None
                    parsed = urlparse(candidate)
                    if parsed.scheme in {"http", "https"} and parsed.netloc:
                        return candidate
                    return None
                return None

            merged['base_url'] = _coerce_base_url(stored.get('base_url'))

            retries = _coerce_int(stored.get('max_retries'))
            merged['max_retries'] = retries or defaults['max_retries']

            retry_min = _coerce_int(stored.get('retry_min_seconds'))
            if retry_min is None:
                retry_min = defaults['retry_min_seconds']
            retry_max_candidate = _coerce_int(stored.get('retry_max_seconds'))
            if retry_max_candidate is None:
                retry_max = max(retry_min, defaults['retry_max_seconds'])
            else:
                retry_max = max(retry_max_candidate, retry_min)
            merged['retry_min_seconds'] = retry_min
            merged['retry_max_seconds'] = retry_max

            return merged

        return dict(defaults)


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

        if not isinstance(model, str) or not model.strip():
            raise ValueError("A default Mistral model must be provided.")

        current = self.get_mistral_llm_settings()
        settings = dict(current)
        settings['model'] = model.strip()

        def _normalize_float(
            value: Optional[Any], *, field: str, default: float, minimum: float, maximum: float
        ) -> float:
            if value is None:
                return default
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a number.") from exc
            if numeric < minimum or numeric > maximum:
                raise ValueError(
                    f"{field} must be between {minimum} and {maximum}."
                )
            return numeric

        def _normalize_positive_int(
            value: Optional[Any],
            field: str,
            *,
            allow_zero: bool = False,
            zero_means_none: bool = False,
        ) -> Optional[int]:
            if value is None or value == "":
                return None
            try:
                numeric = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be an integer.") from exc
            if numeric < 0:
                raise ValueError(f"{field} must be a non-negative integer.")
            if numeric == 0:
                if zero_means_none:
                    return None
                if not allow_zero:
                    raise ValueError(f"{field} must be a positive integer.")
            return numeric

        def _normalize_bool(value: Optional[Any]) -> Optional[bool]:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True
                if normalized in {"0", "false", "no", "off"}:
                    return False
                raise ValueError("Boolean fields must be provided as a boolean or yes/no string.")
            return bool(value)

        def _normalize_tool_choice(value: Optional[Any]) -> Any:
            if value is None:
                return settings.get('tool_choice')
            if isinstance(value, Mapping):
                return dict(value)
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return None
                return cleaned
            raise ValueError(
                "Tool choice must be a mapping, string, or None."
            )

        def _normalize_json_mode(value: Any, existing: bool) -> bool:
            if value is None:
                return existing
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if not normalized:
                    return existing
                if normalized in {"1", "true", "yes", "on", "json", "json_object"}:
                    return True
                if normalized in {"0", "false", "no", "off", "text", "none"}:
                    return False
                return existing
            try:
                return bool(value)
            except Exception:
                return existing

        def _normalize_json_schema(
            value: Any, existing: Optional[Dict[str, Any]]
        ) -> Optional[Dict[str, Any]]:
            if value is None:
                return existing

            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")

            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON schema must be valid JSON: {exc.msg}"
                    ) from exc

            if value is False:
                return None

            if not isinstance(value, dict):
                raise ValueError(
                    "JSON schema must be provided as an object or JSON string."
                )

            if not value:
                return None

            schema_payload = value.get('schema') if isinstance(value, dict) else None
            schema_name = value.get('name') if isinstance(value, dict) else None

            if schema_payload is None:
                schema_payload = value
                schema_like_keys = {
                    '$schema',
                    '$ref',
                    'type',
                    'properties',
                    'items',
                    'oneOf',
                    'anyOf',
                    'allOf',
                    'definitions',
                    'patternProperties',
                }
                if isinstance(schema_payload, dict) and not (
                    schema_like_keys & set(schema_payload.keys())
                ):
                    raise ValueError(
                        "JSON schema must include a 'schema' object or a valid schema definition."
                    )

            if not isinstance(schema_payload, dict):
                raise ValueError(
                    "The 'schema' entry for the JSON schema must be an object."
                )

            if schema_name is None:
                if isinstance(existing, dict):
                    schema_name = existing.get('name')
                if not schema_name:
                    schema_name = 'atlas_response'

            normalized: Dict[str, Any] = {
                'name': str(schema_name).strip() or 'atlas_response',
                'schema': schema_payload,
            }

            strict_value = value.get('strict') if isinstance(value, dict) else None
            if strict_value is None and isinstance(existing, dict):
                strict_value = existing.get('strict')

            if strict_value is not None:
                normalized['strict'] = bool(strict_value)

            try:
                return json.loads(json.dumps(normalized))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"JSON schema contains non-serializable content: {exc}"
                ) from exc

        settings['temperature'] = _normalize_float(
            temperature,
            field='Temperature',
            default=float(settings.get('temperature', 0.0)),
            minimum=0.0,
            maximum=2.0,
        )
        settings['top_p'] = _normalize_float(
            top_p,
            field='Top-p',
            default=float(settings.get('top_p', 1.0)),
            minimum=0.0,
            maximum=1.0,
        )
        tokens = _normalize_positive_int(
            max_tokens,
            'Max tokens',
            zero_means_none=True,
        )
        settings['max_tokens'] = tokens

        normalized_safe_prompt = _normalize_bool(safe_prompt)
        if normalized_safe_prompt is not None:
            settings['safe_prompt'] = normalized_safe_prompt

        normalized_stream = _normalize_bool(stream)
        if normalized_stream is not None:
            settings['stream'] = normalized_stream

        normalized_parallel = _normalize_bool(parallel_tool_calls)
        if normalized_parallel is not None:
            settings['parallel_tool_calls'] = normalized_parallel

        seed = (
            _normalize_positive_int(
                random_seed,
                'Random seed',
                allow_zero=True,
            )
            if random_seed not in {None, ""}
            else None
        )
        settings['random_seed'] = seed

        settings['frequency_penalty'] = _normalize_float(
            frequency_penalty,
            field='Frequency penalty',
            default=float(settings.get('frequency_penalty', 0.0)),
            minimum=-2.0,
            maximum=2.0,
        )
        settings['presence_penalty'] = _normalize_float(
            presence_penalty,
            field='Presence penalty',
            default=float(settings.get('presence_penalty', 0.0)),
            minimum=-2.0,
            maximum=2.0,
        )

        settings['tool_choice'] = _normalize_tool_choice(tool_choice)

        if stop_sequences is not _UNSET:
            settings['stop_sequences'] = self._coerce_stop_sequences(stop_sequences)

        settings['json_mode'] = _normalize_json_mode(
            json_mode,
            bool(settings.get('json_mode', False)),
        )
        settings['json_schema'] = _normalize_json_schema(
            json_schema,
            settings.get('json_schema'),
        )

        def _normalize_prompt_mode(value: Any, existing: Optional[str]) -> Optional[str]:
            if value is _UNSET:
                return existing
            if value in {None, ""}:
                return None
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode("utf-8")
                except Exception as exc:
                    raise ValueError("Prompt mode must be text.") from exc
            if isinstance(value, str):
                normalized = value.strip().lower()
                if not normalized:
                    return None
                if normalized not in _SUPPORTED_MISTRAL_PROMPT_MODES:
                    raise ValueError(
                        "Prompt mode must be one of: reasoning or left unset."
                    )
                return normalized
            raise ValueError("Prompt mode must be provided as text or None.")

        settings['prompt_mode'] = _normalize_prompt_mode(
            prompt_mode,
            settings.get('prompt_mode'),
        )

        def _normalize_base_url(value: Any, existing: Optional[str]) -> Optional[str]:
            if value is _UNSET:
                return existing
            if value in {None, ""}:
                return None
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode('utf-8')
                except Exception as exc:
                    raise ValueError('Base URL must be a valid HTTP(S) URL.') from exc
            if isinstance(value, str):
                candidate = value.strip()
                if not candidate:
                    return None
                parsed = urlparse(candidate)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    return candidate
                raise ValueError('Base URL must include an http:// or https:// scheme.')
            raise ValueError('Base URL must be provided as text.')

        settings['base_url'] = _normalize_base_url(base_url, settings.get('base_url'))

        if max_retries is not None:
            normalized_retries = _normalize_positive_int(
                max_retries,
                'Max retries',
            )
            if normalized_retries is None:
                raise ValueError('Max retries must be a positive integer.')
            settings['max_retries'] = normalized_retries

        current_retry_min = settings.get('retry_min_seconds', 4)
        if not isinstance(current_retry_min, (int, float)) or current_retry_min <= 0:
            current_retry_min = 4

        if retry_min_seconds is not None:
            normalized_retry_min = _normalize_positive_int(
                retry_min_seconds,
                'Retry minimum wait',
            )
            if normalized_retry_min is None:
                raise ValueError('Retry minimum wait must be a positive integer.')
            current_retry_min = normalized_retry_min
            settings['retry_min_seconds'] = current_retry_min
        else:
            settings['retry_min_seconds'] = int(current_retry_min)

        current_retry_max = settings.get('retry_max_seconds', max(current_retry_min, 10))
        if not isinstance(current_retry_max, (int, float)) or current_retry_max <= 0:
            current_retry_max = max(current_retry_min, 10)

        if retry_max_seconds is not None:
            normalized_retry_max = _normalize_positive_int(
                retry_max_seconds,
                'Retry maximum wait',
            )
            if normalized_retry_max is None:
                raise ValueError('Retry maximum wait must be a positive integer.')
            current_retry_max = normalized_retry_max

        if current_retry_max < current_retry_min:
            raise ValueError('Retry maximum wait must be greater than or equal to the minimum wait.')

        settings['retry_max_seconds'] = int(current_retry_max)

        self.yaml_config['MISTRAL_LLM'] = copy.deepcopy(settings)
        self.config['MISTRAL_LLM'] = copy.deepcopy(settings)
        self.config['MISTRAL_BASE_URL'] = settings.get('base_url')
        self.env_config['MISTRAL_BASE_URL'] = settings.get('base_url')

        self._write_yaml_config()

        return copy.deepcopy(settings)


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
        provider_env_keys = self._get_provider_env_keys()

        env_key = provider_env_keys.get(provider_name)
        if not env_key:
            raise ValueError(f"No API key mapping found for provider '{provider_name}'.")

        self._persist_env_value(env_key, new_api_key)
        self.logger.info(f"API key for {provider_name} updated successfully.")

    def has_provider_api_key(self, provider_name: str) -> bool:
        """
        Determine whether an API key is configured for the given provider.

        Args:
            provider_name (str): The name of the provider to check.

        Returns:
            bool: True if an API key exists for the provider, False otherwise.
        """

        return self._is_api_key_set(provider_name)

    def _is_api_key_set(self, provider_name: str) -> bool:
        """
        Checks if the API key for a specified provider is set.

        Args:
            provider_name (str): The name of the provider.

        Returns:
            bool: True if the API key is set, False otherwise.
        """
        env_key = self._get_provider_env_keys().get(provider_name)
        if not env_key:
            return False

        api_key = self.get_config(env_key)
        return bool(api_key)

    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves metadata for available providers without exposing raw secrets.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are provider names and values contain
            availability metadata such as whether the credential is set, a masked hint, and the
            stored length.
        """

        provider_env_keys = self._get_provider_env_keys()
        providers: Dict[str, Dict[str, Any]] = {}

        for provider, env_key in provider_env_keys.items():
            value = self.get_config(env_key)

            if value is None:
                secret = ""
            elif isinstance(value, str):
                secret = value
            else:
                secret = str(value)

            available = bool(secret)
            length = len(secret) if available else 0
            hint = self._mask_secret_preview(secret) if available else ""

            providers[provider] = {
                "available": available,
                "length": length,
                "hint": hint,
            }

        return providers

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
    def _extract_auth_env_keys(auth_block: Optional[Mapping[str, Any]]) -> List[str]:
        """Return normalized environment variable keys declared in a manifest auth block."""

        candidates: List[str] = []
        if isinstance(auth_block, Mapping):
            env_value = auth_block.get("env")
            if isinstance(env_value, str):
                candidates.append(env_value)
            elif isinstance(env_value, Sequence) and not isinstance(env_value, (str, bytes, bytearray)):
                for entry in env_value:
                    if isinstance(entry, str):
                        candidates.append(entry)

            envs_value = auth_block.get("envs")
            if isinstance(envs_value, Mapping):
                for entry in envs_value.values():
                    if isinstance(entry, str):
                        candidates.append(entry)
            elif isinstance(envs_value, Sequence) and not isinstance(envs_value, (str, bytes, bytearray)):
                for entry in envs_value:
                    if isinstance(entry, str):
                        candidates.append(entry)

        normalized: List[str] = []
        for candidate in candidates:
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

            env_keys = self._extract_auth_env_keys(auth_block)
            credentials = self._collect_credential_status(env_keys)

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

            env_keys = self._extract_auth_env_keys(auth_block)
            credentials = self._collect_credential_status(env_keys)

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

    def _write_yaml_config(self):
        """
        Writes the current YAML configuration back to the config.yaml file.
        """
        yaml_path = getattr(self, '_yaml_path', None) or self._compute_yaml_path()
        try:
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, 'w') as file:
                yaml.dump(self.yaml_config, file)
            self.logger.debug("Configuration written to %s", yaml_path)
        except Exception as e:
            self.logger.error(f"Failed to write configuration to {yaml_path}: {e}")

    def _synchronize_openai_speech_block(self):
        """Sync the persisted OpenAI speech YAML block into the in-memory config."""

        block = self.yaml_config.get("OPENAI_SPEECH")

        if not isinstance(block, Mapping):
            self.config.pop("OPENAI_SPEECH", None)
            return

        normalized = dict(block)
        self.config["OPENAI_SPEECH"] = normalized

        mapping = {
            "OPENAI_STT_PROVIDER": "stt_provider",
            "OPENAI_TTS_PROVIDER": "tts_provider",
            "OPENAI_LANGUAGE": "language",
            "OPENAI_TASK": "task",
            "OPENAI_INITIAL_PROMPT": "initial_prompt",
        }

        for config_key, block_key in mapping.items():
            if block_key in normalized:
                self.config[config_key] = normalized[block_key]
