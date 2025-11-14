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

from .core import (
    ConfigCore,
    _DEFAULT_CONVERSATION_STORE_DSN,
    _UNSET,
    find_dotenv,
    load_dotenv,
    set_key,
    setup_logger,
)
from .conversation_summary import ConversationSummaryConfigSection
from .messaging import MessagingConfigSection, setup_message_bus
from .persistence import KV_STORE_UNSET, PersistenceConfigSection
from .persistence import PersistenceConfigMixin
from .providers import ProviderConfigSections
from .providers import ProviderConfigMixin
from .tooling import ToolingConfigSection
from .ui_config import UIConfig
from modules.orchestration.message_bus import MessageBus
from modules.job_store import JobService
from modules.job_store.repository import JobStoreRepository
from modules.task_store import TaskService, TaskStoreRepository
from modules.Tools.Base_Tools.task_queue import (
    TaskQueueService,
    get_default_task_queue_service,
)
from urllib.parse import urlparse

# Legacy compatibility imports retained for modules referencing this file directly
import yaml


if TYPE_CHECKING:
    from modules.orchestration.job_manager import JobManager
    from modules.orchestration.job_scheduler import JobScheduler

class ConfigManager(ProviderConfigMixin, PersistenceConfigMixin, ConfigCore):
    UNSET = _UNSET
    """Manages configuration settings for the application."""

    def _create_logger(self):
        return setup_logger(__name__)

    def _load_dotenv(
        self,
        path: str | None = None,
        *,
        override: bool | None = None,
    ) -> None:
        if path is None and override is None:
            load_dotenv()
            return

        kwargs: dict[str, Any] = {}
        if override is not None:
            kwargs["override"] = override
        load_dotenv(path, **kwargs)

    def _find_dotenv(self) -> str:
        return find_dotenv()

    def _set_key(self, path: str, key: str, value: str) -> None:
        set_key(path, key, value)

    def __init__(self) -> None:
        """Initialise configuration sections and cached collaborators."""

        super().__init__()

        # --- UI helpers ------------------------------------------------
        self.ui_config = UIConfig(
            config=self.config,
            yaml_config=self.yaml_config,
            read_config=self.get_config,
            write_config=self._write_yaml_config,
        )

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

        # --- Conversation summary defaults ----------------------------
        self.conversation_summary = ConversationSummaryConfigSection(
            config=self.config,
            yaml_config=self.yaml_config,
            logger=self.logger,
            write_yaml_callback=self._write_yaml_config,
        )
        self.conversation_summary.apply()

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

        self._message_backend: Optional[Any] = None
        self._message_bus: Optional[MessageBus] = None
        self._task_session_factory: sessionmaker | None = None
        self._task_repository: TaskStoreRepository | None = None
        self._task_service: TaskService | None = None
        self._job_repository: JobStoreRepository | None = None
        self._job_service: JobService | None = None
        self._task_queue_service: TaskQueueService | None = None
        self._job_manager: "JobManager" | None = None
        self._job_scheduler: "JobScheduler" | None = None









    # --- Service factory helpers -------------------------------------













    # --- Provider fallback configuration -----------------------------

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
        self._message_backend = None
        self._message_bus = None
        return dict(block)

    def get_conversation_summary_settings(self) -> Dict[str, Any]:
        """Return the configured automatic conversation summary settings."""

        return self.conversation_summary.get_settings()

    def get_persona_remediation_policies(self) -> Dict[str, Any]:
        """Return tenant specific remediation policies for persona metrics."""

        runtime_block = self.config.get("remediation")
        yaml_block = self.yaml_config.get("remediation")

        result: Dict[str, Any] = {}
        if isinstance(yaml_block, Mapping):
            result = copy.deepcopy(yaml_block)

        def _merge(destination: Dict[str, Any], source: Mapping[str, Any]) -> None:
            for key, value in source.items():
                if isinstance(value, Mapping) and isinstance(destination.get(key), Mapping):
                    merged = dict(destination[key])  # type: ignore[index]
                    _merge(merged, value)
                    destination[key] = merged
                else:
                    destination[key] = copy.deepcopy(value)

        if isinstance(runtime_block, Mapping):
            if not result:
                result = {}
            _merge(result, runtime_block)

        return result

    def set_conversation_summary_settings(self, **settings: Any) -> Dict[str, Any]:
        """Persist conversation summary preferences and update cached state."""

        return self.conversation_summary.set_settings(**settings)










    def configure_message_bus(self) -> MessageBus:
        """Instantiate and configure the global message bus."""

        if self._message_bus is not None:
            return self._message_bus

        settings = self.get_messaging_settings()
        backend, bus = setup_message_bus(settings, logger=self.logger)
        self._message_backend = backend
        self._message_bus = bus
        return bus




    def _get_provider_env_keys(self) -> Dict[str, str]:
        """Return the mapping between provider display names and environment keys."""
        return self.providers.get_env_keys()


















    def get_config(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by key with an optional default."""

        return self.config.get(key, default)




























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













    @staticmethod
    def _extract_auth_env_keys(auth_block: Optional[Mapping[str, Any]]) -> List[str]:
        """Return normalized environment variable keys declared in a manifest auth block."""

        definitions = ConfigCore._extract_auth_env_definitions(auth_block)
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

            env_definitions = ConfigCore._extract_auth_env_definitions(auth_block)
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

            env_definitions = ConfigCore._extract_auth_env_definitions(auth_block)
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

        return self.ui_config.set_debug_log_max_lines(max_lines)

    def set_ui_debug_logger_names(self, logger_names: Optional[Sequence[str]]) -> List[str]:
        """Persist the list of logger names mirrored in the UI debug console."""

        return self.ui_config.set_debug_logger_names(logger_names)

    def get_ui_debug_log_level(self) -> Optional[Any]:
        """Return the configured UI debug log level."""

        return self.ui_config.get_debug_log_level()

    def set_ui_debug_log_level(self, level: Optional[Any]) -> Optional[Any]:
        """Persist the configured UI debug log level."""

        return self.ui_config.set_debug_log_level(level)

    def get_ui_debug_log_max_lines(self, default: Optional[int] = None) -> Optional[int]:
        """Return the configured maximum number of debug log lines."""

        return self.ui_config.get_debug_log_max_lines(default)

    def get_ui_debug_log_initial_lines(self, default: Optional[int] = None) -> Optional[int]:
        """Return the configured number of initial debug log lines."""

        return self.ui_config.get_debug_log_initial_lines(default)

    def get_ui_debug_logger_names(self) -> List[str]:
        """Return configured debug logger names."""

        return self.ui_config.get_debug_logger_names()

    def get_ui_debug_log_format(self) -> Optional[str]:
        """Return the configured debug log format string."""

        return self.ui_config.get_debug_log_format()

    def get_ui_debug_log_file_name(self) -> Optional[str]:
        """Return the configured debug log file name."""

        return self.ui_config.get_debug_log_file_name()

    def get_ui_terminal_wrap_enabled(self) -> bool:
        """Return whether terminal sections in the UI should wrap lines."""

        return self.ui_config.get_terminal_wrap_enabled()

    def set_ui_terminal_wrap_enabled(self, enabled: bool) -> bool:
        """Persist the line wrapping preference for UI terminal sections."""

        return self.ui_config.set_terminal_wrap_enabled(enabled)

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

