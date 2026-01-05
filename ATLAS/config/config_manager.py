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
    _DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND,
    _DEFAULT_CONVERSATION_STORE_BACKENDS,
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
from .rag import RAGSettings
from .tooling import ToolingConfigSection
from .storage import StorageArchitecture
from .ui_config import UIConfig
from .budget import BudgetConfigSection
from modules.logging.audit_templates import get_audit_template
from ATLAS.messaging import AgentBus
from modules.job_store import JobService, MongoJobStoreRepository
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

    @staticmethod
    def _coerce_feature_flag(value: Any, default: bool = False) -> bool:
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
        try:
            return bool(int(value))
        except Exception:
            return bool(value)

    def _initialise_feature_flags(self) -> Dict[str, bool]:
        raw_block = {}
        raw_flags = self.config.get("feature_flags")
        if isinstance(raw_flags, Mapping):
            raw_block = dict(raw_flags)

        plan_env = os.getenv("ATLAS_TENANT_PLAN", raw_block.get("plan", ""))
        enterprise_plan = str(plan_env).strip().lower() == "enterprise"
        enterprise_env = os.getenv("ATLAS_ENTERPRISE_TENANT")
        if enterprise_env is not None:
            enterprise_plan = self._coerce_feature_flag(enterprise_env, enterprise_plan)

        enterprise_enabled = self._coerce_feature_flag(
            raw_block.get("enterprise"), enterprise_plan
        )

        auth_enabled = self._coerce_feature_flag(
            raw_block.get("auth_connectors"), enterprise_enabled
        )
        delegated_admin_enabled = self._coerce_feature_flag(
            raw_block.get("delegated_admin"), enterprise_enabled
        )

        feature_flags = {
            "enterprise": bool(enterprise_enabled),
            "auth_connectors": bool(enterprise_enabled and auth_enabled),
            "delegated_admin": bool(enterprise_enabled and delegated_admin_enabled),
        }
        return feature_flags

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

        self._feature_flags = self._initialise_feature_flags()
        self.config["feature_flags"] = dict(self._feature_flags)
        self.yaml_config.setdefault("feature_flags", {})
        self.yaml_config["feature_flags"].update(dict(self._feature_flags))

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
            default_conversation_dsn_map=_DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND,
            conversation_backend_options=_DEFAULT_CONVERSATION_STORE_BACKENDS,
        )
        self.persistence.apply()

        # --- Budget management configuration -------------------------
        self.budget = BudgetConfigSection(
            config=self.config,
            yaml_config=self.yaml_config,
            env_config=self.env_config,
            logger=self.logger,
            write_yaml_callback=self._write_yaml_config,
        )

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

        # --- Storage architecture defaults ---------------------------
        self.storage_architecture = self._initialise_storage_architecture()

        # --- Local profile defaults -----------------------------------
        profile_cap = self.get_config("LOCAL_PROFILE_LIMIT", None)
        try:
            normalized_profile_cap = int(profile_cap) if profile_cap is not None else 5
        except (TypeError, ValueError):
            normalized_profile_cap = 5
        if normalized_profile_cap <= 0:
            normalized_profile_cap = 5
        self.config["LOCAL_PROFILE_LIMIT"] = normalized_profile_cap
        self.yaml_config.setdefault("LOCAL_PROFILE_LIMIT", normalized_profile_cap)

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
        self._message_bus: Optional[AgentBus] = None
        self._task_session_factory: sessionmaker | None = None
        self._task_repository: TaskStoreRepository | None = None
        self._task_service: TaskService | None = None
        self._job_repository: JobStoreRepository | MongoJobStoreRepository | None = None
        self._job_service: JobService | None = None
        self._task_queue_service: TaskQueueService | None = None
        self._job_manager: "JobManager" | None = None
        self._job_scheduler: "JobScheduler" | None = None
        self._job_mongo_client: Any | None = None

        # --- RAG configuration ----------------------------------------
        self._rag_settings = self._initialise_rag_settings()









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
        initial_offset: Optional[str] = None,
        replay_start: Optional[str] = None,
        batch_size: Optional[int] = None,
        blocking_timeout_ms: Optional[int] = None,
        auto_claim_idle_ms: Optional[int] = None,
        auto_claim_count: Optional[int] = None,
        delete_acknowledged: Optional[bool] = None,
        trim_max_length: Optional[int] = None,
        delete_on_ack: Optional[bool] = None,
        trim_maxlen: Optional[int] = None,
        min_idle_ms: Optional[int] = None,
        policy: Optional[Mapping[str, Any]] = None,
        kafka: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist messaging backend preferences and clear cached bus instances."""

        block = self.messaging.set_settings(
            backend=backend,
            redis_url=redis_url,
            stream_prefix=stream_prefix,
            initial_offset=initial_offset,
            replay_start=replay_start,
            batch_size=batch_size,
            blocking_timeout_ms=blocking_timeout_ms,
            auto_claim_idle_ms=auto_claim_idle_ms,
            auto_claim_count=auto_claim_count,
            delete_acknowledged=delete_acknowledged,
            trim_max_length=trim_max_length,
            delete_on_ack=delete_on_ack,
            trim_maxlen=trim_maxlen,
            min_idle_ms=min_idle_ms,
            policy=policy,
            kafka=kafka,
        )
        self._message_backend = None
        self._message_bus = None
        return dict(block)

    def get_conversation_summary_settings(self) -> Dict[str, Any]:
        """Return the configured automatic conversation summary settings."""

        return self.conversation_summary.get_settings()

    def get_vector_store_settings(self) -> Dict[str, Any]:
        """Return the configured vector store tooling block."""

        tools_block = self.config.get("tools")
        if isinstance(tools_block, Mapping):
            vector_block = tools_block.get("vector_store")
            if isinstance(vector_block, Mapping):
                return dict(vector_block)
        yaml_tools = self.yaml_config.get("tools")
        if isinstance(yaml_tools, Mapping):
            vector_block = yaml_tools.get("vector_store")
            if isinstance(vector_block, Mapping):
                return dict(vector_block)
        return {}

    def get_javascript_executor_settings(self) -> Dict[str, Any]:
        """Return configuration for the JavaScript executor tool."""

        tools_block = self.config.get("tools")
        if isinstance(tools_block, Mapping):
            js_block = tools_block.get("javascript_executor")
            if isinstance(js_block, Mapping):
                return dict(js_block)

        yaml_tools = self.yaml_config.get("tools")
        if isinstance(yaml_tools, Mapping):
            js_block = yaml_tools.get("javascript_executor")
            if isinstance(js_block, Mapping):
                return dict(js_block)

        return {}

    def get_mcp_settings(self) -> Dict[str, Any]:
        """Return configuration for the MCP tool provider block."""

        tools_block = self.config.get("tools")
        if isinstance(tools_block, Mapping):
            mcp_block = tools_block.get("mcp")
            if isinstance(mcp_block, Mapping):
                return dict(mcp_block)

        yaml_tools = self.yaml_config.get("tools")
        if isinstance(yaml_tools, Mapping):
            mcp_block = yaml_tools.get("mcp")
            if isinstance(mcp_block, Mapping):
                return dict(mcp_block)

        return {}

    def set_vector_store_settings(
        self,
        *,
        default_adapter: str,
        adapter_settings: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist the default vector store adapter and optional configuration."""

        normalized_adapter = (default_adapter or "").strip().lower() or "in_memory"

        tools_yaml = dict(self.yaml_config.get("tools") or {})
        vector_yaml = dict(tools_yaml.get("vector_store") or {})
        vector_yaml["default_adapter"] = normalized_adapter

        adapters_yaml = vector_yaml.get("adapters")
        if isinstance(adapters_yaml, Mapping):
            adapters = dict(adapters_yaml)
        else:
            adapters = {}
        if adapter_settings is not None:
            adapters[normalized_adapter] = dict(adapter_settings)
        elif normalized_adapter not in adapters:
            adapters[normalized_adapter] = {}
        adapters.setdefault("in_memory", adapters.get("in_memory", {}))
        if "mongodb" not in adapters:
            if isinstance(adapters_yaml, Mapping) and isinstance(adapters_yaml.get("mongodb"), Mapping):
                adapters["mongodb"] = dict(adapters_yaml["mongodb"])  # type: ignore[index]
            else:
                adapters["mongodb"] = {}
        vector_yaml["adapters"] = adapters

        tools_yaml["vector_store"] = vector_yaml
        self.yaml_config["tools"] = tools_yaml

        tools_config = dict(self.config.get("tools") or {})
        tools_config["vector_store"] = dict(vector_yaml)
        self.config["tools"] = tools_config

        self._write_yaml_config()
        return dict(vector_yaml)

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

    def _persist_conversation_database_url(
        self,
        url: str,
        *,
        backend: str | None = None,
    ) -> None:
        """Persist the conversation database URL and optional backend selection."""

        normalized_backend = backend
        if backend is not None:
            normalized_backend = self.persistence.conversation._normalize_backend(backend)
            if normalized_backend is None and backend:
                normalized_backend = str(backend).strip().lower() or None
            self.persistence.conversation._persist_backend(normalized_backend)

        self.persistence.conversation._persist_url(url)










    def configure_message_bus(self) -> AgentBus:
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

    def _initialise_storage_architecture(self) -> StorageArchitecture:
        """Load the storage architecture block and persist defaults if missing."""

        storage_block = self.yaml_config.get("storage")
        if not isinstance(storage_block, Mapping):
            storage_block = self.config.get("storage")

        architecture_block = None
        if isinstance(storage_block, Mapping):
            architecture_candidate = storage_block.get("architecture")
            if isinstance(architecture_candidate, Mapping):
                architecture_block = architecture_candidate

        architecture = StorageArchitecture.from_mapping(architecture_block)
        self._persist_storage_architecture(architecture, write=False)
        return architecture

    def _persist_storage_architecture(
        self, architecture: StorageArchitecture, *, write: bool = True
    ) -> Dict[str, Any]:
        """Write the storage architecture block to runtime and YAML config."""

        storage_yaml = dict(self.yaml_config.get("storage") or {})
        storage_config = dict(self.config.get("storage") or {})
        block = architecture.to_dict()

        storage_yaml["architecture"] = dict(block)
        storage_config["architecture"] = dict(block)

        self.yaml_config["storage"] = storage_yaml
        self.config["storage"] = storage_config

        if write:
            self._write_yaml_config()

        return dict(block)

    def get_storage_architecture(self) -> StorageArchitecture:
        """Return the normalised storage architecture definition."""

        return self.storage_architecture

    def get_storage_architecture_settings(self) -> Dict[str, Any]:
        """Return the storage architecture block as a mapping."""

        return self.storage_architecture.to_dict()

    def set_storage_architecture(
        self, architecture: StorageArchitecture | Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Persist storage architecture preferences and refresh cached state."""

        if isinstance(architecture, StorageArchitecture):
            normalized = architecture
        elif isinstance(architecture, Mapping):
            normalized = StorageArchitecture.from_mapping(architecture)
        else:
            raise TypeError("storage architecture must be a StorageArchitecture or mapping")

        self.storage_architecture = normalized
        return self._persist_storage_architecture(normalized)

    # --- RAG configuration helpers ------------------------------------

    def _initialise_rag_settings(self) -> RAGSettings:
        """Load RAG settings from config and persist defaults if missing."""
        rag_block = self.yaml_config.get("rag")
        if not isinstance(rag_block, Mapping):
            rag_block = self.config.get("rag")

        settings = RAGSettings.from_mapping(rag_block if isinstance(rag_block, Mapping) else None)
        self._persist_rag_settings(settings, write=False)
        return settings

    def _persist_rag_settings(
        self, settings: RAGSettings, *, write: bool = True
    ) -> Dict[str, Any]:
        """Write RAG settings to runtime and YAML config."""
        block = settings.to_dict()

        self.yaml_config["rag"] = dict(block)
        self.config["rag"] = dict(block)

        if write:
            self._write_yaml_config()

        return dict(block)

    @property
    def rag_settings(self) -> RAGSettings:
        """Return the current RAG settings."""
        return self._rag_settings

    def get_rag_settings(self) -> RAGSettings:
        """Return the current RAG settings."""
        return self._rag_settings

    def get_rag_settings_dict(self) -> Dict[str, Any]:
        """Return RAG settings as a dictionary."""
        return self._rag_settings.to_dict()

    def set_rag_settings(
        self, settings: RAGSettings | Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Persist RAG settings and refresh cached state."""
        if isinstance(settings, RAGSettings):
            normalized = settings
        elif isinstance(settings, Mapping):
            normalized = RAGSettings.from_mapping(settings)
        else:
            raise TypeError("RAG settings must be RAGSettings or mapping")

        self._rag_settings = normalized
        return self._persist_rag_settings(normalized)

    def set_rag_enabled(self, enabled: bool) -> None:
        """Toggle the master RAG enabled flag."""
        from dataclasses import replace
        self._rag_settings = replace(self._rag_settings, enabled=enabled)
        self._persist_rag_settings(self._rag_settings)

    def is_rag_enabled(self) -> bool:
        """Check if RAG is enabled."""
        return self._rag_settings.enabled

    def is_rag_fully_enabled(self) -> bool:
        """Check if RAG pipeline is fully operational."""
        return self._rag_settings.is_fully_enabled

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

    def get_company_identity(self) -> Dict[str, Optional[str]]:
        """Return persisted company identity fields."""

        defaults = {
            "name": None,
            "domain": None,
            "primary_contact": None,
            "contact_email": None,
            "address_line1": None,
            "address_line2": None,
            "city": None,
            "state": None,
            "postal_code": None,
            "country": None,
            "phone_number": None,
        }

        block = self.config.get("company")
        if not isinstance(block, Mapping):
            return defaults

        normalized: Dict[str, Optional[str]] = dict(defaults)

        for key in defaults:
            value = block.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                normalized[key] = cleaned or None
            else:
                normalized[key] = None

        return normalized

    def set_company_identity(
        self,
        *,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        primary_contact: Optional[str] = None,
        contact_email: Optional[str] = None,
        address_line1: Optional[str] = None,
        address_line2: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        postal_code: Optional[str] = None,
        country: Optional[str] = None,
        phone_number: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Persist company identity fields for enterprise setups."""

        cleaned = {
            "name": name.strip() if isinstance(name, str) else None,
            "domain": domain.strip() if isinstance(domain, str) else None,
            "primary_contact": (
                primary_contact.strip() if isinstance(primary_contact, str) else None
            ),
            "contact_email": (
                contact_email.strip().lower() if isinstance(contact_email, str) else None
            ),
            "address_line1": (
                address_line1.strip() if isinstance(address_line1, str) else None
            ),
            "address_line2": (
                address_line2.strip() if isinstance(address_line2, str) else None
            ),
            "city": city.strip() if isinstance(city, str) else None,
            "state": state.strip() if isinstance(state, str) else None,
            "postal_code": postal_code.strip() if isinstance(postal_code, str) else None,
            "country": country.strip() if isinstance(country, str) else None,
            "phone_number": (
                phone_number.strip() if isinstance(phone_number, str) else None
            ),
        }

        if not any(cleaned.values()):
            self.yaml_config.pop("company", None)
            self.config.pop("company", None)
            self._write_yaml_config()
            return {key: None for key in cleaned}

        self.yaml_config["company"] = dict(cleaned)
        self.config["company"] = dict(cleaned)
        self._write_yaml_config()
        return dict(cleaned)

    def set_http_server_autostart(self, enabled: bool) -> bool:
        """Persist whether the embedded HTTP server should start automatically."""

        block = dict(self.yaml_config.get('http_server', {}))
        block['auto_start'] = bool(enabled)
        self.yaml_config['http_server'] = dict(block)
        self.config['http_server'] = dict(block)
        self._write_yaml_config()
        return bool(enabled)

    def get_data_residency_settings(self) -> Dict[str, Optional[str]]:
        """Return normalized data residency preferences."""

        block = self.config.get("data_residency")
        if not isinstance(block, Mapping):
            return {"region": None, "residency_requirement": None}

        region = str(block.get("region")).strip() if block.get("region") else None
        residency_requirement = (
            str(block.get("residency_requirement")).strip()
            if block.get("residency_requirement")
            else None
        )
        return {
            "region": region or None,
            "residency_requirement": residency_requirement or None,
        }

    def set_data_residency(
        self,
        *,
        region: Optional[str] = None,
        residency_requirement: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Persist data residency preferences for the deployment."""

        normalized_region = region.strip() if isinstance(region, str) else None
        normalized_residency = (
            residency_requirement.strip()
            if isinstance(residency_requirement, str)
            else None
        )

        block: Dict[str, str] = {}
        if normalized_region:
            block["region"] = normalized_region
        if normalized_residency:
            block["residency_requirement"] = normalized_residency

        if block:
            self.yaml_config["data_residency"] = dict(block)
            self.config["data_residency"] = dict(block)
        else:
            self.yaml_config.pop("data_residency", None)
            self.config.pop("data_residency", None)

        self._write_yaml_config()
        return {
            "region": block.get("region"),
            "residency_requirement": block.get("residency_requirement"),
        }

    def get_local_profile_limit(self, default: int = 5) -> int:
        """Return the configured cap on locally managed profiles."""

        value = self.get_config("LOCAL_PROFILE_LIMIT", default)
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return default
        return normalized if normalized > 0 else default

    # ------------------------------------------------------------------
    # Feature flag helpers
    # ------------------------------------------------------------------

    def get_feature_flags(self) -> Dict[str, bool]:
        """Return a copy of the normalized feature flag block."""

        flags = self.config.get("feature_flags")
        return copy.deepcopy(flags if isinstance(flags, Mapping) else {})

    def is_enterprise_tenant(self) -> bool:
        """Return ``True`` when the tenant is marked as Enterprise."""

        return bool(self.get_feature_flags().get("enterprise", False))

    def auth_connectors_enabled(self) -> bool:
        """Return ``True`` when Enterprise SSO/SCIM connectors are enabled."""

        flags = self.get_feature_flags()
        return bool(flags.get("enterprise") and flags.get("auth_connectors"))

    def delegated_admin_enabled(self) -> bool:
        """Return ``True`` when delegated admin controls are enabled."""

        flags = self.get_feature_flags()
        return bool(flags.get("enterprise") and flags.get("delegated_admin"))

    # ------------------------------------------------------------------
    # Data loss prevention
    # ------------------------------------------------------------------

    def get_dlp_policy(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Return merged DLP settings for the requested tenant."""

        settings = self.config.get("dlp")
        if not isinstance(settings, Mapping):
            return {"enabled": False}

        enabled = bool(settings.get("enabled", False))
        default_policy = settings.get("default") if isinstance(settings.get("default"), Mapping) else {}
        tenant_policy = {}
        tenant_map = settings.get("tenants") if isinstance(settings.get("tenants"), Mapping) else {}
        if tenant_id and isinstance(tenant_map, Mapping):
            tenant_policy = tenant_map.get(str(tenant_id)) or {}

        merged: Dict[str, Any] = {}
        if isinstance(default_policy, Mapping):
            merged.update(default_policy)
        if isinstance(tenant_policy, Mapping):
            merged.update(tenant_policy)

        merged_enabled = bool(merged.get("enabled", True)) if merged else True
        merged["enabled"] = bool(enabled and merged_enabled)
        return merged

    def set_dlp_policy(
        self,
        policy: Mapping[str, Any],
        *,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist default or tenant-specific DLP rules."""

        settings = dict(self.config.get("dlp") or {})
        enabled = bool(settings.get("enabled", policy.get("enabled", True)))
        settings["enabled"] = enabled

        if tenant_id:
            tenants_block = settings.get("tenants")
            if not isinstance(tenants_block, Mapping):
                tenants_block = {}
            tenants_block = dict(tenants_block)
            tenants_block[str(tenant_id)] = dict(policy)
            settings["tenants"] = tenants_block
        else:
            settings["default"] = dict(policy)

        self.config["dlp"] = copy.deepcopy(settings)
        self.yaml_config["dlp"] = copy.deepcopy(settings)
        self._write_yaml_config()
        return dict(settings)













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

    # ------------------------------------------------------------------
    # Signing key helpers
    # ------------------------------------------------------------------
    def get_bundle_signing_key(
        self,
        asset_type: str,
        *,
        require: bool = True,
    ) -> Optional[str]:
        """Return the configured signing key for the given bundle asset."""

        normalized = str(asset_type or "").strip().lower()
        if not normalized:
            raise ValueError("Asset type is required to resolve a signing key")

        env_keys = [
            f"ATLAS_{normalized.upper()}_BUNDLE_SIGNING_KEY",
            f"{normalized.upper()}_BUNDLE_SIGNING_KEY",
        ]

        secret: Optional[str] = None
        for key in env_keys:
            candidate = self.env_config.get(key)
            if isinstance(candidate, str) and candidate.strip():
                secret = candidate.strip()
                break
            candidate = os.getenv(key)
            if isinstance(candidate, str) and candidate.strip():
                secret = candidate.strip()
                break

        if secret:
            return secret

        if require:
            raise RuntimeError(
                f"Signing key for '{normalized}' bundles is not configured."
            )
        return None

    def get_persona_bundle_signing_key(self, *, require: bool = True) -> Optional[str]:
        return self.get_bundle_signing_key("persona", require=require)

    def get_task_bundle_signing_key(self, *, require: bool = True) -> Optional[str]:
        return self.get_bundle_signing_key("task", require=require)

    def get_tool_bundle_signing_key(self, *, require: bool = True) -> Optional[str]:
        return self.get_bundle_signing_key("tool", require=require)

    def get_skill_bundle_signing_key(self, *, require: bool = True) -> Optional[str]:
        return self.get_bundle_signing_key("skill", require=require)

    def get_job_bundle_signing_key(self, *, require: bool = True) -> Optional[str]:
        return self.get_bundle_signing_key("job", require=require)

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

    # -- audit and retention templates ----------------------------------

    def get_audit_settings(self) -> Dict[str, Any]:
        """Return the persisted audit template block when present."""

        block = self.config.get("audit")
        if isinstance(block, Mapping):
            return dict(block)
        return {}

    def get_audit_template(self) -> Optional[str]:
        """Return the configured audit template key, if any."""

        template = self.get_audit_settings().get("template")
        if isinstance(template, str) and template.strip():
            return template.strip()
        return None

    def apply_audit_template(
        self,
        template_key: Optional[str],
        *,
        apply_retention: bool = True,
        retention_days: Optional[int] = None,
        retention_history_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Persist the selected audit template and optional retention values."""

        template = get_audit_template(template_key)
        settings: Dict[str, Any] = {}

        if template is not None:
            days = retention_days if retention_days is not None else template.retention_days
            history_limit = (
                retention_history_limit
                if retention_history_limit is not None
                else template.retention_history_limit
            )
            settings = {
                "template": template.key,
                "intent": template.intent,
                "persona_sink": template.persona_sink,
                "skill_sink": template.skill_sink,
                "retention_days": days,
                "retention_history_limit": history_limit,
            }
            if apply_retention:
                self.set_conversation_retention(
                    days=days,
                    history_limit=history_limit,
                )

        self.config["audit"] = dict(settings)
        if settings:
            self.yaml_config["audit"] = dict(settings)
        else:
            self.yaml_config.pop("audit", None)
        self._write_yaml_config()
        return dict(settings)

    def resolve_audit_log_paths(self) -> Dict[str, Path]:
        """Return absolute paths for configured audit sinks."""

        settings = self.get_audit_settings()
        base_dir = Path(self.config.get("APP_ROOT", ".")).expanduser().resolve()
        logs_dir = base_dir / "logs"

        def _resolve(value: Any) -> Optional[Path]:
            if not value:
                return None
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = logs_dir / candidate
            return candidate

        persona_path = _resolve(settings.get("persona_sink"))
        skill_path = _resolve(settings.get("skill_sink"))

        return {"persona_sink": persona_path, "skill_sink": skill_path}

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
