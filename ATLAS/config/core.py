"""Core configuration utilities and base class for :mod:`ATLAS.config`."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

from dataclasses import dataclass

import yaml
from dotenv import find_dotenv, load_dotenv, set_key

from modules.logging.logger import setup_logger

_UNSET = object()



@dataclass(frozen=True)
class ConversationStoreBackendOption:
    """Describe the default DSN and metadata for a conversation store backend."""

    name: str
    label: str
    dialect: str
    dsn: str
    description: str


_DEFAULT_CONVERSATION_STORE_BACKENDS: Tuple[ConversationStoreBackendOption, ...] = (
    ConversationStoreBackendOption(
        name="postgresql",
        label="PostgreSQL (production ready)",
        dialect="postgresql",
        dsn="postgresql+psycopg://atlas:atlas@localhost:5432/atlas",
        description=(
            "Connect to a PostgreSQL instance using the psycopg driver. "
            "Suitable for shared or durable deployments."
        ),
    ),
    ConversationStoreBackendOption(
        name="sqlite",
        label="SQLite (file based)",
        dialect="sqlite",
        dsn="sqlite:///atlas.sqlite3",
        description=(
            "Stores conversations in a local SQLite file. Ideal for trials or "
            "single-user environments."
        ),
    ),
    ConversationStoreBackendOption(
        name="mongodb",
        label="MongoDB (experimental)",
        dialect="mongodb",
        dsn="mongodb://localhost:27017/atlas",
        description=(
            "Persist conversations in MongoDB using the experimental document "
            "store repository."
        ),
    ),
)

_DEFAULT_CONVERSATION_STORE_DSN = _DEFAULT_CONVERSATION_STORE_BACKENDS[0].dsn


def default_conversation_store_backend_name() -> str:
    """Return the canonical backend name used for conversation persistence."""

    return _DEFAULT_CONVERSATION_STORE_BACKENDS[0].name


def get_default_conversation_store_backends() -> Tuple[ConversationStoreBackendOption, ...]:
    """Return the tuple of supported conversation store backend defaults."""

    return _DEFAULT_CONVERSATION_STORE_BACKENDS


def get_default_conversation_store_backend(name: str) -> ConversationStoreBackendOption:
    """Return metadata for the named backend, raising ``KeyError`` if missing."""

    normalized = (name or "").strip().lower()
    for option in _DEFAULT_CONVERSATION_STORE_BACKENDS:
        if option.name == normalized:
            return option
    raise KeyError(name)


def infer_conversation_store_backend(value: str | None) -> Optional[str]:
    """Infer the backend name from a SQLAlchemy URL or dialect string."""

    if not value:
        return None

    candidate = value.strip().lower()
    if not candidate:
        return None

    if "://" in candidate:
        try:
            scheme = candidate.split(":", 1)[0]
        except Exception:
            scheme = candidate
    else:
        scheme = candidate

    dialect = scheme.split("+", 1)[0]
    for option in _DEFAULT_CONVERSATION_STORE_BACKENDS:
        if option.dialect == dialect:
            return option.name
    return dialect or None


class ConfigCore:
    """Base class providing shared configuration helpers."""

    UNSET = _UNSET

    def __init__(self) -> None:
        # Load environment variables from .env before initialisation
        self._load_dotenv()
        self.logger = self._create_logger()

        self.env_config: Dict[str, Any] = self._load_env_config()
        self._yaml_path = self._compute_yaml_path()
        self.yaml_config: Dict[str, Any] = self._load_yaml_config()

        # Merge configurations, with YAML config overriding env config if there's overlap
        self.config: Dict[str, Any] = {**self.env_config, **self.yaml_config}

        # Normalise model cache persistence early for downstream users
        self._model_cache: Dict[str, List[str]] = self._normalize_model_cache(
            self.yaml_config.get("MODEL_CACHE")
        )
        normalized_cache = copy.deepcopy(self._model_cache)
        self.config["MODEL_CACHE"] = normalized_cache
        self.yaml_config["MODEL_CACHE"] = copy.deepcopy(self._model_cache)

    # ------------------------------------------------------------------
    # Environment and persistence helpers
    # ------------------------------------------------------------------
    def _load_env_config(self) -> Dict[str, Any]:
        """Load environment variables into the configuration dictionary."""

        app_root = Path(__file__).resolve().parents[2]

        config = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "DEFAULT_PROVIDER": os.getenv("DEFAULT_PROVIDER", "OpenAI"),
            "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "gpt-4o"),
            "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
            "MISTRAL_BASE_URL": os.getenv("MISTRAL_BASE_URL"),
            "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "GROK_API_KEY": os.getenv("GROK_API_KEY"),
            "XI_API_KEY": os.getenv("XI_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "OPENAI_ORGANIZATION": os.getenv("OPENAI_ORGANIZATION"),
            "JAVASCRIPT_EXECUTOR_BIN": os.getenv("JAVASCRIPT_EXECUTOR_BIN"),
            "JAVASCRIPT_EXECUTOR_ARGS": os.getenv("JAVASCRIPT_EXECUTOR_ARGS"),
            "APP_ROOT": str(app_root),
        }
        self.logger.debug("APP_ROOT is set to: %s", config["APP_ROOT"])
        for asset in ("PERSONA", "TASK", "TOOL", "SKILL", "JOB"):
            env_key = f"ATLAS_{asset}_BUNDLE_SIGNING_KEY"
            config[env_key] = os.getenv(env_key)
        return config

    def _compute_yaml_path(self) -> str:
        """Return the absolute path to the persistent YAML configuration file."""

        app_root = self.env_config.get("APP_ROOT", ".")
        return os.path.join(app_root, "ATLAS", "config", "atlas_config.yaml")

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration settings from the YAML configuration file."""

        yaml_path = getattr(self, "_yaml_path", None) or self._compute_yaml_path()

        if not os.path.exists(yaml_path):
            self.logger.error("Configuration file not found: %s", yaml_path)
            return {}

        try:
            with open(yaml_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
                self.logger.debug("Loaded configuration from %s", yaml_path)
                return config
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to load configuration from %s: %s", yaml_path, exc)
            return {}

    def _write_yaml_config(self) -> None:
        """Persist the current YAML configuration back to disk."""

        yaml_path = getattr(self, "_yaml_path", None) or self._compute_yaml_path()
        try:
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, "w", encoding="utf-8") as file:
                yaml.dump(self.yaml_config, file)
                if file.tell() == 0 and self.yaml_config:
                    file.write(json.dumps(self.yaml_config))
            self.logger.debug("Configuration written to %s", yaml_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to write configuration to %s: %s", yaml_path, exc)

    def _persist_env_value(self, env_key: str, value: Optional[str]) -> None:
        """Persist an environment-backed configuration value and refresh state."""

        env_path = self._find_dotenv()
        if not env_path:
            app_root: Optional[str]
            if isinstance(getattr(self, "env_config", None), Mapping):
                app_root = self.env_config.get("APP_ROOT")
            else:
                app_root = os.getenv("APP_ROOT")
            if not app_root:
                app_root = str(Path(__file__).resolve().parents[2])

            env_path = os.path.abspath(os.path.join(app_root, ".env"))
            os.makedirs(os.path.dirname(env_path), exist_ok=True)
            if not os.path.exists(env_path):
                with open(env_path, "a", encoding="utf-8"):
                    pass

        self._set_key(env_path, env_key, value or "")
        self._load_dotenv(env_path, override=True)  # reload to refresh environment

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

    # ------------------------------------------------------------------
    # General utilities shared by configuration subsections
    # ------------------------------------------------------------------
    def _normalize_network_allowlist(self, value: Any) -> Optional[List[str]]:
        """Return a sanitised allowlist for sandboxed tool networking."""

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
        """Normalise persisted provider model caches into a predictable mapping."""

        normalized: Dict[str, List[str]] = {}

        if isinstance(value, Mapping):
            items: Iterable[tuple[Any, Any]] = value.items()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
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
                except Exception:  # pragma: no cover - defensive fallback
                    continue
            else:
                provider_key = provider

            provider_key = provider_key.strip()
            if not provider_key:
                continue

            seen: set[str] = set()
            ordered: List[str] = []

            if isinstance(models, Mapping):
                candidate_iterable: Iterable[Any] = models.values()
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

    @staticmethod
    def _mask_secret_preview(secret: str) -> str:
        """Return a masked preview of a secret without revealing its contents."""

        if not secret:
            return ""

        visible_count = min(len(secret), 8)
        return "•" * visible_count

    @staticmethod
    def _sanitize_tool_value(value: Any) -> Any:
        """Return a JSON-serialisable representation for persisted tool settings."""

        if isinstance(value, Mapping):
            return {
                str(key): ConfigCore._sanitize_tool_value(subvalue)
                for key, subvalue in value.items()
            }

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [ConfigCore._sanitize_tool_value(item) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    @staticmethod
    def _sanitize_tool_settings_block(value: Any) -> Dict[str, Any]:
        """Normalise persisted tool settings into a dictionary."""

        if not isinstance(value, Mapping):
            return {}

        return {
            str(key): ConfigCore._sanitize_tool_value(subvalue)
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
                    metadata = {}
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
                    env_name_candidate = value.get("name")
                    if isinstance(env_name_candidate, str):
                        env_name = env_name_candidate
                    metadata.update({k: v for k, v in value.items() if k != "name"})

                if env_name is None:
                    env_name = str(key)

                if env_required is not None and "required" not in metadata:
                    metadata["required"] = env_required

                _register_env(env_name, metadata)

        return definitions

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_config(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by key with an optional default."""

        return self.config.get(key, default)

    # ------------------------------------------------------------------
    # Hooks for subclasses (primarily ConfigManager) to override
    # ------------------------------------------------------------------
    def _create_logger(self):
        return setup_logger(__name__)

    def _load_dotenv(
        self,
        path: str | None = None,
        *,
        override: bool | None = None,
    ) -> None:
        """Load environment variables from ``.env`` supporting targeted reloads."""

        if path is None and override is None:
            load_dotenv()
            return

        kwargs: dict[str, Any] = {}
        if override is not None:
            kwargs["override"] = override

        # ``load_dotenv`` accepts the file path as the first positional argument;
        # retain this convention for compatibility with test doubles.
        load_dotenv(path, **kwargs)

    def _find_dotenv(self) -> str:
        return find_dotenv()

    def _set_key(self, path: str, key: str, value: str) -> None:
        set_key(path, key, value)


__all__ = [
    "ConfigCore",
    "ConversationStoreBackendOption",
    "_DEFAULT_CONVERSATION_STORE_DSN",
    "_DEFAULT_CONVERSATION_STORE_BACKENDS",
    "_UNSET",
    "default_conversation_store_backend_name",
    "get_default_conversation_store_backend",
    "get_default_conversation_store_backends",
    "infer_conversation_store_backend",
    "find_dotenv",
    "load_dotenv",
    "set_key",
    "setup_logger",
]
