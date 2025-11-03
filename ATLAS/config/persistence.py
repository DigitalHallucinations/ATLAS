"""Persistence-focused configuration helpers for :mod:`ATLAS.config`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, MutableMapping, Optional
from collections.abc import Mapping


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
    default_conversation_dsn: str

    kv_engine_cache: Dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.conversation = ConversationStoreConfigSection(
            config=self.config,
            yaml_config=self.yaml_config,
            env_config=self.env_config,
            logger=self.logger,
            write_yaml_callback=self.write_yaml_callback,
            default_conversation_dsn=self.default_conversation_dsn,
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

        if "postgres" not in normalized["adapters"]:
            normalized["adapters"]["postgres"] = self._normalize_postgres_settings({})

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

@dataclass(kw_only=True)
class ConversationStoreConfigSection:
    """Manage conversation store configuration and lifecycle."""

    config: MutableMapping[str, Any]
    yaml_config: MutableMapping[str, Any]
    env_config: Mapping[str, Any]
    logger: Any
    write_yaml_callback: Callable[[], None]
    default_conversation_dsn: str
    create_engine: Callable[..., Any]
    inspect_engine: Callable[..., Any]
    make_url: Callable[..., Any]
    sessionmaker_factory: Callable[..., Any]
    conversation_required_tables: Callable[[], set[str]]

    _conversation_store_verified: bool = False
    _conversation_engine: Any | None = None
    _conversation_session_factory: Any | None = None

    def apply(self) -> None:
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
                    "Invalid CONVERSATION_DATABASE_RETENTION_DAYS value %r",
                    env_retention_days,
                )

        retention_block.setdefault("history_message_limit", 500)
        conversation_store_block["retention"] = retention_block
        self.config["conversation_database"] = conversation_store_block

    # Exposed helpers --------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        block = self.config.get("conversation_database")
        if not isinstance(block, Mapping):
            return {}
        return dict(block)

    def ensure_postgres_store(self) -> str:
        self._conversation_store_verified = False

        config = self.get_config()
        url_value = config.get("url")
        if isinstance(url_value, str):
            url = url_value.strip()
        else:
            url = url_value or ""

        generated_default = False
        if not url:
            url = self.default_conversation_dsn
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
        if dialect != "postgresql":
            message = (
                "Conversation database URL must use the 'postgresql' dialect; "
                f"received '{parsed_url.drivername}'."
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

        self._persist_url(url)

        if generated_default:
            self.write_yaml_callback()

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
            retention["days"] = int(days)

        if history_limit is None:
            retention.pop("history_message_limit", None)
        else:
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

    def _build_session_factory(self) -> tuple[Any | None, Any | None]:
        ensured_url = self.ensure_postgres_store()

        config = self.get_config()
        url = ensured_url

        try:
            parsed_url = self.make_url(url)
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

        engine = self.create_engine(url, future=True, **engine_kwargs)
        factory = self.sessionmaker_factory(bind=engine, future=True)
        return engine, factory
