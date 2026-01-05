"""Tooling configuration helpers for :mod:`ATLAS.config`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping
from collections.abc import Mapping, Sequence
import shlex


@dataclass
class ToolingConfigSection:
    """Manage tool-related configuration defaults."""

    config: MutableMapping[str, Any]
    yaml_config: MutableMapping[str, Any]
    env_config: Mapping[str, Any]
    logger: Any

    def apply(self) -> None:
        """Populate the shared configuration dictionary with tooling defaults."""

        self._ensure_tool_defaults()
        self._ensure_conversation_block()
        self._ensure_tool_logging()
        self._ensure_tools_block()
        self._ensure_tool_safety()

    # ------------------------------------------------------------------
    # Tool default helpers
    def _ensure_tool_defaults(self) -> None:
        tool_defaults = self.config.get("tool_defaults")
        if not isinstance(tool_defaults, Mapping):
            tool_defaults = {}
        else:
            tool_defaults = dict(tool_defaults)
        tool_defaults.setdefault("timeout_seconds", 30)
        tool_defaults.setdefault("max_cost_per_session", None)
        self.config["tool_defaults"] = tool_defaults

    def _ensure_conversation_block(self) -> None:
        conversation_block = self.config.get("conversation")
        if not isinstance(conversation_block, Mapping):
            conversation_block = {}
        else:
            conversation_block = dict(conversation_block)
        conversation_block.setdefault("max_tool_duration_ms", 120000)
        self.config["conversation"] = conversation_block

    def _ensure_tool_logging(self) -> None:
        tool_logging_block = self.config.get("tool_logging")
        if not isinstance(tool_logging_block, Mapping):
            tool_logging_block = {}
        else:
            tool_logging_block = dict(tool_logging_block)
        tool_logging_block.setdefault("log_full_payloads", False)
        tool_logging_block.setdefault("payload_summary_length", 256)
        self.config["tool_logging"] = tool_logging_block

    def _ensure_tools_block(self) -> None:
        tools_block = self.config.get("tools")
        if not isinstance(tools_block, Mapping):
            tools_block = {}
        else:
            tools_block = dict(tools_block)

        js_block = tools_block.get("javascript_executor")
        if not isinstance(js_block, Mapping):
            js_block = {}
        else:
            js_block = dict(js_block)

        env_executable = self.env_config.get("JAVASCRIPT_EXECUTOR_BIN")
        if env_executable and not js_block.get("executable"):
            js_block["executable"] = env_executable

        env_args = self.env_config.get("JAVASCRIPT_EXECUTOR_ARGS")
        if env_args and not js_block.get("args"):
            try:
                js_block["args"] = shlex.split(env_args)
            except ValueError:
                js_block["args"] = env_args

        js_block.setdefault("default_timeout", 5.0)
        js_block.setdefault("cpu_time_limit", 2.0)
        js_block.setdefault("memory_limit_bytes", 256 * 1024 * 1024)
        js_block.setdefault("max_output_bytes", 64 * 1024)
        js_block.setdefault("max_file_bytes", 128 * 1024)
        js_block.setdefault("max_files", 32)

        tools_block["javascript_executor"] = js_block

        vector_block = tools_block.get("vector_store")
        if not isinstance(vector_block, Mapping):
            vector_block = {}
        else:
            vector_block = dict(vector_block)

        adapter = vector_block.get("default_adapter")
        if isinstance(adapter, str) and adapter.strip():
            normalized_adapter = adapter.strip().lower()
        else:
            env_adapter = self.env_config.get("ATLAS_VECTOR_STORE_ADAPTER")
            if isinstance(env_adapter, str) and env_adapter.strip():
                normalized_adapter = env_adapter.strip().lower()
            else:
                normalized_adapter = "in_memory"
        vector_block["default_adapter"] = normalized_adapter

        adapters_block = vector_block.get("adapters")
        if isinstance(adapters_block, Mapping):
            adapters_block = dict(adapters_block)
        else:
            adapters_block = {}
        adapters_block.setdefault("in_memory", dict(adapters_block.get("in_memory", {})))

        mongo_block = adapters_block.get("mongodb")
        if isinstance(mongo_block, Mapping):
            mongo_block = dict(mongo_block)
        else:
            mongo_block = {}

        uri_override = self.env_config.get("ATLAS_VECTOR_STORE_MONGODB_URI")
        if uri_override and not mongo_block.get("connection_uri"):
            mongo_block["connection_uri"] = uri_override

        database_override = self.env_config.get("ATLAS_VECTOR_STORE_MONGODB_DATABASE")
        if database_override and not mongo_block.get("database"):
            mongo_block["database"] = database_override
        else:
            mongo_block.setdefault("database", "atlas_vector_store")

        collection_override = self.env_config.get("ATLAS_VECTOR_STORE_MONGODB_COLLECTION")
        if collection_override and not mongo_block.get("collection"):
            mongo_block["collection"] = collection_override
        else:
            mongo_block.setdefault("collection", "embeddings")

        index_override = self.env_config.get("ATLAS_VECTOR_STORE_MONGODB_INDEX")
        if index_override and not mongo_block.get("index_name"):
            mongo_block["index_name"] = index_override
        else:
            mongo_block.setdefault("index_name", "vector_index")

        adapters_block["mongodb"] = mongo_block
        vector_block["adapters"] = adapters_block

        tools_block["vector_store"] = vector_block

        mcp_block = tools_block.get("mcp")
        if not isinstance(mcp_block, Mapping):
            mcp_block = {}
        else:
            mcp_block = dict(mcp_block)

        env_enabled = self.env_config.get("ATLAS_MCP_ENABLED")
        if env_enabled is not None and "enabled" not in mcp_block:
            mcp_block["enabled"] = self._coerce_bool(env_enabled)
        mcp_block.setdefault("enabled", False)

        env_default_server = self.env_config.get("ATLAS_MCP_DEFAULT_SERVER")
        if env_default_server and not mcp_block.get("default_server"):
            mcp_block["default_server"] = str(env_default_server).strip()
        mcp_block.setdefault("default_server", "")

        env_timeout = self.env_config.get("ATLAS_MCP_TIMEOUT_SECONDS")
        if env_timeout is not None and "timeout_seconds" not in mcp_block:
            mcp_block["timeout_seconds"] = self._coerce_float(env_timeout, 30.0)
        mcp_block.setdefault("timeout_seconds", 30.0)

        env_health = self.env_config.get("ATLAS_MCP_HEALTH_CHECK_INTERVAL")
        if env_health is not None and "health_check_interval" not in mcp_block:
            mcp_block["health_check_interval"] = self._coerce_float(env_health, 300.0)
        mcp_block.setdefault("health_check_interval", 300.0)

        root_allow = self._normalize_csv_list(mcp_block.get("allow_tools"))
        env_allow = self._normalize_csv_list(self.env_config.get("ATLAS_MCP_ALLOW_TOOLS"))
        mcp_block["allow_tools"] = env_allow if env_allow is not None else root_allow

        root_deny = self._normalize_csv_list(mcp_block.get("deny_tools"))
        env_deny = self._normalize_csv_list(self.env_config.get("ATLAS_MCP_DENY_TOOLS"))
        mcp_block["deny_tools"] = env_deny if env_deny is not None else root_deny

        servers_block = mcp_block.get("servers")
        if isinstance(servers_block, Mapping):
            servers = dict(servers_block)
        else:
            servers = {}

        env_server_config: dict[str, Any] = {}
        env_transport = self.env_config.get("ATLAS_MCP_SERVER_TRANSPORT")
        if env_transport:
            env_server_config["transport"] = env_transport
        env_command = self.env_config.get("ATLAS_MCP_SERVER_COMMAND")
        if env_command:
            env_server_config["command"] = env_command
        env_args = self.env_config.get("ATLAS_MCP_SERVER_ARGS")
        if env_args:
            try:
                env_server_config["args"] = shlex.split(env_args)
            except ValueError:
                env_server_config["args"] = env_args
        env_url = self.env_config.get("ATLAS_MCP_SERVER_URL")
        if env_url:
            env_server_config["url"] = env_url
        env_cwd = self.env_config.get("ATLAS_MCP_SERVER_CWD")
        if env_cwd:
            env_server_config["cwd"] = env_cwd

        if env_server_config:
            default_server_name = mcp_block.get("default_server") or "default"
            if not isinstance(default_server_name, str) or not default_server_name.strip():
                default_server_name = "default"
            servers.setdefault(default_server_name, {})
            env_allow_tools = env_allow
            env_deny_tools = env_deny
            if env_allow_tools is not None:
                env_server_config.setdefault("allow_tools", env_allow_tools)
            if env_deny_tools is not None:
                env_server_config.setdefault("deny_tools", env_deny_tools)
            servers[default_server_name] = {
                **servers.get(default_server_name, {}),
                **env_server_config,
            }

        normalized_servers: dict[str, Mapping[str, Any]] = {}
        for name, raw_server in servers.items():
            server_name = str(name or "").strip()
            if not server_name:
                continue
            normalized_servers[server_name] = self._normalize_mcp_server(
                raw_server,
                defaults={
                    "timeout_seconds": mcp_block["timeout_seconds"],
                    "health_check_interval": mcp_block["health_check_interval"],
                    "allow_tools": mcp_block["allow_tools"],
                    "deny_tools": mcp_block["deny_tools"],
                },
            )

        mcp_block["servers"] = normalized_servers

        mcp_block.setdefault("server", mcp_block.get("default_server", ""))
        mcp_block.setdefault("tool", "")

        tools_block["mcp"] = mcp_block
        self.config["tools"] = tools_block

    def _ensure_tool_safety(self) -> None:
        tool_safety_block = self.config.get("tool_safety")
        if not isinstance(tool_safety_block, Mapping):
            tool_safety_block = {}
        else:
            tool_safety_block = dict(tool_safety_block)

        normalized_allowlist = self.normalize_network_allowlist(
            tool_safety_block.get("network_allowlist")
        )
        tool_safety_block["network_allowlist"] = normalized_allowlist
        self.config["tool_safety"] = tool_safety_block

    # ------------------------------------------------------------------
    # Normalisation helpers
    def normalize_network_allowlist(self, value: Any):
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

    # MCP helpers ------------------------------------------------------
    def _normalize_mcp_server(self, raw: Any, defaults: Mapping[str, Any]) -> Mapping[str, Any]:
        server = dict(raw) if isinstance(raw, Mapping) else {}

        transport = str(server.get("transport") or server.get("type") or "stdio").strip().lower()
        if transport not in {"stdio", "ws", "wss", "websocket", "http", "https"}:
            transport = "stdio"
        server["transport"] = transport

        args = server.get("args") or server.get("command_args") or []
        server["args"] = self._normalize_sequence(args)

        env = server.get("env")
        server["env"] = dict(env) if isinstance(env, Mapping) else {}

        server.setdefault("command", server.get("cmd"))
        server.setdefault("cwd", server.get("working_directory"))
        server.setdefault("url", server.get("uri") or server.get("endpoint"))

        server["allow_tools"] = self._normalize_csv_list(
            server.get("allow_tools"), fallback=defaults.get("allow_tools")
        )
        server["deny_tools"] = self._normalize_csv_list(
            server.get("deny_tools"), fallback=defaults.get("deny_tools")
        )

        server["timeout_seconds"] = self._coerce_float(
            server.get("timeout_seconds", server.get("timeout")),
            defaults.get("timeout_seconds", 30.0),
        )
        server["health_check_interval"] = self._coerce_float(
            server.get("health_check_interval"),
            defaults.get("health_check_interval", 300.0),
        )
        return server

    @staticmethod
    def _normalize_sequence(value: Any) -> list[Any]:
        if isinstance(value, str):
            try:
                return shlex.split(value)
            except ValueError:
                return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [item for item in value]
        return []

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
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
            return default

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        if result < 0:
            return default
        return result

    @staticmethod
    def _normalize_csv_list(value: Any, fallback: Any = None):
        if value is None:
            value = fallback
        if value is None:
            return None
        if isinstance(value, str):
            tokens = [item.strip() for item in value.split(",") if item.strip()]
            return tokens or None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            tokens = [str(item).strip() for item in value if str(item).strip()]
            return tokens or None
        return None
