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
        vector_block["adapters"] = adapters_block

        tools_block["vector_store"] = vector_block
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
