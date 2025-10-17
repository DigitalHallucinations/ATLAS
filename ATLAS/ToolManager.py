# ATLAS/Tools/ToolManager.py

import asyncio
import contextlib
import functools
import copy
import io
import json
import inspect
import importlib.util
import sys
import os
import random
import re
import socket
import threading
import uuid
from collections import deque
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from types import MappingProxyType

from jsonschema import Draft7Validator, ValidationError

from modules.analytics.persona_metrics import record_persona_tool_event
from modules.logging.logger import setup_logger
from modules.Tools.tool_event_system import event_system
from modules.Tools.providers.router import ToolProviderRouter

from ATLAS.config import ConfigManager
logger = setup_logger(__name__)


class ToolExecutionError(RuntimeError):
    """Exception raised when a tool call fails to execute successfully."""

    def __init__(
        self,
        message: str,
        *,
        tool_call_id: Optional[str] = None,
        function_name: Optional[str] = None,
        error_type: Optional[str] = None,
        entry: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.tool_call_id = tool_call_id
        self.function_name = function_name
        self.error_type = error_type
        self.entry = entry


class ToolManifestValidationError(RuntimeError):
    """Raised when a tool manifest fails schema validation."""

    def __init__(
        self,
        message: str,
        *,
        persona: Optional[str] = None,
        errors: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.persona = persona
        self.errors = errors or {}

_function_map_cache: Dict[str, Tuple[float, Optional[Tuple[str, ...]], Dict[str, Any]]] = {}
_function_payload_cache: Dict[str, Tuple[float, Optional[Tuple[str, ...]], Any]] = {}
_function_payload_cache_lock = threading.Lock()
_default_function_map_cache: Optional[Tuple[float, Dict[str, Any]]] = None
_default_function_map_lock = threading.Lock()
_default_config_manager: Optional[ConfigManager] = None
_DEFAULT_FUNCTIONS_CACHE_KEY = "__default__"
_tool_manifest_validator = None
_tool_manifest_validator_lock = threading.Lock()

_KNOWN_METADATA_FIELDS = (
    "version",
    "side_effects",
    "default_timeout",
    "auth",
    "allow_parallel",
    "idempotency_key",
    "safety_level",
    "requires_consent",
    "persona_allowlist",
    "capabilities",
    "cost_per_call",
    "cost_unit",
    "providers",
)

_TOOL_ACTIVITY_EVENT = "tool_activity"
_tool_activity_log: deque = deque(maxlen=100)
_tool_activity_lock = threading.Lock()

_SENSITIVE_PAYLOAD_FIELDS = ("arguments", "result", "stdout", "stderr")
_REDACTION_REPLACEMENT = "<redacted>"
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9\-]{8,}"),
    re.compile(r"rk-[A-Za-z0-9\-]{8,}"),
    re.compile(r"pk-[A-Za-z0-9\-]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{12,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)([A-Za-z0-9\-_=]{6,})"
)
_SECRET_JSON_PATTERN = re.compile(
    r"(?i)(\"(?:api[_-]?key|token|secret|password)\"\s*:\s*\")([^\"\\]{4,})(\")"
)

_DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0
_DEFAULT_CONVERSATION_TOOL_BUDGET_MS = 120000.0

_conversation_tool_runtime_ms: Dict[str, float] = {}
_conversation_runtime_lock = threading.Lock()

_SANDBOX_ENV_FLAG = "ATLAS_SANDBOX_ACTIVE"


@dataclass(frozen=True)
class ToolPolicyDecision:
    """Represents the outcome of a pre-execution policy evaluation."""

    allowed: bool
    reason: Optional[str] = None
    use_sandbox: bool = False
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _load_default_functions_payload(*, refresh: bool = False, config_manager=None):
    """Load the shared functions.json payload for default tools."""

    try:
        app_root = _get_config_manager(config_manager).get_app_root()
    except Exception as exc:
        logger.error(
            "Unable to determine application root when loading shared functions: %s",
            exc,
        )
        return None

    functions_path = os.path.join(
        app_root, "modules", "Tools", "tool_maps", "functions.json"
    )

    cache_key = _DEFAULT_FUNCTIONS_CACHE_KEY

    try:
        file_mtime = os.path.getmtime(functions_path)
    except FileNotFoundError:
        logger.error("Default functions.json not found at path: %s", functions_path)
        with _function_payload_cache_lock:
            _function_payload_cache.pop(cache_key, None)
        return None

    if refresh:
        with _function_payload_cache_lock:
            _function_payload_cache.pop(cache_key, None)

    with _function_payload_cache_lock:
        cache_entry = _function_payload_cache.get(cache_key)
        if cache_entry and not refresh:
            cached_mtime, cached_payload = cache_entry
            if cached_mtime == file_mtime:
                return cached_payload

        try:
            with open(functions_path, "r") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON decoding error in shared functions.json: %s",
                exc,
                exc_info=True,
            )
            _function_payload_cache.pop(cache_key, None)
            return None
        except Exception as exc:  # pragma: no cover - unexpected I/O errors
            logger.error(
                "Unexpected error loading shared functions.json: %s",
                exc,
                exc_info=True,
            )
            _function_payload_cache.pop(cache_key, None)
            return None

        _function_payload_cache[cache_key] = (file_mtime, payload)
        return payload


def _get_tool_manifest_validator(config_manager=None):
    """Return a cached JSON schema validator for tool manifests."""

    global _tool_manifest_validator

    with _tool_manifest_validator_lock:
        if _tool_manifest_validator is not None:
            return _tool_manifest_validator

        try:
            app_root = _get_config_manager(config_manager).get_app_root()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Unable to determine application root when loading tool manifest schema: %s",
                exc,
            )
            return None

        schema_path = os.path.join(
            app_root,
            "modules",
            "Tools",
            "tool_maps",
            "schema.json",
        )

        try:
            with open(schema_path, "r", encoding="utf-8") as schema_file:
                schema = json.load(schema_file)
        except FileNotFoundError:
            logger.warning("Tool manifest schema not found at path: %s", schema_path)
            return None
        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON in tool manifest schema at %s: %s",
                schema_path,
                exc,
                exc_info=True,
            )
            return None
        except Exception as exc:  # pragma: no cover - unexpected I/O errors
            logger.error(
                "Unexpected error loading tool manifest schema from %s: %s",
                schema_path,
                exc,
                exc_info=True,
            )
            return None

        try:
            _tool_manifest_validator = Draft7Validator(schema)
        except Exception as exc:  # pragma: no cover - schema compilation errors
            logger.error(
                "Failed to build tool manifest validator from schema %s: %s",
                schema_path,
                exc,
                exc_info=True,
            )
            _tool_manifest_validator = None

        return _tool_manifest_validator

def _freeze_generation_settings(
    settings: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Return an immutable snapshot of the supplied generation settings."""

    if not settings:
        return MappingProxyType({})

    if isinstance(settings, Mapping):
        try:
            return MappingProxyType(dict(settings))
        except TypeError:
            # Fallback for mappings with uncopyable values
            serialized: Dict[str, Any] = {}
            for key, value in settings.items():
                serialized[key] = value
            return MappingProxyType(serialized)

    raise TypeError("generation settings must be a mapping when provided")


async def _collect_async_chunks(stream: AsyncIterator) -> str:
    """Consume an async iterator of chunks into a single string."""

    chunks = []

    async for chunk in stream:
        if chunk is None:
            continue
        if isinstance(chunk, dict):
            text = chunk.get("content") or chunk.get("text") or chunk.get("message")
            if text is None:
                text = str(chunk)
        else:
            text = str(chunk)
        chunks.append(text)

    return "".join(chunks)


async def _gather_async_iterator(stream: AsyncIterator) -> List[Any]:
    """Consume an async iterator into a list, skipping ``None`` items."""

    items: List[Any] = []

    async for item in stream:
        if item is None:
            continue
        items.append(item)

    return items


def _is_async_stream(candidate: Any) -> bool:
    return isinstance(candidate, AsyncIterator) or inspect.isasyncgen(candidate)


@dataclass
class _ToolStreamCapture:
    items: List[Any]
    text: str
    entry: Dict[str, Any]


async def _stream_tool_iterator(
    stream: AsyncIterator,
    *,
    log_entry: Dict[str, Any],
    active_entry: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable[[Any], None]] = None,
) -> _ToolStreamCapture:
    """Iterate ``stream`` producing incremental tool activity updates."""

    collected_items: List[Any] = []
    text_fragments: List[str] = []
    if active_entry is None:
        active_entry = _record_tool_activity({**log_entry, "result": []})
    else:
        active_entry = _record_tool_activity(
            {**log_entry, "result": []}, replace=active_entry
        )

    async for item in stream:
        if item is None:
            continue

        collected_items.append(item)
        text_value = _stringify_tool_value(item)
        if text_value:
            text_fragments.append(text_value)

        if on_chunk is not None:
            try:
                on_chunk(item)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Tool chunk callback failed")

        update_payload = {
            **log_entry,
            "result": list(collected_items),
            "result_text": "".join(text_fragments),
            "status": log_entry.get("status", "running"),
            "completed_at": datetime.utcnow().isoformat(timespec="milliseconds"),
        }
        active_entry = _record_tool_activity(update_payload, replace=active_entry)

    return _ToolStreamCapture(collected_items, "".join(text_fragments), active_entry)


def _extract_text_and_audio(payload):
    """Return normalized content parts, audio payload, and a text fallback."""

    if payload is None:
        return [], None, ""

    def _normalize_single_part(value):
        if value is None:
            return {"type": "output_text", "text": ""}
        if isinstance(value, str):
            return {"type": "output_text", "text": value}
        if isinstance(value, list):
            return [_normalize_single_part(item) for item in value]
        if isinstance(value, tuple):
            return [_normalize_single_part(item) for item in value]
        if isinstance(value, dict):
            return copy.deepcopy(value)
        return {"type": "output_text", "text": str(value)}

    def _normalize_content_parts(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [_normalize_single_part(item) for item in value]
        return [_normalize_single_part(value)]

    def _collect_text_from_parts(parts):
        texts: List[str] = []

        def _collect(value):
            if isinstance(value, dict):
                if value.get("type") == "output_text":
                    text_value = value.get("text")
                    if isinstance(text_value, str):
                        texts.append(text_value)
                elif "content" in value:
                    _collect(value["content"])
            elif isinstance(value, list):
                for item in value:
                    _collect(item)
            elif isinstance(value, str):
                texts.append(value)

        for part in parts:
            _collect(part)

        return "".join(texts)

    audio_payload = None
    content_parts: List[Any] = []

    if isinstance(payload, dict):
        audio_payload = payload.get("audio")
        if "content" in payload:
            content_parts = _normalize_content_parts(payload.get("content"))
        else:
            for key in ("text", "message", "output_text"):
                value = payload.get(key)
                if value is not None:
                    content_parts = _normalize_content_parts(value)
                    break
    else:
        content_parts = _normalize_content_parts(payload)

    text_fallback = _collect_text_from_parts(content_parts)

    if not text_fallback and isinstance(payload, dict):
        for key in ("text", "message", "output_text"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                text_fallback = value
                if not content_parts:
                    content_parts = _normalize_content_parts(value)
                break
        if not text_fallback:
            content_value = payload.get("content")
            if isinstance(content_value, str) and content_value:
                text_fallback = content_value
                if not content_parts:
                    content_parts = _normalize_content_parts(content_value)

    if not text_fallback and not content_parts:
        text_fallback = "" if payload is None else str(payload)

    return content_parts, audio_payload, text_fallback


def _store_assistant_message(
    conversation_history,
    user,
    conversation_id,
    payload,
    *,
    metadata_overrides: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
):
    """Normalize ``payload`` and persist the assistant message to history."""

    if payload is None:
        return ""

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content_parts, audio_payload, text_fallback = _extract_text_and_audio(payload)

    if not content_parts:
        content_value: Any = text_fallback or ""
    else:
        content_value = content_parts

    entry_kwargs: Dict[str, Any] = {}
    metadata_payload: Optional[Dict[str, Any]] = None
    role = "assistant"

    if isinstance(payload, dict):
        metadata_candidate = payload.get("metadata")
        if isinstance(metadata_candidate, dict):
            metadata_payload = dict(metadata_candidate)
        role = str(payload.get("role", role))
        for key, value in payload.items():
            if key in {"content", "text", "message", "audio", "role", "metadata"}:
                continue
            entry_kwargs[key] = value

        if audio_payload is None and "audio" in payload:
            audio_payload = payload.get("audio")

    if metadata_overrides:
        merged_metadata: Dict[str, Any] = dict(metadata_payload or {})
        for key, value in metadata_overrides.items():
            if value is None:
                continue
            merged_metadata[key] = value
        metadata_payload = merged_metadata

    if audio_payload is not None:
        entry_kwargs["audio"] = audio_payload

    conversation_history.add_message(
        user,
        conversation_id,
        role,
        content_value,
        timestamp,
        metadata=metadata_payload,
        **entry_kwargs,
    )

    return text_fallback


def _proxy_streaming_response(
    stream: AsyncIterator,
    *,
    conversation_history,
    user,
    conversation_id,
    metadata_overrides: Optional[Dict[str, Any]] = None,
):
    """Yield chunks while capturing the final payload for history persistence."""

    async def _generator():
        collected_texts: List[str] = []
        normalized_parts: List[Any] = []
        metadata_payload: Optional[Dict[str, Any]] = None
        entry_kwargs: Dict[str, Any] = {}
        role = "assistant"
        audio_payload = None

        async for chunk in stream:
            if chunk is None:
                continue

            chunk_parts, chunk_audio, chunk_text = _extract_text_and_audio(chunk)

            if chunk_parts:
                normalized_parts.extend(chunk_parts)

            if chunk_audio is not None:
                audio_payload = chunk_audio

            if chunk_text:
                collected_texts.append(chunk_text)

            if isinstance(chunk, dict):
                metadata_candidate = chunk.get("metadata")
                if isinstance(metadata_candidate, dict):
                    metadata_payload = dict(metadata_candidate)

                role = str(chunk.get("role", role))

                for key, value in chunk.items():
                    if key in {"content", "text", "message", "metadata", "role", "audio"}:
                        continue
                    entry_kwargs[key] = value

            yield chunk

        aggregated_text = "".join(collected_texts)

        if metadata_payload or entry_kwargs or audio_payload is not None or role != "assistant" or normalized_parts:
            payload_content: Any = normalized_parts if normalized_parts else aggregated_text
            payload = {"content": payload_content}
            if metadata_payload:
                payload["metadata"] = metadata_payload
            if audio_payload is not None:
                payload["audio"] = audio_payload
            if role:
                payload["role"] = role
            if entry_kwargs:
                payload.update(entry_kwargs)
        else:
            payload = aggregated_text

        _store_assistant_message(
            conversation_history,
            user,
            conversation_id,
            payload,
            metadata_overrides=metadata_overrides,
        )

        return

    return _generator()


def _get_config_manager(candidate=None):
    """Return a :class:`ConfigManager`, caching the default instance."""

    global _default_config_manager

    if candidate is not None:
        return candidate

    if _default_config_manager is None:
        _default_config_manager = ConfigManager()

    return _default_config_manager


def _resolve_provider_manager(provider_manager=None, config_manager=None):
    """Return the active provider manager and (optionally new) config manager."""

    if provider_manager is not None:
        return provider_manager, config_manager

    if config_manager is not None:
        candidate = getattr(config_manager, "provider_manager", None)
        if candidate is not None:
            return candidate, config_manager

    if config_manager is None:
        logger.debug(
            "No provider manager supplied; instantiating ConfigManager to locate one."
        )
        config_manager = ConfigManager()
        candidate = getattr(config_manager, "provider_manager", None)
        if candidate is not None:
            return candidate, config_manager

    raise RuntimeError(
        "Provider manager is required but could not be determined. "
        "Pass provider_manager explicitly or provide a config manager that exposes one."
    )

def get_required_args(function):
    logger.info("Retrieving required arguments for the function.")
    sig = inspect.signature(_resolve_function_callable(function))
    return [
        param.name for param in sig.parameters.values()
        if param.default == param.empty and param.name != 'self'
    ]


def _clone_json_compatible(value: Any) -> Any:
    """Return a JSON-compatible clone of ``value`` when possible."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return str(value)


def _freeze_metadata(metadata: Optional[Dict[str, Any]]) -> Mapping[str, Any]:
    """Return an immutable view of metadata for safe sharing."""

    if not metadata:
        return MappingProxyType({})

    try:
        return MappingProxyType(dict(metadata))
    except TypeError:
        serialized: Dict[str, Any] = {}
        for key, value in dict(metadata).items():
            serialized[key] = value
        return MappingProxyType(serialized)


def _extract_persona_name(current_persona: Any) -> Optional[str]:
    """Return the persona display name when available."""

    if not current_persona:
        return None

    if isinstance(current_persona, Mapping):
        for key in ("name", "persona_name", "display_name"):
            value = current_persona.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("name", "persona_name", "display_name"):
        value = getattr(current_persona, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(current_persona, str) and current_persona.strip():
        return current_persona.strip()

    return None


def _normalize_persona_allowlist(raw_allowlist: Any) -> Optional[set[str]]:
    """Return a normalized set of persona names from ``raw_allowlist``."""

    if raw_allowlist is None:
        return None

    if isinstance(raw_allowlist, str):
        candidate = raw_allowlist.strip()
        return {candidate} if candidate else None

    if isinstance(raw_allowlist, Mapping):
        values = raw_allowlist.values()
    elif isinstance(raw_allowlist, (list, tuple, set)):
        values = raw_allowlist
    else:
        return None

    names = {str(item).strip() for item in values if str(item).strip()}
    return names or None


def _has_tool_consent(
    conversation_manager: Any,
    conversation_id: Any,
    function_name: str,
) -> bool:
    """Return ``True`` when the conversation has already approved ``function_name``."""

    if conversation_manager is None:
        return False

    checker = getattr(conversation_manager, "has_tool_consent", None)
    if not callable(checker):
        return False

    try:
        return bool(
            checker(conversation_id=conversation_id, tool_name=function_name)
        )
    except TypeError:
        try:
            return bool(checker(conversation_id, function_name))
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Consent checker failed for tool '%s'.", function_name)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Consent checker failed for tool '%s'.", function_name)

    return False


def _request_tool_consent(
    conversation_manager: Any,
    conversation_id: Any,
    function_name: str,
    metadata: Mapping[str, Any],
) -> bool:
    """Request consent from the active conversation for ``function_name``."""

    if conversation_manager is None:
        return False

    requester = getattr(conversation_manager, "request_tool_consent", None)
    if not callable(requester):
        return False

    try:
        return bool(
            requester(
                conversation_id=conversation_id,
                tool_name=function_name,
                metadata=dict(metadata) if metadata else {},
            )
        )
    except TypeError:
        try:
            return bool(requester(conversation_id, function_name, metadata))
        except TypeError:
            try:
                return bool(requester(conversation_id, function_name))
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Consent request failed for tool '%s'.", function_name)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Consent request failed for tool '%s'.", function_name)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Consent request failed for tool '%s'.", function_name)

    return False


def _evaluate_tool_policy(
    *,
    function_name: str,
    metadata: Mapping[str, Any],
    current_persona: Any,
    conversation_manager: Any,
    conversation_id: Any,
) -> ToolPolicyDecision:
    """Return the pre-execution policy decision for ``function_name``."""

    persona_name = _extract_persona_name(current_persona)
    allowlist = _normalize_persona_allowlist(metadata.get("persona_allowlist"))

    if allowlist and (persona_name is None or persona_name not in allowlist):
        reason = (
            f"Tool '{function_name}' is restricted to approved personas."
        )
        return ToolPolicyDecision(
            allowed=False,
            reason=reason,
            metadata=_freeze_metadata(dict(metadata)),
        )

    safety_level = str(metadata.get("safety_level") or "standard").lower()
    requires_consent = bool(metadata.get("requires_consent"))
    if safety_level == "high":
        requires_consent = True

    if requires_consent and not _has_tool_consent(
        conversation_manager, conversation_id, function_name
    ):
        if not _request_tool_consent(
            conversation_manager, conversation_id, function_name, metadata
        ):
            reason = (
                f"Tool '{function_name}' requires approval before it can be used."
            )
            return ToolPolicyDecision(
                allowed=False,
                reason=reason,
                metadata=_freeze_metadata(dict(metadata)),
            )

    use_sandbox = safety_level in {"high"}

    return ToolPolicyDecision(
        allowed=True,
        use_sandbox=use_sandbox,
        metadata=_freeze_metadata(dict(metadata)),
    )


def _get_sandbox_runner(config_manager=None):
    """Return a sandbox runner bound to ``config_manager``."""

    return SandboxedToolRunner(config_manager)


class SandboxedToolRunner:
    """Provides a minimal sandbox environment for high-risk tool execution."""

    def __init__(self, config_manager=None):
        self._config_manager = config_manager

    def _resolve_network_allowlist(
        self, metadata: Optional[Mapping[str, Any]]
    ) -> Optional[set[str]]:
        config_block = _get_config_section(self._config_manager, "tool_safety")
        allowlist = None
        if isinstance(config_block, Mapping):
            allowlist = config_block.get("network_allowlist")
        if allowlist is None and metadata is not None:
            allowlist = metadata.get("network_allowlist")

        if allowlist is None:
            return None

        if isinstance(allowlist, str):
            allowlist = [allowlist]

        if isinstance(allowlist, Mapping):
            values = allowlist.values()
        else:
            values = allowlist

        allowed_hosts: set[str] = set()
        for item in values:
            host = str(item).strip()
            if not host:
                continue
            allowed_hosts.add(host)
            allowed_hosts.add(host.lower())

        return allowed_hosts or None

    @staticmethod
    def _extract_host(address: Any) -> Optional[str]:
        if isinstance(address, (list, tuple)) and address:
            candidate = address[0]
        else:
            candidate = address

        if isinstance(candidate, bytes):
            candidate = candidate.decode(errors="ignore")

        if isinstance(candidate, str):
            candidate = candidate.strip()
            if not candidate or candidate.startswith("/"):
                return None
            return candidate

        return None

    @staticmethod
    def _ensure_host_allowed(host: Optional[str], allowed_hosts: Optional[set[str]]):
        if host is None or allowed_hosts is None:
            return

        if host in allowed_hosts or host.lower() in allowed_hosts:
            return

        raise PermissionError(
            f"Network access to '{host}' is not permitted by the sandbox allowlist."
        )

    @contextlib.contextmanager
    def activate(self, *, metadata: Optional[Mapping[str, Any]] = None):
        """Apply sandbox constraints for the duration of the context."""

        allowed_hosts = self._resolve_network_allowlist(metadata)
        original_create_connection = socket.create_connection
        original_connect = socket.socket.connect

        def _guarded_create_connection(address, *args, **kwargs):
            host = self._extract_host(address)
            self._ensure_host_allowed(host, allowed_hosts)
            return original_create_connection(address, *args, **kwargs)

        def _guarded_connect(sock, address):
            host = self._extract_host(address)
            self._ensure_host_allowed(host, allowed_hosts)
            return original_connect(sock, address)

        previous_flag = os.environ.get(_SANDBOX_ENV_FLAG)
        os.environ[_SANDBOX_ENV_FLAG] = "1"

        socket.create_connection = _guarded_create_connection
        socket.socket.connect = _guarded_connect

        try:
            yield
        finally:
            socket.create_connection = original_create_connection
            socket.socket.connect = original_connect
            if previous_flag is None:
                os.environ.pop(_SANDBOX_ENV_FLAG, None)
            else:
                os.environ[_SANDBOX_ENV_FLAG] = previous_flag


def _get_config_section(config_manager, key: str):
    """Return a configuration block by key when available."""

    manager = config_manager
    if manager is None:
        try:
            manager = _get_config_manager()
        except Exception:  # pragma: no cover - best effort fallback
            return None

    getter = getattr(manager, "get_config", None)
    if callable(getter):
        try:
            value = getter(key, ConfigManager.UNSET)
        except TypeError:
            value = getter(key)
        if value is ConfigManager.UNSET:
            return None
        return value

    raw_config = getattr(manager, "config", None)
    if isinstance(raw_config, Mapping):
        return raw_config.get(key)

    return None


def _resolve_tool_timeout_seconds(config_manager, metadata_timeout: Optional[Any]) -> Optional[float]:
    """Determine the timeout for a tool call in seconds."""

    if isinstance(metadata_timeout, (int, float)):
        if metadata_timeout <= 0:
            return None
        return float(metadata_timeout)

    section = _get_config_section(config_manager, "tool_defaults")
    if isinstance(section, Mapping):
        candidate = section.get("timeout_seconds")
        if isinstance(candidate, (int, float)):
            if candidate <= 0:
                return None
            return float(candidate)

    return _DEFAULT_TOOL_TIMEOUT_SECONDS


def _get_conversation_tool_budget_ms(config_manager) -> Optional[float]:
    """Return the configured per-conversation tool runtime budget in milliseconds."""

    section = _get_config_section(config_manager, "conversation")
    if isinstance(section, Mapping):
        candidate = section.get("max_tool_duration_ms")
        if isinstance(candidate, (int, float)):
            if candidate <= 0:
                return None
            return float(candidate)

    return _DEFAULT_CONVERSATION_TOOL_BUDGET_MS


def _get_conversation_runtime_ms(conversation_id: Optional[str]) -> float:
    """Return the accumulated runtime for ``conversation_id`` in milliseconds."""

    if not conversation_id:
        return 0.0

    with _conversation_runtime_lock:
        return _conversation_tool_runtime_ms.get(conversation_id, 0.0)


def _increment_conversation_runtime_ms(conversation_id: Optional[str], duration_ms: float) -> None:
    """Add ``duration_ms`` to the tracked runtime for ``conversation_id``."""

    if not conversation_id:
        return

    try:
        increment = float(duration_ms)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return

    with _conversation_runtime_lock:
        previous = _conversation_tool_runtime_ms.get(conversation_id, 0.0)
        _conversation_tool_runtime_ms[conversation_id] = previous + increment


def _generate_idempotency_key() -> str:
    """Return a unique key suitable for idempotent tool invocations."""

    return uuid.uuid4().hex


def _is_tool_idempotent(metadata: Optional[Mapping[str, Any]]) -> bool:
    """Return ``True`` when ``metadata`` marks a tool as idempotent."""

    if not metadata:
        return False

    candidate = metadata.get("idempotency_key")
    if isinstance(candidate, Mapping):
        return bool(candidate.get("required"))
    if isinstance(candidate, bool):
        return candidate
    return bool(candidate)


async def _apply_idempotent_retry_backoff(
    attempt: int,
    *,
    idempotent: bool,
    base_delay: float = 0.5,
    jitter: float = 0.5,
) -> bool:
    """Sleep using exponential backoff with jitter when ``idempotent`` is True."""

    if not idempotent:
        return False

    interval = base_delay * (2 ** max(0, attempt - 1))
    interval += random.uniform(0.0, jitter)
    await asyncio.sleep(interval)
    return True


async def _run_with_timeout(awaitable, timeout: Optional[float]):
    """Await ``awaitable`` enforcing ``timeout`` seconds when positive."""

    if timeout is None or timeout <= 0:
        return await awaitable

    if asyncio.isfuture(awaitable) or isinstance(awaitable, asyncio.Task):
        task = awaitable
        created_task = False
    elif inspect.isawaitable(awaitable):
        task = asyncio.create_task(awaitable)
        created_task = True
    else:  # pragma: no cover - defensive guard
        return await awaitable

    try:
        return await asyncio.wait_for(task, timeout)
    except asyncio.TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise
    finally:
        if created_task and task.done():
            # Ensure any exception state is retrieved without suppressing the original error
            with contextlib.suppress(BaseException):
                task.result()


def _build_metadata_lookup(functions_payload: Any) -> Dict[str, Mapping[str, Any]]:
    """Create a lookup map of tool metadata keyed by function name."""

    lookup: Dict[str, Mapping[str, Any]] = {}

    if isinstance(functions_payload, list):
        entries = functions_payload
    elif isinstance(functions_payload, dict):
        entries = functions_payload.values()
    else:
        return lookup

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        metadata: Dict[str, Any] = {}
        for field in _KNOWN_METADATA_FIELDS:
            if field in entry:
                metadata[field] = copy.deepcopy(entry[field])
        lookup[name] = _freeze_metadata(metadata)

    return lookup


def _annotate_function_map(
    function_map: Optional[Dict[str, Any]],
    *,
    metadata_lookup: Optional[Dict[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Attach metadata to each callable in the supplied function map."""

    if not isinstance(function_map, dict):
        return {}

    annotated: Dict[str, Dict[str, Any]] = {}
    for name, func in function_map.items():
        metadata = MappingProxyType({})
        if metadata_lookup and name in metadata_lookup:
            metadata = metadata_lookup[name]

        entry: Dict[str, Any] = {"metadata": metadata}

        providers: List[Mapping[str, Any]] = []
        raw_providers = metadata.get("providers") if isinstance(metadata, Mapping) else None
        if isinstance(raw_providers, Iterable):
            for provider in raw_providers:
                if isinstance(provider, Mapping):
                    providers.append(provider)

        if providers:
            router = ToolProviderRouter(
                tool_name=name,
                provider_specs=providers,
                fallback_callable=func,
            )
            entry["callable"] = router.call
            entry["provider_router"] = router
        else:
            entry["callable"] = func

        annotated[name] = entry

    return annotated


def _build_function_entry_lookup(functions_payload: Any) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    if isinstance(functions_payload, list):
        entries = functions_payload
    elif isinstance(functions_payload, Mapping):
        entries = functions_payload.values()
    else:
        return lookup

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not name:
            continue
        lookup[str(name)] = entry
    return lookup


def _extract_allowed_tools(current_persona: Any) -> Optional[List[str]]:
    if not isinstance(current_persona, Mapping):
        return None

    allowed = current_persona.get("allowed_tools")
    if allowed is None:
        return None

    names: List[str] = []

    if isinstance(allowed, str):
        candidate = allowed.strip()
        if candidate:
            names.append(candidate)
    elif isinstance(allowed, Iterable):
        for item in allowed:
            if isinstance(item, str):
                candidate = item.strip()
            elif isinstance(item, Mapping):
                raw_name = item.get("name")
                candidate = str(raw_name).strip() if raw_name is not None else ""
            else:
                candidate = ""
            if candidate and candidate not in names:
                names.append(candidate)

    return names


def _filter_function_map_by_allowlist(
    function_map: Optional[Dict[str, Any]],
    allowed_names: Optional[List[str]],
) -> Dict[str, Any]:
    if allowed_names is None:
        return dict(function_map or {}) if isinstance(function_map, dict) else {}

    if not allowed_names:
        return {}

    if not isinstance(function_map, dict):
        return {}

    filtered: Dict[str, Any] = {}
    for name in allowed_names:
        if name in function_map and name not in filtered:
            filtered[name] = function_map[name]
    return filtered


def _select_allowed_functions(
    functions_payload: Any,
    allowed_names: Optional[List[str]],
    *,
    config_manager=None,
    refresh: bool = False,
) -> Any:
    if allowed_names is None:
        return functions_payload

    if not allowed_names:
        return []

    lookup = _build_function_entry_lookup(functions_payload)
    shared_lookup: Optional[Dict[str, Mapping[str, Any]]] = None
    selected: List[Dict[str, Any]] = []

    for name in allowed_names:
        entry = lookup.get(name)
        if entry is None:
            if shared_lookup is None:
                shared_payload = _load_default_functions_payload(
                    refresh=refresh,
                    config_manager=config_manager,
                )
                shared_lookup = _build_function_entry_lookup(shared_payload)
            entry = (shared_lookup or {}).get(name)
        if entry is not None:
            selected.append(copy.deepcopy(dict(entry)))

    return selected


def _resolve_function_callable(entry: Any) -> Any:
    """Return the executable callable from a function map entry."""

    if isinstance(entry, dict):
        candidate = entry.get("callable")
        if candidate is not None:
            return candidate
    return entry


def _stringify_tool_value(value: Any) -> str:
    """Return a human-readable string for tool payload data."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _redact_text(value: str) -> str:
    """Mask sensitive tokens within ``value`` using regex patterns."""

    if not value:
        return value

    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_REDACTION_REPLACEMENT, redacted)

    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTION_REPLACEMENT}",
        redacted,
    )

    redacted = _SECRET_JSON_PATTERN.sub(
        lambda match: f"{match.group(1)}{_REDACTION_REPLACEMENT}{match.group(3)}",
        redacted,
    )

    return redacted


def _redact_payload_value(value: Any) -> Any:
    """Recursively redact sensitive content in payload values."""

    if isinstance(value, str):
        return _redact_text(value)

    if isinstance(value, Mapping):
        return {key: _redact_payload_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_redact_payload_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_redact_payload_value(item) for item in value)

    if isinstance(value, set):
        return {_redact_payload_value(item) for item in value}

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    return _redact_text(str(value))


def _summarize_payload_value(value: Any, *, limit: int) -> str:
    """Return a truncated, human-readable representation of ``value``."""

    text = _stringify_tool_value(value).strip()
    if limit <= 0 or len(text) <= limit:
        return text

    if limit == 1:
        return "…"

    return text[: limit - 1] + "…"


def _get_tool_logging_preferences(config_manager=None) -> Dict[str, Any]:
    """Return tool logging preferences from configuration."""

    section = _get_config_section(config_manager, "tool_logging")
    if not isinstance(section, Mapping):
        return {"log_full_payloads": False, "payload_summary_length": 256}

    preferences = dict(section)
    preferences.setdefault("log_full_payloads", False)
    preferences.setdefault("payload_summary_length", 256)
    return preferences


def _build_tool_metrics(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Construct the structured metrics block for a tool activity entry."""

    metrics: Dict[str, Any] = {}

    status = entry.get("status")
    if status is not None:
        metrics["status"] = status

    latency = entry.get("duration_ms")
    if latency is not None:
        metrics["latency_ms"] = latency

    started = entry.get("started_at")
    if started is not None:
        metrics["started_at"] = started

    completed = entry.get("completed_at")
    if completed is not None:
        metrics["completed_at"] = completed

    timeout_seconds = entry.get("timeout_seconds")
    if timeout_seconds is not None:
        metrics["timeout_seconds"] = timeout_seconds

    error_type = entry.get("error_type")
    if error_type is not None:
        metrics["error_type"] = error_type

    result_value = entry.get("result")
    if isinstance(result_value, list):
        metrics["result_item_count"] = len(result_value)

    tool_version = entry.get("tool_version")
    metadata = entry.get("metadata")
    if tool_version is not None:
        metrics["tool_version"] = tool_version
    elif isinstance(metadata, Mapping):
        candidate = metadata.get("version") or metadata.get("tool_version")
        if candidate is not None:
            metrics["tool_version"] = candidate

    return metrics


def _build_public_tool_entry(
    entry: Mapping[str, Any], *, config_manager=None
) -> Dict[str, Any]:
    """Return a redacted, configuration-aware view of ``entry``."""

    preferences = _get_tool_logging_preferences(config_manager)
    log_full_payloads = bool(preferences.get("log_full_payloads"))
    summary_limit = int(preferences.get("payload_summary_length", 256) or 0)

    public_entry: Dict[str, Any] = {}
    for key, value in entry.items():
        if key in _SENSITIVE_PAYLOAD_FIELDS:
            continue
        if key.startswith("_"):
            continue
        if isinstance(value, str) and key in {"arguments_text", "result_text", "error"}:
            redacted_value = _redact_text(value)
            if not log_full_payloads and key in {"arguments_text", "result_text"}:
                redacted_value = _summarize_payload_value(
                    redacted_value, limit=summary_limit
                )
            public_entry[key] = redacted_value
            continue
        public_entry[key] = _clone_json_compatible(value)

    sanitized_payload: Dict[str, Any] = {}
    payload_preview: Dict[str, str] = {}
    for field in _SENSITIVE_PAYLOAD_FIELDS:
        sanitized_value = _redact_payload_value(entry.get(field))
        sanitized_payload[field] = _clone_json_compatible(sanitized_value)
        payload_preview[field] = _summarize_payload_value(
            sanitized_value, limit=summary_limit
        )

    if log_full_payloads:
        public_entry.update({
            field: sanitized_payload[field] for field in _SENSITIVE_PAYLOAD_FIELDS
        })
        public_entry["payload"] = dict(sanitized_payload)
    else:
        public_entry.update({field: payload_preview[field] for field in _SENSITIVE_PAYLOAD_FIELDS})
        public_entry["payload"] = None

    public_entry["payload_preview"] = payload_preview
    public_entry["payload_included"] = log_full_payloads
    public_entry["metrics"] = _build_tool_metrics(entry)

    return public_entry


def _record_tool_activity(
    entry: Dict[str, Any], *, replace: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Append or update a tool activity entry and publish an event."""

    # Ensure we only store JSON-friendly copies to avoid mutating UI consumers.
    stored_entry = {
        **entry,
        "arguments": _clone_json_compatible(entry.get("arguments")),
        "result": _clone_json_compatible(entry.get("result")),
        "stdout": _clone_json_compatible(entry.get("stdout")),
        "stderr": _clone_json_compatible(entry.get("stderr")),
    }

    with _tool_activity_lock:
        if replace is not None:
            replace.clear()
            replace.update(stored_entry)
            stored_entry = replace
        else:
            _tool_activity_log.append(stored_entry)

    public_entry = _build_public_tool_entry(stored_entry)
    event_system.publish(_TOOL_ACTIVITY_EVENT, public_entry)
    return stored_entry


def _record_tool_failure(
    conversation_history,
    user,
    conversation_id,
    *,
    tool_call_id: Optional[str],
    function_name: Optional[str],
    message: str,
    error_type: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Persist a structured tool failure message in the conversation history."""

    if conversation_history is None or not hasattr(conversation_history, "add_message"):
        return None

    timestamp_value = timestamp or datetime.now()
    metadata: Dict[str, Any] = {"status": "error"}
    if function_name:
        metadata["name"] = function_name
    if error_type:
        metadata["error_type"] = error_type

    entry_kwargs: Dict[str, Any] = {}
    if tool_call_id is not None:
        entry_kwargs["tool_call_id"] = tool_call_id

    try:
        return conversation_history.add_message(
            user,
            conversation_id,
            "tool",
            [{"type": "output_text", "text": message}],
            timestamp_value.strftime("%Y-%m-%d %H:%M:%S"),
            metadata=metadata,
            **entry_kwargs,
        )
    except Exception:  # pragma: no cover - history failures should not crash tools
        logger.exception("Failed to record tool failure in conversation history.")
        return None


def get_tool_activity_log(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return a copy of the recorded tool activity log."""

    with _tool_activity_lock:
        entries = list(_tool_activity_log)

    if limit is not None and isinstance(limit, int) and limit >= 0:
        entries = entries[-limit:]

    return [_build_public_tool_entry(entry) for entry in entries]

def load_default_function_map(*, refresh: bool = False, config_manager=None):
    """Load the shared default tool function map."""

    logger.info("Attempting to load shared default function map.")

    try:
        app_root = _get_config_manager(config_manager).get_app_root()
    except Exception as exc:
        logger.error(
            "Unable to determine application root when loading shared tools: %s",
            exc,
        )
        return None

    maps_path = os.path.join(app_root, "modules", "Tools", "tool_maps", "maps.py")
    module_name = "modules.Tools.tool_maps.maps"

    global _default_function_map_cache

    try:
        with _default_function_map_lock:
            try:
                file_mtime = os.path.getmtime(maps_path)
            except FileNotFoundError:
                logger.error("Default maps.py not found at path: %s", maps_path)
                _default_function_map_cache = None
                return None

            cache_entry = None if refresh else _default_function_map_cache
            if cache_entry:
                cached_mtime, cached_map = cache_entry
                if cached_mtime == file_mtime:
                    logger.info("Returning cached shared function map without reloading module.")
                    metadata_lookup = _build_metadata_lookup(
                        _load_default_functions_payload(
                            refresh=refresh, config_manager=config_manager
                        )
                    )
                    return _annotate_function_map(
                        cached_map, metadata_lookup=metadata_lookup
                    )

                logger.info(
                    "Detected updated shared maps.py (cached mtime %s, current mtime %s); reloading.",
                    cached_mtime,
                    file_mtime,
                )
                sys.modules.pop(module_name, None)
                _default_function_map_cache = None

            if refresh:
                logger.info("Refresh requested for shared tool map; clearing cached module.")
                sys.modules.pop(module_name, None)
                _default_function_map_cache = None

            module = sys.modules.get(module_name)

            if module is None:
                logger.info("Loading shared tool map module '%s' from '%s'.", module_name, maps_path)
                spec = importlib.util.spec_from_file_location(module_name, maps_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load specification for shared tool map from {maps_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            else:
                logger.info("Reusing already loaded shared tool map module '%s'.", module_name)

            function_map = getattr(module, "function_map", None)

            if isinstance(function_map, dict):
                _default_function_map_cache = (file_mtime, function_map)
                metadata_lookup = _build_metadata_lookup(
                    _load_default_functions_payload(
                        refresh=refresh, config_manager=config_manager
                    )
                )
                return _annotate_function_map(
                    function_map, metadata_lookup=metadata_lookup
                )

            logger.warning("Shared tool map module '%s' does not define 'function_map'.", module_name)
            _default_function_map_cache = None
            return None
    except Exception as exc:
        logger.error("Error loading shared default function map: %s", exc, exc_info=True)
        with _default_function_map_lock:
            _default_function_map_cache = None
        return None


def load_function_map_from_current_persona(
    current_persona,
    *,
    refresh=False,
    config_manager=None,
):
    logger.info("Attempting to load function map from current persona.")
    if not current_persona or "name" not in current_persona:
        logger.error("Current persona is None or does not have a 'name' key.")
        return load_default_function_map(refresh=refresh, config_manager=config_manager)

    persona_name = current_persona["name"]
    allowed_names = _extract_allowed_tools(current_persona)
    allowed_signature = tuple(allowed_names) if allowed_names is not None else None
    try:
        app_root = _get_config_manager(config_manager).get_app_root()
    except Exception as exc:
        logger.error(
            "Unable to determine application root when loading persona '%s': %s",
            persona_name,
            exc,
        )
        return load_default_function_map(refresh=refresh, config_manager=config_manager)

    toolbox_root = os.path.join(app_root, "modules", "Personas", persona_name, "Toolbox")
    maps_path = os.path.join(toolbox_root, "maps.py")
    module_name = f'persona_{persona_name}_maps'
    try:
        if refresh:
            logger.info(
                "Refresh requested for persona '%s'; clearing cached module and function map.",
                persona_name,
            )
            sys.modules.pop(module_name, None)
            _function_map_cache.pop(persona_name, None)
            _function_payload_cache.pop(persona_name, None)

        file_mtime = os.path.getmtime(maps_path)

        cache_entry = _function_map_cache.get(persona_name)

        if not refresh and cache_entry:
            cached_mtime, cached_signature, cached_map = cache_entry
            if cached_mtime == file_mtime and cached_signature == allowed_signature:
                logger.info(
                    "Returning cached function map for persona '%s' without reloading module.",
                    persona_name,
                )
                metadata_lookup = _build_metadata_lookup(
                    load_functions_from_json(
                        current_persona,
                        refresh=refresh,
                        config_manager=config_manager,
                    )
                )
                return _annotate_function_map(
                    cached_map,
                    metadata_lookup=metadata_lookup,
                )

            logger.info(
                "Detected updated maps.py for persona '%s' (cached mtime %s, current mtime %s); reloading.",
                persona_name,
                cached_mtime,
                file_mtime,
            )
            sys.modules.pop(module_name, None)
            _function_map_cache.pop(persona_name, None)

        module = sys.modules.get(module_name)

        if module is None:
            logger.info(
                "Module '%s' not found in sys.modules; loading from '%s'.",
                module_name,
                maps_path,
            )
            spec = importlib.util.spec_from_file_location(module_name, maps_path)
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Could not load specification for persona '{persona_name}' from {maps_path}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            logger.info(
                "Reusing already loaded module '%s' from sys.modules for persona '%s'.",
                module_name,
                persona_name,
            )

        if hasattr(module, 'function_map'):
            logger.info(
                "Function map successfully loaded for persona '%s': %s",
                persona_name,
                module.function_map,
            )
            filtered_map = _filter_function_map_by_allowlist(
                module.function_map, allowed_names
            )
            _function_map_cache[persona_name] = (
                file_mtime,
                allowed_signature,
                filtered_map,
            )
            metadata_lookup = _build_metadata_lookup(
                load_functions_from_json(
                    current_persona,
                    refresh=refresh,
                    config_manager=config_manager,
                )
            )
            return _annotate_function_map(
                filtered_map,
                metadata_lookup=metadata_lookup,
            )
        else:
            logger.warning(
                "No 'function_map' found in maps.py for persona '%s'.",
                persona_name,
            )
            _function_map_cache.pop(persona_name, None)
    except FileNotFoundError:
        logger.error(f"maps.py file not found for persona '{persona_name}' at path: {maps_path}")
    except Exception as e:
        logger.error(f"Error loading function map for persona '{persona_name}': {e}", exc_info=True)

    logger.info(
        "Falling back to shared default function map for persona '%s'.",
        persona_name,
    )
    fallback_map = load_default_function_map(refresh=refresh, config_manager=config_manager)
    return _filter_function_map_by_allowlist(fallback_map, allowed_names)


def load_functions_from_json(
    current_persona,
    *,
    refresh=False,
    config_manager=None,
):
    logger.info("Attempting to load functions from JSON for the current persona.")
    if not current_persona or "name" not in current_persona:
        logger.error("Current persona is None or does not have a 'name' key.")
        return None

    persona_name = current_persona["name"]
    allowed_names = _extract_allowed_tools(current_persona)
    allowed_signature = tuple(allowed_names) if allowed_names is not None else None
    try:
        app_root = _get_config_manager(config_manager).get_app_root()
    except Exception as exc:
        logger.error(
            "Unable to determine application root when loading persona '%s': %s",
            persona_name,
            exc,
        )
        return None

    toolbox_root = os.path.join(app_root, "modules", "Personas", persona_name, "Toolbox")
    functions_json_path = os.path.join(toolbox_root, "functions.json")

    try:
        try:
            file_mtime = os.path.getmtime(functions_json_path)
        except FileNotFoundError:
            file_mtime = None

        cache_key_mtime = file_mtime if file_mtime is not None else -1.0

        with _function_payload_cache_lock:
            if not refresh:
                cache_entry = _function_payload_cache.get(persona_name)
                if cache_entry:
                    cached_mtime, cached_signature, cached_functions = cache_entry
                    if cached_mtime == cache_key_mtime and cached_signature == allowed_signature:
                        logger.info(
                            "Returning cached functions for persona '%s' (mtime %s).",
                            persona_name,
                            cached_mtime,
                        )
                        return cached_functions

        functions = None
        if file_mtime is not None:
            with open(functions_json_path, 'r', encoding='utf-8') as file:
                functions = json.load(file)

            logger.info(
                "Functions successfully loaded from JSON for persona '%s': %s",
                persona_name,
                functions,
            )

            validator = _get_tool_manifest_validator(config_manager=config_manager)
            if validator is not None:
                try:
                    validator.validate(functions)
                except ValidationError as exc:
                    error_details = {
                        'persona': persona_name,
                        'path': list(exc.absolute_path),
                        'message': exc.message,
                    }
                    logger.error(
                        "Tool manifest validation error for persona '%s': %s",
                        persona_name,
                        error_details,
                    )
                    with _function_payload_cache_lock:
                        _function_payload_cache.pop(persona_name, None)
                    raise ToolManifestValidationError(
                        f"Invalid tool manifest for persona '{persona_name}': {exc.message}",
                        persona=persona_name,
                        errors=error_details,
                    ) from exc

        selected = _select_allowed_functions(
            functions,
            allowed_names,
            config_manager=config_manager,
            refresh=refresh,
        )

        with _function_payload_cache_lock:
            _function_payload_cache[persona_name] = (
                cache_key_mtime,
                allowed_signature,
                selected,
            )

        return selected
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error in functions.json for persona '{persona_name}': {e}", exc_info=True)
        with _function_payload_cache_lock:
            _function_payload_cache.pop(persona_name, None)
    except ToolManifestValidationError:
        raise
    except FileNotFoundError:
        logger.error(f"functions.json file not found for persona '{persona_name}' at path: {functions_json_path}")
        with _function_payload_cache_lock:
            _function_payload_cache.pop(persona_name, None)
        return _select_allowed_functions(None, allowed_names, config_manager=config_manager, refresh=refresh)
    except Exception as e:
        logger.error(f"Unexpected error loading functions for persona '{persona_name}': {e}", exc_info=True)
        with _function_payload_cache_lock:
            _function_payload_cache.pop(persona_name, None)
    return None

async def use_tool(
    user,
    conversation_id,
    message,
    conversation_history,
    function_map,
    functions,
    current_persona,
    temperature_var,
    top_p_var,
    frequency_penalty_var,
    presence_penalty_var,
    conversation_manager,
    provider_manager=None,
    config_manager=None,
    *,
    stream: Optional[bool] = None,
    generation_settings: Optional[Mapping[str, Any]] = None,
):
    logger.info(f"use_tool called for user: {user}, conversation_id: {conversation_id}")
    logger.info(f"Message received: {message}")

    if conversation_manager is None:
        conversation_manager = conversation_history

    try:
        provider_manager, config_manager = _resolve_provider_manager(
            provider_manager, config_manager
        )
    except RuntimeError as exc:
        logger.error("Unable to resolve provider manager: %s", exc)
        raise

    try:
        generation_context = _freeze_generation_settings(generation_settings)
    except TypeError as exc:
        logger.warning("Ignoring invalid generation settings: %s", exc)
        generation_context = MappingProxyType({})

    if not isinstance(message, dict):
        normalized_message = {}
        for attr in ("function_call", "tool_calls", "tool_call"):
            value = getattr(message, attr, None)
            if value is not None:
                normalized_message[attr] = value
        message = normalized_message

    def _safe_get(target, key, default=None):
        if isinstance(target, dict):
            return target.get(key, default)
        return getattr(target, key, default)

    tool_streaming_enabled = bool(stream)
    persona_name = _extract_persona_name(current_persona)

    def _record_persona_metric(
        tool_name: Optional[str],
        *,
        success: bool,
        latency_ms: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        if not persona_name or not tool_name:
            return
        try:
            record_persona_tool_event(
                persona=persona_name,
                tool=tool_name,
                success=success,
                latency_ms=latency_ms,
                timestamp=timestamp,
                config_manager=config_manager,
            )
        except Exception:
            logger.exception(
                "Failed to record persona metrics for persona '%s' and tool '%s'",
                persona_name,
                tool_name,
            )

    def _normalize_tool_call_entry(raw_entry):
        if not raw_entry:
            return None

        function_payload = _safe_get(raw_entry, "function") or raw_entry
        name = _safe_get(function_payload, "name") or _safe_get(raw_entry, "name")
        arguments = _safe_get(function_payload, "arguments")
        if arguments is None:
            arguments = _safe_get(raw_entry, "arguments")
        tool_call_id = _safe_get(raw_entry, "id") or _safe_get(function_payload, "id")

        if not name:
            return None

        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments)
            except (TypeError, ValueError):
                arguments = str(arguments)

        entry: Dict[str, Any] = {"name": name, "arguments": arguments}
        if tool_call_id:
            entry["id"] = tool_call_id
        return entry

    tool_call_entries: List[Dict[str, Any]] = []
    tool_calls_payload = message.get("tool_calls")

    def _append_tool_entry(raw_entry):
        normalized = _normalize_tool_call_entry(raw_entry)
        if normalized:
            tool_call_entries.append(normalized)

    if isinstance(tool_calls_payload, list):
        for raw_entry in tool_calls_payload:
            _append_tool_entry(raw_entry)
    elif tool_calls_payload:
        _append_tool_entry(tool_calls_payload)
    else:
        _append_tool_entry(message.get("function_call"))
        _append_tool_entry(message.get("tool_call"))

    if tool_call_entries and not message.get("function_call"):
        message = dict(message)
        message["function_call"] = dict(tool_call_entries[0])

    if not tool_call_entries:
        return None

    conversation_budget_ms = _get_conversation_tool_budget_ms(config_manager)

    executed_calls: List[Dict[str, Any]] = []

    for index, tool_call_entry in enumerate(tool_call_entries):
        function_name = tool_call_entry.get("name")
        tool_call_id = tool_call_entry.get("id")
        logger.info(f"Function call detected: {function_name} (index {index})")
        function_args_json = tool_call_entry.get("arguments", "{}")
        logger.info(f"Function arguments (JSON): {function_args_json}")

        if conversation_budget_ms is not None:
            consumed_ms = _get_conversation_runtime_ms(conversation_id)
            if consumed_ms >= conversation_budget_ms:
                budget_message = (
                    "Tool runtime budget exceeded for conversation "
                    f"'{conversation_id}': {consumed_ms:.0f}ms used of "
                    f"{conversation_budget_ms:.0f}ms allowed."
                )
                _record_persona_metric(
                    function_name,
                    success=False,
                    latency_ms=0.0,
                    timestamp=datetime.utcnow(),
                )
                error_entry = _record_tool_failure(
                    conversation_history,
                    user,
                    conversation_id,
                    tool_call_id=tool_call_id,
                    function_name=function_name,
                    message=budget_message,
                    error_type="tool_runtime_budget_exceeded",
                )
                raise ToolExecutionError(
                    budget_message,
                    tool_call_id=tool_call_id,
                    function_name=function_name,
                    error_type="tool_runtime_budget_exceeded",
                    entry=error_entry,
                )

        try:
            function_args = json.loads(function_args_json)
            logger.info(f"Function arguments (parsed): {function_args}")
        except json.JSONDecodeError as exc:
            logger.error(
                "Error decoding JSON for function arguments: %s", exc, exc_info=True
            )
            error_message = f"Invalid JSON in function arguments: {exc}"
            _record_persona_metric(
                function_name,
                success=False,
                latency_ms=0.0,
                timestamp=datetime.utcnow(),
            )
            error_entry = _record_tool_failure(
                conversation_history,
                user,
                conversation_id,
                tool_call_id=tool_call_id,
                function_name=function_name,
                message=error_message,
                error_type="invalid_arguments",
            )
            raise ToolExecutionError(
                error_message,
                tool_call_id=tool_call_id,
                function_name=function_name,
                error_type="invalid_arguments",
                entry=error_entry,
            ) from exc

        if function_name not in function_map:
            logger.error(f"Function '{function_name}' not found in function map.")
            error_message = f"Function '{function_name}' not found."
            _record_persona_metric(
                function_name,
                success=False,
                latency_ms=0.0,
                timestamp=datetime.utcnow(),
            )
            error_entry = _record_tool_failure(
                conversation_history,
                user,
                conversation_id,
                tool_call_id=tool_call_id,
                function_name=function_name,
                message=error_message,
                error_type="function_not_found",
            )
            raise ToolExecutionError(
                error_message,
                tool_call_id=tool_call_id,
                function_name=function_name,
                error_type="function_not_found",
                entry=error_entry,
            )

        function_entry = function_map[function_name]
        entry_metadata: Dict[str, Any] = {}
        if isinstance(function_entry, dict):
            metadata_candidate = function_entry.get("metadata")
            if isinstance(metadata_candidate, Mapping):
                entry_metadata = dict(metadata_candidate)

        policy_decision = _evaluate_tool_policy(
            function_name=function_name,
            metadata=entry_metadata,
            current_persona=current_persona,
            conversation_manager=conversation_manager,
            conversation_id=conversation_id,
        )

        if not policy_decision.allowed:
            policy_message = policy_decision.reason or (
                f"Tool '{function_name}' is blocked by the current policy."
            )
            _record_persona_metric(
                function_name,
                success=False,
                latency_ms=0.0,
                timestamp=datetime.utcnow(),
            )
            error_entry = _record_tool_failure(
                conversation_history,
                user,
                conversation_id,
                tool_call_id=tool_call_id,
                function_name=function_name,
                message=policy_message,
                error_type="tool_policy_violation",
            )
            raise ToolExecutionError(
                policy_message,
                tool_call_id=tool_call_id,
                function_name=function_name,
                error_type="tool_policy_violation",
                entry=error_entry,
            )

        required_args = get_required_args(function_entry)
        logger.info(f"Required arguments for function '{function_name}': {required_args}")
        provided_args = list(function_args.keys())
        logger.info(f"Provided arguments for function '{function_name}': {provided_args}")
        missing_args = set(required_args) - set(function_args.keys())

        if missing_args:
            logger.error(
                "Missing required arguments for function '%s': %s",
                function_name,
                missing_args,
            )
            error_message = (
                "Missing required arguments for function "
                f"'{function_name}': {', '.join(missing_args)}"
            )
            _record_persona_metric(
                function_name,
                success=False,
                latency_ms=0.0,
                timestamp=datetime.utcnow(),
            )
            error_entry = _record_tool_failure(
                conversation_history,
                user,
                conversation_id,
                tool_call_id=tool_call_id,
                function_name=function_name,
                message=error_message,
                error_type="missing_arguments",
            )
            raise ToolExecutionError(
                error_message,
                tool_call_id=tool_call_id,
                function_name=function_name,
                error_type="missing_arguments",
                entry=error_entry,
            )

        try:
            logger.info(f"Calling function '{function_name}' with arguments: {function_args}")
            func = _resolve_function_callable(function_entry)
            entry_metadata = dict(entry_metadata)

            is_write_tool = entry_metadata.get("side_effects") == "write" if entry_metadata else False
            is_idempotent_tool = _is_tool_idempotent(entry_metadata)

            metadata_timeout = None
            if entry_metadata:
                metadata_timeout = entry_metadata.get("default_timeout")

            if is_write_tool:
                context_payload = function_args.get("context")
                if isinstance(context_payload, Mapping):
                    context_payload = dict(context_payload)
                else:
                    context_payload = {}
                context_payload.setdefault("idempotency_key", _generate_idempotency_key())
                function_args["context"] = context_payload

            timeout_seconds = _resolve_tool_timeout_seconds(
                config_manager, metadata_timeout
            )

            max_attempts = 3 if is_idempotent_tool else 1
            attempt = 0
            last_exception: Optional[Exception] = None
            final_log_entry: Optional[Dict[str, Any]] = None
            final_response: Any = None
            final_completed_at: Optional[datetime] = None

            sandbox_runner = None
            if policy_decision.use_sandbox:
                sandbox_runner = _get_sandbox_runner(config_manager)

            while attempt < max_attempts:
                attempt += 1
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                started_at = datetime.utcnow()
                function_response = None
                call_error: Optional[Exception] = None
                stream_capture: Optional[_ToolStreamCapture] = None
                active_log_entry: Optional[Dict[str, Any]] = None

                base_log_entry = {
                    "tool_name": function_name,
                    "tool_call_id": tool_call_id,
                    "arguments": function_args,
                    "arguments_text": _stringify_tool_value(function_args),
                    "started_at": started_at.isoformat(timespec="milliseconds"),
                    "status": "running",
                    "result_text": "",
                    "stdout": "",
                    "stderr": "",
                }
                if entry_metadata:
                    base_log_entry["metadata"] = entry_metadata

                sandbox_context = (
                    sandbox_runner.activate(metadata=entry_metadata)
                    if sandbox_runner is not None
                    else contextlib.nullcontext()
                )

                try:
                    with sandbox_context:
                        if inspect.isasyncgenfunction(func):

                            async def _execute_async_gen():
                                nonlocal active_log_entry
                                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                                    stderr_buffer
                                ):
                                    async_stream = func(**function_args)
                                    if tool_streaming_enabled:
                                        if active_log_entry is None:
                                            active_log_entry = _record_tool_activity(
                                                {**base_log_entry, "result": []}
                                            )
                                        return await _stream_tool_iterator(
                                            async_stream,
                                            log_entry=base_log_entry,
                                            active_entry=active_log_entry,
                                        )
                                    return await _gather_async_iterator(async_stream)

                            execution_result = await _run_with_timeout(
                                _execute_async_gen(), timeout_seconds
                            )
                        elif asyncio.iscoroutinefunction(func):

                            async def _execute_coroutine():
                                nonlocal active_log_entry
                                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                                    stderr_buffer
                                ):
                                    result = await func(**function_args)
                                    if _is_async_stream(result):
                                        if tool_streaming_enabled:
                                            if active_log_entry is None:
                                                active_log_entry = _record_tool_activity(
                                                    {**base_log_entry, "result": []}
                                                )
                                            return await _stream_tool_iterator(
                                                result,
                                                log_entry=base_log_entry,
                                                active_entry=active_log_entry,
                                            )
                                        return await _gather_async_iterator(result)
                                    return result

                            execution_result = await _run_with_timeout(
                                _execute_coroutine(), timeout_seconds
                            )
                        else:

                            def _run_sync_function():
                                with contextlib.redirect_stdout(
                                    stdout_buffer
                                ), contextlib.redirect_stderr(stderr_buffer):
                                    return func(**function_args)

                            execution_result = await _run_with_timeout(
                                asyncio.to_thread(_run_sync_function), timeout_seconds
                            )

                            if _is_async_stream(execution_result):

                                async def _consume_iterator():
                                    nonlocal active_log_entry
                                    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                                        stderr_buffer
                                    ):
                                        if tool_streaming_enabled:
                                            if active_log_entry is None:
                                                active_log_entry = _record_tool_activity(
                                                    {**base_log_entry, "result": []}
                                                )
                                            return await _stream_tool_iterator(
                                                execution_result,
                                                log_entry=base_log_entry,
                                                active_entry=active_log_entry,
                                            )
                                        return await _gather_async_iterator(execution_result)

                                execution_result = await _run_with_timeout(
                                    _consume_iterator(), timeout_seconds
                                )

                    if isinstance(execution_result, _ToolStreamCapture):
                        stream_capture = execution_result
                        function_response = stream_capture.items
                    else:
                        function_response = execution_result

                except Exception as exc:
                    call_error = exc

                if function_response is None and active_log_entry is not None:
                    candidate_result = active_log_entry.get("result")
                    if candidate_result is not None:
                        function_response = candidate_result

                completed_at = datetime.utcnow()
                final_completed_at = completed_at
                duration_ms = (completed_at - started_at).total_seconds() * 1000
                _increment_conversation_runtime_ms(conversation_id, duration_ms)
                log_entry = {
                    "tool_name": function_name,
                    "tool_call_id": tool_call_id,
                    "arguments": function_args,
                    "arguments_text": _stringify_tool_value(function_args),
                    "started_at": started_at.isoformat(timespec="milliseconds"),
                    "completed_at": completed_at.isoformat(timespec="milliseconds"),
                    "duration_ms": duration_ms,
                    "status": "success" if call_error is None else "error",
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                    "result": function_response,
                    "result_text": _stringify_tool_value(function_response),
                    "error": str(call_error) if call_error else None,
                }
                if timeout_seconds is not None:
                    log_entry["timeout_seconds"] = timeout_seconds
                if call_error is not None:
                    if isinstance(call_error, asyncio.TimeoutError):
                        log_entry["error_type"] = "timeout"
                    else:
                        log_entry["error_type"] = "execution_error"
                if entry_metadata:
                    log_entry["metadata"] = entry_metadata
                if stream_capture is not None:
                    log_entry["result"] = stream_capture.items
                    log_entry["result_text"] = stream_capture.text

                active_entry_ref = (
                    stream_capture.entry if stream_capture is not None else active_log_entry
                )
                if active_entry_ref is not None:
                    _record_tool_activity(log_entry, replace=active_entry_ref)
                else:
                    _record_tool_activity(log_entry)

                if call_error is None:
                    final_log_entry = log_entry
                    final_response = function_response
                    last_exception = None
                    break

                last_exception = call_error
                final_log_entry = log_entry

                if attempt >= max_attempts:
                    break

                await _apply_idempotent_retry_backoff(
                    attempt, idempotent=is_idempotent_tool
                )

            if last_exception is not None:
                if isinstance(last_exception, asyncio.TimeoutError):
                    if timeout_seconds is not None:
                        timeout_display = f"{timeout_seconds:g}" if timeout_seconds is not None else "configured"
                        error_message = (
                            f"Function '{function_name}' exceeded the timeout of {timeout_display} seconds."
                        )
                    else:
                        error_message = (
                            f"Function '{function_name}' exceeded the configured timeout."
                        )
                    error_type = "timeout"
                else:
                    error_message = (
                        f"Exception during function '{function_name}' execution: {last_exception}"
                    )
                    error_type = "execution_error"
                failure_timestamp = final_completed_at or datetime.utcnow()
                _record_persona_metric(
                    function_name,
                    success=False,
                    latency_ms=(final_log_entry or {}).get("duration_ms"),
                    timestamp=failure_timestamp,
                )
                error_entry = _record_tool_failure(
                    conversation_history,
                    user,
                    conversation_id,
                    tool_call_id=tool_call_id,
                    function_name=function_name,
                    message=error_message,
                    error_type=error_type,
                    timestamp=failure_timestamp,
                )
                raise ToolExecutionError(
                    error_message,
                    tool_call_id=tool_call_id,
                    function_name=function_name,
                    error_type=error_type,
                    entry=error_entry,
                ) from last_exception

            log_entry = final_log_entry or {}
            function_response = final_response

            _record_persona_metric(
                function_name,
                success=True,
                latency_ms=log_entry.get("duration_ms"),
                timestamp=final_completed_at or datetime.utcnow(),
            )

            logger.info(
                "Function '%s' executed successfully. Response: %s",
                function_name,
                function_response,
            )

            if function_name == "execute_python":
                command = function_args.get('command')
                event_system.publish("code_executed", command, function_response)
                logger.info("Published 'code_executed' event.")

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conversation_history.add_response(
                user,
                conversation_id,
                function_response,
                current_time,
                tool_call_id=tool_call_id,
            )
            logger.info("Function response added to conversation history.")

            executed_calls.append(
                {
                    "id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args,
                    "result": function_response,
                    "result_text": log_entry.get("result_text"),
                    "metadata": entry_metadata,
                }
            )

        except ToolExecutionError:
            raise
        except Exception as exc:
            logger.error(
                "Exception during function '%s' execution: %s",
                function_name,
                exc,
                exc_info=True,
            )
            error_message = (
                f"Exception during function '{function_name}' execution: {exc}"
            )
            error_entry = _record_tool_failure(
                conversation_history,
                user,
                conversation_id,
                tool_call_id=tool_call_id,
                function_name=function_name,
                message=error_message,
                error_type="execution_error",
            )
            raise ToolExecutionError(
                error_message,
                tool_call_id=tool_call_id,
                function_name=function_name,
                error_type="execution_error",
                entry=error_entry,
            ) from exc

    if not executed_calls:
        return None

    messages = conversation_history.get_history(user, conversation_id)
    logger.info(f"Conversation history after tool execution: {messages}")

    effective_stream = bool(stream) if stream is not None else False

    executed_call_ids = [
        call.get("id") for call in executed_calls if call.get("id")
    ]
    metadata_overrides: Optional[Dict[str, Any]] = None
    if executed_call_ids or current_persona:
        metadata_overrides = {}
        if executed_call_ids:
            metadata_overrides["tool_call_ids"] = list(executed_call_ids)
        if current_persona:
            persona_name = getattr(current_persona, "name", None)
            if persona_name is None:
                persona_name = getattr(current_persona, "persona_name", None)
            if persona_name is None:
                persona_name = str(current_persona)
            metadata_overrides["persona"] = persona_name

    new_text = await call_model_with_new_prompt(
        messages,
        current_persona,
        temperature_var,
        top_p_var,
        frequency_penalty_var,
        presence_penalty_var,
        functions,
        config_manager,
        provider_manager=provider_manager,
        conversation_manager=conversation_manager,
        conversation_id=conversation_id,
        user=user,
        stream=effective_stream,
        generation_settings=generation_context,
    )

    logger.info(f"Model response after function execution: {new_text}")

    if effective_stream and (
        isinstance(new_text, AsyncIterator) or inspect.isasyncgen(new_text)
    ):
        logger.info("Returning streaming response produced after tool execution.")
        return _proxy_streaming_response(
            new_text,
            conversation_history=conversation_history,
            user=user,
            conversation_id=conversation_id,
            metadata_overrides=metadata_overrides,
        )

    if new_text is None:
        logger.warning("Model returned None response. Using default fallback message.")
        new_text = (
            "Tool Manager says: Sorry, I couldn't generate a meaningful response. "
            "Please try again or provide more context."
        )

    if isinstance(new_text, AsyncIterator) or inspect.isasyncgen(new_text):
        logger.info(
            "Received async iterator response but streaming was disabled; collecting chunks."
        )
        new_text = await _collect_async_chunks(new_text)

    if new_text:
        new_text = _store_assistant_message(
            conversation_history,
            user,
            conversation_id,
            new_text,
            metadata_overrides=metadata_overrides,
        )
        logger.info("Assistant's message added to conversation history.")

    return new_text

async def call_model_with_new_prompt(
    messages,
    current_persona,
    temperature_var,
    top_p_var,
    frequency_penalty_var,
    presence_penalty_var,
    functions,
    config_manager=None,
    *,
    provider_manager=None,
    conversation_manager=None,
    conversation_id=None,
    user=None,
    prompt=None,
    stream: bool = False,
    generation_settings: Optional[Mapping[str, Any]] = None,
):
    logger.info("Calling model after tool execution.")
    logger.info(f"Messages provided to model: {messages}")
    if prompt:
        logger.info(f"Additional user prompt supplied: {prompt}")

    provider_manager, config_manager = _resolve_provider_manager(
        provider_manager, config_manager
    )

    try:
        generation_context = _freeze_generation_settings(generation_settings)
    except TypeError as exc:
        logger.warning("Ignoring invalid follow-up generation settings: %s", exc)
        generation_context = MappingProxyType({})

    try:
        messages_payload: List[Dict[str, Any]] = list(messages or [])
        if prompt:
            messages_payload.append({"role": "user", "content": prompt})

        persona_overrides = generation_context.get("persona_overrides")
        provider_override = generation_context.get("provider")
        model_override = generation_context.get("model")
        if isinstance(persona_overrides, Mapping):
            provider_override = provider_override or persona_overrides.get("provider")
            model_override = model_override or persona_overrides.get("model")

        response = await provider_manager.generate_response(
            messages=messages_payload,
            model=model_override or provider_manager.get_current_model(),
            provider=provider_override,
            temperature=temperature_var,
            top_p=top_p_var,
            frequency_penalty=frequency_penalty_var,
            presence_penalty=presence_penalty_var,
            functions=functions,
            conversation_manager=conversation_manager,
            conversation_id=conversation_id,
            user=user,
            stream=stream,
            tool_choice=generation_context.get("tool_choice"),
            parallel_tool_calls=generation_context.get("parallel_tool_calls"),
            json_mode=generation_context.get("json_mode"),
            json_schema=generation_context.get("json_schema"),
            audio_enabled=generation_context.get("audio_enabled"),
            audio_voice=generation_context.get("audio_voice"),
            audio_format=generation_context.get("audio_format"),
        )

        if stream:
            return response

        if isinstance(response, AsyncIterator) or inspect.isasyncgen(response):
            logger.info("Received streaming response while streaming disabled; collecting chunks into text.")
            response = await _collect_async_chunks(response)

        logger.info(f"Model's response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error calling model with new prompt: {e}", exc_info=True)
        return None
