"""Tool execution orchestration helpers."""
from __future__ import annotations

import asyncio
import contextlib
import functools
import io
import inspect
import json
import os
import random
import socket
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from types import MappingProxyType

from modules.Personas.utils import (
    coerce_persona_flag as _coerce_persona_flag_value,
    collect_missing_flag_requirements as _collect_missing_flag_requirements,
    format_denied_operations_summary as _format_denied_operations_summary,
    join_with_and as _join_with_and,
    normalize_persona_allowlist as _normalize_persona_allowlist,
    normalize_requires_flags as _normalize_requires_flags,
    persona_flag_enabled as _persona_flag_enabled,
)
from modules.analytics.persona_metrics import record_persona_tool_event
from modules.logging.logger import setup_logger
from modules.Tools.tool_event_system import event_system, publish_bus_event
from modules.orchestration import budget_tracker
from modules.orchestration.capability_registry import get_capability_registry
from ATLAS.messaging import MessagePriority

from ATLAS.config import ConfigManager
from ATLAS.tools.cache import (
    clone_json_compatible as _clone_json_compatible,
    get_config_manager as _get_config_manager,
    get_config_section as _get_config_section,
    get_tool_activity_log,
    record_tool_activity as _record_tool_activity,
    record_tool_failure as _record_tool_failure,
    stringify_tool_value as _stringify_tool_value,
)
from ATLAS.tools.errors import ToolExecutionError, ToolManifestValidationError
from ATLAS.tools.streaming import (
    ToolStreamCapture as _ToolStreamCapture,
    collect_async_chunks as _collect_async_chunks,
    gather_async_iterator as _gather_async_iterator,
    is_async_stream as _is_async_stream,
    stream_tool_iterator as _stream_tool_iterator,
)

logger = setup_logger(__name__)


_DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0

_SANDBOX_ENV_FLAG = "ATLAS_SANDBOX_ACTIVE"

_SANDBOX_JAIL_ENV = "ATLAS_TERMINAL_JAIL"

@dataclass(frozen=True)
class ToolPolicyDecision:
    """Represents the outcome of a pre-execution policy evaluation."""

    allowed: bool
    reason: Optional[str] = None
    use_sandbox: bool = False
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    denied_operations: Mapping[str, Tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

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
    logger.debug("Retrieving required arguments for the function.")
    sig = inspect.signature(_resolve_function_callable(function))
    return [
        param.name for param in sig.parameters.values()
        if param.default == param.empty and param.name != 'self'
    ]

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

def _format_operation_flag_reason(
    function_name: str,
    operation: str,
    flags: Tuple[str, ...],
) -> str:
    """Return a human-friendly reason for blocking an operation."""

    flag_phrase = _join_with_and([f"'{flag}'" for flag in flags])
    plural = "s" if len(flags) > 1 else ""
    return (
        f"Operation '{operation}' for tool '{function_name}' requires persona flag"
        f"{plural} {flag_phrase} to be enabled."
    )

def _build_persona_context_snapshot(current_persona: Any) -> Optional[Dict[str, Any]]:
    """Extract a lightweight persona snapshot for downstream tools."""

    if not isinstance(current_persona, Mapping):
        return None

    snapshot: Dict[str, Any] = {}
    type_payload = current_persona.get("type")
    if isinstance(type_payload, Mapping):
        try:
            snapshot["type"] = copy.deepcopy(type_payload)
        except Exception:
            snapshot["type"] = dict(type_payload)

    flags_payload = current_persona.get("flags")
    if isinstance(flags_payload, Mapping):
        try:
            snapshot["flags"] = copy.deepcopy(flags_payload)
        except Exception:
            snapshot["flags"] = dict(flags_payload)

    name = current_persona.get("name")
    if isinstance(name, str) and name.strip():
        snapshot["name"] = name.strip()

    return snapshot or None

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
    tool_arguments: Optional[Mapping[str, Any]] = None,
) -> ToolPolicyDecision:
    """Return the pre-execution policy decision for ``function_name``."""

    metadata_dict: Dict[str, Any] = {}
    if isinstance(metadata, Mapping):
        metadata_dict = dict(metadata)
    persona_name = _extract_persona_name(current_persona)
    allowlist = _normalize_persona_allowlist(metadata_dict.get("persona_allowlist"))

    if allowlist and (persona_name is None or persona_name not in allowlist):
        reason = (
            f"Tool '{function_name}' is restricted to approved personas."
        )
        return ToolPolicyDecision(
            allowed=False,
            reason=reason,
            metadata=_freeze_metadata(metadata_dict),
            denied_operations=MappingProxyType({}),
        )

    requires_flags = _normalize_requires_flags(metadata_dict.get("requires_flags"))
    missing_flags = _collect_missing_flag_requirements(
        requires_flags, current_persona
    )
    denied_operations = MappingProxyType({
        op: tuple(flags) for op, flags in missing_flags.items()
    })
    summary_reason = _format_denied_operations_summary(
        function_name, missing_flags
    )

    requested_operation: Optional[str] = None
    if isinstance(tool_arguments, Mapping):
        operation_candidate = tool_arguments.get("operation")
        if isinstance(operation_candidate, str):
            requested_operation = operation_candidate.strip().lower()
        elif operation_candidate is not None:
            requested_operation = str(operation_candidate).strip().lower()

    if requested_operation and requested_operation in missing_flags:
        reason = _format_operation_flag_reason(
            function_name, requested_operation, missing_flags[requested_operation]
        )
        return ToolPolicyDecision(
            allowed=False,
            reason=reason,
            metadata=_freeze_metadata(metadata_dict),
            denied_operations=denied_operations,
        )

    if requested_operation is None:
        for blocking_op in ("read", "execute"):
            if blocking_op in missing_flags:
                reason = _format_operation_flag_reason(
                    function_name, blocking_op, missing_flags[blocking_op]
                )
                return ToolPolicyDecision(
                    allowed=False,
                    reason=reason,
                    metadata=_freeze_metadata(metadata_dict),
                    denied_operations=denied_operations,
                )

    safety_level = str(metadata_dict.get("safety_level") or "standard").lower()
    requires_consent = bool(metadata_dict.get("requires_consent"))
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
                metadata=_freeze_metadata(metadata_dict),
                denied_operations=denied_operations,
            )

    use_sandbox = safety_level in {"high"}

    return ToolPolicyDecision(
        allowed=True,
        reason=summary_reason,
        use_sandbox=use_sandbox,
        metadata=_freeze_metadata(metadata_dict),
        denied_operations=denied_operations,
    )

def _get_sandbox_runner(config_manager=None):
    """Return a sandbox runner bound to ``config_manager``."""

    return SandboxedToolRunner(config_manager)

def compute_tool_policy_snapshot(
    function_map: Mapping[str, Any],
    *,
    current_persona: Any,
    conversation_manager: Any = None,
    conversation_id: Any = None,
) -> Dict[str, ToolPolicyDecision]:
    """Return policy decisions for each tool declared in ``function_map``."""

    snapshot: Dict[str, ToolPolicyDecision] = {}
    if not isinstance(function_map, Mapping):
        return snapshot

    for name, entry in function_map.items():
        entry_metadata: Mapping[str, Any] = {}
        if isinstance(entry, Mapping):
            candidate = entry.get("metadata")
            if isinstance(candidate, Mapping):
                entry_metadata = candidate

        decision = _evaluate_tool_policy(
            function_name=name,
            metadata=entry_metadata,
            current_persona=current_persona,
            conversation_manager=conversation_manager,
            conversation_id=conversation_id,
        )
        snapshot[name] = decision

    return snapshot

class SandboxedToolRunner:
    """Provides a minimal sandbox environment for high-risk tool execution."""

    def __init__(self, config_manager=None):
        self._config_manager = config_manager

    def _resolve_filesystem_root(
        self, metadata: Optional[Mapping[str, Any]]
    ) -> Path:
        config_block = _get_config_section(self._config_manager, "tool_safety")
        root_candidate = None
        if isinstance(metadata, Mapping):
            root_candidate = metadata.get("filesystem_root")
        if root_candidate is None and isinstance(config_block, Mapping):
            root_candidate = config_block.get("filesystem_root")
        if root_candidate is None:
            return Path.cwd()
        try:
            resolved = Path(str(root_candidate)).expanduser().resolve()
        except (OSError, RuntimeError):
            logger.warning("Invalid sandbox filesystem root '%s'; defaulting to current directory.", root_candidate)
            return Path.cwd()
        if not resolved.exists():
            try:
                resolved.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.warning("Unable to create sandbox filesystem root '%s'. Using current directory.", resolved)
                return Path.cwd()
        return resolved

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
        filesystem_root = self._resolve_filesystem_root(metadata)
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
        previous_jail = os.environ.get(_SANDBOX_JAIL_ENV)
        os.environ[_SANDBOX_ENV_FLAG] = "1"
        os.environ[_SANDBOX_JAIL_ENV] = str(filesystem_root)

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
            if previous_jail is None:
                os.environ.pop(_SANDBOX_JAIL_ENV, None)
            else:
                os.environ[_SANDBOX_JAIL_ENV] = previous_jail
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

def _resolve_function_callable(entry: Any) -> Any:
    """Return the executable callable from a function map entry."""

    if isinstance(entry, dict):
        candidate = entry.get("callable")
        if candidate is not None:
            return candidate
    return entry

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
    logger.debug(f"use_tool called for user: {user}, conversation_id: {conversation_id}")
    logger.debug(f"Message received: {message}")

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
        for attr in ("tool_calls", "tool_call"):
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
    capability_registry = get_capability_registry(config_manager=config_manager)

    def _record_persona_metric(
        tool_name: Optional[str],
        *,
        success: bool,
        latency_ms: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        metric_timestamp: Optional[float] = None
        if isinstance(timestamp, datetime):
            metric_timestamp = timestamp.timestamp()
        if tool_name:
            try:
                capability_registry.record_tool_execution(
                    persona=persona_name,
                    tool_name=tool_name,
                    success=success,
                    latency_ms=latency_ms,
                    timestamp=metric_timestamp,
                )
            except Exception:  # pragma: no cover - defensive guard
                logger.debug(
                    "Failed to record capability metrics for tool '%s'", tool_name, exc_info=True
                )

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
        _append_tool_entry(message.get("tool_call"))

    if not tool_call_entries:
        return None

    conversation_budget_ms = budget_tracker.resolve_conversation_budget_ms(
        config_manager
    )

    executed_calls: List[Dict[str, Any]] = []

    for index, tool_call_entry in enumerate(tool_call_entries):
        function_name = tool_call_entry.get("name")
        tool_call_id = tool_call_entry.get("id")
        logger.debug(f"Function call detected: {function_name} (index {index})")
        function_args_json = tool_call_entry.get("arguments", "{}")
        logger.debug(f"Function arguments (JSON): {function_args_json}")

        if conversation_budget_ms is not None:
            consumed_ms = await budget_tracker.get_consumed_runtime_ms(
                conversation_id
            )
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
            logger.debug(f"Function arguments (parsed): {function_args}")
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
            tool_arguments=function_args,
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
        logger.debug(f"Required arguments for function '{function_name}': {required_args}")
        provided_args = list(function_args.keys())
        logger.debug(f"Provided arguments for function '{function_name}': {provided_args}")
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
            logger.debug(f"Calling function '{function_name}' with arguments: {function_args}")
            func = _resolve_function_callable(function_entry)
            entry_metadata = dict(entry_metadata)

            is_write_tool = entry_metadata.get("side_effects") == "write" if entry_metadata else False
            is_idempotent_tool = _is_tool_idempotent(entry_metadata)

            metadata_timeout = None
            if entry_metadata:
                metadata_timeout = entry_metadata.get("default_timeout")

            persona_snapshot = _build_persona_context_snapshot(current_persona)
            context_payload = function_args.get("context")
            if isinstance(context_payload, Mapping):
                context_payload = dict(context_payload)
            else:
                context_payload = {}

            if persona_snapshot:
                context_payload.setdefault("persona", persona_snapshot)

            if is_write_tool:
                context_payload.setdefault("idempotency_key", _generate_idempotency_key())

            if context_payload:
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
                await budget_tracker.reserve_runtime_ms(conversation_id, duration_ms)
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

            logger.debug(
                "Function '%s' executed successfully. Response: %s",
                function_name,
                function_response,
            )

            if function_name == "execute_python":
                command = function_args.get('command')
                publish_bus_event(
                    "code_executed",
                    {"command": command, "result": function_response},
                    priority=MessagePriority.NORMAL,
                    correlation_id=tool_call_id,
                    tracing={
                        "conversation_id": conversation_id,
                        "tool_name": function_name,
                        "persona": persona_name,
                    },
                    metadata={"component": "ToolManager"},
                )
                logger.debug("Published 'code_executed' event.")

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conversation_history.add_response(
                user,
                conversation_id,
                function_response,
                current_time,
                tool_call_id=tool_call_id,
            )
            logger.debug("Function response added to conversation history.")

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
    logger.debug(f"Conversation history after tool execution: {messages}")

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

    logger.debug(f"Model response after function execution: {new_text}")

    if effective_stream and (
        isinstance(new_text, AsyncIterator) or inspect.isasyncgen(new_text)
    ):
        logger.debug("Returning streaming response produced after tool execution.")
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
        logger.debug(
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
        logger.debug("Assistant's message added to conversation history.")

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
    logger.debug("Calling model after tool execution.")
    logger.debug(f"Messages provided to model: {messages}")
    if prompt:
        logger.debug(f"Additional user prompt supplied: {prompt}")

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
            logger.debug("Received streaming response while streaming disabled; collecting chunks into text.")
            response = await _collect_async_chunks(response)

        logger.debug(f"Model's response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error calling model with new prompt: {e}", exc_info=True)
        return None
