# ATLAS/Tools/ToolManager.py

import asyncio
import contextlib
import copy
import io
import json
import inspect
import importlib.util
import sys
import os
import threading
from collections import deque
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from types import MappingProxyType

from jsonschema import Draft7Validator, ValidationError

from modules.logging.logger import setup_logger
from modules.Tools.tool_event_system import event_system

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

_function_map_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_function_payload_cache: Dict[str, Any] = {}
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
)

_TOOL_ACTIVITY_EVENT = "tool_activity"
_tool_activity_log: deque = deque(maxlen=100)
_tool_activity_lock = threading.Lock()


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
        annotated[name] = {"callable": func, "metadata": metadata}

    return annotated


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


def _record_tool_activity(entry: Dict[str, Any]) -> None:
    """Append a tool activity entry to the ring buffer and publish an event."""

    # Ensure we only store JSON-friendly copies to avoid mutating UI consumers.
    stored_entry = {
        **entry,
        "arguments": _clone_json_compatible(entry.get("arguments")),
        "result": _clone_json_compatible(entry.get("result")),
    }

    with _tool_activity_lock:
        _tool_activity_log.append(stored_entry)

    event_system.publish(_TOOL_ACTIVITY_EVENT, dict(stored_entry))


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

    return [dict(entry) for entry in entries]

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
            cached_mtime, cached_map = cache_entry
            if cached_mtime == file_mtime:
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
                    cached_map, metadata_lookup=metadata_lookup
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
            _function_map_cache[persona_name] = (file_mtime, module.function_map)
            metadata_lookup = _build_metadata_lookup(
                load_functions_from_json(
                    current_persona,
                    refresh=refresh,
                    config_manager=config_manager,
                )
            )
            return _annotate_function_map(
                module.function_map, metadata_lookup=metadata_lookup
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
    return load_default_function_map(refresh=refresh, config_manager=config_manager)


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
        file_mtime = os.path.getmtime(functions_json_path)

        with _function_payload_cache_lock:
            if not refresh:
                cache_entry = _function_payload_cache.get(persona_name)
                if cache_entry:
                    cached_mtime, cached_functions = cache_entry
                    if cached_mtime == file_mtime:
                        logger.info(
                            "Returning cached functions for persona '%s' (mtime %s).",
                            persona_name,
                            cached_mtime,
                        )
                        return cached_functions

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
                    _function_payload_cache.pop(persona_name, None)
                    raise ToolManifestValidationError(
                        f"Invalid tool manifest for persona '{persona_name}': {exc.message}",
                        persona=persona_name,
                        errors=error_details,
                    ) from exc

            _function_payload_cache[persona_name] = (file_mtime, functions)

        return functions
    except FileNotFoundError:
        logger.error(f"functions.json file not found for persona '{persona_name}' at path: {functions_json_path}")
        with _function_payload_cache_lock:
            _function_payload_cache.pop(persona_name, None)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error in functions.json for persona '{persona_name}': {e}", exc_info=True)
        with _function_payload_cache_lock:
            _function_payload_cache.pop(persona_name, None)
    except ToolManifestValidationError:
        raise
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

    executed_calls: List[Dict[str, Any]] = []

    for index, tool_call_entry in enumerate(tool_call_entries):
        function_name = tool_call_entry.get("name")
        tool_call_id = tool_call_entry.get("id")
        logger.info(f"Function call detected: {function_name} (index {index})")
        function_args_json = tool_call_entry.get("arguments", "{}")
        logger.info(f"Function arguments (JSON): {function_args_json}")

        try:
            function_args = json.loads(function_args_json)
            logger.info(f"Function arguments (parsed): {function_args}")
        except json.JSONDecodeError as exc:
            logger.error(
                "Error decoding JSON for function arguments: %s", exc, exc_info=True
            )
            error_message = f"Invalid JSON in function arguments: {exc}"
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
            entry_metadata: Dict[str, Any] = {}
            if isinstance(function_entry, dict):
                metadata_candidate = function_entry.get("metadata")
                if isinstance(metadata_candidate, Mapping):
                    entry_metadata = dict(metadata_candidate)
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            started_at = datetime.utcnow()
            function_response = None
            call_error: Optional[Exception] = None

            try:
                if inspect.isasyncgenfunction(func):
                    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                        stderr_buffer
                    ):
                        async_stream = func(**function_args)
                        function_response = await _gather_async_iterator(async_stream)
                elif asyncio.iscoroutinefunction(func):
                    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                        stderr_buffer
                    ):
                        result = await func(**function_args)
                        if inspect.isasyncgen(result) or isinstance(result, AsyncIterator):
                            function_response = await _gather_async_iterator(result)
                        else:
                            function_response = result
                else:
                    def _run_sync_function():
                        with contextlib.redirect_stdout(
                            stdout_buffer
                        ), contextlib.redirect_stderr(stderr_buffer):
                            return func(**function_args)

                    function_response = await asyncio.to_thread(_run_sync_function)

                    if inspect.isasyncgen(function_response) or isinstance(
                        function_response, AsyncIterator
                    ):
                        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                            stderr_buffer
                        ):
                            function_response = await _gather_async_iterator(function_response)
            except Exception as exc:
                call_error = exc

            completed_at = datetime.utcnow()
            duration_ms = (completed_at - started_at).total_seconds() * 1000
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
            if entry_metadata:
                log_entry["metadata"] = entry_metadata
            _record_tool_activity(log_entry)

            if call_error is not None:
                error_message = (
                    f"Exception during function '{function_name}' execution: {call_error}"
                )
                error_entry = _record_tool_failure(
                    conversation_history,
                    user,
                    conversation_id,
                    tool_call_id=tool_call_id,
                    function_name=function_name,
                    message=error_message,
                    error_type="execution_error",
                    timestamp=completed_at,
                )
                raise ToolExecutionError(
                    error_message,
                    tool_call_id=tool_call_id,
                    function_name=function_name,
                    error_type="execution_error",
                    entry=error_entry,
                ) from call_error

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
                    "result_text": log_entry["result_text"],
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
