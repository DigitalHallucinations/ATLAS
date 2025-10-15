# ATLAS/Tools/ToolManager.py

import asyncio
import contextlib
import io
import json
import inspect
import importlib.util
import sys
import os
import threading
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from modules.logging.logger import setup_logger
from modules.Tools.tool_event_system import event_system

from ATLAS.config import ConfigManager
logger = setup_logger(__name__)

_function_map_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_function_payload_cache: Dict[str, Any] = {}
_default_config_manager: Optional[ConfigManager] = None

_TOOL_ACTIVITY_EVENT = "tool_activity"
_tool_activity_log: deque = deque(maxlen=100)
_tool_activity_lock = threading.Lock()


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


def _extract_text_and_audio(payload):
    """Return textual content and optional audio payload from ``payload``."""

    if payload is None:
        return None, None

    audio = None
    text = None

    if isinstance(payload, dict):
        audio = payload.get("audio")
        for key in ("content", "text", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                text = value
                break

    if text is None:
        if isinstance(payload, str):
            text = payload
        else:
            text = str(payload)

    return text, audio


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
    sig = inspect.signature(function)
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


def get_tool_activity_log(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return a copy of the recorded tool activity log."""

    with _tool_activity_lock:
        entries = list(_tool_activity_log)

    if limit is not None and isinstance(limit, int) and limit >= 0:
        entries = entries[-limit:]

    return [dict(entry) for entry in entries]

def load_function_map_from_current_persona(
    current_persona,
    *,
    refresh=False,
    config_manager=None,
):
    logger.info("Attempting to load function map from current persona.")
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
                return cached_map

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
            return module.function_map
        else:
            logger.warning(
                "No 'function_map' found in maps.py for persona '%s'.",
                persona_name,
            )
            _function_map_cache.pop(persona_name, None)
            return None
    except FileNotFoundError:
        logger.error(f"maps.py file not found for persona '{persona_name}' at path: {maps_path}")
    except Exception as e:
        logger.error(f"Error loading function map for persona '{persona_name}': {e}", exc_info=True)
    return None

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
        cache_entry = _function_payload_cache.get(persona_name)
        if not refresh and cache_entry:
            cached_mtime, cached_functions = cache_entry
            if cached_mtime == file_mtime:
                logger.info(
                    "Returning cached functions for persona '%s' (mtime %s).",
                    persona_name,
                    cached_mtime,
                )
                return cached_functions

        with open(functions_json_path, 'r') as file:
            functions = json.load(file)
            logger.info(f"Functions successfully loaded from JSON for persona '{persona_name}': {functions}")
            _function_payload_cache[persona_name] = (file_mtime, functions)
            return functions
    except FileNotFoundError:
        logger.error(f"functions.json file not found for persona '{persona_name}' at path: {functions_json_path}")
        _function_payload_cache.pop(persona_name, None)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error in functions.json for persona '{persona_name}': {e}", exc_info=True)
        _function_payload_cache.pop(persona_name, None)
    except Exception as e:
        logger.error(f"Unexpected error loading functions for persona '{persona_name}': {e}", exc_info=True)
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
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for function arguments: {e}", exc_info=True)
            return f"Error: Invalid JSON in function arguments: {e}", True

        if function_name not in function_map:
            logger.error(f"Function '{function_name}' not found in function map.")
            return f"Error: Function '{function_name}' not found.", True

        required_args = get_required_args(function_map[function_name])
        logger.info(f"Required arguments for function '{function_name}': {required_args}")
        provided_args = list(function_args.keys())
        logger.info(f"Provided arguments for function '{function_name}': {provided_args}")
        missing_args = set(required_args) - set(function_args.keys())

        if missing_args:
            logger.error(f"Missing required arguments for function '{function_name}': {missing_args}")
            return (
                f"Error: Missing required arguments for function '{function_name}': {', '.join(missing_args)}",
                True,
            )

        try:
            logger.info(f"Calling function '{function_name}' with arguments: {function_args}")
            func = function_map[function_name]
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            started_at = datetime.utcnow()
            function_response = None
            call_error: Optional[Exception] = None

            try:
                if asyncio.iscoroutinefunction(func):
                    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                        stderr_buffer
                    ):
                        function_response = await func(**function_args)
                else:
                    def _run_sync_function():
                        with contextlib.redirect_stdout(
                            stdout_buffer
                        ), contextlib.redirect_stderr(stderr_buffer):
                            return func(**function_args)

                    function_response = await asyncio.to_thread(_run_sync_function)
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
            _record_tool_activity(log_entry)

            if call_error is not None:
                raise call_error

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
                }
            )

        except Exception as e:
            logger.error(f"Exception during function '{function_name}' execution: {e}", exc_info=True)
            return f"Error: Exception during function '{function_name}' execution: {e}", True

    if not executed_calls:
        return None

    messages = conversation_history.get_history(user, conversation_id)
    logger.info(f"Conversation history after tool execution: {messages}")

    effective_stream = bool(stream) if stream is not None else False

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
    )

    logger.info(f"Model response after function execution: {new_text}")

    if effective_stream and (
        isinstance(new_text, AsyncIterator) or inspect.isasyncgen(new_text)
    ):
        logger.info("Returning streaming response produced after tool execution.")
        return new_text

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
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text_payload, audio_payload = _extract_text_and_audio(new_text)
        entry_kwargs: Dict[str, Any] = {}
        metadata_payload: Optional[Dict[str, Any]] = None
        role = "assistant"

        if isinstance(new_text, dict):
            metadata_candidate = new_text.get("metadata")
            if isinstance(metadata_candidate, dict):
                metadata_payload = dict(metadata_candidate)
            role = str(new_text.get("role", role))
            for key, value in new_text.items():
                if key in {"content", "text", "message", "audio", "role", "metadata"}:
                    continue
                entry_kwargs[key] = value

        if audio_payload is not None:
            entry_kwargs["audio"] = audio_payload

        conversation_history.add_message(
            user,
            conversation_id,
            role,
            text_payload,
            current_time,
            metadata=metadata_payload,
            **entry_kwargs,
        )
        new_text = text_payload
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
):
    logger.info("Calling model after tool execution.")
    logger.info(f"Messages provided to model: {messages}")
    if prompt:
        logger.info(f"Additional user prompt supplied: {prompt}")

    provider_manager, config_manager = _resolve_provider_manager(
        provider_manager, config_manager
    )

    try:
        messages_payload: List[Dict[str, Any]] = list(messages or [])
        if prompt:
            messages_payload.append({"role": "user", "content": prompt})

        response = await provider_manager.generate_response(
            messages=messages_payload,
            model=provider_manager.get_current_model(),
            temperature=temperature_var,
            top_p=top_p_var,
            frequency_penalty=frequency_penalty_var,
            presence_penalty=presence_penalty_var,
            functions=functions,
            conversation_manager=conversation_manager,
            conversation_id=conversation_id,
            user=user,
            stream=stream,
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
