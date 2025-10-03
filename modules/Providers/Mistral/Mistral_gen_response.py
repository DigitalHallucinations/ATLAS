# modules/Providers/Mistral/Mistral_gen_response.py

import asyncio
import json
import threading
from collections.abc import Iterable, Mapping
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from mistralai import Mistral
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from ATLAS.config import ConfigManager
from ATLAS.model_manager import ModelManager
from ATLAS.ToolManager import (
    load_function_map_from_current_persona,
    load_functions_from_json,
    use_tool,
)
from modules.logging.logger import setup_logger

class MistralGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.api_key = self.config_manager.get_mistral_api_key()
        if not self.api_key:
            self.logger.error("Mistral API key not found in configuration")
            raise ValueError("Mistral API key not found in configuration")
        self._base_url: Optional[str] = None
        self.client = self._instantiate_client(None)
        self.model_manager = ModelManager(config_manager)
        settings = self._get_settings_snapshot()
        default_model = settings.get("model")
        if isinstance(default_model, str) and default_model.strip():
            self.model_manager.set_model(default_model.strip(), "Mistral")

    def _instantiate_client(self, base_url: Optional[str]) -> Mistral:
        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if base_url:
            client_kwargs["server_url"] = base_url
        return Mistral(**client_kwargs)

    def _refresh_client(self, base_url: Optional[str]) -> None:
        if base_url == self._base_url and getattr(self, "client", None) is not None:
            return

        previous_client = getattr(self, "client", None)
        self._base_url = base_url
        self.client = self._instantiate_client(base_url)

        closer = getattr(previous_client, "close", None)
        if callable(closer):  # pragma: no cover - cleanup best effort
            try:
                closer()
            except Exception:
                pass

    def _get_settings_snapshot(self) -> Dict[str, Any]:
        settings = self.config_manager.get_mistral_llm_settings()
        base_url = settings.get("base_url") if isinstance(settings, Mapping) else None
        if isinstance(base_url, str):
            base_url = base_url.strip() or None
        self._refresh_client(base_url)
        return settings

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        safe_prompt: Optional[bool] = None,
        random_seed: Optional[int] = None,
        stream: Optional[bool] = None,
        current_persona=None,
        functions=None,
        conversation_manager=None,
        user=None,
        conversation_id=None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
        stop_sequences: Optional[Any] = None,
    ) -> Union[str, AsyncIterator[str]]:
        settings = self._get_settings_snapshot()

        def _coerce_positive_int(value: Optional[Any], default: int) -> int:
            try:
                number = int(value) if value is not None else default
            except (TypeError, ValueError):
                return default
            if number <= 0:
                return default
            return number

        max_attempts = _coerce_positive_int(settings.get('max_retries'), 3)
        min_wait_seconds = _coerce_positive_int(settings.get('retry_min_seconds'), 4)
        max_wait_seconds = _coerce_positive_int(settings.get('retry_max_seconds'), max(min_wait_seconds, 10))
        if max_wait_seconds < min_wait_seconds:
            max_wait_seconds = min_wait_seconds

        retry_loop = AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait_seconds, max=max_wait_seconds),
            reraise=True,
        )

        async for attempt in retry_loop:
            with attempt:
                try:
                    mistral_messages = self.convert_messages_to_mistral_format(messages)
                    settings = self.config_manager.get_mistral_llm_settings()

                    def _resolve_model(preferred: Optional[str]) -> str:
                        current = self.model_manager.get_current_model()
                        if preferred and preferred.strip():
                            return preferred.strip()
                        stored = settings.get("model")
                        if isinstance(stored, str) and stored.strip():
                            return stored.strip()
                        if current and isinstance(current, str) and current.strip():
                            return current.strip()
                        return "mistral-large-latest"

                    effective_model = _resolve_model(model)
                    current_model = self.model_manager.get_current_model()
                    if effective_model != current_model:
                        self.model_manager.set_model(effective_model, "Mistral")

                    def _resolve_float(
                        candidate: Optional[Any],
                        stored_key: str,
                        default: float,
                        *,
                        minimum: float,
                        maximum: float,
                    ) -> float:
                        value = candidate
                        if value is None:
                            value = settings.get(stored_key, default)
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            return default
                        if numeric < minimum or numeric > maximum:
                            return default
                        return numeric

                    def _resolve_int(
                        candidate: Optional[Any],
                        stored_key: str,
                        default: Optional[int],
                    ) -> Optional[int]:
                        value = candidate
                        if value is None:
                            value = settings.get(stored_key, default)
                        if value in {None, ""}:
                            return None if default is None else default
                        try:
                            numeric = int(value)
                        except (TypeError, ValueError):
                            return default
                        if numeric <= 0:
                            return None if default is None else default
                        return numeric

                    def _resolve_bool(candidate: Optional[Any], stored_key: str, default: bool) -> bool:
                        value = candidate
                        if value is None:
                            value = settings.get(stored_key, default)
                        if isinstance(value, bool):
                            return value
                        if isinstance(value, str):
                            lowered = value.strip().lower()
                            if lowered in {"1", "true", "yes", "on"}:
                                return True
                            if lowered in {"0", "false", "no", "off"}:
                                return False
                            return default
                        return bool(value)

                    def _resolve_optional_int(candidate: Optional[Any], stored_key: str) -> Optional[int]:
                        value = candidate
                        if value is None:
                            value = settings.get(stored_key)
                        if value in {None, ""}:
                            return None
                        try:
                            numeric = int(value)
                        except (TypeError, ValueError):
                            return None
                        return numeric

                    effective_temperature = _resolve_float(
                        temperature,
                        'temperature',
                        0.0,
                        minimum=0.0,
                        maximum=2.0,
                    )
                    effective_top_p = _resolve_float(
                        top_p,
                        'top_p',
                        1.0,
                        minimum=0.0,
                        maximum=1.0,
                    )
                    effective_max_tokens = _resolve_int(
                        max_tokens,
                        'max_tokens',
                        None,
                    )
                    effective_frequency_penalty = _resolve_float(
                        frequency_penalty,
                        'frequency_penalty',
                        0.0,
                        minimum=-2.0,
                        maximum=2.0,
                    )
                    effective_presence_penalty = _resolve_float(
                        presence_penalty,
                        'presence_penalty',
                        0.0,
                        minimum=-2.0,
                        maximum=2.0,
                    )
                    effective_safe_prompt = _resolve_bool(
                        safe_prompt,
                        'safe_prompt',
                        False,
                    )
                    effective_stream = _resolve_bool(
                        stream,
                        'stream',
                        True,
                    )
                    effective_parallel = _resolve_bool(
                        parallel_tool_calls,
                        'parallel_tool_calls',
                        True,
                    )
                    effective_random_seed = _resolve_optional_int(random_seed, 'random_seed')

                    configured_tool_choice = (
                        tool_choice if tool_choice is not None else settings.get('tool_choice')
                    )
                    if isinstance(configured_tool_choice, str):
                        configured_tool_choice = configured_tool_choice.strip() or None
                    elif isinstance(configured_tool_choice, dict):
                        configured_tool_choice = dict(configured_tool_choice)
                    else:
                        configured_tool_choice = None if configured_tool_choice is None else configured_tool_choice

                    if stop_sequences is None:
                        candidate_stop_sequences = settings.get('stop_sequences')
                    else:
                        candidate_stop_sequences = stop_sequences
                    try:
                        effective_stop_sequences = self.config_manager._coerce_stop_sequences(
                            candidate_stop_sequences
                        )
                    except ValueError:
                        effective_stop_sequences = []

                    self.logger.info(
                        "Generating response with Mistral AI using model: %s",
                        effective_model,
                    )

                    provided_functions = functions
                    if provided_functions is None and current_persona is not None:
                        try:
                            provided_functions = load_functions_from_json(current_persona)
                        except Exception as exc:  # pragma: no cover - defensive logging
                            self.logger.warning(
                                "Failed to load functions from persona for Mistral: %s",
                                exc,
                            )

                    tools_payload = self._convert_functions_to_tools(provided_functions)

                    function_map = None
                    if current_persona is not None:
                        try:
                            function_map = load_function_map_from_current_persona(current_persona)
                        except Exception as exc:  # pragma: no cover - defensive logging
                            self.logger.warning(
                                "Failed to load function map for Mistral persona: %s",
                                exc,
                            )

                    response_format_payload: Optional[Dict[str, Any]] = None
                    try:
                        normalized_schema = self._prepare_json_schema(
                            settings.get('json_schema')
                        )
                    except ValueError as exc:
                        self.logger.warning(
                            "Ignoring invalid JSON schema for Mistral: %s",
                            exc,
                        )
                        normalized_schema = None

                    if normalized_schema:
                        response_format_payload = {
                            'type': 'json_schema',
                            'json_schema': normalized_schema,
                        }
                    elif settings.get('json_mode'):
                        response_format_payload = {'type': 'json_object'}

                    request_kwargs: Dict[str, Any] = {
                        'model': effective_model,
                        'messages': mistral_messages,
                        'temperature': effective_temperature,
                        'top_p': effective_top_p,
                        'safe_prompt': effective_safe_prompt,
                        'frequency_penalty': effective_frequency_penalty,
                        'presence_penalty': effective_presence_penalty,
                    }

                    if effective_max_tokens is not None:
                        request_kwargs['max_tokens'] = effective_max_tokens

                    if effective_random_seed is not None:
                        request_kwargs['random_seed'] = effective_random_seed

                    if effective_stop_sequences:
                        request_kwargs['stop'] = effective_stop_sequences

                    if tools_payload:
                        request_kwargs['tools'] = tools_payload
                        if configured_tool_choice is not None:
                            request_kwargs['tool_choice'] = configured_tool_choice
                        request_kwargs['parallel_tool_calls'] = effective_parallel
                    else:
                        if configured_tool_choice is not None:
                            request_kwargs['tool_choice'] = configured_tool_choice

                    if response_format_payload:
                        request_kwargs['response_format'] = response_format_payload

                    if effective_stream:
                        response_stream = await asyncio.to_thread(
                            self.client.chat.stream,
                            **request_kwargs,
                        )
                        return self.process_streaming_response(
                            response_stream,
                            user=user,
                            conversation_id=conversation_id,
                            conversation_manager=conversation_manager,
                            function_map=function_map,
                            functions=provided_functions,
                            current_persona=current_persona,
                            temperature=effective_temperature,
                            top_p=effective_top_p,
                            frequency_penalty=effective_frequency_penalty,
                            presence_penalty=effective_presence_penalty,
                        )

                    response = await asyncio.to_thread(
                        self.client.chat.complete,
                        **request_kwargs,
                    )
                    message = self._safe_get(response.choices[0], "message")
                    tool_messages = self._extract_tool_messages(message)
                    if tool_messages:
                        tool_result = await self._handle_tool_messages(
                            tool_messages,
                            user=user,
                            conversation_id=conversation_id,
                            conversation_manager=conversation_manager,
                            function_map=function_map,
                            functions=provided_functions,
                            current_persona=current_persona,
                            temperature=effective_temperature,
                            top_p=effective_top_p,
                            frequency_penalty=effective_frequency_penalty,
                            presence_penalty=effective_presence_penalty,
                        )
                        if tool_result is not None:
                            return tool_result
                    return self._safe_get(message, "content")

                except Mistral.APIError as exc:
                    self.logger.error(f"Mistral API error: {str(exc)}")
                    raise
                except Exception as exc:
                    self.logger.error(f"Unexpected error with Mistral: {str(exc)}")
                    raise

    def convert_messages_to_mistral_format(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        mistral_messages: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue

            role = message.get("role")
            if not isinstance(role, str) or not role:
                continue

            mistral_message: Dict[str, Any] = {"role": role}

            content = message.get("content")
            if content is None:
                mistral_message["content"] = ""
            else:
                mistral_message["content"] = self._clone_message_value(content)

            for optional_key in (
                "name",
                "tool_call_id",
                "id",
                "function_call",
                "tool_calls",
                "metadata",
                "audio",
                "modalities",
            ):
                if optional_key in message:
                    mistral_message[optional_key] = self._clone_message_value(
                        message[optional_key]
                    )

            for key, value in message.items():
                if key in {"role", *mistral_message.keys()}:
                    continue
                mistral_message[key] = self._clone_message_value(value)

            mistral_messages.append(mistral_message)

        return mistral_messages

    def _clone_message_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: self._clone_message_value(val) for key, val in value.items()}

        if isinstance(value, list):
            return [self._clone_message_value(item) for item in value]

        if isinstance(value, tuple):
            return [self._clone_message_value(item) for item in value]

        if isinstance(value, Iterable) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [self._clone_message_value(item) for item in value]

        return value

    def _prepare_json_schema(self, schema: Any) -> Optional[Dict[str, Any]]:
        if schema is None:
            return None

        if isinstance(schema, (bytes, bytearray)):
            schema = schema.decode("utf-8")

        if isinstance(schema, str):
            text = schema.strip()
            if not text:
                return None
            try:
                schema = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON schema: {exc.msg}") from exc

        if not isinstance(schema, dict):
            raise ValueError("JSON schema must be provided as an object.")

        if not schema:
            return None

        payload = dict(schema)
        schema_payload = payload.get('schema') if isinstance(payload, dict) else None
        schema_name = payload.get('name') if isinstance(payload, dict) else None

        if schema_payload is None:
            schema_payload = payload
            schema_like_keys = {
                '$schema',
                '$ref',
                'type',
                'properties',
                'items',
                'oneOf',
                'anyOf',
                'allOf',
                'definitions',
                'patternProperties',
            }
            if isinstance(schema_payload, dict) and not (
                schema_like_keys & set(schema_payload.keys())
            ):
                raise ValueError(
                    "JSON schema must include a 'schema' object or a valid schema definition."
                )

        if not isinstance(schema_payload, dict):
            raise ValueError("JSON schema payload must include a 'schema' object.")

        if schema_name is None:
            schema_name = payload.get('name') if isinstance(payload, dict) else None
        if not schema_name:
            schema_name = 'atlas_response'

        normalized: Dict[str, Any] = {
            'name': str(schema_name).strip() or 'atlas_response',
            'schema': schema_payload,
        }

        if 'strict' in payload:
            normalized['strict'] = bool(payload['strict'])

        try:
            return json.loads(json.dumps(normalized))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"JSON schema payload could not be serialized: {exc}"
            ) from exc

    def _convert_functions_to_tools(self, functions: Optional[Any]) -> Optional[List[Dict[str, Any]]]:
        if not functions:
            return None

        tools: List[Dict[str, Any]] = []

        def _normalize_function(entry: Any) -> Optional[Dict[str, Any]]:
            if entry is None:
                return None
            payload = entry
            if isinstance(entry, Mapping) and 'function' in entry:
                payload = entry.get('function')
            if not isinstance(payload, Mapping):
                return None
            name = payload.get('name')
            if not isinstance(name, str) or not name.strip():
                return None
            function_spec: Dict[str, Any] = {'name': name.strip()}
            description = payload.get('description')
            if isinstance(description, str) and description.strip():
                function_spec['description'] = description.strip()
            if 'parameters' in payload:
                function_spec['parameters'] = payload['parameters']
            return {'type': 'function', 'function': function_spec}

        candidates: Iterable[Any]
        if isinstance(functions, Mapping):
            candidates = [functions]
        elif isinstance(functions, Iterable) and not isinstance(functions, (str, bytes, bytearray)):
            candidates = list(functions)
        else:
            candidates = []

        for item in candidates:
            tool = _normalize_function(item)
            if tool:
                tools.append(tool)

        return tools or None

    async def process_streaming_response(
        self,
        response,
        *,
        user=None,
        conversation_id=None,
        conversation_manager=None,
        function_map=None,
        functions=None,
        current_persona=None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
        DONE = object()

        def iterate_stream():
            pending_tool_calls: Dict[int, Dict[str, Any]] = {}
            try:
                for chunk in response:
                    data = getattr(chunk, "data", None)
                    if not data or not getattr(data, "choices", None):
                        continue
                    choice = data.choices[0]
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("chunk", content)
                        )
                    if delta:
                        tool_calls_delta = getattr(delta, "tool_calls", None)
                        if tool_calls_delta:
                            for tool_call_delta in tool_calls_delta:
                                self._update_pending_tool_calls(
                                    pending_tool_calls, tool_call_delta
                                )
                        single_tool_call = getattr(delta, "tool_call", None)
                        if single_tool_call:
                            self._update_pending_tool_calls(
                                pending_tool_calls, single_tool_call
                            )
                        legacy_call = getattr(delta, "function_call", None)
                        if legacy_call:
                            self._update_pending_tool_calls(
                                pending_tool_calls,
                                {"function": legacy_call, "type": "function"},
                            )
                    if getattr(choice, "finish_reason", None):
                        if pending_tool_calls:
                            finalized = self._finalize_pending_tool_calls(
                                pending_tool_calls
                            )
                            if finalized:
                                loop.call_soon_threadsafe(
                                    queue.put_nowait, ("tools", finalized)
                                )
                        break
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # pragma: no cover - defensive
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", DONE))

        threading.Thread(target=iterate_stream, daemon=True).start()

        while True:
            kind, payload = await queue.get()
            if kind == "chunk":
                yield payload
            elif kind == "tools":
                tool_result = await self._handle_tool_messages(
                    payload,
                    user=user,
                    conversation_id=conversation_id,
                    conversation_manager=conversation_manager,
                    function_map=function_map,
                    functions=functions,
                    current_persona=current_persona,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                if tool_result is not None:
                    yield tool_result
            elif kind == "error":
                raise payload
            elif kind == "done":
                break

    def _safe_get(self, target: Any, attribute: str, default=None):
        if isinstance(target, dict):
            return target.get(attribute, default)
        return getattr(target, attribute, default)

    def _stringify_function_arguments(self, arguments) -> str:
        if arguments is None:
            return "{}"
        if isinstance(arguments, str):
            return arguments
        try:
            return json.dumps(arguments)
        except (TypeError, ValueError):
            return str(arguments)

    def _normalize_tool_call(self, call: Any) -> Optional[Dict[str, Any]]:
        if not call:
            return None

        payload: Dict[str, Any] = {}
        if isinstance(call, dict):
            payload = dict(call)
        else:
            for key in ("type", "id", "function", "name", "arguments", "function_call"):
                value = getattr(call, key, None)
                if value is not None:
                    payload[key] = value

        raw_call = call if isinstance(call, dict) else payload or call
        call_type = payload.get("type")
        function_payload = payload.get("function")
        if function_payload is None and isinstance(payload.get("function_call"), dict):
            function_payload = payload.get("function_call")
            call_type = call_type or "function"

        if function_payload is None and (
            payload.get("name") or payload.get("arguments")
        ):
            function_payload = {
                "name": payload.get("name"),
                "arguments": payload.get("arguments"),
            }

        if call_type is None and function_payload is not None:
            call_type = "function"

        if call_type != "function":
            return None

        function_payload = function_payload or {}
        name = self._safe_get(function_payload, "name") or payload.get("name")
        arguments = self._safe_get(function_payload, "arguments")
        if arguments is None:
            arguments = payload.get("arguments")

        if not name:
            return None

        normalized: Dict[str, Any] = {
            "type": "function",
            "function_call": {
                "name": name,
                "arguments": self._stringify_function_arguments(arguments),
            },
            "raw_call": raw_call,
        }

        for optional_key in ("id", "tool_call_id"):
            value = payload.get(optional_key)
            if value:
                normalized[optional_key] = value

        return normalized

    def _extract_tool_messages(self, message: Any) -> List[Dict[str, Any]]:
        if not message:
            return []

        collected: List[Any] = []
        tool_calls = self._safe_get(message, "tool_calls")
        if isinstance(tool_calls, list):
            collected.extend(tool_calls)
        elif tool_calls:
            collected.append(tool_calls)

        legacy_call = self._safe_get(message, "function_call")
        if legacy_call:
            collected.append({"type": "function", "function": legacy_call})

        return self._prepare_tool_messages(collected)

    def _prepare_tool_messages(
        self, tool_messages: Iterable[Any]
    ) -> List[Dict[str, Any]]:
        normalized_messages: List[Dict[str, Any]] = []
        if not tool_messages:
            return normalized_messages

        for entry in tool_messages:
            normalized_entry: Optional[Dict[str, Any]] = None
            if isinstance(entry, Mapping) and entry.get("type") == "function":
                function_call = entry.get("function_call") or entry.get("function")
                if isinstance(function_call, Mapping):
                    cloned_entry: Dict[str, Any] = {}
                    for key, value in entry.items():
                        if key == "raw_call" and value is entry:
                            cloned_entry[key] = value
                        else:
                            cloned_entry[key] = self._clone_message_value(value)
                    normalized_entry = cloned_entry
                    normalized_entry.setdefault("type", "function")
                    function_payload = dict(function_call)
                    normalized_entry["function_call"] = {
                        "name": function_payload.get("name"),
                        "arguments": self._stringify_function_arguments(
                            function_payload.get("arguments")
                        ),
                    }
                    normalized_entry.setdefault("raw_call", entry)
                else:
                    normalized_entry = self._normalize_tool_call(entry)
            else:
                normalized_entry = self._normalize_tool_call(entry)

            if normalized_entry:
                normalized_messages.append(normalized_entry)

        return normalized_messages

    def _update_pending_tool_calls(
        self, pending: Dict[int, Dict[str, Any]], tool_call_delta: Any
    ) -> None:
        if not tool_call_delta:
            return

        try:
            index = int(self._safe_get(tool_call_delta, "index"))
        except (TypeError, ValueError):
            index = 0

        entry = pending.setdefault(index, {"call": {}})
        call = entry.setdefault("call", {})

        call_type = self._safe_get(tool_call_delta, "type")
        function_payload = self._safe_get(tool_call_delta, "function") or {}
        if call_type:
            call["type"] = call_type

        call_id = self._safe_get(tool_call_delta, "id")
        if call_id:
            call["id"] = call_id

        if function_payload:
            function_entry = call.setdefault("function", {})
            name = self._safe_get(function_payload, "name")
            if name:
                function_entry["name"] = name
            arguments_delta = self._safe_get(function_payload, "arguments")
            if arguments_delta:
                existing_arguments = function_entry.get("arguments", "")
                function_entry["arguments"] = existing_arguments + str(arguments_delta)
            call.setdefault("type", "function")

    def _finalize_pending_tool_calls(
        self, pending: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for _, payload in sorted(pending.items(), key=lambda item: item[0]):
            call = payload.get("call") if isinstance(payload, dict) else None
            normalized = self._normalize_tool_call(call)
            if normalized:
                messages.append(normalized)
        return messages

    async def _handle_tool_messages(
        self,
        tool_messages: List[Dict[str, Any]],
        *,
        user=None,
        conversation_id=None,
        conversation_manager=None,
        function_map=None,
        functions=None,
        current_persona=None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> Optional[Any]:
        if not tool_messages:
            return None

        prepared_messages = self._prepare_tool_messages(tool_messages)

        for message in prepared_messages:
            if message.get("type") != "function":
                continue
            function_payload = message.get("function_call")
            if not function_payload:
                continue
            tool_response = await use_tool(
                user,
                conversation_id,
                {"function_call": function_payload},
                conversation_manager,
                function_map,
                functions,
                current_persona,
                temperature,
                top_p,
                frequency_penalty,
                presence_penalty,
                conversation_manager,
                self.config_manager,
            )
            if tool_response is not None:
                return tool_response

        return None

    async def process_response(self, response: Union[str, AsyncIterator[str]]) -> str:
        if isinstance(response, str):
            return response
        else:
            full_response = ""
            async for chunk in response:
                full_response += chunk
            return full_response

def setup_mistral_generator(config_manager: ConfigManager):
    return MistralGenerator(config_manager)

async def generate_response(
    config_manager: ConfigManager,
    messages: List[Dict[str, str]],
    model: str = "mistral-large-latest",
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    stream: bool = True,
    current_persona=None,
    functions=None,
) -> Union[str, AsyncIterator[str]]:
    generator = setup_mistral_generator(config_manager)
    return await generator.generate_response(messages, model, max_tokens, temperature, stream, current_persona, functions)

async def process_response(response: Union[str, AsyncIterator[str]]) -> str:
    if isinstance(response, str):
        return response

    chunks: List[str] = []
    async for chunk in response:
        chunks.append(chunk)
    return "".join(chunks)

def generate_response_sync(config_manager: ConfigManager, messages: List[Dict[str, str]], model: str = "mistral-large-latest", stream: bool = False) -> str:
    """
    Synchronous version of generate_response for compatibility with non-async code.
    """
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        generate_response(
            config_manager,
            messages,
            model=model,
            stream=stream,
        )
    )
    if stream:
        return loop.run_until_complete(process_response(response))
    return response
