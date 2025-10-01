# modules/Providers/Google/GG_gen_response.py

import asyncio
import contextlib
import copy
import json
from typing import Any, AsyncIterator, Dict, Iterable, List, Mapping, Optional, Set, Union

import google.generativeai as genai
from google.generativeai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential

from ATLAS.ToolManager import (
    load_function_map_from_current_persona,
    load_functions_from_json,
)
from ATLAS.config import ConfigManager
from modules.Providers.Google.settings_resolver import GoogleSettingsResolver
from modules.logging.logger import setup_logger


class GoogleGeminiGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.api_key = self.config_manager.get_google_api_key()
        if not self.api_key:
            self.logger.error("Google API key not found in configuration")
            raise ValueError("Google API key not found in configuration")
        genai.configure(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None,
        stop_sequences: Optional[Iterable[str]] = None,
        stream: Optional[bool] = None,
        current_persona=None,
        functions=None,
        safety_settings: Optional[Any] = None,
        response_mime_type: Optional[str] = None,
        system_instruction: Optional[str] = None,
        enable_functions: bool = True,
        response_schema: Optional[Any] = None,
        seed: Optional[int] = None,
        response_logprobs: Optional[bool] = None,
    ) -> Union[str, AsyncIterator[Union[str, Dict[str, Dict[str, str]]]]]:
        try:
            contents = self._convert_messages_to_contents(messages)
            tools = None
            declared_function_names: Set[str] = set()
            if enable_functions:
                tools = self._build_tools_payload(functions, current_persona)
                declared_function_names = self._extract_declared_function_names(tools)

            stored_settings: Dict[str, Any] = {}
            getter = getattr(self.config_manager, "get_google_llm_settings", None)
            if callable(getter):
                try:
                    candidate_settings = getter() or {}
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning(
                        "Failed to load Google LLM defaults: %s", exc, exc_info=True
                    )
                    candidate_settings = {}
                if isinstance(candidate_settings, dict):
                    stored_settings = candidate_settings

            defaults = {
                'model': 'gemini-1.5-pro-latest',
                'temperature': 0.0,
                'top_p': 1.0,
                'top_k': None,
                'candidate_count': 1,
                'stop_sequences': [],
                'max_output_tokens': 32000,
                'stream': True,
                'response_schema': None,
                'seed': None,
                'response_logprobs': False,
            }

            resolver = GoogleSettingsResolver(
                stored=stored_settings,
                defaults=defaults,
            )

            effective_stream = resolver.resolve_bool(
                'stream',
                stream,
                default=defaults['stream'],
            )

            def _resolve_model(candidate: Optional[str]) -> str:
                if candidate:
                    cleaned = str(candidate).strip()
                    if cleaned:
                        return cleaned
                stored_model = stored_settings.get('model')
                if isinstance(stored_model, str) and stored_model.strip():
                    return stored_model.strip()
                return defaults['model']

            effective_model = _resolve_model(model)

            effective_temperature = resolver.resolve_float(
                'temperature',
                temperature,
                field='Temperature',
                minimum=0.0,
                maximum=2.0,
                allow_invalid_stored=True,
            )
            effective_top_p = resolver.resolve_float(
                'top_p',
                top_p,
                field='Top-p',
                minimum=0.0,
                maximum=1.0,
                allow_invalid_stored=True,
            )
            effective_top_k = resolver.resolve_optional_int(
                'top_k',
                top_k,
                field='Top-k',
                minimum=1,
                allow_invalid_stored=True,
            )
            effective_candidate_count = resolver.resolve_int(
                'candidate_count',
                candidate_count,
                field='Candidate count',
                minimum=1,
                allow_invalid_stored=True,
            )

            effective_seed = resolver.resolve_seed(
                seed,
                allow_invalid_stored=True,
            )

            effective_response_logprobs = resolver.resolve_bool(
                'response_logprobs',
                response_logprobs,
                default=defaults['response_logprobs'],
            )

            if stop_sequences is not None:
                effective_stop_sequences = [
                    str(item).strip()
                    for item in stop_sequences
                    if isinstance(item, str) and item.strip()
                ]
            else:
                stored_stop_sequences = stored_settings.get('stop_sequences')
                if isinstance(stored_stop_sequences, (list, tuple, set)):
                    effective_stop_sequences = [
                        str(item).strip()
                        for item in stored_stop_sequences
                        if isinstance(item, str) and item.strip()
                    ]
                else:
                    effective_stop_sequences = []

            effective_max_tokens = resolver.resolve_max_output_tokens(
                max_tokens,
                allow_invalid_stored=True,
            )

            effective_safety_settings = (
                safety_settings
                if safety_settings is not None
                else stored_settings.get('safety_settings')
            )

            effective_response_mime_type = (
                response_mime_type
                if response_mime_type is not None
                else stored_settings.get('response_mime_type')
            )
            if isinstance(effective_response_mime_type, str):
                effective_response_mime_type = (
                    effective_response_mime_type.strip() or None
                )

            effective_system_instruction = (
                system_instruction
                if system_instruction is not None
                else stored_settings.get('system_instruction')
            )
            if isinstance(effective_system_instruction, str):
                effective_system_instruction = (
                    effective_system_instruction.strip() or None
                )

            resolved_schema = resolver.resolve_response_schema(response_schema)
            effective_response_schema = resolved_schema or None

            if (
                effective_response_schema is not None
                and not effective_response_mime_type
            ):
                effective_response_mime_type = "application/json"

            valid_modes = {"auto", "any", "none", "require"}

            stored_mode = stored_settings.get('function_call_mode')
            if isinstance(stored_mode, str):
                cleaned_mode = stored_mode.strip().lower()
                if cleaned_mode in valid_modes:
                    resolved_function_call_mode = cleaned_mode
                else:
                    resolved_function_call_mode = 'auto'
            else:
                resolved_function_call_mode = 'auto'

            allowed_names_source = stored_settings.get('allowed_function_names')
            resolved_allowed_names: List[str] = []
            if isinstance(allowed_names_source, str):
                resolved_allowed_names = [
                    token.strip()
                    for token in allowed_names_source.split(',')
                    if token.strip()
                ]
            elif isinstance(allowed_names_source, Iterable) and not isinstance(
                allowed_names_source, (bytes, bytearray, str)
            ):
                seen_names = set()
                for item in allowed_names_source:
                    if not isinstance(item, str):
                        continue
                    cleaned = item.strip()
                    if cleaned and cleaned not in seen_names:
                        resolved_allowed_names.append(cleaned)
                        seen_names.add(cleaned)

            discarded_allowed_names: List[str] = []

            if not enable_functions:
                effective_function_call_mode = 'none'
                effective_allowed_names: List[str] = []
            else:
                effective_function_call_mode = resolved_function_call_mode
                effective_allowed_names = list(resolved_allowed_names)

                declared_tools_available = bool(tools) and bool(
                    declared_function_names
                )

                if not declared_tools_available:
                    if effective_allowed_names:
                        discarded_allowed_names.extend(effective_allowed_names)
                    effective_allowed_names = []
                    if effective_function_call_mode == 'require':
                        effective_function_call_mode = 'auto'
                else:
                    filtered_allowed_names: List[str] = []
                    for name in effective_allowed_names:
                        if name in declared_function_names:
                            filtered_allowed_names.append(name)
                        else:
                            discarded_allowed_names.append(name)
                    effective_allowed_names = filtered_allowed_names

            function_calling_config: Dict[str, Any] = {
                "mode": effective_function_call_mode.upper(),
            }
            if effective_allowed_names:
                function_calling_config["allowed_function_names"] = list(
                    effective_allowed_names
                )

            if discarded_allowed_names:
                unique_discarded = list(dict.fromkeys(discarded_allowed_names))
                self.logger.warning(
                    "Discarded Gemini allowlist entries not declared as tools: %s",
                    ", ".join(unique_discarded),
                )
                self._notify_allowlist_pruned(unique_discarded)

            tool_config_payload: Dict[str, Any] = {
                "function_calling_config": function_calling_config
            }

            model_instance = genai.GenerativeModel(model_name=effective_model)
            self.logger.info(
                "Generating response with Google Gemini using model: %s",
                effective_model,
            )

            generation_config_payload = {
                "max_output_tokens": effective_max_tokens,
                "temperature": effective_temperature,
                "top_p": effective_top_p,
                "top_k": effective_top_k,
                "candidate_count": effective_candidate_count,
                "stop_sequences": effective_stop_sequences or None,
                "response_schema": copy.deepcopy(effective_response_schema)
                if effective_response_schema is not None
                else None,
            }

            if effective_seed is not None:
                generation_config_payload["seed"] = effective_seed

            if effective_response_logprobs:
                generation_config_payload["response_logprobs"] = bool(
                    effective_response_logprobs
                )

            request_kwargs = {
                "generation_config": genai_types.GenerationConfig(
                    **{key: value for key, value in generation_config_payload.items() if value is not None}
                ),
                "stream": effective_stream,
            }
            if tools:
                request_kwargs["tools"] = tools

            if tool_config_payload:
                request_kwargs["tool_config"] = tool_config_payload

            if effective_safety_settings:
                if isinstance(effective_safety_settings, (list, dict)):
                    request_kwargs["safety_settings"] = copy.deepcopy(
                        effective_safety_settings
                    )
                else:
                    request_kwargs["safety_settings"] = effective_safety_settings

            if effective_response_mime_type:
                request_kwargs["response_mime_type"] = effective_response_mime_type

            if effective_system_instruction:
                request_kwargs["system_instruction"] = effective_system_instruction

            response = await asyncio.to_thread(
                model_instance.generate_content,
                contents,
                **request_kwargs,
            )

            if effective_stream:
                return self.stream_response(response)

            function_calls = self._extract_function_calls(response)
            if function_calls:
                return {"function_call": function_calls[0]}

            return response.text

        except Exception as e:
            self.logger.error(f"Error in Google Gemini API call: {str(e)}")
            raise

    def _convert_messages_to_contents(
        self, messages: List[Dict[str, Union[str, Dict, List]]]
    ) -> List[genai_types.ContentDict]:
        contents: List[genai_types.ContentDict] = []
        for message in messages:
            role = message.get("role")
            if not role:
                continue

            content_payload = message.get("content")
            parts = list(self._normalize_parts(content_payload))

            function_call = message.get("function_call")
            if function_call:
                normalized_call = self._normalize_function_call(function_call)
                if normalized_call:
                    parts.append({"function_call": normalized_call})

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, Iterable):
                for tool_call in tool_calls:
                    normalized_tool_call = self._normalize_function_call(tool_call)
                    if normalized_tool_call:
                        parts.append({"function_call": normalized_tool_call})

            content_dict: genai_types.ContentDict = {
                "role": role,
                "parts": parts or [""],
            }

            if "metadata" in message and isinstance(message["metadata"], dict):
                content_dict["metadata"] = dict(message["metadata"])

            for optional_key in ("name", "tool_call_id", "id"):
                if optional_key in message:
                    content_dict[optional_key] = message[optional_key]

            contents.append(content_dict)

        return contents

    def _normalize_parts(self, content_payload) -> Iterable[genai_types.PartDict]:
        if content_payload is None:
            return []

        if isinstance(content_payload, str):
            return [{"text": content_payload}]

        if isinstance(content_payload, dict):
            return [self._normalize_part_dict(content_payload)]

        if isinstance(content_payload, Iterable):
            normalized_parts: List[genai_types.PartDict] = []
            for item in content_payload:
                if isinstance(item, str):
                    normalized_parts.append({"text": item})
                elif isinstance(item, dict):
                    normalized_parts.append(self._normalize_part_dict(item))
                else:
                    normalized_parts.append({"text": str(item)})
            return normalized_parts

        return [{"text": str(content_payload)}]

    def _normalize_part_dict(self, payload: Dict) -> genai_types.PartDict:
        if not isinstance(payload, dict):
            return {"text": str(payload)}

        if payload.get("type") == "text" and "text" in payload:
            return {"text": payload["text"]}

        if "text" in payload:
            return {"text": payload["text"]}

        if "function_call" in payload:
            normalized_call = self._normalize_function_call(payload["function_call"])
            if normalized_call:
                return {"function_call": normalized_call}

        if payload.get("type") == "function_call":
            normalized_call = self._normalize_function_call(payload)
            if normalized_call:
                return {"function_call": normalized_call}

        return payload

    def _build_tools_payload(self, functions, current_persona) -> Optional[List[genai_types.Tool]]:
        declared_functions: List[genai_types.FunctionDeclaration] = []
        seen_names = set()

        provided_functions = functions
        if provided_functions is None and current_persona:
            provided_functions = load_functions_from_json(current_persona)

        if isinstance(provided_functions, dict):
            provided_functions = provided_functions.get("functions") or provided_functions.get(
                "items"
            )

        if isinstance(provided_functions, Iterable):
            for function_payload in provided_functions:
                declaration = self._to_function_declaration(function_payload)
                if declaration and declaration.name not in seen_names:
                    declared_functions.append(declaration)
                    seen_names.add(declaration.name)

        function_map = (
            load_function_map_from_current_persona(current_persona)
            if current_persona
            else None
        )
        if isinstance(function_map, dict):
            for name in function_map.keys():
                if name in seen_names:
                    continue
                declared_functions.append(
                    genai_types.FunctionDeclaration(
                        name=name,
                        description=f"Function available in persona toolbox: {name}",
                        parameters={"type": "object", "properties": {}},
                    )
                )
                seen_names.add(name)

        if not declared_functions:
            return None

        return [genai_types.Tool(function_declarations=declared_functions)]

    def _extract_declared_function_names(
        self, tools_payload: Optional[List[genai_types.Tool]]
    ) -> Set[str]:
        declared: Set[str] = set()
        if not tools_payload:
            return declared

        for tool in tools_payload:
            declarations = getattr(tool, "function_declarations", None)
            if not declarations:
                continue
            try:
                iterator = list(declarations)
            except TypeError:
                iterator = [declarations]
            for declaration in iterator:
                name = getattr(declaration, "name", None)
                if isinstance(name, str):
                    cleaned = name.strip()
                    if cleaned:
                        declared.add(cleaned)

        return declared

    def _to_function_declaration(self, payload) -> Optional[genai_types.FunctionDeclaration]:
        if not isinstance(payload, dict):
            return None

        name = payload.get("name")
        if not name:
            return None

        description = payload.get("description") or ""
        parameters = payload.get("parameters")
        if parameters is None:
            parameters = {"type": "object", "properties": {}}

        return genai_types.FunctionDeclaration(
            name=name,
            description=description,
            parameters=parameters,
        )

    def _notify_allowlist_pruned(self, discarded_names: Iterable[str]) -> None:
        if not discarded_names:
            return

        message = (
            "Removed unsupported Google Gemini tool allowlist entries: "
            + ", ".join(discarded_names)
        )

        for attr in (
            "notify_ui_warning",
            "notify_ui",
            "queue_notification",
            "queue_ui_notification",
            "send_notification",
        ):
            notifier = getattr(self.config_manager, attr, None)
            if callable(notifier):
                with contextlib.suppress(Exception):
                    notifier(message)
                break

    def _normalize_function_call(self, payload) -> Optional[Dict[str, str]]:
        if payload is None:
            return None

        if isinstance(payload, dict):
            name = payload.get("name") or payload.get("function", {}).get("name")
            args = (
                payload.get("arguments")
                or payload.get("args")
                or payload.get("function", {}).get("arguments")
            )
        else:
            name = getattr(payload, "name", None)
            args = getattr(payload, "arguments", None)
            if args is None:
                args = getattr(payload, "args", None)

        if not name:
            return None

        if isinstance(args, str):
            arguments_json = args
        elif args is None:
            arguments_json = "{}"
        else:
            try:
                arguments_json = json.dumps(dict(args))
            except Exception:
                try:
                    arguments_json = json.dumps(args)
                except TypeError:
                    arguments_json = str(args)

        return {"name": name, "arguments": arguments_json}

    def _extract_function_calls(self, response) -> List[Dict[str, str]]:
        calls: List[Dict[str, str]] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                for part in parts:
                    normalized = None
                    if isinstance(part, dict):
                        normalized = self._normalize_function_call(part.get("function_call"))
                    else:
                        normalized = self._normalize_function_call(
                            getattr(part, "function_call", None)
                        )
                    if normalized:
                        calls.append(normalized)
        return calls

    def _iter_stream_payloads(self, chunk):
        if getattr(chunk, "text", None):
            yield chunk.text

        for candidate in getattr(chunk, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not parts:
                continue

            for part in parts:
                text_value = None
                function_call = None

                if isinstance(part, dict):
                    if "text" in part:
                        text_value = part.get("text")
                    function_call = part.get("function_call")
                else:
                    text_value = getattr(part, "text", None)
                    function_call = getattr(part, "function_call", None)

                if text_value:
                    yield text_value

                normalized_call = self._normalize_function_call(function_call)
                if normalized_call:
                    yield {"function_call": normalized_call}

    async def stream_response(
        self, response
    ) -> AsyncIterator[Union[str, Dict[str, Dict[str, str]]]]:
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[tuple[str, Optional[Union[str, Dict[str, Dict[str, str]]]]]]" = (
            asyncio.Queue()
        )

        def producer():
            try:
                for chunk in response:
                    for payload in self._iter_stream_payloads(chunk):
                        loop.call_soon_threadsafe(queue.put_nowait, ("data", payload))
            except Exception as exc:  # pragma: no cover - defensive
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        producer_task = asyncio.create_task(asyncio.to_thread(producer))

        try:
            while True:
                kind, payload = await queue.get()
                if kind == "data":
                    yield payload  # type: ignore[misc]
                elif kind == "error":
                    raise payload  # type: ignore[misc]
                elif kind == "done":
                    break
        finally:
            if not producer_task.done():
                producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task

    async def process_response(self, response) -> str:
        if isinstance(response, AsyncIterator):
            full_response = ""
            async for chunk in response:
                if isinstance(chunk, str):
                    full_response += chunk
            return full_response
        else:
            return response


def setup_google_gemini_generator(config_manager: ConfigManager):
    return GoogleGeminiGenerator(config_manager)


async def generate_response(
    config_manager: ConfigManager,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    *,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    candidate_count: Optional[int] = None,
    stop_sequences: Optional[Iterable[str]] = None,
    stream: bool = True,
    current_persona=None,
    functions=None,
    safety_settings: Optional[Any] = None,
    response_mime_type: Optional[str] = None,
    system_instruction: Optional[str] = None,
    enable_functions: bool = True,
):
    generator = setup_google_gemini_generator(config_manager)
    return await generator.generate_response(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        candidate_count=candidate_count,
        stop_sequences=stop_sequences,
        stream=stream,
        current_persona=current_persona,
        functions=functions,
        safety_settings=safety_settings,
        response_mime_type=response_mime_type,
        system_instruction=system_instruction,
        enable_functions=enable_functions,
    )


async def process_response(response: Union[str, AsyncIterator[str]]) -> str:
    if isinstance(response, str):
        return response
    elif isinstance(response, AsyncIterator):
        full_response = ""
        async for chunk in response:
            if isinstance(chunk, str):
                full_response += chunk
        return full_response
    else:
        raise ValueError(f"Unexpected response type: {type(response)}")

