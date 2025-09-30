# modules/Providers/Google/GG_gen_response.py

import asyncio
import copy
import json
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Union

import google.generativeai as genai
from google.generativeai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential

from ATLAS.ToolManager import (
    load_function_map_from_current_persona,
    load_functions_from_json,
)
from ATLAS.config import ConfigManager
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
    ) -> Union[str, AsyncIterator[Union[str, Dict[str, Dict[str, str]]]]]:
        try:
            contents = self._convert_messages_to_contents(messages)
            tools = None
            if enable_functions:
                tools = self._build_tools_payload(functions, current_persona)

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
            }

            def _parse_bool(value: Optional[Any]) -> Optional[bool]:
                if isinstance(value, bool):
                    return value
                if value is None:
                    return None
                if isinstance(value, str):
                    cleaned = value.strip().lower()
                    if cleaned in {"", "none"}:
                        return None
                    if cleaned in {"true", "1", "yes", "on"}:
                        return True
                    if cleaned in {"false", "0", "no", "off"}:
                        return False
                return bool(value)

            provided_stream = _parse_bool(stream)
            if provided_stream is None:
                stored_stream = _parse_bool(stored_settings.get('stream'))
                if stored_stream is None:
                    effective_stream = defaults['stream']
                else:
                    effective_stream = stored_stream
            else:
                effective_stream = provided_stream

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

            def _resolve_float(
                provided: Optional[float],
                key: str,
                default: float,
            ) -> float:
                if provided is not None:
                    return float(provided)
                stored_value = stored_settings.get(key)
                if stored_value is not None:
                    try:
                        return float(stored_value)
                    except (TypeError, ValueError):
                        pass
                return default

            def _resolve_optional_int(
                provided: Optional[int],
                key: str,
                default: Optional[int],
            ) -> Optional[int]:
                if provided is not None:
                    return int(provided)
                stored_value = stored_settings.get(key)
                if stored_value is None:
                    return default
                try:
                    return int(stored_value)
                except (TypeError, ValueError):
                    return default

            def _resolve_int(
                provided: Optional[int],
                key: str,
                default: int,
            ) -> int:
                if provided is not None:
                    return int(provided)
                stored_value = stored_settings.get(key)
                if stored_value is not None:
                    try:
                        return int(stored_value)
                    except (TypeError, ValueError):
                        pass
                return default

            effective_temperature = _resolve_float(
                temperature, 'temperature', defaults['temperature']
            )
            effective_top_p = _resolve_float(top_p, 'top_p', defaults['top_p'])
            effective_top_k = _resolve_optional_int(top_k, 'top_k', defaults['top_k'])
            effective_candidate_count = _resolve_int(
                candidate_count, 'candidate_count', defaults['candidate_count']
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

            stored_max_tokens = stored_settings.get('max_output_tokens')
            if max_tokens is not None:
                effective_max_tokens = int(max_tokens)
            elif stored_max_tokens is not None:
                try:
                    effective_max_tokens = int(stored_max_tokens)
                except (TypeError, ValueError):
                    effective_max_tokens = defaults['max_output_tokens']
            else:
                effective_max_tokens = defaults['max_output_tokens']

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

            model_instance = genai.GenerativeModel(model_name=effective_model)
            self.logger.info(
                "Generating response with Google Gemini using model: %s",
                effective_model,
            )

            request_kwargs = {
                "generation_config": genai.types.GenerationConfig(
                    **{
                        key: value
                        for key, value in {
                            "max_output_tokens": effective_max_tokens,
                            "temperature": effective_temperature,
                            "top_p": effective_top_p,
                            "top_k": effective_top_k,
                            "candidate_count": effective_candidate_count,
                            "stop_sequences": effective_stop_sequences
                            or None,
                        }.items()
                        if value is not None
                    }
                ),
                "stream": effective_stream,
            }
            if tools:
                request_kwargs["tools"] = tools

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

    async def stream_response(
        self, response
    ) -> AsyncIterator[Union[str, Dict[str, Dict[str, str]]]]:
        for chunk in response:
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

            await asyncio.sleep(0)

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

