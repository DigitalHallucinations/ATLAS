# modules/Providers/Google/GG_gen_response.py

import asyncio
import json
from typing import AsyncIterator, Dict, Iterable, List, Optional, Union

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
        model: str = "gemini-1.5-pro-latest",
        max_tokens: int = 32000,
        temperature: float = 0.0,
        stream: bool = True,
        current_persona=None,
        functions=None,
    ) -> Union[str, AsyncIterator[Union[str, Dict[str, Dict[str, str]]]]]:
        try:
            contents = self._convert_messages_to_contents(messages)
            tools = self._build_tools_payload(functions, current_persona)

            model_instance = genai.GenerativeModel(model_name=model)
            self.logger.info(
                "Generating response with Google Gemini using model: %s", model
            )

            request_kwargs = {
                "generation_config": genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
                "stream": stream,
            }
            if tools:
                request_kwargs["tools"] = tools

            response = await asyncio.to_thread(
                model_instance.generate_content,
                contents,
                **request_kwargs,
            )

            if stream:
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
    model: str = "gemini-1.5-pro-latest",
    max_tokens: int = 32000,
    temperature: float = 0.0,
    stream: bool = True,
    current_persona=None,
    functions=None,
):
    generator = setup_google_gemini_generator(config_manager)
    return await generator.generate_response(
        messages, model, max_tokens, temperature, stream, current_persona, functions
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

