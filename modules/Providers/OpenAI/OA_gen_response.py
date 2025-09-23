# modules/Providers/OpenAI/OA_gen_response.py

import json

from openai import AsyncOpenAI
from ATLAS.model_manager import ModelManager

from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Union, AsyncIterator, Optional, Any
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from ATLAS.ToolManager import (
    load_function_map_from_current_persona,
    load_functions_from_json,
    use_tool
)

class OpenAIGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = setup_logger(__name__) 
        self.api_key = self.config_manager.get_openai_api_key()
        if not self.api_key:
            self.logger.error("OpenAI API key not found in configuration")
            raise ValueError("OpenAI API key not found in configuration")
        settings = self.config_manager.get_openai_llm_settings()
        client_kwargs = {"api_key": self.api_key}
        base_url = settings.get("base_url")
        if base_url:
            client_kwargs["base_url"] = base_url
        organization = settings.get("organization")
        if organization:
            client_kwargs["organization"] = organization

        self.client = AsyncOpenAI(**client_kwargs)
        self.model_manager = ModelManager(config_manager)
        default_model = settings.get("model")
        if default_model:
            self.model_manager.set_model(default_model, "OpenAI")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: Optional[bool] = None,
        current_persona=None,
        conversation_manager=None,
        user=None,
        conversation_id=None,
        functions=None,
        reasoning_effort: Optional[str] = None,
        function_calling: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[Any] = None,
        json_mode: Optional[Any] = None,
        json_schema: Optional[Any] = None,
    ) -> Union[str, AsyncIterator[str]]:
        try:
            settings = self.config_manager.get_openai_llm_settings()
            current_model = self.model_manager.get_current_model()

            if model and model != current_model:
                self.model_manager.set_model(model, "OpenAI")
                current_model = model
                self.logger.info(f"Model changed to {model}")
            elif not model:
                model = settings.get("model") or current_model
                if model and model != current_model:
                    self.model_manager.set_model(model, "OpenAI")
                    current_model = model
                self.logger.info(f"Using current model: {model}")

            if max_tokens is None:
                max_tokens = int(settings.get("max_tokens", 4000))
            if max_output_tokens is None:
                stored_max_output = settings.get("max_output_tokens")
                if stored_max_output is not None:
                    try:
                        max_output_tokens = int(stored_max_output)
                    except (TypeError, ValueError):
                        max_output_tokens = None
            if temperature is None:
                temperature = float(settings.get("temperature", 0.0))
            if top_p is None:
                top_p = float(settings.get("top_p", 1.0))
            if frequency_penalty is None:
                frequency_penalty = float(settings.get("frequency_penalty", 0.0))
            if presence_penalty is None:
                presence_penalty = float(settings.get("presence_penalty", 0.0))
            if stream is None:
                stream = bool(settings.get("stream", True))
            if reasoning_effort is None:
                reasoning_effort = settings.get("reasoning_effort")

            if function_calling is None:
                allow_function_calls = bool(settings.get("function_calling", True))
            else:
                allow_function_calls = bool(function_calling)

            if parallel_tool_calls is None:
                effective_parallel_tool_calls = bool(settings.get("parallel_tool_calls", True))
            else:
                effective_parallel_tool_calls = bool(parallel_tool_calls)

            configured_tool_choice = (
                tool_choice if tool_choice is not None else settings.get("tool_choice")
            )
            normalized_tool_choice = self._normalize_tool_choice(configured_tool_choice)
            if not allow_function_calls:
                normalized_tool_choice = "none"

            configured_schema = (
                json_schema if json_schema is not None else settings.get("json_schema")
            )
            normalized_schema = self._prepare_json_schema(configured_schema)
            force_json_mode = False
            if normalized_schema is None:
                force_json_mode = self._should_force_json(
                    json_mode if json_mode is not None else settings.get("json_mode")
                )

            self.logger.info(f"Starting API call to OpenAI with model {model}")
            self.logger.info(f"Current persona: {current_persona}")
            if normalized_schema is not None:
                self.logger.info(
                    "JSON schema response mode enabled for this request (schema name: %s).",
                    normalized_schema.get("name"),
                )
            elif force_json_mode:
                self.logger.info("JSON response mode enabled for this request.")

            # Load functions if not provided
            if functions is None and current_persona:
                self.logger.info("No functions provided; attempting to load from current persona.")
                functions = load_functions_from_json(current_persona)
                if functions:
                    self.logger.info(f"Functions loaded from JSON: {functions}")
                else:
                    self.logger.warning("No functions loaded from JSON.")
            elif functions:
                self.logger.info(f"Using provided functions: {functions}")
            else:
                self.logger.warning("No functions to load or provide.")

            # Load function map
            function_map = load_function_map_from_current_persona(current_persona) if current_persona else None
            if function_map:
                self.logger.info(f"Function map loaded: {function_map}")
            else:
                self.logger.warning("No function map loaded.")

            # Ensure functions and function_map are sent for all models
            if not functions and not function_map:
                self.logger.warning("Neither functions nor function map available to send to the model.")

            tools_payload = self._convert_functions_to_tools(functions)

            # Log what is being sent to the API
            self.logger.info(f"Sending tools to OpenAI API: {tools_payload}")
            self.logger.info(f"Sending function map to OpenAI API: {function_map}")

            if self._is_reasoning_model(model):
                self.logger.info("Routing request through the Responses API for reasoning model support.")
                return await self._generate_via_responses_api(
                    messages=messages,
                    model=model,
                    stream=stream,
                    user=user,
                    conversation_id=conversation_id,
                    function_map=function_map,
                    functions=functions,
                    current_persona=current_persona,
                    conversation_manager=conversation_manager,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    allow_function_calls=allow_function_calls,
                    parallel_tool_calls=effective_parallel_tool_calls,
                    tool_choice=normalized_tool_choice,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                )

            function_call_mode = None
            if tools_payload:
                function_call_mode = "auto" if allow_function_calls else "none"
                self.logger.info(
                    "Automatic tool calling %s for this request.",
                    "enabled" if allow_function_calls else "disabled",
                )

            request_kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "n": 1,
                "stop": None,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stream": stream,
            }

            if tools_payload:
                request_kwargs["tools"] = tools_payload
                request_kwargs["parallel_tool_calls"] = effective_parallel_tool_calls
                if normalized_tool_choice is not None:
                    request_kwargs["tool_choice"] = normalized_tool_choice
            if function_call_mode is not None:
                request_kwargs["function_call"] = function_call_mode
            if normalized_schema is not None:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": normalized_schema,
                }
            elif force_json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**request_kwargs)

            self.logger.info("Received response from OpenAI API.")

            if stream:
                self.logger.info("Processing streaming response.")
                return self.process_streaming_response(
                    response,
                    user,
                    conversation_id,
                    function_map,
                    functions,
                    current_persona,
                    temperature,
                    conversation_manager,
                    model,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                )
            else:
                message = response.choices[0].message
                tool_messages = self._extract_chat_tool_calls(message)
                if tool_messages:
                    results = []
                    for tool_message in tool_messages:
                        self.logger.info(
                            "Tool call detected in response: %s",
                            tool_message.get("function_call"),
                        )
                        result = await self.handle_function_call(
                            user,
                            conversation_id,
                            tool_message,
                            conversation_manager,
                            function_map,
                            functions,
                            current_persona,
                            temperature,
                            model,
                            top_p,
                            frequency_penalty,
                            presence_penalty,
                        )
                        if result:
                            results.append(result)
                    if results:
                        return "\n".join(str(item) for item in results)
                self.logger.info("No function or tool call detected in response.")
                return self._coerce_content_to_text(getattr(message, "content", ""))

        except Exception as e:
            self.logger.error(f"Error in OpenAI API call: {str(e)}", exc_info=True)
            raise

    def _is_reasoning_model(self, model: Optional[str]) -> bool:
        return isinstance(model, str) and model.lower().startswith("o")

    def _map_messages_to_responses_input(self, messages: List[Dict[str, str]]):
        formatted = []
        for message in messages:
            role = message.get("role", "user") if isinstance(message, dict) else "user"
            content = message.get("content") if isinstance(message, dict) else message
            if content is None:
                continue
            if isinstance(content, list):
                normalized_content = []
                for item in content:
                    if isinstance(item, dict):
                        normalized_content.append(item)
                    else:
                        normalized_content.append({"type": "text", "text": str(item)})
            else:
                normalized_content = [{"type": "text", "text": str(content)}]

            formatted.append({"role": role, "content": normalized_content})
        return formatted

    def _convert_functions_to_tools(self, functions):
        tools = []
        try:
            settings = self.config_manager.get_openai_llm_settings()
        except Exception:
            settings = {}

        if isinstance(settings, dict):
            if settings.get("enable_code_interpreter"):
                tools.append({"type": "code_interpreter"})
            if settings.get("enable_file_search"):
                tools.append({"type": "file_search"})

        if not functions:
            return tools

        for function in functions:
            if isinstance(function, dict):
                tools.append({"type": "function", "function": function})
        return tools

    def _normalize_tool_call(self, call: Any) -> Optional[Dict[str, Any]]:
        if not call:
            return None

        if isinstance(call, dict) and call.get("type") in {"function", "code_interpreter", "file_search"}:
            normalized = dict(call)
            if "raw_call" not in normalized:
                normalized["raw_call"] = call
            if normalized.get("type") == "function" and "function_call" not in normalized:
                function_payload = normalized.get("function") or {}
                if isinstance(function_payload, dict):
                    name = self._safe_get(function_payload, "name")
                    arguments = self._safe_get(function_payload, "arguments")
                    if name:
                        normalized["function_call"] = {
                            "name": name,
                            "arguments": self._stringify_function_arguments(arguments),
                        }
            return normalized

        payload: Dict[str, Any] = {}
        for key in (
            "type",
            "function",
            "function_call",
            "name",
            "arguments",
            "code_interpreter",
            "file_search",
            "id",
        ):
            value = self._safe_get(call, key)
            if value is not None:
                payload[key] = value

        if not payload:
            return None

        call_type = payload.get("type")
        function_payload = payload.get("function")
        legacy_function_call = payload.get("function_call")

        normalized: Dict[str, Any] = {"raw_call": dict(payload)}

        if function_payload is None and isinstance(legacy_function_call, dict):
            function_payload = legacy_function_call
            call_type = call_type or "function"

        if call_type is None and function_payload is not None:
            call_type = "function"

        if call_type == "function":
            function_payload = function_payload or {}
            name = self._safe_get(function_payload, "name") or payload.get("name")
            arguments = self._safe_get(function_payload, "arguments")
            if arguments is None:
                arguments = payload.get("arguments")
            if not name:
                return None
            normalized.update(
                {
                    "type": "function",
                    "function_call": {
                        "name": name,
                        "arguments": self._stringify_function_arguments(arguments),
                    },
                }
            )
            return normalized

        if call_type in {"code_interpreter", "file_search"}:
            normalized["type"] = call_type
            builtin_payload = payload.get(call_type)
            if isinstance(builtin_payload, dict):
                normalized[call_type] = builtin_payload
            return normalized

        return None

    def _merge_builtin_tool_payload(
        self,
        tool_type: str,
        existing: Any,
        delta: Any,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        if not isinstance(delta, dict):
            return base

        def _merge_values(key: str, value: Any) -> None:
            if isinstance(value, list):
                previous = base.get(key)
                if isinstance(previous, list):
                    base[key] = previous + value
                else:
                    base[key] = list(value)
            elif isinstance(value, dict):
                previous = base.get(key)
                merged_child = self._merge_builtin_tool_payload(tool_type, previous, value)
                base[key] = merged_child
            elif value is not None:
                if key == "input" and tool_type == "code_interpreter":
                    existing_text = str(base.get(key, ""))
                    base[key] = existing_text + str(value)
                else:
                    base[key] = value

        for key, value in delta.items():
            _merge_values(key, value)

        return base

    def _format_builtin_tool_output(self, payload: Dict[str, Any]) -> Optional[str]:
        tool_type = payload.get("type")
        if tool_type == "code_interpreter":
            interpreter = payload.get("code_interpreter")
            if not isinstance(interpreter, dict):
                interpreter = self._safe_get(payload.get("raw_call"), "code_interpreter") or {}
            outputs = interpreter.get("outputs")
            parts: List[str] = []
            if isinstance(outputs, list):
                for output in outputs:
                    if not isinstance(output, dict):
                        continue
                    output_type = output.get("type")
                    if output_type in {"logs", "output_text"}:
                        text_value = output.get("text") or output.get("logs")
                        if isinstance(text_value, list):
                            text_value = "".join(str(item) for item in text_value if item is not None)
                        if text_value:
                            parts.append(str(text_value))
                    elif output_type == "image":
                        parts.append(
                            "[Code Interpreter produced an image output that cannot be rendered here.]"
                        )
                    elif output_type == "file":
                        file_id = output.get("id") or output.get("file_id")
                        if file_id:
                            parts.append(f"[Code Interpreter produced a downloadable file: {file_id}]")
                    else:
                        text_value = output.get("text") or output.get("logs") or output.get("output")
                        if text_value:
                            parts.append(str(text_value))
            if not parts:
                input_text = interpreter.get("input")
                if input_text:
                    parts.append(f"Code Interpreter executed:\n{input_text}")
            combined = "\n".join(part.strip() for part in parts if part).strip()
            return combined or None

        if tool_type == "file_search":
            search_payload = payload.get("file_search")
            if not isinstance(search_payload, dict):
                search_payload = self._safe_get(payload.get("raw_call"), "file_search") or {}
            parts: List[str] = []
            summary = search_payload.get("summary")
            if summary:
                parts.append(str(summary))
            results = search_payload.get("results") or search_payload.get("output")
            if isinstance(results, list):
                for result in results:
                    if not isinstance(result, dict):
                        continue
                    text_fragments: List[str] = []
                    text_value = result.get("text")
                    if text_value:
                        text_fragments.append(str(text_value))
                    content_entries = result.get("content")
                    if isinstance(content_entries, list):
                        for entry in content_entries:
                            if isinstance(entry, dict):
                                text = entry.get("text")
                                if text:
                                    text_fragments.append(str(text))
                    if text_fragments:
                        parts.append("\n".join(fragment.strip() for fragment in text_fragments if fragment))
            content_entries = search_payload.get("content")
            if isinstance(content_entries, list):
                for entry in content_entries:
                    if isinstance(entry, dict):
                        text = entry.get("text")
                        if text:
                            parts.append(str(text))
            if not parts:
                query = search_payload.get("query")
                if query:
                    parts.append(f"File search query: {query}")
            combined = "\n".join(part.strip() for part in parts if part).strip()
            return combined or None

        return None

    def _normalize_tool_choice(self, value):
        if value is None:
            return None

        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return None
            if normalized in {"auto", "none", "required"}:
                return normalized

        return None

    def _safe_get(self, target, attribute: str, default=None):
        if target is None:
            return default
        if isinstance(target, dict):
            return target.get(attribute, default)
        return getattr(target, attribute, default)

    def _coerce_content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for entry in content:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if text is None and "content" in entry:
                        text = entry.get("content")
                    if text is not None:
                        parts.append(str(text))
                elif entry is not None:
                    parts.append(str(entry))
            return "".join(parts)
        try:
            return str(content)
        except Exception:
            return ""

    def _should_force_json(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return False
            if normalized in {"1", "true", "yes", "on", "json", "json_object"}:
                return True
            if normalized in {"0", "false", "no", "off", "text", "none"}:
                return False
            return False
        try:
            return bool(value)
        except Exception:
            return False

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
        inner_schema = payload.get("schema")
        if not isinstance(inner_schema, dict):
            raise ValueError("JSON schema payload must include a 'schema' object.")

        schema_name = payload.get("name") or "atlas_response"
        normalized: Dict[str, Any] = {
            "name": str(schema_name),
            "schema": inner_schema,
        }

        if "strict" in payload:
            normalized["strict"] = bool(payload["strict"])

        try:
            return json.loads(json.dumps(normalized))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"JSON schema payload could not be serialized: {exc}"
            ) from exc

    def _stringify_function_arguments(self, arguments) -> str:
        if arguments is None:
            return "{}"
        if isinstance(arguments, str):
            return arguments
        try:
            return json.dumps(arguments)
        except (TypeError, ValueError):
            return str(arguments)

    def _extract_responses_tool_calls(self, response) -> List[Dict[str, Any]]:
        tool_messages: List[Dict[str, Any]] = []
        outputs = self._safe_get(response, "output") or self._safe_get(response, "outputs")
        if not outputs:
            return tool_messages

        for item in outputs:
            item_type = self._safe_get(item, "type")
            if item_type != "tool_call":
                continue

            tool_calls = self._safe_get(item, "tool_calls")
            if tool_calls is None:
                single_call = self._safe_get(item, "tool_call")
                tool_calls = [single_call] if single_call else []

            for call in tool_calls:
                normalized = self._normalize_tool_call(call)
                if normalized:
                    tool_messages.append(normalized)

        return tool_messages

    def _extract_chat_tool_calls(self, message) -> List[Dict[str, Any]]:
        tool_messages: List[Dict[str, Any]] = []
        tool_calls = self._safe_get(message, "tool_calls")

        if not tool_calls:
            single_call = self._safe_get(message, "tool_call")
            tool_calls = [single_call] if single_call else []

        for call in tool_calls or []:
            normalized = self._normalize_tool_call(call)
            if normalized:
                tool_messages.append(normalized)

        if tool_messages:
            return tool_messages

        legacy_call = self._safe_get(message, "function_call")
        if legacy_call:
            normalized = self._normalize_tool_call({"type": "function", "function": legacy_call})
            if normalized:
                tool_messages.append(normalized)

        return tool_messages

    def _update_pending_tool_calls(self, pending: Dict[int, Dict[str, Any]], tool_call_delta):
        if not tool_call_delta:
            return

        try:
            index = int(self._safe_get(tool_call_delta, "index"))
        except (TypeError, ValueError):
            index = 0

        entry = pending.setdefault(index, {"call": {}})
        call = entry.setdefault("call", {})

        function_payload = self._safe_get(tool_call_delta, "function") or {}
        call_type = self._safe_get(tool_call_delta, "type") or self._safe_get(function_payload, "type")
        if call_type:
            call["type"] = call_type

        call_id = self._safe_get(tool_call_delta, "id")
        if call_id:
            call["id"] = call_id

        name = self._safe_get(function_payload, "name") or self._safe_get(tool_call_delta, "name")
        arguments_delta = self._safe_get(function_payload, "arguments")
        if arguments_delta is None:
            arguments_delta = self._safe_get(tool_call_delta, "arguments")

        if function_payload or name or call_type == "function":
            function_entry = call.setdefault("function", {})
            if name:
                function_entry["name"] = name
            if arguments_delta:
                existing_arguments = function_entry.get("arguments", "")
                function_entry["arguments"] = existing_arguments + str(arguments_delta)
            call.setdefault("type", "function")

        for builtin_key in ("code_interpreter", "file_search"):
            payload_delta = self._safe_get(tool_call_delta, builtin_key)
            if payload_delta:
                existing_payload = call.get(builtin_key) or {}
                merged_payload = self._merge_builtin_tool_payload(
                    builtin_key, existing_payload, payload_delta
                )
                call[builtin_key] = merged_payload
                call["type"] = builtin_key

    def _finalize_pending_tool_calls(self, pending: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for _, payload in sorted(pending.items(), key=lambda item: item[0]):
            call = payload.get("call") if isinstance(payload, dict) else None
            normalized = self._normalize_tool_call(call)
            if normalized:
                messages.append(normalized)
        return messages

    def _get_response_text(self, response) -> str:
        text = self._safe_get(response, "output_text")
        if text:
            return text

        outputs = self._safe_get(response, "output") or self._safe_get(response, "outputs")
        if not outputs:
            return ""

        parts = []
        for item in outputs:
            item_type = self._safe_get(item, "type")
            if item_type == "output_text":
                snippet = self._safe_get(item, "text")
                if snippet:
                    parts.append(snippet)
            elif item_type == "message":
                content_list = self._safe_get(item, "content") or []
                for entry in content_list:
                    snippet = self._safe_get(entry, "text")
                    if snippet:
                        parts.append(snippet)

        return "".join(parts)

    async def _generate_via_responses_api(
        self,
        messages,
        model,
        stream,
        user,
        conversation_id,
        function_map,
        functions,
        current_persona,
        conversation_manager,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        allow_function_calls,
        parallel_tool_calls,
        tool_choice,
        max_output_tokens,
        reasoning_effort,
    ):
        request_kwargs = {
            "model": model,
            "input": self._map_messages_to_responses_input(messages),
        }

        if allow_function_calls:
            tools = self._convert_functions_to_tools(functions)
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["parallel_tool_calls"] = parallel_tool_calls
                if tool_choice is not None:
                    request_kwargs["tool_choice"] = tool_choice

        if max_output_tokens:
            request_kwargs["max_output_tokens"] = int(max_output_tokens)

        if reasoning_effort:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}

        if stream:
            return self._process_responses_stream(
                request_kwargs=request_kwargs,
                user=user,
                conversation_id=conversation_id,
                function_map=function_map,
                functions=functions,
                current_persona=current_persona,
                conversation_manager=conversation_manager,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                model=model,
                allow_function_calls=allow_function_calls,
            )

        response = await self.client.responses.create(**request_kwargs)
        return await self._handle_responses_completion(
            response=response,
            user=user,
            conversation_id=conversation_id,
            function_map=function_map,
            functions=functions,
            current_persona=current_persona,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            model=model,
            allow_function_calls=allow_function_calls,
            conversation_manager=conversation_manager,
        )

    async def _handle_responses_completion(
        self,
        response,
        user,
        conversation_id,
        function_map,
        functions,
        current_persona,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        model,
        allow_function_calls,
        conversation_manager,
    ):
        if allow_function_calls:
            tool_messages = self._extract_responses_tool_calls(response)
            if tool_messages:
                results = []
                for tool_message in tool_messages:
                    result = await self.handle_function_call(
                        user,
                        conversation_id,
                        tool_message,
                        conversation_manager,
                        function_map=function_map,
                        functions=functions,
                        current_persona=current_persona,
                        temperature=temperature,
                        model=model,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                    )
                    if result:
                        results.append(result)
                if results:
                    return "\n".join(results)

        return self._get_response_text(response)

    async def _process_responses_stream(
        self,
        *,
        request_kwargs,
        user,
        conversation_id,
        function_map,
        functions,
        current_persona,
        conversation_manager,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        model,
        allow_function_calls,
    ):
        full_response = ""
        final_response = None

        async with self.client.responses.stream(**request_kwargs) as stream_response:
            async for event in stream_response:
                event_type = self._safe_get(event, "type") or self._safe_get(event, "event")
                if event_type == "response.output_text.delta":
                    delta = self._safe_get(event, "delta")
                    if delta:
                        yield delta
                        full_response += delta
                elif event_type == "response.refusal.delta":
                    delta = self._safe_get(event, "delta")
                    if delta:
                        yield delta
                        full_response += delta

            getter = getattr(stream_response, "get_final_response", None)
            if callable(getter):
                final_response = await getter()

        if allow_function_calls and final_response is not None:
            tool_messages = self._extract_responses_tool_calls(final_response)
            for tool_message in tool_messages:
                tool_result = await self.handle_function_call(
                    user,
                    conversation_id,
                    tool_message,
                    conversation_manager,
                    function_map,
                    functions,
                    current_persona,
                    temperature,
                    model,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                )
                if tool_result:
                    yield tool_result
                    full_response += tool_result

        if conversation_manager:
            conversation_manager.add_message(user, conversation_id, "assistant", full_response)
            self.logger.info("Full streaming response added to conversation history.")

    async def process_streaming_response(
        self,
        response: AsyncIterator[Dict],
        user,
        conversation_id,
        function_map,
        functions,
        current_persona,
        temperature,
        conversation_manager,
        model,
        top_p,
        frequency_penalty,
        presence_penalty,
    ):
        full_response = ""
        pending_tool_calls: Dict[int, Dict[str, Any]] = {}

        async for chunk in response:
            choices = self._safe_get(chunk, "choices") or []
            choice = choices[0] if choices else None
            delta = self._safe_get(choice, "delta") or {}
            delta_content = self._safe_get(delta, "content")
            text_delta = self._coerce_content_to_text(delta_content)
            if text_delta:
                yield text_delta
                full_response += text_delta

            tool_calls_delta = self._safe_get(delta, "tool_calls")
            if tool_calls_delta:
                for tool_call_delta in tool_calls_delta:
                    self._update_pending_tool_calls(pending_tool_calls, tool_call_delta)

            single_tool_call = self._safe_get(delta, "tool_call")
            if single_tool_call:
                self._update_pending_tool_calls(pending_tool_calls, single_tool_call)

            function_call_delta = self._safe_get(delta, "function_call")
            if function_call_delta:
                self.logger.info(
                    "Function call detected during streaming: %s",
                    function_call_delta,
                )
                result = await self.handle_function_call(
                    user,
                    conversation_id,
                    {"function_call": function_call_delta},
                    conversation_manager,
                    function_map,
                    functions,
                    current_persona,
                    temperature,
                    model,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                )
                if result:
                    yield result
                    full_response += str(result)

        if pending_tool_calls:
            tool_messages = self._finalize_pending_tool_calls(pending_tool_calls)
            for tool_message in tool_messages:
                self.logger.info(
                    "Processing aggregated streaming tool call: %s",
                    tool_message.get("function_call"),
                )
                tool_result = await self.handle_function_call(
                    user,
                    conversation_id,
                    tool_message,
                    conversation_manager,
                    function_map,
                    functions,
                    current_persona,
                    temperature,
                    model,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                )
                if tool_result:
                    yield tool_result
                    full_response += str(tool_result)

        if conversation_manager:
            conversation_manager.add_message(user, conversation_id, "assistant", full_response)
            self.logger.info("Full streaming response added to conversation history.")

    async def handle_function_call(
        self,
        user,
        conversation_id,
        message,
        conversation_manager,
        function_map,
        functions,
        current_persona,
        temperature,
        model,
        top_p,
        frequency_penalty,
        presence_penalty,
    ):
        normalized = self._normalize_tool_call(message)
        if not normalized and isinstance(message, dict):
            normalized = self._normalize_tool_call(dict(message))

        if not normalized:
            self.logger.warning("Received unrecognized tool payload: %s", message)
            return None

        tool_type = normalized.get("type") or (
            "function" if normalized.get("function_call") else None
        )

        if tool_type in {"code_interpreter", "file_search"}:
            formatted_output = self._format_builtin_tool_output(normalized)
            if formatted_output:
                self.logger.info(
                    "Returning output from OpenAI %s tool call.", tool_type
                )
                return formatted_output
            self.logger.info(
                "OpenAI %s tool call produced no textual output.", tool_type
            )
            return None

        function_payload = normalized.get("function_call")
        if not function_payload:
            self.logger.warning("Tool payload missing function_call: %s", normalized)
            return None

        self.logger.info("Handling function call: %s", function_payload)
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
            self.config_manager
        )

        if tool_response:
            self.logger.info(f"Tool response generated: {tool_response}")
            return tool_response

        self.logger.warning("No tool response generated; sending default message.")
        return "Sorry, I couldn't process the function call. Please try again or provide more context."

async def generate_response(
    config_manager: ConfigManager,
    messages: List[Dict[str, str]],
    model: str = None,
    max_tokens: int = 4000,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stream: bool = True,
    current_persona=None,
    conversation_manager=None,
    user=None,
    conversation_id=None,
    functions=None,
    reasoning_effort: Optional[str] = None,
    function_calling: Optional[bool] = None,
    parallel_tool_calls: Optional[bool] = None,
    tool_choice: Optional[Any] = None,
    json_mode: Optional[Any] = None,
    json_schema: Optional[Any] = None
) -> Union[str, AsyncIterator[str]]:
    generator = OpenAIGenerator(config_manager)
    return await generator.generate_response(
        messages,
        model,
        max_tokens,
        max_output_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        stream,
        current_persona,
        conversation_manager,
        user,
        conversation_id,
        functions,
        reasoning_effort,
        function_calling,
        parallel_tool_calls,
        tool_choice,
        json_mode,
        json_schema
    )

async def process_streaming_response(response: AsyncIterator[Dict]) -> str:
    content = ""
    async for chunk in response:
        delta_content = chunk.choices[0].delta.content
        if delta_content is None:
            continue
        if isinstance(delta_content, str):
            content += delta_content
        elif isinstance(delta_content, list):
            for entry in delta_content:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if text:
                        content += str(text)
                elif entry is not None:
                    content += str(entry)
        else:
            content += str(delta_content)
    return content
