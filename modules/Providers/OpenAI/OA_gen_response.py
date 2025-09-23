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
        json_mode: Optional[Any] = None
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

            force_json_mode = self._should_force_json(
                json_mode if json_mode is not None else settings.get("json_mode")
            )

            self.logger.info(f"Starting API call to OpenAI with model {model}")
            self.logger.info(f"Current persona: {current_persona}")
            if force_json_mode:
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

            # Log what is being sent to the API
            self.logger.info(f"Sending functions to OpenAI API: {functions}")
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
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                )

            function_call_mode = None
            if functions:
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

            if functions:
                request_kwargs["functions"] = functions
            if function_call_mode is not None:
                request_kwargs["function_call"] = function_call_mode
            if force_json_mode:
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
                if hasattr(message, 'function_call') and message.function_call:
                    self.logger.info(f"Function call detected in response: {message.function_call}")
                    return await self.handle_function_call(
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
                    )
                self.logger.info("No function call detected in response.")
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
        if not functions:
            return []

        tools = []
        for function in functions:
            if isinstance(function, dict):
                tools.append({"type": "function", "function": function})
        return tools

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

    def _stringify_function_arguments(self, arguments) -> str:
        if arguments is None:
            return "{}"
        if isinstance(arguments, str):
            return arguments
        try:
            return json.dumps(arguments)
        except (TypeError, ValueError):
            return str(arguments)

    def _extract_responses_tool_calls(self, response) -> List[Dict[str, Dict[str, str]]]:
        tool_messages = []
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
                if not call:
                    continue
                function_payload = self._safe_get(call, "function") or {}
                name = self._safe_get(function_payload, "name") or self._safe_get(call, "name")
                arguments = self._safe_get(function_payload, "arguments")
                if arguments is None:
                    arguments = self._safe_get(call, "arguments")
                if not name:
                    continue
                tool_messages.append(
                    {
                        "function_call": {
                            "name": name,
                            "arguments": self._stringify_function_arguments(arguments),
                        }
                    }
                )

        return tool_messages

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
        async for chunk in response:
            delta_content = chunk.choices[0].delta.content
            text_delta = self._coerce_content_to_text(delta_content)
            if text_delta:
                yield text_delta
                full_response += text_delta
            elif chunk.choices[0].delta.function_call:
                function_call = chunk.choices[0].delta.function_call
                self.logger.info(f"Function call detected during streaming: {function_call}")
                result = await self.handle_function_call(
                    user,
                    conversation_id,
                    {"function_call": function_call},
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
        self.logger.info(f"Handling function call: {message.get('function_call')}")
        tool_response = await use_tool(
            user,
            conversation_id,
            message,
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
    json_mode: Optional[Any] = None
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
        json_mode
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
