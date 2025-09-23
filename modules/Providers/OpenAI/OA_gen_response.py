# modules/Providers/OpenAI/OA_gen_response.py

from openai import AsyncOpenAI
from ATLAS.model_manager import ModelManager

from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Union, AsyncIterator, Optional
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
        function_calling: Optional[bool] = None
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

            if function_calling is None:
                allow_function_calls = bool(settings.get("function_calling", True))
            else:
                allow_function_calls = bool(function_calling)

            json_mode_enabled = bool(settings.get("json_mode", False))

            self.logger.info(f"Starting API call to OpenAI with model {model}")
            self.logger.info(f"Current persona: {current_persona}")

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

            function_call_mode = None
            if functions:
                function_call_mode = "auto" if allow_function_calls else "none"
                self.logger.info(
                    "Automatic tool calling %s for this request.",
                    "enabled" if allow_function_calls else "disabled",
                )

            request_kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=stream,
                functions=functions,
                function_call=function_call_mode,
            )

            if json_mode_enabled:
                request_kwargs["response_format"] = {"type": "json_object"}
                self.logger.info("Requesting JSON-formatted responses from OpenAI.")

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
                    conversation_manager
                )
            else:
                message = response.choices[0].message
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            parts.append(part.get("text", ""))
                        elif hasattr(part, "text"):
                            parts.append(getattr(part, "text") or "")
                    content = "".join(parts)
                else:
                    content = content if content is not None else ""
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
                return content

        except Exception as e:
            self.logger.error(f"Error in OpenAI API call: {str(e)}", exc_info=True)
            raise

    async def process_streaming_response(
        self,
        response: AsyncIterator[Dict],
        user,
        conversation_id,
        function_map,
        functions,
        current_persona,
        temperature,
        conversation_manager
    ):
        full_response = ""
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                full_response += chunk.choices[0].delta.content
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
                    response.model,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                )
                yield result
                full_response += result

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
    function_calling: Optional[bool] = None
) -> Union[str, AsyncIterator[str]]:
    generator = OpenAIGenerator(config_manager)
    return await generator.generate_response(
        messages,
        model,
        max_tokens,
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
        function_calling
    )

async def process_streaming_response(response: AsyncIterator[Dict]) -> str:
    content = ""
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content
    return content
