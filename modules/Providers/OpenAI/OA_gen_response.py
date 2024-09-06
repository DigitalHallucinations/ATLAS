# modules/Providers/OpenAI/OA_gen_response.py

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Union, AsyncIterator
from modules.config import ConfigManager
from modules.Tools.ToolManager import load_function_map_from_current_persona, load_functions_from_json, use_tool
from SCOUT.model_manager import ModelManager  

class OpenAIGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = self.config_manager.logger  
        self.api_key = self.config_manager.get_openai_api_key()
        if not self.api_key:
            self.logger.error("OpenAI API key not found in configuration")
            raise ValueError("OpenAI API key not found in configuration")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model_manager = ModelManager(config_manager)  

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, messages: List[Dict[str, str]], model: str = None, max_tokens: int = 4000, temperature: float = 0.0, stream: bool = True, current_persona=None, conversation_manager=None, user=None, conversation_id=None, functions=None) -> Union[str, AsyncIterator[str]]:
        try:
            current_model = self.model_manager.get_current_model()
            
            if model and model != current_model:
                self.model_manager.set_model(model, "OpenAI")
                current_model = model
            elif not model:
                model = current_model

            self.logger.debug(f"Starting API call to OpenAI with model {model}")

            if functions is None and current_persona:
                functions = load_functions_from_json(current_persona)
            self.logger.debug(f"Loaded functions: {functions}")
            function_map = load_function_map_from_current_persona(current_persona) if current_persona else None

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=temperature,
                stream=stream,
                functions=functions,
                function_call="auto" if functions else None
            )
            
            self.logger.debug(f"Received response: {response}")
            
            if stream:
                return self.process_streaming_response(response, user, conversation_id, function_map, functions, current_persona, temperature, conversation_manager)
            else:
                message = response.choices[0].message
                if hasattr(message, 'function_call') and message.function_call:
                    self.logger.debug(f"Function call detected: {message.function_call}")
                    return await self.handle_function_call(user, conversation_id, message, conversation_manager, function_map, functions, current_persona, temperature, model)
                return message.content

        except Exception as e:
            self.logger.error(f"Error in OpenAI API call: {str(e)}")
            raise

    async def process_streaming_response(self, response: AsyncIterator[Dict], user, conversation_id, function_map, functions, current_persona, temperature, conversation_manager):
        full_response = ""
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                full_response += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.function_call:
                function_call = chunk.choices[0].delta.function_call
                result = await self.handle_function_call(user, conversation_id, {"function_call": function_call}, conversation_manager, function_map, functions, current_persona, temperature, response.model)
                yield result
                full_response += result

        if conversation_manager:
            conversation_manager.add_message(user, conversation_id, "assistant", full_response)

    async def handle_function_call(self, user, conversation_id, message, conversation_manager, function_map, functions, current_persona, temperature, model):
        tool_response = await use_tool(user, conversation_id, message, conversation_manager, function_map, functions, current_persona, temperature, 1.0, conversation_manager, self.config_manager)
        
        if tool_response:
            return tool_response
        
        return "Sorry, I couldn't process the function call. Please try again or provide more context."

async def generate_response(config_manager: ConfigManager, messages: List[Dict[str, str]], model: str = None, max_tokens: int = 4000, temperature: float = 0.0, stream: bool = True, current_persona=None, conversation_manager=None, user=None, conversation_id=None, functions=None) -> Union[str, AsyncIterator[str]]:
    generator = OpenAIGenerator(config_manager)
    return await generator.generate_response(messages, model, max_tokens, temperature, stream, current_persona, conversation_manager, user, conversation_id, functions)

async def process_streaming_response(response: AsyncIterator[Dict]) -> str:
    content = ""
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content
    return content