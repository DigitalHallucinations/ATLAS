# modules/Providers/Anthropic/Anthropic_gen_response.py

import asyncio
from typing import List, Dict, Union, AsyncIterator, Optional, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from anthropic import AsyncAnthropic, APIError, RateLimitError
import json

class AnthropicGenerator:
    def __init__(self, config_manager=ConfigManager):
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.api_key = self.config_manager.get_anthropic_api_key()
        if not self.api_key:
            self.logger.error("Anthropic API key not found in configuration")
            raise ValueError("Anthropic API key not found in configuration")
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.default_model = "claude-3-opus-20240229"
        self.streaming_enabled = True
        self.function_calling_enabled = False
        self.max_retries = 3
        self.retry_delay = 5
        self.timeout = 60

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError)
    )
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stream: Optional[bool] = None,
        current_persona=None,
        functions: Optional[List[Dict]] = None,
        **kwargs
    ) -> Union[str, AsyncIterator[Union[str, Dict[str, Any]]]]:
        try:
            model = model or self.default_model
            stream = self.streaming_enabled if stream is None else stream

            self.logger.info(f"Generating response with Anthropic AI using model: {model}")

            system_prompt, message_payload = self._prepare_messages(messages)

            message_params = {
                "model": model,
                "messages": message_payload,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }

            if system_prompt:
                message_params["system"] = system_prompt

            if self.function_calling_enabled and functions:
                message_params["tools"] = functions

            if stream:
                return self.process_streaming_response(message_params)
            else:
                response = await asyncio.wait_for(
                    self.client.messages.create(**message_params),
                    timeout=self.timeout
                )
                return self.process_function_call(response) if self.function_calling_enabled else self._extract_text_content(response.content)

        except RateLimitError as e:
            self.logger.warning(f"Rate limit reached. Retrying: {str(e)}")
            raise
        except APIError as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise
        except asyncio.TimeoutError:
            self.logger.error(f"Request timed out after {self.timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error with Anthropic: {str(e)}")
            raise

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        system_messages: List[str] = []
        formatted_messages: List[Dict[str, Any]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_messages.append(content)
                else:
                    system_messages.append(str(content))
                continue

            if isinstance(content, list):
                formatted_content = content
            elif isinstance(content, str):
                formatted_content = [{"type": "text", "text": content}]
            else:
                formatted_content = [{"type": "text", "text": str(content)}]

            formatted_messages.append({
                "role": role,
                "content": formatted_content
            })

        system_prompt = "\n\n".join(system_messages).strip()
        return system_prompt, formatted_messages

    async def process_streaming_response(self, message_params: Dict[str, Any]) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        async with self.client.messages.stream(**message_params) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta and getattr(delta, "type", None) == "text_delta":
                        text = getattr(delta, "text", "")
                        if text:
                            yield text
                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block is None:
                        continue
                    block_type = getattr(block, "type", None)
                    if block_type is None and isinstance(block, dict):
                        block_type = block.get("type")
                    if block_type == "tool_use":
                        block_dict = block if isinstance(block, dict) else None
                        tool_payload = {
                            "function_call": {
                                "name": getattr(block, "name", None) or (block_dict.get("name") if block_dict else None),
                                "arguments": getattr(block, "input", None) or (block_dict.get("input") if block_dict else {}),
                                "id": getattr(block, "id", None) or (block_dict.get("id") if block_dict else None),
                            }
                        }
                        yield tool_payload

            final_response = await stream.get_final_response()
            if self.function_calling_enabled:
                function_call = self.process_function_call(final_response)
                if isinstance(function_call, dict):
                    yield function_call

    def _extract_text_content(self, content_blocks: List[Any]) -> str:
        collected_text: List[str] = []
        for block in content_blocks or []:
            block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if block_type == "text":
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    collected_text.append(text)
        return "".join(collected_text)

    async def process_response(self, response: Union[str, AsyncIterator[Union[str, Dict[str, Any]]]]) -> str:
        if isinstance(response, str):
            return response
        else:
            full_response = ""
            async for chunk in response:
                if isinstance(chunk, str):
                    full_response += chunk
            return full_response

    def process_function_call(self, response):
        content_blocks = getattr(response, "content", None) or []
        for block in content_blocks:
            block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if block_type == "tool_use":
                name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
                arguments = getattr(block, "input", None) or (block.get("input") if isinstance(block, dict) else None)
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                return {
                    "function_call": {
                        "name": name,
                        "arguments": arguments
                    }
                }

        text_content = self._extract_text_content(content_blocks)
        if text_content:
            return text_content
        return ""

    def set_streaming(self, enabled: bool):
        self.streaming_enabled = enabled
        self.logger.info(f"Streaming {'enabled' if enabled else 'disabled'}")

    def set_function_calling(self, enabled: bool):
        self.function_calling_enabled = enabled
        self.logger.info(f"Function calling {'enabled' if enabled else 'disabled'}")

    def set_default_model(self, model: str):
        self.default_model = model
        self.logger.info(f"Default model set to: {model}")

    def set_timeout(self, timeout: int):
        self.timeout = timeout
        self.logger.info(f"Timeout set to: {timeout} seconds")

    def set_max_retries(self, max_retries: int):
        self.max_retries = max_retries
        self.logger.info(f"Max retries set to: {max_retries}")

    def set_retry_delay(self, retry_delay: int):
        self.retry_delay = retry_delay
        self.logger.info(f"Retry delay set to: {retry_delay} seconds")

def setup_anthropic_generator(config_manager: ConfigManager):
    return AnthropicGenerator(config_manager)

async def generate_response(
    config_manager: ConfigManager,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    stream: Optional[bool] = None,
    current_persona=None,
    functions: Optional[List[Dict]] = None,
    **kwargs
) -> Union[str, AsyncIterator[Union[str, Dict[str, Any]]]]:
    generator = setup_anthropic_generator(config_manager)
    return await generator.generate_response(messages, model, max_tokens, temperature, stream, current_persona, functions, **kwargs)

async def process_response(response: Union[str, AsyncIterator[Union[str, Dict[str, Any]]]]) -> str:
    generator = AnthropicGenerator(ConfigManager())
    return await generator.process_response(response)

def generate_response_sync(
    config_manager: ConfigManager,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    stream: Optional[bool] = None,
    **kwargs
) -> str:
    """
    Synchronous version of generate_response for compatibility with non-async code.
    """
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(generate_response(config_manager, messages, model, stream=stream, **kwargs))
    if stream:
        return loop.run_until_complete(process_response(response))
    return response
