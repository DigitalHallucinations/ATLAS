# modules/Providers/Anthropic/Anthropic_gen_response.py

"""Async response generation helpers for the Anthropic provider."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from anthropic import APIError, AsyncAnthropic, RateLimitError

from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger


@dataclass(frozen=True)
class _RetrySchedule:
    """Simple container describing retry attempt and delay information."""

    attempts: int
    base_delay: int


def _normalise_positive_int(value: Any, fallback: int, *, minimum: int = 0) -> int:
    """Validate and coerce integer configuration values."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(fallback, minimum)

    return max(parsed, minimum)

class AnthropicGenerator:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
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

        settings: Dict[str, Any] = {}
        getter = getattr(self.config_manager, "get_anthropic_settings", None)
        if callable(getter):
            try:
                settings = getter() or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(
                    "Unable to load persisted Anthropic settings: %s", exc, exc_info=True
                )

        if isinstance(settings, dict):
            model = settings.get("model")
            if isinstance(model, str) and model.strip():
                self.default_model = model.strip()

            stream = settings.get("stream")
            if stream is not None:
                self.streaming_enabled = bool(stream)

            fn_calling = settings.get("function_calling")
            if fn_calling is not None:
                self.function_calling_enabled = bool(fn_calling)

            timeout = settings.get("timeout")
            if isinstance(timeout, (int, float)) and timeout > 0:
                self.timeout = int(timeout)

            max_retries = settings.get("max_retries")
            if isinstance(max_retries, (int, float)) and max_retries >= 0:
                self.max_retries = int(max_retries)

            retry_delay = settings.get("retry_delay")
            if isinstance(retry_delay, (int, float)) and retry_delay >= 0:
                self.retry_delay = int(retry_delay)

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

            response = await self._create_message_with_retry(message_params)
            if self.function_calling_enabled:
                return self.process_function_call(response)
            return self._extract_text_content(response.content)

        except asyncio.TimeoutError:
            self.logger.error(f"Request timed out after {self.timeout} seconds")
            raise
        except RateLimitError as e:
            self.logger.warning(f"Rate limit reached. Retrying: {str(e)}")
            raise
        except APIError as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error with Anthropic: {str(e)}")
            raise

    async def _create_message_with_retry(self, message_params: Dict[str, Any]):
        """Send a non-streaming request honouring the configured retry policy."""

        schedule = self._build_retry_schedule()
        base_delay = max(1, schedule.base_delay)
        attempt = 1

        while True:
            try:
                return await asyncio.wait_for(
                    self.client.messages.create(**message_params),
                    timeout=self.timeout,
                )
            except RateLimitError as exc:
                if attempt >= schedule.attempts:
                    self.logger.warning(
                        "Rate limit reached during attempt %s; no more retries.", attempt
                    )
                    raise

                delay = min(base_delay * (2 ** (attempt - 1)), 60)
                self.logger.warning(
                    "Rate limit reached during attempt %s. Retrying in %s seconds.",
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1

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
        schedule = self._build_retry_schedule()
        base_delay = max(1, schedule.base_delay)

        attempt = 1
        timeout_ctx = getattr(asyncio, "timeout", None)
        while True:
            try:
                if timeout_ctx is not None:
                    async with timeout_ctx(self.timeout):
                        async with self.client.messages.stream(**message_params) as stream:
                            async for event in stream:
                                for chunk in self._translate_stream_event(event):
                                    yield chunk

                            final_response = await stream.get_final_response()
                            if self.function_calling_enabled:
                                function_call = self.process_function_call(final_response)
                                if isinstance(function_call, dict):
                                    yield function_call
                            return
                else:
                    stream_cm = self.client.messages.stream(**message_params)
                    stream = await asyncio.wait_for(
                        stream_cm.__aenter__(), timeout=self.timeout
                    )
                    try:
                        async for event in self._iterate_stream_with_wait_for(stream):
                            for chunk in self._translate_stream_event(event):
                                yield chunk

                        final_response = await asyncio.wait_for(
                            stream.get_final_response(), timeout=self.timeout
                        )
                        if self.function_calling_enabled:
                            function_call = self.process_function_call(final_response)
                            if isinstance(function_call, dict):
                                yield function_call
                        return
                    finally:
                        await asyncio.wait_for(
                            stream_cm.__aexit__(None, None, None), timeout=self.timeout
                        )

            except asyncio.TimeoutError:
                self.logger.error(
                    "Streaming request timed out after %s seconds without receiving data",
                    self.timeout,
                )
                raise
            except RateLimitError as exc:
                if attempt >= schedule.attempts:
                    self.logger.warning(
                        "Rate limit reached during streaming attempt %s; no more retries.",
                        attempt,
                    )
                    raise

                delay = min(base_delay * (2 ** (attempt - 1)), 60)
                self.logger.warning(
                    "Rate limit reached during streaming attempt %s. Retrying in %s seconds.",
                    attempt,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def _iterate_stream_with_wait_for(self, stream):
        while True:
            try:
                event = await asyncio.wait_for(stream.__anext__(), timeout=self.timeout)
            except StopAsyncIteration:
                break
            yield event

    def _translate_stream_event(self, event):
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
                return
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "tool_use":
                block_dict = block if isinstance(block, dict) else None
                tool_payload = {
                    "function_call": {
                        "name": getattr(block, "name", None)
                        or (block_dict.get("name") if block_dict else None),
                        "arguments": getattr(block, "input", None)
                        or (block_dict.get("input") if block_dict else {}),
                        "id": getattr(block, "id", None)
                        or (block_dict.get("id") if block_dict else None),
                    }
                }
                yield tool_payload

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

    async def process_response(
        self, response: Union[str, AsyncIterator[Union[str, Dict[str, Any]]]]
    ) -> Union[str, Dict[str, Any]]:
        """Consume a streamed response and return aggregated text or structured data.

        The first non-text chunk (typically a function-call payload) is returned
        immediately. Otherwise, all text fragments are concatenated and returned
        as a single string.
        """

        if isinstance(response, str):
            return response

        collected_text: List[str] = []
        async for chunk in response:
            if isinstance(chunk, str):
                collected_text.append(chunk)
                continue

            if isinstance(chunk, dict):
                return chunk

            if isinstance(chunk, Mapping):
                return dict(chunk)

            return chunk  # type: ignore[return-value]

        return "".join(collected_text)

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
        timeout_value = _normalise_positive_int(timeout, self.timeout, minimum=1)
        self.timeout = timeout_value
        self.logger.info(f"Timeout set to: {timeout_value} seconds")

    def set_max_retries(self, max_retries: int):
        retries_value = _normalise_positive_int(max_retries, self.max_retries, minimum=0)
        self.max_retries = retries_value
        self.logger.info(
            f"Max retries (additional attempts) set to: {retries_value}"
        )

    def set_retry_delay(self, retry_delay: int):
        retry_delay_value = _normalise_positive_int(retry_delay, self.retry_delay, minimum=0)
        self.retry_delay = retry_delay_value
        self.logger.info(f"Retry delay set to: {retry_delay_value} seconds")

    def _build_retry_schedule(self) -> _RetrySchedule:
        configured_attempts = _normalise_positive_int(
            self.max_retries, self.max_retries, minimum=0
        )
        attempts = max(1, configured_attempts + 1)
        delay = _normalise_positive_int(self.retry_delay, self.retry_delay, minimum=0)
        return _RetrySchedule(attempts=attempts, base_delay=delay or 1)

def setup_anthropic_generator(config_manager: Optional[ConfigManager] = None):
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

async def process_response(
    response: Union[str, AsyncIterator[Union[str, Dict[str, Any]]]]
) -> Union[str, Dict[str, Any]]:
    generator = setup_anthropic_generator()
    return await generator.process_response(response)

def generate_response_sync(
    config_manager: ConfigManager,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    stream: Optional[bool] = None,
    **kwargs
) -> Union[str, Dict[str, Any]]:
    """Synchronous wrapper for :meth:`AnthropicGenerator.generate_response`.

    Returns either the aggregated text content or the first structured payload
    encountered when streaming is enabled.
    """

    generator = setup_anthropic_generator(config_manager)
    resolved_stream = generator.streaming_enabled if stream is None else bool(stream)

    async def _execute() -> Union[str, Dict[str, Any]]:
        response = await generator.generate_response(
            messages,
            model,
            stream=resolved_stream,
            **kwargs,
        )
        if resolved_stream:
            return await generator.process_response(response)
        return response  # type: ignore[return-value]

    coroutine = _execute()
    try:
        return asyncio.run(coroutine)
    except RuntimeError as exc:
        coroutine.close()
        if "asyncio.run() cannot be called" in str(exc):
            raise RuntimeError(
                "generate_response_sync cannot be called while an event loop is running; "
                "use the async Anthropic API instead."
            ) from exc
        raise
