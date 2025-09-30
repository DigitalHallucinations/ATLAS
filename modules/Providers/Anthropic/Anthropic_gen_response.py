# modules/Providers/Anthropic/Anthropic_gen_response.py

"""Async response generation helpers for the Anthropic provider."""

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
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


def _normalise_probability(
    value: Any,
    fallback: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Normalise probability-like parameters to the expected range."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback

    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _normalise_optional_positive_int(
    value: Any,
    fallback: Optional[int],
    *,
    minimum: int = 1,
) -> Optional[int]:
    """Return ``None`` or a positive integer respecting the configured minimum."""

    if value in {None, ""}:
        return fallback if fallback is None else max(int(fallback), minimum)

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback if fallback is None else max(int(fallback), minimum)

    if parsed < minimum:
        return fallback if fallback is None else max(int(fallback), minimum)

    return parsed


def _normalise_optional_bounded_int(
    value: Any,
    fallback: Optional[int],
    *,
    minimum: int,
    maximum: int,
) -> Optional[int]:
    """Return ``None`` or an integer constrained to the provided bounds."""

    if value in {None, ""}:
        return None if fallback is None else max(min(int(fallback), maximum), minimum)

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None if fallback is None else max(min(int(fallback), maximum), minimum)

    if parsed < minimum or parsed > maximum:
        return None if fallback is None else max(min(int(fallback), maximum), minimum)

    return parsed


def _normalise_stop_sequences(value: Any) -> List[str]:
    """Convert stop sequences into a cleaned list limited to four entries."""

    if value is None or value == "":
        return []

    if isinstance(value, str):
        entries = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        entries = []
        for item in value:
            if item in {None, ""}:
                continue
            text = str(item).strip()
            if text:
                entries.append(text)
    else:
        return []

    return entries[:4]


def _normalise_metadata(value: Any) -> Dict[str, str]:
    """Normalise metadata inputs to Anthropic's expected mapping structure."""

    if value in (None, "", {}):
        return {}

    metadata: Dict[str, str] = {}

    def _record(entry_key: Any, entry_value: Any) -> None:
        if entry_key in {None, ""}:
            return
        key_text = str(entry_key).strip()
        if not key_text:
            return
        metadata[key_text] = "" if entry_value is None else str(entry_value).strip()

    if isinstance(value, Mapping):
        for key, val in value.items():
            _record(key, val)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        parsed: Any
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, Mapping):
            for key, val in parsed.items():
                _record(key, val)
        elif isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
            for entry in parsed:
                if isinstance(entry, Mapping):
                    for key, val in entry.items():
                        _record(key, val)
                elif isinstance(entry, Sequence) and len(entry) == 2:
                    _record(entry[0], entry[1])
        else:
            segments = [segment.strip() for segment in text.replace("\n", ",").split(",")]
            for segment in segments:
                if not segment or "=" not in segment:
                    continue
                key, val = segment.split("=", 1)
                _record(key, val)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for entry in value:
            if isinstance(entry, Mapping):
                for key, val in entry.items():
                    _record(key, val)
            elif isinstance(entry, Sequence) and len(entry) == 2:
                _record(entry[0], entry[1])

    if len(metadata) > 16:
        return dict(list(metadata.items())[:16])

    return metadata


def _normalise_tool_choice(value: Any, name: Any) -> Tuple[str, Optional[str]]:
    """Convert tool selection preferences into Anthropic's tool_choice schema."""

    choice: Optional[str]
    provided_name = name

    if isinstance(value, Mapping):
        choice = str(value.get("type", "")).strip().lower()
        if value.get("name") is not None:
            provided_name = value.get("name")
    elif isinstance(value, str):
        choice = value.strip().lower()
    else:
        choice = None

    alias_map = {"required": "any"}
    resolved_choice = alias_map.get(choice or "", choice or "")

    if resolved_choice not in {"auto", "any", "none", "tool"}:
        resolved_choice = "auto"

    if resolved_choice == "tool":
        tool_name = None if provided_name in {None, ""} else str(provided_name).strip()
        if not tool_name:
            return "auto", None
        return "tool", tool_name

    return resolved_choice, None


def _build_tool_choice_payload(choice: str, tool_name: Optional[str]) -> Optional[Dict[str, str]]:
    """Return the payload for Anthropic's ``tool_choice`` argument."""

    if choice == "auto" or not choice:
        return None

    if choice == "tool" and tool_name:
        return {"type": "tool", "name": tool_name}

    return {"type": choice}


def _build_thinking_payload(enabled: bool, budget: Optional[int]) -> Optional[Dict[str, Any]]:
    """Construct the ``thinking`` parameter respecting optional budget hints."""

    if not enabled:
        return None

    payload: Dict[str, Any] = {"type": "enabled"}
    if isinstance(budget, (int, float)) and int(budget) > 0:
        payload["budget_tokens"] = int(budget)
    return payload


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
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k: Optional[int] = None
        self.max_output_tokens: Optional[int] = None
        self.max_retries = 3
        self.retry_delay = 5
        self.timeout = 60
        self.stop_sequences: List[str] = []
        self.tool_choice: str = "auto"
        self.tool_choice_name: Optional[str] = None
        self.metadata: Dict[str, str] = {}
        self.thinking_enabled: bool = False
        self.thinking_budget: Optional[int] = None

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

            stored_temperature = settings.get("temperature")
            if isinstance(stored_temperature, (int, float)):
                self.temperature = _normalise_probability(
                    stored_temperature, self.temperature
                )

            stored_top_p = settings.get("top_p")
            if isinstance(stored_top_p, (int, float)):
                self.top_p = _normalise_probability(
                    stored_top_p, self.top_p
                )

            stored_top_k = settings.get("top_k")
            if stored_top_k is not None:
                self.top_k = _normalise_optional_bounded_int(
                    stored_top_k,
                    self.top_k,
                    minimum=1,
                    maximum=500,
                )

            stored_max_output = settings.get("max_output_tokens")
            if stored_max_output is not None:
                self.max_output_tokens = _normalise_optional_positive_int(
                    stored_max_output, self.max_output_tokens
                )
            else:
                self.max_output_tokens = None

            stored_stop_sequences = settings.get("stop_sequences")
            if stored_stop_sequences is not None:
                self.stop_sequences = _normalise_stop_sequences(stored_stop_sequences)

            stored_tool_choice = settings.get("tool_choice")
            stored_tool_choice_name = settings.get("tool_choice_name")
            choice, choice_name = _normalise_tool_choice(
                stored_tool_choice,
                stored_tool_choice_name,
            )
            self.tool_choice = choice
            self.tool_choice_name = choice_name

            stored_metadata = settings.get("metadata")
            if stored_metadata is not None:
                self.metadata = _normalise_metadata(stored_metadata)

            stored_thinking = settings.get("thinking")
            if stored_thinking is not None:
                self.thinking_enabled = bool(stored_thinking)

            stored_thinking_budget = settings.get("thinking_budget")
            if stored_thinking_budget is not None:
                self.thinking_budget = _normalise_optional_positive_int(
                    stored_thinking_budget,
                    self.thinking_budget,
                )

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
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        current_persona=None,
        functions: Optional[List[Dict]] = None,
        stop_sequences: Optional[Any] = None,
        **kwargs
    ) -> Union[str, AsyncIterator[Union[str, Dict[str, Any]]]]:
        try:
            model = model or self.default_model
            stream = self.streaming_enabled if stream is None else stream
            if "max_tokens" in kwargs and max_output_tokens is None:
                max_output_tokens = kwargs.pop("max_tokens")
            temperature = _normalise_probability(
                temperature if temperature is not None else self.temperature,
                self.temperature,
            )
            top_p = _normalise_probability(
                top_p if top_p is not None else self.top_p,
                self.top_p,
            )
            resolved_max_output_tokens = _normalise_optional_positive_int(
                max_output_tokens,
                self.max_output_tokens,
            )

            resolved_top_k = _normalise_optional_bounded_int(
                top_k if top_k is not None else self.top_k,
                self.top_k,
                minimum=1,
                maximum=500,
            )

            resolved_stop_sequences = _normalise_stop_sequences(
                stop_sequences if stop_sequences is not None else self.stop_sequences
            )

            self.logger.info(f"Generating response with Anthropic AI using model: {model}")

            system_prompt, message_payload = self._prepare_messages(messages)

            message_params = {
                "model": model,
                "messages": message_payload,
                "temperature": temperature,
                "top_p": top_p,
                **kwargs
            }

            if resolved_max_output_tokens is not None:
                message_params["max_output_tokens"] = resolved_max_output_tokens

            if resolved_top_k is not None:
                message_params["top_k"] = resolved_top_k

            if resolved_stop_sequences:
                message_params["stop_sequences"] = resolved_stop_sequences

            if system_prompt:
                message_params["system"] = system_prompt

            if self.metadata:
                message_params["metadata"] = dict(self.metadata)

            thinking_payload = _build_thinking_payload(
                self.thinking_enabled,
                self.thinking_budget,
            )
            if thinking_payload is not None:
                message_params["thinking"] = thinking_payload

            if self.function_calling_enabled and functions:
                message_params["tools"] = functions
                tool_choice_payload = _build_tool_choice_payload(
                    self.tool_choice,
                    self.tool_choice_name,
                )
                if tool_choice_payload is not None:
                    message_params["tool_choice"] = tool_choice_payload

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
                if not getattr(self.logger, "propagate", True):
                    logging.getLogger().error(
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

    def set_temperature(self, temperature: float):
        self.temperature = _normalise_probability(temperature, self.temperature)
        self.logger.info(f"Temperature set to: {self.temperature}")

    def set_top_p(self, top_p: float):
        self.top_p = _normalise_probability(top_p, self.top_p)
        self.logger.info(f"Top-p set to: {self.top_p}")

    def set_top_k(self, top_k: Optional[int]):
        if top_k in {None, ""}:
            self.top_k = None
            self.logger.info("Top-k cleared (default behavior will be used)")
            return

        resolved = _normalise_optional_bounded_int(
            top_k,
            self.top_k,
            minimum=1,
            maximum=500,
        )
        self.top_k = resolved
        if resolved is None:
            self.logger.info("Top-k cleared (default behavior will be used)")
        else:
            self.logger.info(f"Top-k set to: {resolved}")

    def set_max_output_tokens(self, max_output_tokens: Optional[int]):
        if max_output_tokens in {None, ""}:
            self.max_output_tokens = None
            self.logger.info("Max output tokens cleared (no limit)")
            return
        self.max_output_tokens = _normalise_optional_positive_int(
            max_output_tokens, self.max_output_tokens
        )
        if self.max_output_tokens is None:
            self.logger.info("Max output tokens cleared (no limit)")
        else:
            self.logger.info(
                "Max output tokens set to: %s", self.max_output_tokens
            )

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

    def set_stop_sequences(self, stop_sequences: Optional[Any]):
        sequences = _normalise_stop_sequences(stop_sequences)
        self.stop_sequences = sequences
        if sequences:
            self.logger.info(
                "Stop sequences set to: %s",
                ", ".join(sequences),
            )
        else:
            self.logger.info("Stop sequences cleared")

    def set_tool_choice(self, tool_choice: Any, tool_name: Optional[str] = None):
        choice, name = _normalise_tool_choice(tool_choice, tool_name)
        self.tool_choice = choice
        self.tool_choice_name = name
        if choice == "tool" and name:
            self.logger.info("Tool choice set to specific tool: %s", name)
        else:
            self.logger.info("Tool choice set to: %s", choice)

    def set_metadata(self, metadata: Optional[Any]):
        self.metadata = _normalise_metadata(metadata)
        if self.metadata:
            self.logger.info("Metadata set with %s entries", len(self.metadata))
        else:
            self.logger.info("Metadata cleared")

    def set_thinking(self, enabled: Optional[bool], budget: Optional[Any] = None):
        if enabled is not None:
            self.thinking_enabled = bool(enabled)
        if budget is not None:
            self.thinking_budget = _normalise_optional_positive_int(budget, self.thinking_budget)
        if self.thinking_enabled:
            if self.thinking_budget:
                self.logger.info(
                    "Thinking enabled with budget: %s tokens",
                    self.thinking_budget,
                )
            else:
                self.logger.info("Thinking enabled with provider defaults")
        else:
            self.logger.info("Thinking disabled")

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
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    stream: Optional[bool] = None,
    current_persona=None,
    functions: Optional[List[Dict]] = None,
    stop_sequences: Optional[Any] = None,
    **kwargs
) -> Union[str, AsyncIterator[Union[str, Dict[str, Any]]]]:
    generator = setup_anthropic_generator(config_manager)
    return await generator.generate_response(
        messages,
        model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        stream=stream,
        current_persona=current_persona,
        functions=functions,
        stop_sequences=stop_sequences,
        **kwargs,
    )

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
