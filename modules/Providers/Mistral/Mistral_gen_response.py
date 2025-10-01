# modules/Providers/Mistral/Mistral_gen_response.py

import asyncio
import threading
from collections.abc import Iterable, Mapping
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

from ATLAS.config import ConfigManager
from ATLAS.model_manager import ModelManager
from ATLAS.ToolManager import load_functions_from_json
from modules.logging.logger import setup_logger

class MistralGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.api_key = self.config_manager.get_mistral_api_key()
        if not self.api_key:
            self.logger.error("Mistral API key not found in configuration")
            raise ValueError("Mistral API key not found in configuration")
        self.client = Mistral(api_key=self.api_key)
        self.model_manager = ModelManager(config_manager)
        settings = self.config_manager.get_mistral_llm_settings()
        default_model = settings.get("model")
        if isinstance(default_model, str) and default_model.strip():
            self.model_manager.set_model(default_model.strip(), "Mistral")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        safe_prompt: Optional[bool] = None,
        random_seed: Optional[int] = None,
        stream: Optional[bool] = None,
        current_persona=None,
        functions=None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Union[str, AsyncIterator[str]]:
        try:
            mistral_messages = self.convert_messages_to_mistral_format(messages)
            settings = self.config_manager.get_mistral_llm_settings()

            def _resolve_model(preferred: Optional[str]) -> str:
                current = self.model_manager.get_current_model()
                if preferred and preferred.strip():
                    return preferred.strip()
                stored = settings.get("model")
                if isinstance(stored, str) and stored.strip():
                    return stored.strip()
                if current and isinstance(current, str) and current.strip():
                    return current.strip()
                return "mistral-large-latest"

            effective_model = _resolve_model(model)
            current_model = self.model_manager.get_current_model()
            if effective_model != current_model:
                self.model_manager.set_model(effective_model, "Mistral")

            def _resolve_float(
                candidate: Optional[Any],
                stored_key: str,
                default: float,
                *,
                minimum: float,
                maximum: float,
            ) -> float:
                value = candidate
                if value is None:
                    value = settings.get(stored_key, default)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return default
                if numeric < minimum or numeric > maximum:
                    return default
                return numeric

            def _resolve_int(
                candidate: Optional[Any],
                stored_key: str,
                default: Optional[int],
            ) -> Optional[int]:
                value = candidate
                if value is None:
                    value = settings.get(stored_key, default)
                if value in {None, ""}:
                    return None if default is None else default
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    return default
                if numeric <= 0:
                    return None if default is None else default
                return numeric

            def _resolve_bool(candidate: Optional[Any], stored_key: str, default: bool) -> bool:
                value = candidate
                if value is None:
                    value = settings.get(stored_key, default)
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"1", "true", "yes", "on"}:
                        return True
                    if lowered in {"0", "false", "no", "off"}:
                        return False
                    return default
                return bool(value)

            def _resolve_optional_int(candidate: Optional[Any], stored_key: str) -> Optional[int]:
                value = candidate
                if value is None:
                    value = settings.get(stored_key)
                if value in {None, ""}:
                    return None
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    return None
                return numeric

            effective_temperature = _resolve_float(
                temperature,
                'temperature',
                0.0,
                minimum=0.0,
                maximum=2.0,
            )
            effective_top_p = _resolve_float(
                top_p,
                'top_p',
                1.0,
                minimum=0.0,
                maximum=1.0,
            )
            effective_max_tokens = _resolve_int(
                max_tokens,
                'max_tokens',
                None,
            )
            effective_frequency_penalty = _resolve_float(
                frequency_penalty,
                'frequency_penalty',
                0.0,
                minimum=-2.0,
                maximum=2.0,
            )
            effective_presence_penalty = _resolve_float(
                presence_penalty,
                'presence_penalty',
                0.0,
                minimum=-2.0,
                maximum=2.0,
            )
            effective_safe_prompt = _resolve_bool(
                safe_prompt,
                'safe_prompt',
                False,
            )
            effective_stream = _resolve_bool(
                stream,
                'stream',
                True,
            )
            effective_parallel = _resolve_bool(
                parallel_tool_calls,
                'parallel_tool_calls',
                True,
            )
            effective_random_seed = _resolve_optional_int(random_seed, 'random_seed')

            configured_tool_choice = (
                tool_choice if tool_choice is not None else settings.get('tool_choice')
            )
            if isinstance(configured_tool_choice, str):
                configured_tool_choice = configured_tool_choice.strip() or None
            elif isinstance(configured_tool_choice, dict):
                configured_tool_choice = dict(configured_tool_choice)
            else:
                configured_tool_choice = None if configured_tool_choice is None else configured_tool_choice

            self.logger.info(
                "Generating response with Mistral AI using model: %s",
                effective_model,
            )

            provided_functions = functions
            if provided_functions is None and current_persona is not None:
                try:
                    provided_functions = load_functions_from_json(current_persona)
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning(
                        "Failed to load functions from persona for Mistral: %s",
                        exc,
                    )

            tools_payload = self._convert_functions_to_tools(provided_functions)

            request_kwargs: Dict[str, Any] = {
                'model': effective_model,
                'messages': mistral_messages,
                'temperature': effective_temperature,
                'top_p': effective_top_p,
                'safe_prompt': effective_safe_prompt,
                'frequency_penalty': effective_frequency_penalty,
                'presence_penalty': effective_presence_penalty,
            }

            if effective_max_tokens is not None:
                request_kwargs['max_tokens'] = effective_max_tokens

            if effective_random_seed is not None:
                request_kwargs['random_seed'] = effective_random_seed

            if tools_payload:
                request_kwargs['tools'] = tools_payload
                if configured_tool_choice is not None:
                    request_kwargs['tool_choice'] = configured_tool_choice
                request_kwargs['parallel_tool_calls'] = effective_parallel
            else:
                if configured_tool_choice is not None:
                    request_kwargs['tool_choice'] = configured_tool_choice

            if effective_stream:
                response_stream = await asyncio.to_thread(
                    self.client.chat.stream,
                    **request_kwargs,
                )
                return self.process_streaming_response(response_stream)

            response = await asyncio.to_thread(
                self.client.chat.complete,
                **request_kwargs,
            )
            return response.choices[0].message.content

        except Mistral.APIError as e:
            self.logger.error(f"Mistral API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error with Mistral: {str(e)}")
            raise

    def convert_messages_to_mistral_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        mistral_messages = []
        for message in messages:
            mistral_messages.append({
                "role": message['role'],
                "content": message['content']
            })
        return mistral_messages

    def _convert_functions_to_tools(self, functions: Optional[Any]) -> Optional[List[Dict[str, Any]]]:
        if not functions:
            return None

        tools: List[Dict[str, Any]] = []

        def _normalize_function(entry: Any) -> Optional[Dict[str, Any]]:
            if entry is None:
                return None
            payload = entry
            if isinstance(entry, Mapping) and 'function' in entry:
                payload = entry.get('function')
            if not isinstance(payload, Mapping):
                return None
            name = payload.get('name')
            if not isinstance(name, str) or not name.strip():
                return None
            function_spec: Dict[str, Any] = {'name': name.strip()}
            description = payload.get('description')
            if isinstance(description, str) and description.strip():
                function_spec['description'] = description.strip()
            if 'parameters' in payload:
                function_spec['parameters'] = payload['parameters']
            return {'type': 'function', 'function': function_spec}

        candidates: Iterable[Any]
        if isinstance(functions, Mapping):
            candidates = [functions]
        elif isinstance(functions, Iterable) and not isinstance(functions, (str, bytes, bytearray)):
            candidates = list(functions)
        else:
            candidates = []

        for item in candidates:
            tool = _normalize_function(item)
            if tool:
                tools.append(tool)

        return tools or None

    async def process_streaming_response(self, response) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
        DONE = object()

        def iterate_stream():
            try:
                for chunk in response:
                    data = getattr(chunk, "data", None)
                    if not data or not getattr(data, "choices", None):
                        continue
                    choice = data.choices[0]
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("chunk", content)
                        )
                    if getattr(choice, "finish_reason", None):
                        break
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # pragma: no cover - defensive
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", DONE))

        threading.Thread(target=iterate_stream, daemon=True).start()

        while True:
            kind, payload = await queue.get()
            if kind == "chunk":
                yield payload
            elif kind == "error":
                raise payload
            elif kind == "done":
                break

    async def process_response(self, response: Union[str, AsyncIterator[str]]) -> str:
        if isinstance(response, str):
            return response
        else:
            full_response = ""
            async for chunk in response:
                full_response += chunk
            return full_response

def setup_mistral_generator(config_manager: ConfigManager):
    return MistralGenerator(config_manager)

async def generate_response(
    config_manager: ConfigManager,
    messages: List[Dict[str, str]],
    model: str = "mistral-large-latest",
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    stream: bool = True,
    current_persona=None,
    functions=None,
) -> Union[str, AsyncIterator[str]]:
    generator = setup_mistral_generator(config_manager)
    return await generator.generate_response(messages, model, max_tokens, temperature, stream, current_persona, functions)

async def process_response(response: Union[str, AsyncIterator[str]]) -> str:
    if isinstance(response, str):
        return response

    chunks: List[str] = []
    async for chunk in response:
        chunks.append(chunk)
    return "".join(chunks)

def generate_response_sync(config_manager: ConfigManager, messages: List[Dict[str, str]], model: str = "mistral-large-latest", stream: bool = False) -> str:
    """
    Synchronous version of generate_response for compatibility with non-async code.
    """
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        generate_response(
            config_manager,
            messages,
            model=model,
            stream=stream,
        )
    )
    if stream:
        return loop.run_until_complete(process_response(response))
    return response
