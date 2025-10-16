import asyncio
import inspect
import json
import threading
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Union

from xai_sdk import Client
from xai_sdk.chat import assistant as grok_assistant
from xai_sdk.chat import system as grok_system
from xai_sdk.chat import tool as grok_tool
from xai_sdk.chat import tool_result as grok_tool_result
from xai_sdk.chat import user as grok_user

from ATLAS.ToolManager import (
    ToolExecutionError,
    load_function_map_from_current_persona,
    load_functions_from_json,
    use_tool,
)
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger


class GrokStreamResponse:
    """Wrap an async generator so we can attach Grok-specific context."""

    def __init__(self, agen: AsyncIterator[str], context: Dict[str, Any]):
        self._agen = agen
        self.grok_context = context

    def __aiter__(self) -> "GrokStreamResponse":
        return self

    async def __anext__(self) -> str:
        return await self._agen.__anext__()

    async def aclose(self) -> None:
        await self._agen.aclose()


class GrokGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.api_key = self.config_manager.get_grok_api_key()
        if not self.api_key:
            self.logger.error("Grok API key not found in configuration")
            raise ValueError("Grok API key not found in configuration")

        # Initialize Grok Client
        self.client = Client(api_key=self.api_key)
        self.logger.info("Grok client initialized")

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: str = "grok-2",
        max_tokens: int = 1000,
        stream: bool = False,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        current_persona: Optional[Dict[str, Any]] = None,
        functions: Optional[Any] = None,
        conversation_manager: Optional[Any] = None,
        user: Optional[str] = None,
        conversation_id: Optional[str] = None,
        generation_settings: Optional[Any] = None,
        function_calling: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[Any] = None,
        tool_choice_name: Optional[str] = None,
        allowed_function_names: Optional[Any] = None,
        function_call_mode: Optional[str] = None,
        tool_prompt_data: Optional[Any] = None,
        **_kwargs: Any,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate a response using the Grok chat API."""

        self.logger.info("Generating Grok response with model %s (stream=%s)", model, stream)

        resolved_functions = functions
        if resolved_functions is None and current_persona is not None:
            self.logger.debug(
                "No functions provided; loading persona toolbox for '%s'.",
                current_persona.get("name"),
            )
            resolved_functions = load_functions_from_json(
                current_persona,
                config_manager=self.config_manager,
            )

        function_map: Optional[Mapping[str, Any]] = None
        if current_persona is not None:
            function_map = load_function_map_from_current_persona(
                current_persona,
                config_manager=self.config_manager,
            )
        if function_map is None:
            function_map = {}

        tools = self._convert_functions_to_tools(resolved_functions)
        merged_generation_settings = self._merge_generation_settings(
            generation_settings,
            function_calling=function_calling,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            tool_choice_name=tool_choice_name,
            allowed_function_names=allowed_function_names,
            function_call_mode=function_call_mode,
            tool_prompt_data=tool_prompt_data,
        )
        chat_kwargs = self._build_chat_kwargs(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tools=tools,
            user=user,
            conversation_id=conversation_id,
        )

        context = {
            "function_map": function_map,
            "functions": resolved_functions,
            "current_persona": current_persona,
            "conversation_manager": conversation_manager,
            "user": user,
            "conversation_id": conversation_id,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "generation_settings": merged_generation_settings,
        }

        if function_calling is not None:
            context["function_calling"] = function_calling
        if parallel_tool_calls is not None:
            context["parallel_tool_calls"] = parallel_tool_calls
        if tool_choice is not None:
            context["tool_choice"] = tool_choice
        if tool_choice_name is not None:
            context["tool_choice_name"] = tool_choice_name
        if allowed_function_names is not None:
            context["allowed_function_names"] = allowed_function_names
        if function_call_mode is not None:
            context["function_call_mode"] = function_call_mode
        if tool_prompt_data is not None:
            context["tool_prompt_data"] = tool_prompt_data

        try:
            if stream:
                return self._stream_response(chat_kwargs, context)

            response = await self._sample_chat_once(chat_kwargs)
            return await self._finalize_response(response, context, stream=False)
        except Exception as exc:
            self.logger.error("Error generating Grok response: %s", exc, exc_info=True)
            raise

    async def _sample_chat_once(self, chat_kwargs: Dict[str, Any]):
        chat = self.client.chat.create(**chat_kwargs)
        return await asyncio.to_thread(chat.sample)

    def _stream_response(
        self,
        chat_kwargs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """Return an async iterator that yields Grok streaming chunks."""

        async def agen() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Any] = asyncio.Queue()
            sentinel = object()

            def produce() -> None:
                try:
                    chat = self.client.chat.create(**chat_kwargs)
                    response_obj = None
                    for response_obj, chunk in chat.stream():
                        text = getattr(chunk, "content", None)
                        if text:
                            loop.call_soon_threadsafe(queue.put_nowait, text)
                    context["final_response"] = response_obj
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            threading.Thread(target=produce, daemon=True).start()

            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

        return GrokStreamResponse(agen(), context)

    async def process_streaming_response(self, response: AsyncIterator[str]) -> str:
        """Consume a streaming Grok response and return the combined output."""

        context = getattr(response, "grok_context", {})
        collected_chunks: List[str] = []

        async for chunk in response:
            if chunk:
                collected_chunks.append(chunk)

        final_response = context.get("final_response")
        if final_response is not None:
            tool_result = await self._process_tool_calls_from_response(
                final_response,
                context,
                stream=True,
            )
            if tool_result is not None:
                return await self._finalize_tool_result(tool_result, stream=False)

        final_text = "".join(collected_chunks)
        if not final_text and final_response is not None:
            fallback = getattr(final_response, "content", None)
            if fallback:
                final_text = fallback

        if collected_chunks:
            self.logger.debug(
                "Completed Grok streaming response with %d chunks", len(collected_chunks)
            )
        else:
            self.logger.debug("Received empty Grok streaming response")

        return final_text

    async def _finalize_response(
        self,
        response_obj: Any,
        context: Dict[str, Any],
        *,
        stream: bool,
    ) -> Any:
        tool_result = await self._process_tool_calls_from_response(
            response_obj,
            context,
            stream=stream,
        )
        if tool_result is not None:
            return await self._finalize_tool_result(tool_result, stream=stream)

        content = getattr(response_obj, "content", None)
        if content is None:
            return ""
        return content

    async def _process_tool_calls_from_response(
        self,
        response_obj: Any,
        context: Dict[str, Any],
        *,
        stream: bool,
    ) -> Optional[Any]:
        tool_calls = getattr(response_obj, "tool_calls", None)
        if not tool_calls:
            return None

        normalized_calls = [self._normalize_tool_call(call) for call in tool_calls]
        normalized_calls = [call for call in normalized_calls if call]
        if not normalized_calls:
            return None

        function_map = context.get("function_map")
        if function_map is None:
            function_map = {}

        return await self._handle_tool_calls(
            normalized_calls,
            user=context.get("user"),
            conversation_id=context.get("conversation_id"),
            conversation_manager=context.get("conversation_manager"),
            function_map=function_map,
            functions=context.get("functions"),
            current_persona=context.get("current_persona"),
            temperature=context.get("temperature"),
            top_p=context.get("top_p"),
            frequency_penalty=context.get("frequency_penalty"),
            presence_penalty=context.get("presence_penalty"),
            stream=stream,
            generation_settings=context.get("generation_settings"),
        )

    async def _handle_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        user: Optional[str],
        conversation_id: Optional[str],
        conversation_manager: Optional[Any],
        function_map: Mapping[str, Any],
        functions: Optional[Any],
        current_persona: Optional[Dict[str, Any]],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        stream: bool,
        generation_settings: Optional[Any],
    ) -> Optional[Any]:
        if not tool_calls:
            return None

        message_payload: Dict[str, Any] = {"tool_calls": tool_calls}
        first_call = tool_calls[0].get("function") if tool_calls else None
        if first_call:
            message_payload["function_call"] = dict(first_call)

        provider_manager = None
        if conversation_manager is not None:
            atlas = getattr(conversation_manager, "ATLAS", None)
            if atlas is not None:
                provider_manager = getattr(atlas, "provider_manager", None)
        if provider_manager is None:
            provider_manager = getattr(self.config_manager, "provider_manager", None)

        try:
            return await use_tool(
                user=user,
                conversation_id=conversation_id,
                message=message_payload,
                conversation_history=conversation_manager,
                function_map=function_map,
                functions=functions,
                current_persona=current_persona,
                temperature_var=temperature,
                top_p_var=top_p,
                frequency_penalty_var=frequency_penalty,
                presence_penalty_var=presence_penalty,
                conversation_manager=conversation_manager,
                provider_manager=provider_manager,
                config_manager=self.config_manager,
                stream=stream,
                generation_settings=generation_settings,
            )
        except ToolExecutionError as exc:
            self.logger.error(
                "Tool execution failed for %s: %s",
                exc.function_name or (first_call.get("name") if first_call else None),
                exc,
                exc_info=True,
            )
            return self._format_tool_error_payload(exc)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Unexpected error while executing Grok tool call: %s",
                exc,
                exc_info=True,
            )
            return {
                "role": "tool",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Tool execution failed: {exc}",
                    }
                ],
            }

    async def _finalize_tool_result(self, tool_result: Any, *, stream: bool) -> Any:
        if tool_result is None:
            return None
        if inspect.isawaitable(tool_result):
            tool_result = await tool_result
        if stream:
            return tool_result
        if self._is_async_stream(tool_result):
            return await self._collect_async_chunks(tool_result)
        return tool_result

    async def _collect_async_chunks(self, stream: AsyncIterator[Any]) -> str:
        chunks: List[str] = []
        async for item in stream:
            if item is None:
                continue
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = (
                    item.get("content")
                    or item.get("text")
                    or item.get("message")
                )
                if text is not None:
                    chunks.append(str(text))
                    continue
            chunks.append(str(item))
        return "".join(chunks)

    def _is_async_stream(self, value: Any) -> bool:
        return inspect.isasyncgen(value) or (
            hasattr(value, "__aiter__") and hasattr(value, "__anext__")
        )

    def _convert_functions_to_tools(self, functions: Optional[Any]):
        if not functions:
            return None

        candidates: List[Dict[str, Any]] = []
        if isinstance(functions, dict):
            candidates.append(functions)
        elif isinstance(functions, list):
            for item in functions:
                if isinstance(item, dict):
                    candidates.append(item)

        tools = []
        for entry in candidates:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not name:
                continue
            description = ""
            if isinstance(entry.get("description"), str):
                description = entry["description"]
            parameters = entry.get("parameters") if isinstance(entry, dict) else None
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    parameters = None
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            tools.append(
                grok_tool(
                    name=name,
                    description=description or "",
                    parameters=parameters,
                )
            )

        return tools or None

    def _merge_generation_settings(
        self,
        generation_settings: Optional[Any],
        *,
        function_calling: Optional[bool],
        parallel_tool_calls: Optional[bool],
        tool_choice: Optional[Any],
        tool_choice_name: Optional[str],
        allowed_function_names: Optional[Any],
        function_call_mode: Optional[str],
        tool_prompt_data: Optional[Any],
    ) -> Optional[Any]:
        extras = {
            "function_calling": function_calling,
            "parallel_tool_calls": parallel_tool_calls,
            "tool_choice": tool_choice,
            "tool_choice_name": tool_choice_name,
            "allowed_function_names": allowed_function_names,
            "function_call_mode": function_call_mode,
            "tool_prompt_data": tool_prompt_data,
        }
        extras = {key: value for key, value in extras.items() if value is not None}
        if not extras:
            return generation_settings

        if isinstance(generation_settings, Mapping):
            merged = dict(generation_settings)
            for key, value in extras.items():
                merged.setdefault(key, value)
            return merged

        if generation_settings is not None:
            extras.setdefault("raw_generation_settings", generation_settings)

        return extras

    def _build_chat_kwargs(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        tools: Optional[Any],
        user: Optional[str],
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if user:
            payload["user"] = user
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if tools:
            payload["tools"] = tools
        return payload

    def _convert_messages(self, messages: List[Dict[str, Any]]):
        converted = []
        for message in messages or []:
            role = str(message.get("role", "")).lower()
            content = self._normalize_message_content(message.get("content"))
            if role == "system":
                converted.append(grok_system(content))
            elif role == "assistant":
                converted.append(grok_assistant(content))
            elif role == "tool":
                converted.append(grok_tool_result(content))
            else:
                converted.append(grok_user(content))
        return converted

    def _normalize_message_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            text = content.get("text") or content.get("content") or content.get("message")
            if text is not None:
                if isinstance(text, (dict, list)):
                    return json.dumps(text)
                return str(text)
            return json.dumps(content)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("message")
                    if text is None and "type" in item and isinstance(item["type"], str):
                        text = item.get("value") or item.get("data")
                    if text is not None:
                        if isinstance(text, (dict, list)):
                            parts.append(json.dumps(text))
                        else:
                            parts.append(str(text))
                    else:
                        parts.append(json.dumps(item))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _normalize_tool_call(self, tool_call: Any) -> Optional[Dict[str, Any]]:
        if tool_call is None:
            return None
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None) if function is not None else None
        if not name:
            return None
        arguments = getattr(function, "arguments", "") if function is not None else ""
        if not isinstance(arguments, str):
            arguments = str(arguments)
        payload: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
        identifier = getattr(tool_call, "id", None)
        if identifier:
            payload["id"] = identifier
            payload["function"]["id"] = identifier
        return payload

    def _format_tool_error_payload(self, error: ToolExecutionError) -> Dict[str, Any]:
        if error.entry:
            return error.entry

        payload: Dict[str, Any] = {
            "role": "tool",
            "content": [{"type": "output_text", "text": str(error)}],
        }

        if error.tool_call_id is not None:
            payload["tool_call_id"] = error.tool_call_id

        metadata: Dict[str, Any] = {"status": "error"}
        if error.function_name:
            metadata["name"] = error.function_name
        if error.error_type:
            metadata["error_type"] = error.error_type

        if metadata:
            payload["metadata"] = metadata

        return payload

    async def unload_model(self) -> None:
        """Release the underlying Grok client resources safely."""

        client = getattr(self, "client", None)
        if client is None:
            self.logger.debug("Grok client already released; nothing to unload.")
            return

        async_close = getattr(client, "aclose", None)
        sync_close = getattr(client, "close", None)

        try:
            if callable(async_close):
                maybe_coro = async_close()
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro
            elif callable(sync_close):
                maybe_result = sync_close()
                if inspect.isawaitable(maybe_result):
                    await maybe_result
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to close Grok client cleanly: %s", exc, exc_info=True)
        finally:
            self.client = None
            self.logger.info("Grok client unloaded.")
