# modules/Providers/HuggingFace/components/response_generator.py

import asyncio
import copy
import json
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Set, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_exponential

from .huggingface_model_manager import HuggingFaceModelManager
from ..utils.cache_manager import CacheManager
from ..utils.logger import setup_logger
from ATLAS.ToolManager import (
    ToolExecutionError,
    load_function_map_from_current_persona,
    load_functions_from_json,
    use_tool,
)


class ResponseGenerator:
    """
    Generates responses using HuggingFace models. Supports both local pipeline inference
    and ONNX Runtime-based inference if an ONNX session is available.
    """

    def __init__(
        self,
        model_manager: HuggingFaceModelManager,
        cache_manager: CacheManager,
        *,
        config_manager: Optional[Any] = None,
    ):
        """
        Initializes the ResponseGenerator with a model manager and cache manager.

        Args:
            model_manager (HuggingFaceModelManager): The model manager instance.
            cache_manager (CacheManager): The cache manager instance.
        """
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.logger = setup_logger()
        self.model_settings = self.model_manager.base_config.model_settings
        self.config_manager = (
            config_manager
            if config_manager is not None
            else getattr(self.model_manager.base_config, "config_manager", None)
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = True,
        *,
        skill_signature: Optional[Any] = None,
        current_persona: Optional[Dict[str, Any]] = None,
        functions: Optional[Any] = None,
        conversation_manager: Optional[Any] = None,
        user: Optional[str] = None,
        conversation_id: Optional[str] = None,
        generation_settings: Optional[Mapping[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        **_kwargs: Any,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generates a response based on the provided messages using the specified model.

        Args:
            messages (List[Dict[str, str]]): A list of messages in the conversation.
            model (str): The name of the HuggingFace model to use.
            stream (bool, optional): Whether to stream the response. Defaults to True.

        Returns:
            Union[str, AsyncIterator[str]]: The generated response as a string or an asynchronous iterator for streaming.
        """
        try:
            skill_version: Optional[str] = None
            capability_tags: Optional[Any] = None
            if skill_signature is not None:
                if isinstance(skill_signature, Mapping):
                    skill_version = skill_signature.get("version") or skill_signature.get("skill_version")
                    capability_tags = (
                        skill_signature.get("required_capabilities")
                        or skill_signature.get("capability_tags")
                    )
                else:
                    skill_version = getattr(skill_signature, "version", None)
                    if not skill_version:
                        skill_version = getattr(skill_signature, "skill_version", None)
                    capability_tags = getattr(skill_signature, "required_capabilities", None)
                    if not capability_tags:
                        capability_tags = getattr(skill_signature, "capability_tags", None)

            persona_functions: Optional[Any] = None
            if current_persona is not None:
                try:
                    persona_functions = load_functions_from_json(
                        current_persona,
                        config_manager=self.config_manager,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning(
                        "Failed to load persona functions for %s: %s",
                        current_persona.get("name"),
                        exc,
                        exc_info=True,
                    )

            resolved_functions = functions
            if resolved_functions is None:
                resolved_functions = persona_functions
            elif persona_functions:
                try:
                    existing_names = {
                        entry.get("name")
                        for entry in resolved_functions
                        if isinstance(entry, Mapping)
                    }
                except TypeError:
                    existing_names = set()
                merged = []
                if isinstance(resolved_functions, list):
                    merged.extend(resolved_functions)
                else:
                    merged = list(resolved_functions or [])
                for entry in persona_functions:
                    if not isinstance(entry, Mapping):
                        merged.append(entry)
                        continue
                    name = entry.get("name")
                    if name and name in existing_names:
                        continue
                    merged.append(entry)
                resolved_functions = merged

            function_map: Dict[str, Any] = {}
            if current_persona is not None:
                try:
                    persona_map = load_function_map_from_current_persona(
                        current_persona,
                        config_manager=self.config_manager,
                    )
                    if isinstance(persona_map, Mapping):
                        function_map = dict(persona_map)
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning(
                        "Failed to load persona function map for %s: %s",
                        current_persona.get("name"),
                        exc,
                        exc_info=True,
                    )

            effective_temperature = (
                temperature
                if temperature is not None
                else self.model_settings.get("temperature")
            )
            effective_top_p = (
                top_p if top_p is not None else self.model_settings.get("top_p")
            )
            effective_frequency_penalty = (
                frequency_penalty
                if frequency_penalty is not None
                else self.model_settings.get("frequency_penalty")
            )
            effective_presence_penalty = (
                presence_penalty
                if presence_penalty is not None
                else self.model_settings.get("presence_penalty")
            )
            generation_settings_snapshot: Dict[str, Any] = {
                "model": model,
                "temperature": effective_temperature,
                "top_p": effective_top_p,
                "frequency_penalty": effective_frequency_penalty,
                "presence_penalty": effective_presence_penalty,
                "max_tokens": self.model_settings.get("max_tokens"),
                "model_settings": copy.deepcopy(self.model_settings),
            }
            merged_generation_settings: Mapping[str, Any]
            if generation_settings:
                merged = dict(generation_settings)
                for key, value in generation_settings_snapshot.items():
                    merged.setdefault(key, value)
                merged_generation_settings = merged
            else:
                merged_generation_settings = generation_settings_snapshot

            # Generate a unique cache key based on the input messages, model, and settings
            cache_key = self.cache_manager.generate_cache_key(
                messages,
                model,
                self.model_settings,
                skill_version=skill_version,
                capability_tags=capability_tags,
            )
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                self.logger.info("Returning cached response")
                return cached_response

            # Load the model if it's not already loaded
            if self.model_manager.current_model != model:
                self.logger.info(f"Loading model: {model}")
                await self.model_manager.load_model(model)

            # Determine whether to use ONNX Runtime or the local pipeline
            if model in self.model_manager.ort_sessions:
                self.logger.info("Using ONNX Runtime for inference")
                response_payload = await self._generate_with_onnx(messages, model)
            else:
                self.logger.info("Using local pipeline for inference")
                response_payload = await self._generate_local_response(messages, model)

            tool_result, normalized_response = await self._process_tool_response(
                response_payload,
                stream=stream,
                user=user,
                conversation_id=conversation_id,
                conversation_manager=conversation_manager,
                function_map=function_map,
                functions=resolved_functions,
                current_persona=current_persona,
                temperature=effective_temperature,
                top_p=effective_top_p,
                frequency_penalty=effective_frequency_penalty,
                presence_penalty=effective_presence_penalty,
                generation_settings=merged_generation_settings,
            )

            if tool_result is not None:
                if stream:
                    return self._ensure_async_stream(tool_result)
                return tool_result

            response_text = normalized_response

            # Cache the response if streaming is not enabled and the response is a string
            if not stream and isinstance(response_text, str):
                self.cache_manager.set(cache_key, response_text)

            if stream:
                return self._stream_response(response_text)

            return response_text
        except Exception as e:
            self.logger.error(f"Error in HuggingFace API call: {str(e)}")
            raise

    async def _process_tool_response(
        self,
        response_payload: Any,
        *,
        stream: bool,
        user: Optional[str],
        conversation_id: Optional[str],
        conversation_manager: Optional[Any],
        function_map: Optional[Mapping[str, Any]],
        functions: Optional[Any],
        current_persona: Optional[Dict[str, Any]],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        generation_settings: Mapping[str, Any],
    ) -> Tuple[Optional[Any], str]:
        tool_payload: Optional[Dict[str, Any]] = None

        if hasattr(response_payload, "__aiter__"):
            collected_chunks: List[str] = []
            async for chunk in response_payload:  # type: ignore[attr-defined]
                if tool_payload is None:
                    tool_payload = self._extract_tool_payload_from_object(chunk)
                collected_chunks.append(self._normalize_response_text(chunk))

            normalized_response = "".join(collected_chunks)
            if tool_payload is None and normalized_response:
                tool_payload = self._extract_tool_payload(normalized_response)

            if tool_payload:
                tool_result = await self._dispatch_tool_payload(
                    tool_payload,
                    stream=stream,
                    user=user,
                    conversation_id=conversation_id,
                    conversation_manager=conversation_manager,
                    function_map=function_map,
                    functions=functions,
                    current_persona=current_persona,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    generation_settings=generation_settings,
                )
                return tool_result, ""

            return None, normalized_response

        tool_payload = self._extract_tool_payload_from_object(response_payload)
        if tool_payload is None and isinstance(response_payload, str):
            tool_payload = self._extract_tool_payload(response_payload)

        if tool_payload:
            tool_result = await self._dispatch_tool_payload(
                tool_payload,
                stream=stream,
                user=user,
                conversation_id=conversation_id,
                conversation_manager=conversation_manager,
                function_map=function_map,
                functions=functions,
                current_persona=current_persona,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                generation_settings=generation_settings,
            )
            return tool_result, ""

        normalized_response = self._normalize_response_text(response_payload)
        if not normalized_response and isinstance(response_payload, str):
            normalized_response = response_payload

        fallback_payload = None
        if normalized_response:
            fallback_payload = self._extract_tool_payload(normalized_response)

        if fallback_payload:
            tool_result = await self._dispatch_tool_payload(
                fallback_payload,
                stream=stream,
                user=user,
                conversation_id=conversation_id,
                conversation_manager=conversation_manager,
                function_map=function_map,
                functions=functions,
                current_persona=current_persona,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                generation_settings=generation_settings,
            )
            return tool_result, ""

        return None, normalized_response

    async def _generate_local_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
    ) -> str:
        """
        Generates a response using the local HuggingFace pipeline.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            model (str): The model to use.
            stream (bool): Whether to stream the response.

        Returns:
            Union[str, AsyncIterator[str]]: The generated response.
        """
        prompt = self._convert_messages_to_prompt(messages)
        self.logger.debug(f"Generated prompt for local inference: {prompt}")

        return await self._generate_text(prompt, model)

    async def _generate_with_onnx(
        self,
        messages: List[Dict[str, str]],
        model: str,
    ) -> str:
        """
        Generates a response using ONNX Runtime.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            model (str): The model to use.
            stream (bool): Whether to stream the response.

        Returns:
            Union[str, AsyncIterator[str]]: The generated response.
        """
        # Ensure that the requested model is currently loaded
        if self.model_manager.current_model != model:
            self.logger.info(f"Loading model for ONNX inference: {model}")
            await self.model_manager.load_model(model)

        prompt = self._convert_messages_to_prompt(messages)
        self.logger.debug(f"Generated prompt for ONNX inference: {prompt}")

        # Tokenize the prompt
        inputs = self.model_manager.tokenizer(prompt, return_tensors='np')
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)

        # Prepare the inputs for ONNX Runtime
        ort_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            ort_inputs["attention_mask"] = attention_mask

        self.logger.debug(f"ONNX Runtime inputs: {ort_inputs}")

        # Run inference using ONNX Runtime
        try:
            ort_session = self.model_manager.ort_sessions.get(model)
            if ort_session is None:
                raise ValueError(f"No ONNX Runtime session found for model: {model}")

            outputs = await asyncio.to_thread(
                ort_session.run, None, ort_inputs
            )
            self.logger.debug(f"ONNX Runtime outputs: {outputs}")

            # Assuming the first output contains the generated token IDs
            generated_ids = outputs[0]
            generated_text = self.model_manager.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            self.logger.debug(f"Generated text from ONNX Runtime: {generated_text}")

            return generated_text
        except Exception as e:
            self.logger.error(f"Error during ONNX Runtime inference: {str(e)}")
            raise

    async def _generate_text(self, prompt: str, model: str) -> str:
        """
        Generates text using the HuggingFace pipeline.

        Args:
            prompt (str): The input prompt.
            model (str): The model to use.

        Returns:
            str: The generated text.
        """
        generation_kwargs = self._get_generation_config()
        generation_kwargs.pop('prompt', None)  # Ensure 'prompt' isn't duplicated
        self.logger.debug(f"Generation kwargs: {generation_kwargs}")

        # Run the pipeline in a separate thread to avoid blocking
        try:
            output = await asyncio.to_thread(self.model_manager.pipeline, prompt, **generation_kwargs)
            self.logger.debug(f"Pipeline output: {output}")

            # Extract the generated text
            return output[0]['generated_text']
        except Exception as e:
            self.logger.error(f"Error during pipeline inference: {str(e)}")
            raise

    async def _stream_response(self, text: str) -> AsyncIterator[str]:
        """
        Streams the generated text token by token.

        Args:
            text (str): The generated text.

        Yields:
            AsyncIterator[str]: Tokens of the generated text.
        """
        for token in text.split():
            yield token + " "
            await asyncio.sleep(0)  # Yield control to the event loop

    async def _single_chunk_stream(self, payload: Any) -> AsyncIterator[Any]:
        yield payload

    def _ensure_async_stream(self, payload: Any) -> AsyncIterator[Any]:
        if hasattr(payload, "__aiter__"):
            return payload  # type: ignore[return-value]
        return self._single_chunk_stream(payload)

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Converts a list of messages into a prompt string.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.

        Returns:
            str: The formatted prompt.
        """
        prompt = ""
        for message in messages:
            role = message['role']
            content = message.get('content')
            if not isinstance(content, str):
                content = "" if content is None else str(content)

            metadata = message.get('metadata')
            if isinstance(metadata, dict) and metadata:
                try:
                    metadata_text = json.dumps(metadata, ensure_ascii=False)
                except (TypeError, ValueError):
                    metadata_text = str(metadata)
                content = f"{content}\n[metadata: {metadata_text}]".strip()

            if role == 'system':
                prompt += f"System: {content}\n"
            elif role == 'user':
                prompt += f"Human: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt.strip()

    def _get_generation_config(self) -> Dict:
        """
        Retrieves the generation configuration settings.

        Returns:
            Dict: The generation configuration.
        """
        defaults = {}
        base_config = getattr(self.model_manager, "base_config", None)
        if base_config is not None:
            defaults = getattr(base_config, "DEFAULT_MODEL_SETTINGS", {})

        def _resolve(key: str, fallback: Any) -> Any:
            if key in self.model_settings:
                return self.model_settings[key]
            if isinstance(defaults, Mapping) and key in defaults:
                return defaults[key]
            return fallback

        config = {
            "max_new_tokens": _resolve("max_tokens", 100),
            "temperature": _resolve("temperature", 0.7),
            "top_p": _resolve("top_p", 1.0),
            "top_k": _resolve("top_k", 50),
            "repetition_penalty": _resolve("repetition_penalty", 1.0),
            "length_penalty": _resolve("length_penalty", 1.0),
            "early_stopping": _resolve("early_stopping", False),
            "do_sample": _resolve("do_sample", False),
        }
        self.logger.debug(f"Using generation config: {config}")
        return config

    async def _dispatch_tool_payload(
        self,
        payload: Mapping[str, Any],
        *,
        stream: bool,
        user: Optional[str],
        conversation_id: Optional[str],
        conversation_manager: Optional[Any],
        function_map: Optional[Mapping[str, Any]],
        functions: Optional[Any],
        current_persona: Optional[Dict[str, Any]],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        generation_settings: Mapping[str, Any],
    ) -> Optional[Any]:
        if not isinstance(payload, Mapping):
            return None

        tool_messages = self._prepare_tool_messages(payload)
        if not tool_messages:
            return None

        provider_manager = None
        if conversation_manager is not None:
            atlas = getattr(conversation_manager, "ATLAS", None)
            if atlas is not None:
                provider_manager = getattr(atlas, "provider_manager", None)
        if provider_manager is None:
            provider_manager = getattr(self.config_manager, "provider_manager", None)

        for message in tool_messages:
            try:
                tool_response = await use_tool(
                    user=user,
                    conversation_id=conversation_id,
                    message=message,
                    conversation_history=conversation_manager,
                    function_map=function_map or {},
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
                    exc.function_name,
                    exc,
                    exc_info=True,
                )
                return str(exc)

            if tool_response is not None:
                return tool_response

        return None

    def _extract_tool_payload_from_object(
        self, payload: Any, *, _visited: Optional[Set[int]] = None
    ) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        if isinstance(payload, Mapping):
            return self._extract_tool_payload_from_mapping(payload, _visited=_visited)
        if isinstance(payload, (list, tuple)):
            for item in payload:
                candidate = self._extract_tool_payload_from_object(
                    item, _visited=_visited
                )
                if candidate:
                    return candidate
        if isinstance(payload, str):
            return self._extract_tool_payload(payload)
        return None

    def _extract_tool_payload_from_mapping(
        self,
        payload: Mapping[str, Any],
        *,
        _visited: Optional[Set[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        if _visited is None:
            _visited = set()
        payload_id = id(payload)
        if payload_id in _visited:
            return None
        _visited.add(payload_id)

        captured: Dict[str, Any] = {}
        for key in ("function_call", "tool_call", "tool_calls", "id", "tool_call_id"):
            if key in payload and payload[key]:
                captured[key] = payload[key]

        if "tool_call" in captured and "tool_calls" not in captured:
            tool_call_entry = captured.pop("tool_call")
            if tool_call_entry:
                captured["tool_calls"] = [tool_call_entry]

        if captured:
            try:
                return json.loads(json.dumps(captured))
            except (TypeError, ValueError):
                return dict(captured)

        for key in (
            "message",
            "delta",
            "response",
            "content",
            "data",
            "choices",
            "messages",
            "outputs",
        ):
            nested = payload.get(key)
            candidate = self._search_for_tool_payload(nested, _visited=_visited)
            if candidate:
                return candidate

        return None

    def _search_for_tool_payload(
        self, value: Any, *, _visited: Optional[Set[int]] = None
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return self._extract_tool_payload_from_mapping(value, _visited=_visited)
        if isinstance(value, (list, tuple, set)):
            for item in value:
                candidate = self._search_for_tool_payload(item, _visited=_visited)
                if candidate:
                    return candidate
        if isinstance(value, str):
            return self._extract_tool_payload(value)
        return None

    def _normalize_response_text(
        self, payload: Any, *, _visited: Optional[Set[int]] = None
    ) -> str:
        text = self._coerce_content_to_text(payload, _visited=_visited)
        return text if text is not None else ""

    def _coerce_content_to_text(
        self, value: Any, *, _visited: Optional[Set[int]] = None
    ) -> Optional[str]:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if _visited is None:
            _visited = set()
        value_id = id(value)
        if value_id in _visited:
            return ""
        _visited.add(value_id)

        if isinstance(value, Mapping):
            for key in ("generated_text", "text", "content"):
                if key in value:
                    candidate = self._coerce_content_to_text(
                        value[key], _visited=_visited
                    )
                    if candidate:
                        return candidate
            token = value.get("token")
            if isinstance(token, Mapping):
                candidate = self._coerce_content_to_text(
                    token.get("text"), _visited=_visited
                )
                if candidate:
                    return candidate
            delta = value.get("delta")
            if isinstance(delta, Mapping):
                candidate = self._coerce_content_to_text(
                    delta.get("content"), _visited=_visited
                )
                if candidate:
                    return candidate
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)

        if isinstance(value, (list, tuple, set)):
            fragments: List[str] = []
            for item in value:
                fragment = self._coerce_content_to_text(item, _visited=_visited)
                if fragment:
                    fragments.append(fragment)
            return "".join(fragments)

        return str(value)

    def _extract_tool_payload(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        candidates = []
        sanitized = text.strip()
        candidates.append(sanitized)
        stripped = self._strip_code_fences(sanitized)
        if stripped != sanitized:
            candidates.append(stripped)

        for candidate in candidates:
            payload = self._try_parse_json(candidate)
            if payload:
                return payload

        return None

    def _strip_code_fences(self, text: str) -> str:
        if text.startswith("```") and text.endswith("```"):
            without_ticks = text.strip("`")
            newline_index = without_ticks.find("\n")
            if newline_index != -1:
                content = without_ticks[newline_index + 1 :]
            else:
                content = without_ticks
            if content.endswith("\n"):
                content = content[:-1]
            return content
        return text

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            snippet = text[start : end + 1]
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                return None
        if isinstance(payload, Mapping):
            if "function_call" in payload or "tool_calls" in payload:
                return dict(payload)
        return None

    def _prepare_tool_messages(self, payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        function_call = payload.get("function_call")
        if isinstance(function_call, Mapping):
            tool_entry: Dict[str, Any] = {
                "type": "function",
                "function": dict(function_call),
            }
            identifier = payload.get("tool_call_id") or payload.get("id")
            if identifier:
                tool_entry["id"] = identifier
            messages.append({"tool_calls": [tool_entry]})

        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                function_payload = call.get("function") or call.get("function_call")
                if not isinstance(function_payload, Mapping):
                    continue
                tool_entry: Dict[str, Any] = {
                    "type": "function",
                    "function": dict(function_payload),
                }
                identifier = call.get("id") or call.get("tool_call_id")
                if identifier:
                    tool_entry["id"] = identifier
                messages.append({"tool_calls": [tool_entry]})

        return messages
