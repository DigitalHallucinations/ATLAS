# ATLAS/provider_manager.py

import asyncio
from collections.abc import Mapping
import inspect
import json
import time
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Union, AsyncIterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from huggingface_hub import HfApi
from ATLAS.model_manager import ModelManager
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from modules.Providers.HuggingFace.HF_gen_response import (
    HuggingFaceGenerator,
    search_models as hf_search_models,
    download_model as hf_download_model,
    update_model_settings as hf_update_model_settings,
    clear_cache as hf_clear_cache,
)
from modules.Providers.Grok.grok_generate_response import GrokGenerator

# Import other necessary provider generators
from modules.Providers.OpenAI.OA_gen_response import get_generator as get_openai_generator
from modules.Providers.Mistral.Mistral_gen_response import get_generator as get_mistral_generator
from modules.Providers.Google.GG_gen_response import get_generator as get_google_generator
from modules.Providers.Anthropic.Anthropic_gen_response import AnthropicGenerator

try:  # pragma: no cover - import guard for optional detailed exceptions
    from mistralai.exceptions import MistralException
except Exception:  # pragma: no cover - fallback when the optional module layout changes
    MistralException = Exception

try:  # pragma: no cover - optional dependency at runtime
    from mistralai import Mistral  # type: ignore
except Exception:  # pragma: no cover - degrade gracefully when SDK is missing
    Mistral = None  # type: ignore[assignment]


class ProviderManager:
    """
    Manages interactions with different LLM providers, ensuring only one instance exists.
    """

    AVAILABLE_PROVIDERS = ["OpenAI", "Mistral", "Google", "HuggingFace", "Anthropic", "Grok"]

    _instance = None  # Class variable to hold the singleton instance
    _lock: asyncio.Lock | None = None  # Lazily instantiated lock for thread-safe creation

    def __init__(self, config_manager: ConfigManager):
        """
        Private constructor to prevent direct instantiation.
        """
        if ProviderManager._instance is not None:
            raise Exception("This class is a singleton! Use ProviderManager.create() to get the instance.")
        
        # Initialization code
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.model_manager = ModelManager(self.config_manager)
        self.current_llm_provider = self.config_manager.get_default_provider()
        self.current_background_provider = self.config_manager.get_default_provider()
        self.current_model = None
        self.generate_response_func = None
        self.process_streaming_response_func = None
        self.huggingface_generator = None
        self.grok_generator = None
        self.anthropic_generator: Optional[AnthropicGenerator] = None
        self._openai_generator = None
        self._mistral_generator = None
        self._google_generator = None
        self.current_functions = None
        self.providers = {}
        self.chat_session = None
        self.conversation_manager = None
        self.current_conversation_id: Optional[str] = None
        self._pending_models: Dict[str, Optional[str]] = {}
        self._provider_model_ready: Dict[str, bool] = {}
        self._provider_invocation_strategies: Dict[
            str,
            Callable[[Callable[..., Awaitable[Any]], Dict[str, Any]], Awaitable[Any]],
        ] = {
            "HuggingFace": self._invoke_huggingface_generator,
            "Grok": self._invoke_grok_generator,
        }
        self._config_injection_cache: Dict[Tuple[Any, bool], bool] = {}

    @classmethod
    async def create(cls, config_manager: ConfigManager):
        """
        Asynchronous factory method to create or retrieve the singleton instance.
        
        Args:
            config_manager (ConfigManager): An instance of ConfigManager.
        
        Returns:
            ProviderManager: The singleton instance of ProviderManager.
        """
        if cls._lock is None:
            cls._lock = asyncio.Lock()

        async with cls._lock:
            if cls._instance is None:
                instance = cls(config_manager)
                cls._instance = instance
                try:
                    await instance.initialize_all_providers()
                except Exception:
                    try:
                        await instance.close()
                    except Exception:  # pragma: no cover - defensive cleanup logging
                        logger = getattr(instance, "logger", None)
                        if logger is not None:
                            logger.warning(
                                "ProviderManager cleanup failed after initialization error.",
                                exc_info=True,
                            )
                    cls._instance = None
                    raise
            return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the cached singleton instance and synchronization lock."""

        cls._instance = None
        cls._lock = None

    async def initialize_all_providers(self):
        """
        Initializes the provider managers and sets the default provider.
        """
        await self._prime_openai_models()
        if self.current_llm_provider and self.config_manager.has_provider_api_key(
            self.current_llm_provider
        ):
            await self.switch_llm_provider(self.current_llm_provider)
        else:
            self.logger.warning(
                "Skipping automatic activation of provider '%s' because its API key is missing.",
                self.current_llm_provider,
            )

    async def _prime_openai_models(self) -> None:
        """Attempt to refresh cached OpenAI models during startup."""

        checker = getattr(self.config_manager, "has_provider_api_key", None)
        if not callable(checker) or not checker("OpenAI"):
            return

        try:
            result = await self.list_openai_models()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Unable to refresh OpenAI models during startup: %s", exc, exc_info=True
            )
            return

        if not isinstance(result, dict):
            return

        if result.get("error"):
            self.logger.info(
                "Skipping cached OpenAI model refresh because discovery failed: %s",
                result["error"],
            )
            return

        models = result.get("models") if isinstance(result.get("models"), list) else []
        if models:
            self.logger.info(
                "Primed cached OpenAI model list with %d entries during startup.",
                len(models),
            )


    @staticmethod
    def _build_result(success: bool, *, message: str = "", error: str = "", data: Any = None) -> Dict[str, Any]:
        """Create a structured result payload for provider actions."""
        payload: Dict[str, Any] = {"success": success}
        if success:
            if message:
                payload["message"] = message
            if data is not None:
                payload["data"] = data
        else:
            payload["error"] = error or message or "Unknown error"
        return payload

    def _ensure_anthropic_generator(self) -> AnthropicGenerator:
        if self.anthropic_generator is None:
            self.anthropic_generator = AnthropicGenerator(self.config_manager)
        return self.anthropic_generator

    def _ensure_openai_generator(self):
        if self._openai_generator is None:
            try:
                self._openai_generator = get_openai_generator(
                    self.config_manager,
                    model_manager=self.model_manager,
                )
            except TypeError:
                self._openai_generator = get_openai_generator(self.config_manager)
                if hasattr(self._openai_generator, "model_manager"):
                    self._openai_generator.model_manager = self.model_manager
        return self._openai_generator

    def _ensure_mistral_generator(self):
        if self._mistral_generator is None:
            try:
                self._mistral_generator = get_mistral_generator(
                    self.config_manager,
                    model_manager=self.model_manager,
                )
            except TypeError:
                self._mistral_generator = get_mistral_generator(self.config_manager)
                if hasattr(self._mistral_generator, "model_manager"):
                    self._mistral_generator.model_manager = self.model_manager
        return self._mistral_generator

    def _ensure_google_generator(self):
        if self._google_generator is None:
            self._google_generator = get_google_generator(self.config_manager)
        return self._google_generator

    async def _close_generator(self, generator: Any, provider_name: str) -> None:
        """Attempt to close or aclose a provider generator instance."""

        if generator is None:
            return

        closer = getattr(generator, "close", None)
        try:
            if callable(closer):
                result = closer()
                if inspect.isawaitable(result):
                    await result
                return

            closer = getattr(generator, "aclose", None)
            if callable(closer):
                result = closer()
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:  # pragma: no cover - defensive cleanup logging
            self.logger.warning(
                "Failed to close %s generator cleanly: %s", provider_name, exc, exc_info=True
            )

    async def _cleanup_provider_generator(self, provider_name: Optional[str]) -> None:
        """Release resources associated with a provider generator when deactivating it."""

        if not provider_name:
            return

        if provider_name == "HuggingFace":
            generator = self.huggingface_generator
            if generator is None:
                return
            try:
                result = await self.unload_hf_model()
                if not result.get("success"):
                    self.logger.warning(
                        "HuggingFace cleanup reported an error: %s",
                        result.get("error", "Unknown error"),
                    )
            finally:
                self.huggingface_generator = None
        elif provider_name == "Grok":
            generator = self.grok_generator
            if generator is None:
                return
            unload = getattr(generator, "unload_model", None)
            if callable(unload):
                try:
                    result = unload()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:  # pragma: no cover - defensive cleanup logging
                    self.logger.warning(
                        "Failed to unload Grok generator cleanly: %s", exc, exc_info=True
                    )
            self.grok_generator = None
        elif provider_name == "OpenAI":
            generator = self._openai_generator
            if generator is None:
                return
            await self._close_generator(generator, "OpenAI")
            self._openai_generator = None
        elif provider_name == "Mistral":
            generator = self._mistral_generator
            if generator is None:
                return
            await self._close_generator(generator, "Mistral")
            self._mistral_generator = None
        elif provider_name == "Anthropic":
            generator = self.anthropic_generator
            if generator is None:
                return
            await self._close_generator(generator, "Anthropic")
            self.anthropic_generator = None
        else:
            return

        self._provider_model_ready[provider_name] = False
        self._pending_models.pop(provider_name, None)

    async def _ensure_provider_callable(
        self, provider: str
    ) -> Callable[..., Awaitable[Any]]:
        """Ensure a callable is registered for the requested provider."""

        existing = self.providers.get(provider)
        if callable(existing):
            return existing

        if provider == "OpenAI":
            generator = self._ensure_openai_generator()
            func = generator.generate_response
        elif provider == "Mistral":
            generator = self._ensure_mistral_generator()
            func = generator.generate_response
        elif provider == "Google":
            generator = self._ensure_google_generator()
            func = generator.generate_response
        elif provider == "HuggingFace":
            ensure_result = self.ensure_huggingface_ready()
            if not ensure_result.get("success"):
                raise ValueError(
                    ensure_result.get(
                        "error", "Failed to initialize HuggingFace generator."
                    )
                )
            if not self.huggingface_generator:
                raise ValueError("HuggingFace generator is not available.")
            func = self.huggingface_generator.generate_response
        elif provider == "Anthropic":
            self._ensure_anthropic_generator()
            func = self._anthropic_generate_response
        elif provider == "Grok":
            if self.grok_generator is None:
                self.grok_generator = GrokGenerator(self.config_manager)
            func = self.grok_generator.generate_response
        else:
            raise ValueError(f"Provider '{provider}' is not recognized for fallback handling.")

        self.providers[provider] = func
        return func

    async def _anthropic_generate_response(self, _config_manager, **kwargs):
        generator = self._ensure_anthropic_generator()
        return await generator.generate_response(**kwargs)

    def register_provider_invoker(
        self,
        provider_name: str,
        adapter: Callable[[Callable[..., Awaitable[Any]], Dict[str, Any]], Awaitable[Any]],
    ) -> None:
        """Register or override a provider-specific invocation adapter."""

        if not callable(adapter):
            raise ValueError("adapter must be callable")
        self._config_injection_cache.clear()
        self._provider_invocation_strategies[provider_name] = adapter

    async def _invoke_provider_callable(
        self,
        provider_name: str,
        func: Callable[..., Awaitable[Any]],
        call_kwargs: Dict[str, Any],
    ) -> Any:
        strategy = self._provider_invocation_strategies.get(
            provider_name, self._invoke_with_config_manager
        )
        return await strategy(func, call_kwargs)

    async def _invoke_with_config_manager(
        self,
        func: Callable[..., Awaitable[Any]],
        call_kwargs: Dict[str, Any],
    ) -> Any:
        bound_instance = getattr(func, "__self__", None)
        cache_key = (getattr(func, "__func__", func), bound_instance is not None)
        needs_config = self._config_injection_cache.get(cache_key)
        if needs_config is None:
            if bound_instance is None:
                needs_config = True
            else:
                signature = inspect.signature(func)
                parameters = list(signature.parameters.values())
                needs_config = False
                if parameters:
                    first_param = parameters[0]
                    needs_config = (
                        first_param.kind
                        in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                        and first_param.name in {"config_manager", "_config_manager"}
                    )
            self._config_injection_cache[cache_key] = needs_config

        if needs_config:
            return await func(self.config_manager, **call_kwargs)
        return await func(**call_kwargs)

    async def _invoke_huggingface_generator(
        self,
        func: Callable[..., Awaitable[Any]],
        call_kwargs: Dict[str, Any],
    ) -> Any:
        payload = {
            "messages": call_kwargs.get("messages"),
            "model": call_kwargs.get("model"),
        }
        if "stream" in call_kwargs:
            payload["stream"] = call_kwargs["stream"]
        return await func(**payload)

    async def _invoke_grok_generator(
        self,
        func: Callable[..., Awaitable[Any]],
        call_kwargs: Dict[str, Any],
    ) -> Any:
        payload = {"messages": call_kwargs.get("messages")}
        model = call_kwargs.get("model")
        if model:
            payload["model"] = model
        if "max_tokens" in call_kwargs and call_kwargs["max_tokens"] is not None:
            payload["max_tokens"] = call_kwargs["max_tokens"]
        if "stream" in call_kwargs and call_kwargs["stream"] is not None:
            payload["stream"] = call_kwargs["stream"]
        return await func(**payload)

    async def update_provider_api_key(
        self, provider_name: str, new_api_key: Optional[str]
    ) -> Dict[str, Any]:
        """Persist provider credentials and refresh active sessions when needed."""

        normalized_key = (new_api_key or "").strip()
        if not normalized_key:
            return self._build_result(False, error="API key cannot be empty.")

        try:
            self.config_manager.update_api_key(provider_name, normalized_key)
        except FileNotFoundError as exc:
            self.logger.error(
                "Failed to update API key for %s because the environment file could not be located.",
                provider_name,
                exc_info=True,
            )
            return self._build_result(
                False,
                error="Unable to save API key because the environment file could not be located.",
            )
        except ValueError as exc:
            self.logger.error("Rejected API key update for %s: %s", provider_name, exc)
            return self._build_result(False, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Unexpected error while updating API key for %s: %s",
                provider_name,
                exc,
                exc_info=True,
            )
            return self._build_result(False, error=str(exc))

        message = f"API Key for {provider_name} saved successfully."

        if provider_name == self.current_llm_provider:
            try:
                await self.set_current_provider(provider_name)
                message = f"{message} Provider {provider_name} refreshed with new API key."
            except Exception as exc:
                self.logger.error(
                    "API key saved but failed to refresh provider %s: %s",
                    provider_name,
                    exc,
                    exc_info=True,
                )
                return self._build_result(
                    False,
                    error=f"API key saved but failed to refresh {provider_name}: {exc}",
                )

        self.logger.info("API key for %s updated via provider manager.", provider_name)
        return self._build_result(True, message=message)

    def set_openai_llm_settings(
        self,
        *,
        model: Optional[str],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        json_mode: Optional[Any] = None,
        json_schema: Optional[Any] = None,
        audio_enabled: Optional[bool] = None,
        audio_voice: Optional[str] = None,
        audio_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist OpenAI LLM defaults via the config manager and refresh runtime state."""

        try:
            settings = self.config_manager.set_openai_llm_settings(
                model=model,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                max_output_tokens=max_output_tokens,
                stream=stream,
                function_calling=function_calling,
                parallel_tool_calls=parallel_tool_calls,
                tool_choice=tool_choice,
                base_url=base_url,
                organization=organization,
                reasoning_effort=reasoning_effort,
                json_mode=json_mode,
                json_schema=json_schema,
                audio_enabled=audio_enabled,
                audio_voice=audio_voice,
                audio_format=audio_format,
            )
        except Exception as exc:
            self.logger.error("Failed to persist OpenAI settings: %s", exc, exc_info=True)
            return self._build_result(False, error=str(exc))

        promoted_model = settings.get("model")
        if promoted_model:
            with self.model_manager.lock:
                available = list(self.model_manager.models.get("OpenAI", []))
                new_order = [promoted_model] + [name for name in available if name != promoted_model]
                self.model_manager.models["OpenAI"] = new_order
            self.model_manager.set_model(promoted_model, "OpenAI")
            if self.current_llm_provider == "OpenAI":
                self.current_model = promoted_model

        message = "OpenAI settings saved."
        refreshed = self.get_openai_llm_settings()
        return self._build_result(True, message=message, data=refreshed)

    def get_openai_llm_settings(self) -> Dict[str, Any]:
        """Return configured OpenAI defaults or an empty payload on failure."""

        getter = getattr(self.config_manager, "get_openai_llm_settings", None)
        if not callable(getter):
            self.logger.warning("Config manager does not expose OpenAI settings accessor.")
            return {}

        try:
            settings = getter() or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to load OpenAI settings: %s", exc, exc_info=True)
            return {}

        return settings

    def set_google_llm_settings(
        self,
        *,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[Any] = None,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[Any] = None,
        stop_sequences: Optional[Any] = None,
        safety_settings: Optional[Any] = None,
        response_mime_type: Optional[str] = None,
        system_instruction: Optional[str] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        function_call_mode: Optional[str] = None,
        allowed_function_names: Optional[Any] = None,
        response_schema: Optional[Any] = None,
        cached_allowed_function_names: Optional[Any] = None,
        seed: Optional[Any] = None,
        response_logprobs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Persist Google Gemini defaults and promote the saved model when possible.

        Args:
            model: Default Gemini model identifier.
            temperature: Sampling temperature between 0 and 2.
            top_p: Nucleus sampling parameter between 0 and 1.
            top_k: Optional integer limiting candidate tokens considered.
            candidate_count: Optional integer specifying number of candidates.
            max_output_tokens: Optional integer limiting the maximum response length.
            stop_sequences: Optional iterable of stop strings.
            safety_settings: Optional safety configuration.
            response_mime_type: Optional MIME hint for the response.
            system_instruction: Optional default system instruction.
            stream: Optional flag toggling streaming responses by default.
            function_calling: Optional flag toggling Gemini tool calling by default.
            function_call_mode: Optional Gemini tool calling mode preference.
            allowed_function_names: Optional whitelist restricting Gemini tool access.
            response_schema: Optional JSON schema enforced for responses.
            seed: Optional deterministic seed for Gemini responses.
            response_logprobs: Optional flag requesting log probability details.
        """

        setter = getattr(self.config_manager, "set_google_llm_settings", None)
        if not callable(setter):
            self.logger.error("Config manager does not expose Google settings setter.")
            return self._build_result(
                False,
                error="Configuration backend does not support Google provider defaults.",
            )

        try:
            settings = setter(
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                candidate_count=candidate_count,
                max_output_tokens=max_output_tokens,
                stop_sequences=stop_sequences,
                safety_settings=safety_settings,
                response_mime_type=response_mime_type,
                system_instruction=system_instruction,
                stream=stream,
                function_calling=function_calling,
                function_call_mode=function_call_mode,
                allowed_function_names=allowed_function_names,
                response_schema=response_schema,
                cached_allowed_function_names=cached_allowed_function_names,
                seed=seed,
                response_logprobs=response_logprobs,
            )
        except Exception as exc:
            self.logger.error("Failed to persist Google settings: %s", exc, exc_info=True)
            return self._build_result(False, error=str(exc))

        promoted_model = settings.get("model") if isinstance(settings, dict) else None
        if isinstance(promoted_model, str) and promoted_model:
            with self.model_manager.lock:
                current = list(self.model_manager.models.get("Google", []))
                reordered = [promoted_model] + [name for name in current if name != promoted_model]
                self.model_manager.models["Google"] = reordered or [promoted_model]
            self.model_manager.set_model(promoted_model, "Google")
            if self.current_llm_provider == "Google":
                self.current_model = promoted_model

        message = "Google settings saved."
        return self._build_result(True, message=message, data=settings)

    def get_google_llm_settings(self) -> Dict[str, Any]:
        """Return persisted Google Gemini defaults, if available."""

        getter = getattr(self.config_manager, "get_google_llm_settings", None)
        if not callable(getter):
            self.logger.warning("Config manager does not expose Google settings accessor.")
            return {}

        try:
            settings = getter() or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to load Google settings: %s", exc, exc_info=True)
            return {}

        return settings

    def set_anthropic_settings(
        self,
        *,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Any = ConfigManager.UNSET,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None,
        stop_sequences: Any = ConfigManager.UNSET,
        tool_choice: Any = ConfigManager.UNSET,
        tool_choice_name: Any = ConfigManager.UNSET,
        metadata: Any = ConfigManager.UNSET,
        thinking: Optional[bool] = None,
        thinking_budget: Any = ConfigManager.UNSET,
    ) -> Dict[str, Any]:
        """Persist Anthropic defaults and refresh the active generator when possible."""

        try:
            settings = self.config_manager.set_anthropic_settings(
                model=model,
                stream=stream,
                function_calling=function_calling,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                stop_sequences=stop_sequences,
                tool_choice=tool_choice,
                tool_choice_name=tool_choice_name,
                metadata=metadata,
                thinking=thinking,
                thinking_budget=thinking_budget,
            )
        except Exception as exc:
            self.logger.error("Failed to persist Anthropic settings: %s", exc, exc_info=True)
            return self._build_result(False, error=str(exc))

        generator: Optional[AnthropicGenerator] = None
        try:
            generator = self._ensure_anthropic_generator()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Anthropic settings saved but generator could not be refreshed: %s",
                exc,
                exc_info=True,
            )
        else:
            try:
                generator.set_default_model(settings.get("model"))
                generator.set_streaming(settings.get("stream", True))
                generator.set_function_calling(settings.get("function_calling", False))
                generator.set_temperature(settings.get("temperature", 0.0))
                generator.set_top_p(settings.get("top_p", 1.0))
                generator.set_top_k(settings.get("top_k"))
                generator.set_max_output_tokens(settings.get("max_output_tokens"))
                generator.set_timeout(settings.get("timeout", 60))
                generator.set_max_retries(settings.get("max_retries", 3))
                generator.set_retry_delay(settings.get("retry_delay", 5))
                generator.set_stop_sequences(settings.get("stop_sequences", []))
                generator.set_tool_choice(
                    settings.get("tool_choice"),
                    settings.get("tool_choice_name"),
                )
                generator.set_metadata(settings.get("metadata"))
                generator.set_thinking(
                    settings.get("thinking"),
                    settings.get("thinking_budget"),
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(
                    "Unable to apply Anthropic settings to the live generator: %s",
                    exc,
                    exc_info=True,
                )

        promoted_model = settings.get("model")
        if promoted_model:
            with self.model_manager.lock:
                current = list(self.model_manager.models.get("Anthropic", []))
                reordered = [promoted_model] + [name for name in current if name != promoted_model]
                self.model_manager.models["Anthropic"] = reordered or [promoted_model]
            self.model_manager.set_model(promoted_model, "Anthropic")
            if self.current_llm_provider == "Anthropic":
                self.current_model = promoted_model

        message = "Anthropic settings saved."
        return self._build_result(True, message=message, data=settings)

    def get_anthropic_settings(self) -> Dict[str, Any]:
        """Retrieve Anthropic defaults, returning an empty mapping on failure."""

        getter = getattr(self.config_manager, "get_anthropic_settings", None)
        if not callable(getter):
            self.logger.warning("Config manager does not expose Anthropic settings accessor.")
            return {}

        try:
            settings = getter() or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to load Anthropic settings: %s", exc, exc_info=True)
            return {}

        return settings

    def get_models_for_provider(self, provider: str) -> List[str]:
        """Return cached model names for the requested provider."""

        try:
            models = self.model_manager.get_available_models(provider).get(provider, [])
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to read cached models for %s: %s", provider, exc, exc_info=True
            )
            return []

        return list(models)

    async def list_openai_models(
        self,
        *,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Retrieve OpenAI models using stored credentials and optional overrides."""

        settings = self.get_openai_llm_settings()
        configured_base_url = settings.get("base_url") if isinstance(settings, dict) else None
        configured_org = settings.get("organization") if isinstance(settings, dict) else None

        effective_base_url = (base_url if base_url is not None else configured_base_url) or "https://api.openai.com/v1"
        effective_org = organization if organization is not None else configured_org

        getter = getattr(self.config_manager, "get_openai_api_key", None)
        if not callable(getter):
            self.logger.error("Configuration backend does not expose an OpenAI API key accessor.")
            return {
                "models": [],
                "error": "OpenAI credentials are unavailable.",
                "base_url": effective_base_url,
                "organization": effective_org,
            }

        api_key = getter() or ""
        if not api_key:
            return {
                "models": [],
                "error": "OpenAI API key is not configured.",
                "base_url": effective_base_url,
                "organization": effective_org,
            }

        endpoint = f"{effective_base_url.rstrip('/')}/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if effective_org:
            headers["OpenAI-Organization"] = effective_org

        def _fetch_models() -> Dict[str, Any]:
            request = Request(endpoint, headers=headers, method="GET")
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - trusted URL built from config
                encoding = response.headers.get_content_charset("utf-8")
                payload = response.read().decode(encoding)
            return json.loads(payload)

        try:
            raw_response = await asyncio.to_thread(_fetch_models)
        except HTTPError as exc:
            detail = f"HTTP {exc.code}: {exc.reason}"
            try:
                body = exc.read()
                if body:
                    detail = f"{detail} - {body.decode('utf-8', 'ignore')}"
            except Exception:  # pragma: no cover - best effort logging
                pass
            self.logger.error("OpenAI model listing failed with HTTP error: %s", detail, exc_info=True)
            return {
                "models": [],
                "error": detail,
                "base_url": effective_base_url,
                "organization": effective_org,
            }
        except URLError as exc:
            detail = getattr(exc, "reason", None) or str(exc)
            self.logger.error("OpenAI model listing failed with network error: %s", detail, exc_info=True)
            return {
                "models": [],
                "error": str(detail),
                "base_url": effective_base_url,
                "organization": effective_org,
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Unexpected error while listing OpenAI models: %s", exc, exc_info=True)
            return {
                "models": [],
                "error": str(exc),
                "base_url": effective_base_url,
                "organization": effective_org,
            }

        entries: List[Any] = []
        if isinstance(raw_response, dict):
            data = raw_response.get("data")
            if isinstance(data, list):
                entries = data
        elif isinstance(raw_response, list):
            entries = raw_response

        models: List[str] = []
        for entry in entries:
            model_id = None
            if isinstance(entry, dict):
                model_id = entry.get("id")
            else:
                model_id = getattr(entry, "id", None)

            if isinstance(model_id, str) and model_id:
                models.append(model_id)

        unique_models = sorted(set(models))
        prioritized = [
            name
            for name in unique_models
            if any(token in name for token in ("gpt", "omni", "o1", "o3", "chat"))
        ]
        if prioritized:
            prioritized_set = set(prioritized)
            trailing = [name for name in unique_models if name not in prioritized_set]
            unique_models = prioritized + trailing

        if unique_models:
            try:
                self.model_manager.update_models_for_provider("OpenAI", unique_models)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(
                    "Failed to update cached OpenAI models after discovery: %s",
                    exc,
                    exc_info=True,
                )

        return {
            "models": unique_models,
            "error": None,
            "base_url": effective_base_url,
            "organization": effective_org,
        }

    async def fetch_mistral_models(
        self, *, base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Discover available models from the Mistral API and cache them locally."""

        if Mistral is None:
            self.logger.error(
                "Mistral SDK is unavailable; cannot discover remote models."
            )
            return self._build_result(False, error="Mistral SDK is not installed.")

        getter = getattr(self.config_manager, "get_mistral_api_key", None)
        if not callable(getter):
            self.logger.error(
                "Configuration backend does not expose a Mistral API key accessor."
            )
            return self._build_result(False, error="Mistral credentials are unavailable.")

        api_key = getter() or ""
        if not api_key:
            return self._build_result(False, error="Mistral API key is not configured.")

        settings: Dict[str, Any] = {}
        try:
            settings = self.config_manager.get_mistral_llm_settings()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Unable to load persisted Mistral defaults before discovery: %s",
                exc,
                exc_info=True,
            )

        effective_base_url = base_url
        if isinstance(effective_base_url, str):
            effective_base_url = effective_base_url.strip() or None
        if effective_base_url is None:
            candidate = settings.get("base_url") if isinstance(settings, dict) else None
            if isinstance(candidate, str):
                candidate = candidate.strip()
                effective_base_url = candidate or None

        def _list_models() -> Any:
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if effective_base_url:
                client_kwargs["server_url"] = effective_base_url
            client = Mistral(**client_kwargs)
            try:
                return client.models.list()
            finally:  # pragma: no cover - best effort cleanup
                closer = getattr(client, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass

        try:
            raw_response = await asyncio.to_thread(_list_models)
        except MistralException as exc:
            detail = str(exc) or exc.__class__.__name__
            self.logger.error(
                "Mistral model listing failed with API error: %s", detail, exc_info=True
            )
            return self._build_result(False, error=detail)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Unexpected error while listing Mistral models: %s", exc, exc_info=True
            )
            return self._build_result(False, error=str(exc))

        entries: List[Any] = []
        if isinstance(raw_response, dict):
            entries = raw_response.get("data") or raw_response.get("models") or []
        elif isinstance(raw_response, list):
            entries = raw_response
        else:
            entries = getattr(raw_response, "data", [])

        if not isinstance(entries, list):
            entries = []

        seen: set[str] = set()
        discovered: List[str] = []

        for entry in entries:
            model_id: Optional[str] = None
            if isinstance(entry, str):
                model_id = entry
            elif isinstance(entry, dict):
                model_id = (
                    entry.get("id")
                    or entry.get("slug")
                    or entry.get("name")
                    or entry.get("model")
                )
            else:
                for attr in ("id", "slug", "name", "model"):
                    value = getattr(entry, attr, None)
                    if value:
                        model_id = value
                        break

            if model_id is None:
                continue

            normalized = str(model_id).strip()
            if normalized and normalized not in seen:
                discovered.append(normalized)
                seen.add(normalized)

        cached_models: List[str] = list(discovered)
        try:
            cached_models = self.model_manager.update_models_for_provider(
                "Mistral", discovered
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Failed to update cached Mistral models after discovery: %s",
                exc,
                exc_info=True,
            )

        persisted_path = self._persist_mistral_model_cache(cached_models)

        count = len(cached_models)
        message = (
            f"Retrieved {count} Mistral model{'s' if count != 1 else ''}."
            if cached_models
            else "No Mistral models were returned."
        )

        data: Dict[str, Any] = {"models": cached_models, "source": "mistral"}
        if effective_base_url:
            data["base_url"] = effective_base_url
        if persisted_path is not None:
            data["persisted_to"] = str(persisted_path)

        return self._build_result(True, message=message, data=data)

    def _persist_mistral_model_cache(self, models: List[str]) -> Optional[Path]:
        """Write the cached model list to disk for offline usage."""

        payload = {"models": models}

        candidates: List[Path] = []
        try:
            root = self.config_manager.get_app_root()
        except Exception:  # pragma: no cover - defensive logging
            root = None

        if root:
            candidates.append(
                Path(root)
                / "modules"
                / "Providers"
                / "Mistral"
                / "M_models.json"
            )

        candidates.append(
            Path(__file__).resolve().parent.parent
            / "modules"
            / "Providers"
            / "Mistral"
            / "M_models.json"
        )

        for path in candidates:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(
                    "Failed to persist Mistral model cache to %s: %s", path, exc
                )
                continue

            return path

        return None

    @staticmethod
    def _mask_secret(secret: str) -> str:
        """Return a sanitized preview for a stored secret without leaking its value."""

        if not secret:
            return ""

        visible_count = min(len(secret), 8)
        return "•" * visible_count

    def get_provider_api_key_status(self, provider_name: str) -> Dict[str, Any]:
        """Return whether a provider key exists along with sanitized metadata."""

        metadata: Dict[str, Any] = {}
        has_key = False

        provider_values: Dict[str, Any] = {}
        getter = getattr(self.config_manager, "get_available_providers", None)
        if callable(getter):
            try:
                provider_values = getter() or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(
                    "Unable to read provider credentials for %s: %s",
                    provider_name,
                    exc,
                    exc_info=True,
                )
                provider_values = {}

        raw_value = provider_values.get(provider_name)

        if isinstance(raw_value, Mapping):
            has_key = bool(raw_value.get("available"))
            if has_key:
                metadata = {}
                for key in ("length", "hint", "source"):
                    if key in raw_value:
                        metadata[key] = raw_value[key]

                if "length" not in metadata:
                    metadata["length"] = int(raw_value.get("length", 0))

                if "hint" not in metadata:
                    metadata["hint"] = self._mask_secret("x" * metadata["length"])

                metadata["source"] = metadata.get("source", "environment")
        else:
            if isinstance(raw_value, str):
                stored_secret = raw_value
            elif raw_value is None:
                stored_secret = ""
            else:
                stored_secret = str(raw_value)

            if stored_secret:
                has_key = True
                metadata = {
                    "length": len(stored_secret),
                    "hint": self._mask_secret(stored_secret),
                    "source": "environment",
                }

        if not has_key:
            checker = getattr(self.config_manager, "has_provider_api_key", None)
            if callable(checker):
                try:
                    has_key = bool(checker(provider_name))
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.error(
                        "Failed to verify API key presence for %s: %s",
                        provider_name,
                        exc,
                        exc_info=True,
                    )

        return {"has_key": has_key, "metadata": metadata}

    def save_huggingface_token(self, token: Optional[str]) -> Dict[str, Any]:
        """Persist a Hugging Face API token and refresh provider state when needed."""

        normalized = (token or "").strip()
        if not normalized:
            return self._build_result(False, error="Hugging Face token cannot be empty.")

        setter = getattr(self.config_manager, "set_hf_token", None)
        if not callable(setter):
            self.logger.error("Config manager does not support saving Hugging Face tokens.")
            return self._build_result(
                False,
                error="Configuration backend does not support saving a Hugging Face token.",
            )

        try:
            setter(normalized)
        except FileNotFoundError as exc:
            self.logger.error("Failed to persist Hugging Face token: %s", exc, exc_info=True)
            return self._build_result(
                False,
                error="Unable to save Hugging Face token because the .env file could not be located.",
            )
        except ValueError as exc:
            self.logger.error("Rejected Hugging Face token: %s", exc)
            return self._build_result(False, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Unexpected error while saving Hugging Face token: %s", exc, exc_info=True)
            return self._build_result(False, error=str(exc))

        refresh_message = ""
        if self.huggingface_generator is not None:
            previous_generator = self.huggingface_generator
            try:
                previous_generator.unload_model()
            except Exception as exc:  # pragma: no cover - defensive cleanup logging
                self.logger.warning(
                    "Failed to unload HuggingFace model during token refresh: %s",
                    exc,
                    exc_info=True,
                )
            if self.current_llm_provider == "HuggingFace":
                self.current_model = None
            self._provider_model_ready["HuggingFace"] = False
            self.huggingface_generator = None
            refresh_result = self.ensure_huggingface_ready()
            if not refresh_result.get("success"):
                self.logger.error(
                    "Token saved but failed to refresh HuggingFace generator: %s",
                    refresh_result.get("error"),
                )
                self.huggingface_generator = previous_generator
                return self._build_result(
                    False,
                    error="Hugging Face token saved but provider refresh failed: "
                    + refresh_result.get("error", "Unknown error"),
                )
            refresh_message = refresh_result.get("message", "")

        message = "Hugging Face token saved."
        if refresh_message:
            message = f"{message} {refresh_message}"

        self.logger.info("Hugging Face token updated and provider refreshed.")
        return self._build_result(True, message=message)

    def ensure_huggingface_ready(self) -> Dict[str, Any]:
        """Create the HuggingFace generator if it does not already exist."""
        if self.huggingface_generator is not None:
            return self._build_result(True, message="HuggingFace generator already initialized.")

        try:
            self.huggingface_generator = HuggingFaceGenerator(self.config_manager)
            self.logger.info("HuggingFace generator initialized successfully.")
            return self._build_result(True, message="HuggingFace generator initialized.")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to initialize HuggingFace generator: %s", exc, exc_info=True)
            self.huggingface_generator = None
            return self._build_result(False, error=str(exc))

    async def test_huggingface_token(self, token: Optional[str]) -> Dict[str, Any]:
        """Validate a HuggingFace token using the hub API."""

        ensure_result = self.ensure_huggingface_ready()
        if not ensure_result.get("success"):
            return ensure_result

        configured_token: str = token or ""
        if not configured_token:
            getter = getattr(self.config_manager, "get_huggingface_api_key", None)
            if callable(getter):
                configured_token = getter() or ""

        if not configured_token:
            return self._build_result(False, error="No HuggingFace token provided.")

        try:
            api = HfApi()
            whoami_data = await asyncio.to_thread(api.whoami, token=configured_token)
            display_name = ""
            if isinstance(whoami_data, dict):
                display_name = (
                    whoami_data.get("name")
                    or whoami_data.get("fullname")
                    or whoami_data.get("email")
                    or ""
                )
            message = "Token verified successfully."
            if display_name:
                message = f"Token OK. Signed in as: {display_name}"
            return self._build_result(True, message=message, data=whoami_data)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to validate HuggingFace token: %s", exc, exc_info=True)
            return self._build_result(False, error=str(exc))

    async def load_hf_model(self, model_name: str, force_download: bool = False) -> Dict[str, Any]:
        """Load a HuggingFace model, instantiating the generator when needed."""
        ensure_result = self.ensure_huggingface_ready()
        if not ensure_result.get("success"):
            return ensure_result

        try:
            await self.huggingface_generator.load_model(model_name, force_download)
            self.model_manager.set_model(model_name, "HuggingFace")
            if self.current_llm_provider == "HuggingFace":
                self.current_model = model_name
            self._provider_model_ready["HuggingFace"] = True
            self._pending_models.pop("HuggingFace", None)
            message = f"Model '{model_name}' loaded successfully."
            return self._build_result(True, message=message)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to load HuggingFace model %s: %s", model_name, exc, exc_info=True)
            return self._build_result(False, error=str(exc))

    async def unload_hf_model(self) -> Dict[str, Any]:
        """Unload the currently active HuggingFace model if one is loaded."""
        if not self.huggingface_generator:
            return self._build_result(True, message="No HuggingFace model loaded.")

        try:
            await asyncio.to_thread(self.huggingface_generator.unload_model)
            if self.current_llm_provider == "HuggingFace":
                self.current_model = None
            self._provider_model_ready["HuggingFace"] = False
            return self._build_result(True, message="HuggingFace model unloaded.")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to unload HuggingFace model: %s", exc, exc_info=True)
            return self._build_result(False, error=str(exc))

    async def remove_hf_model(self, model_name: str) -> Dict[str, Any]:
        """Remove a cached HuggingFace model from disk."""
        ensure_result = self.ensure_huggingface_ready()
        if not ensure_result.get("success"):
            return ensure_result

        try:
            await asyncio.to_thread(
                self.huggingface_generator.model_manager.remove_installed_model,
                model_name,
            )
            message = f"Model '{model_name}' removed successfully."
            return self._build_result(True, message=message)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to remove HuggingFace model %s: %s", model_name, exc, exc_info=True)
            return self._build_result(False, error=str(exc))

    def list_hf_models(self) -> Dict[str, Any]:
        """List installed HuggingFace models."""
        ensure_result = self.ensure_huggingface_ready()
        if not ensure_result.get("success"):
            return ensure_result

        try:
            models = self.huggingface_generator.get_installed_models()
            return self._build_result(True, data=models, message="Retrieved installed HuggingFace models.")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to list HuggingFace models: %s", exc, exc_info=True)
            return self._build_result(False, error=str(exc))

    async def search_huggingface_models(
        self,
        search_query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search HuggingFace models using shared backend helpers."""

        ensure_result = self.ensure_huggingface_ready()
        if not ensure_result.get("success"):
            return ensure_result

        try:
            results = await hf_search_models(self.huggingface_generator, search_query, filters, limit)
            return self._build_result(True, data=results, message="Search completed.")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to search HuggingFace models '%s': %s", search_query, exc, exc_info=True)
            return self._build_result(False, error=str(exc))

    async def download_huggingface_model(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        """Download a HuggingFace model without loading it into memory."""

        ensure_result = self.ensure_huggingface_ready()
        if not ensure_result.get("success"):
            return ensure_result

        try:
            await hf_download_model(self.huggingface_generator, model_id, force)
            message = f"Model '{model_id}' downloaded successfully."
            return self._build_result(True, message=message)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to download HuggingFace model %s: %s", model_id, exc, exc_info=True)
            return self._build_result(False, error=str(exc))

    def update_huggingface_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Persist HuggingFace model settings via shared helper."""

        ensure_result = self.ensure_huggingface_ready()
        if not ensure_result.get("success"):
            return ensure_result

        try:
            updated = hf_update_model_settings(self.huggingface_generator, settings)
            return self._build_result(True, data=updated, message="Settings updated successfully.")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to update HuggingFace settings: %s", exc, exc_info=True)
            return self._build_result(False, error=str(exc))

    def clear_huggingface_cache(self) -> Dict[str, Any]:
        """Clear HuggingFace caches using shared helper."""

        ensure_result = self.ensure_huggingface_ready()
        if not ensure_result.get("success"):
            return ensure_result

        try:
            hf_clear_cache(self.huggingface_generator)
            return self._build_result(True, message="HuggingFace cache cleared.")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to clear HuggingFace cache: %s", exc, exc_info=True)
            return self._build_result(False, error=str(exc))

    @classmethod
    def providers(cls) -> List[str]:
        """
        Get a list of all available LLM providers.

        Returns:
            List[str]: A list of provider names.
        """
        return cls.AVAILABLE_PROVIDERS

    def get_available_providers(self) -> List[str]:
        """
        Get a list of all available LLM providers.

        Returns:
            List[str]: A list of provider names.
        """
        return self.__class__.providers()

    async def switch_llm_provider(self, llm_provider: str):
        """
        Switches the current LLM provider to the specified provider.

        Args:
            llm_provider (str): The name of the LLM provider to switch to.
        """
        # Only skip initialization if the provider is already set and generate_response_func is initialized
        if llm_provider == self.current_llm_provider and self.generate_response_func is not None:
            self.logger.info(f"Provider {llm_provider} is already the current provider and is initialized. No action taken.")
            return

        # Validate provider
        if llm_provider not in self.AVAILABLE_PROVIDERS:
            self.logger.warning(f"Provider {llm_provider} is not implemented. Reverting to default provider OpenAI.")
            llm_provider = "OpenAI"

        self.logger.info(f"Attempting to switch to provider: {llm_provider}")

        previous_provider = self.current_llm_provider

        try:
            if previous_provider and previous_provider != llm_provider:
                await self._cleanup_provider_generator(previous_provider)

            self.current_llm_provider = llm_provider

            if llm_provider == "OpenAI":
                self._provider_model_ready["OpenAI"] = False
                self._pending_models.pop("OpenAI", None)
                await self._cleanup_provider_generator("Grok")
                await self._cleanup_provider_generator("HuggingFace")
                openai_generator = self._ensure_openai_generator()
                self.generate_response_func = openai_generator.generate_response
                self.process_streaming_response_func = openai_generator.process_streaming_response

                default_model = self.get_default_model_for_provider("OpenAI")
                if default_model:
                    await self.set_model(default_model)
                else:
                    self.logger.error("No default model found for OpenAI. Ensure models are configured correctly.")
                    raise ValueError("No default model available for OpenAI provider.")

            elif llm_provider == "Mistral":
                self._provider_model_ready["Mistral"] = False
                self._pending_models.pop("Mistral", None)
                mistral_generator = self._ensure_mistral_generator()
                self.generate_response_func = mistral_generator.generate_response
                self.process_streaming_response_func = getattr(
                    mistral_generator, "process_response", None
                )
                await self._cleanup_provider_generator("Grok")
                await self._cleanup_provider_generator("HuggingFace")
                default_model = self.get_default_model_for_provider("Mistral")
                if default_model:
                    await self.set_model(default_model)
                else:
                    self.logger.error("No default model found for Mistral. Ensure models are configured correctly.")
                    raise ValueError("No default model available for Mistral provider.")

            elif llm_provider == "Google":
                self._provider_model_ready["Google"] = False
                self._pending_models.pop("Google", None)
                google_generator = self._ensure_google_generator()
                self.generate_response_func = google_generator.generate_response
                self.process_streaming_response_func = getattr(
                    google_generator, "process_response", None
                )
                await self._cleanup_provider_generator("Grok")
                await self._cleanup_provider_generator("HuggingFace")
                default_model = self.get_default_model_for_provider("Google")
                if default_model:
                    await self.set_model(default_model)
                else:
                    self.logger.error("No default model found for Google. Ensure models are configured correctly.")
                    raise ValueError("No default model available for Google provider.")

            elif llm_provider == "HuggingFace":
                await self._cleanup_provider_generator("Grok")
                # Reset any previously selected model when switching to HuggingFace.
                self.current_model = None
                self._provider_model_ready["HuggingFace"] = False
                self._pending_models.pop("HuggingFace", None)
                ensure_result = self.ensure_huggingface_ready()
                if not ensure_result.get("success"):
                    raise ValueError(ensure_result.get("error", "Failed to initialize HuggingFace generator."))

                self.generate_response_func = self.huggingface_generator.generate_response
                self.process_streaming_response_func = self.huggingface_generator.process_streaming_response
                default_model = self.get_default_model_for_provider("HuggingFace")
                if default_model:
                    self._pending_models["HuggingFace"] = default_model
                    self.logger.info(
                        "HuggingFace default model '%s' recorded for deferred loading.",
                        default_model,
                    )
                else:
                    self.logger.warning(
                        "No default model found for HuggingFace. The provider is active without a loaded model."
                    )
                    # Ensure downstream logic knows no model is loaded yet.
                    if hasattr(self.model_manager, "current_model"):
                        self.model_manager.current_model = None
                    if hasattr(self.model_manager, "current_provider"):
                        self.model_manager.current_provider = "HuggingFace"
                    self._pending_models["HuggingFace"] = None
            elif llm_provider == "Anthropic":
                self._provider_model_ready["Anthropic"] = False
                self._pending_models.pop("Anthropic", None)
                await self._cleanup_provider_generator("Grok")
                await self._cleanup_provider_generator("HuggingFace")
                generator = self._ensure_anthropic_generator()
                self.generate_response_func = self._anthropic_generate_response
                self.process_streaming_response_func = generator.process_streaming_response

                default_model = generator.default_model or self.get_default_model_for_provider("Anthropic")
                if default_model:
                    await self.set_model(default_model)
                else:
                    self.logger.error("No default model found for Anthropic. Ensure models are configured correctly.")
                    raise ValueError("No default model available for Anthropic provider.")

            elif llm_provider == "Grok":
                self._provider_model_ready["Grok"] = False
                self._pending_models.pop("Grok", None)
                await self._cleanup_provider_generator("HuggingFace")
                await self._cleanup_provider_generator("Grok")
                self.grok_generator = GrokGenerator(self.config_manager)
                self.generate_response_func = self.grok_generator.generate_response
                self.process_streaming_response_func = self.grok_generator.process_streaming_response
                default_model = self.get_default_model_for_provider("Grok")
                if default_model:
                    self.current_model = default_model
                    self._provider_model_ready["Grok"] = True
                else:
                    self.logger.error("No default model found for Grok. Ensure models are configured correctly.")
                    raise ValueError("No default model available for Grok provider.")
                # Initialize Grok-specific settings if necessary

            else:
                self.logger.warning(f"Provider {llm_provider} is not recognized. Reverting to OpenAI.")
                openai_generator = self._ensure_openai_generator()
                self.generate_response_func = openai_generator.generate_response
                self.process_streaming_response_func = openai_generator.process_streaming_response
                await self._cleanup_provider_generator("Grok")
                await self._cleanup_provider_generator("HuggingFace")
                default_model = self.get_default_model_for_provider("OpenAI")
                if default_model:
                    await self.set_model(default_model)
                else:
                    self.logger.error("No default model found for OpenAI. Ensure models are configured correctly.")
                    raise ValueError("No default model available for OpenAI provider.")

            self.current_llm_provider = llm_provider
            self.providers[llm_provider] = self.generate_response_func

            self.logger.info(f"Switched to LLM provider: {self.current_llm_provider}")
            if self.current_model:
                self.logger.info(f"Current model set to: {self.current_model}")

        except Exception as e:
            self.logger.error(f"Failed to switch to provider {llm_provider}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.current_llm_provider = previous_provider
            raise

    async def set_current_provider(self, provider: str):
        """
        Set the current provider to the specified provider.

        Args:
            provider (str): The name of the provider to set.
        """
        if not self.config_manager.has_provider_api_key(provider):
            raise ValueError(
                f"API key for provider '{provider}' is not configured. "
                "Please add it in the provider settings before selecting this provider."
            )

        await self.switch_llm_provider(provider)
        self.logger.info(f"Current provider set to {self.current_llm_provider}")
        if self.current_model:
            self.logger.info(f"Current model set to: {self.current_model}")

    def get_current_model(self) -> str:
        """
        Get the current model being used.

        Returns:
            str or None: The name of the current model, or None if not set.
        """
        if self.current_llm_provider == "HuggingFace" and self.huggingface_generator:
            getter = getattr(self.huggingface_generator, "get_current_model", None)
            if callable(getter):
                return getter()
        return self.current_model

    def set_current_functions(self, functions):
        """
        Set the current functions for the provider.

        Args:
            functions: The functions to set.
        """
        self.current_functions = functions
        self.logger.debug(f"Updated current functions: {self.current_functions}")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        provider: str = None,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None,
        stop_sequences: Optional[Any] = None,
        safety_settings: Optional[Any] = None,
        response_mime_type: Optional[str] = None,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Any] = None,
        seed: Optional[int] = None,
        response_logprobs: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: Optional[bool] = None,
        current_persona=None,
        functions=None,
        reasoning_effort: Optional[str] = None,
        llm_call_type: str = None,
        user: Optional[str] = None,
        conversation_id: Optional[str] = None,
        conversation_manager: Optional[Any] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate a response using the specified provider and model, or the current ones if not specified.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            model (str, optional): The model to use. If None, uses the current model.
            provider (str, optional): The provider to use. If None, uses the current provider.
            max_tokens (int, optional): Maximum number of tokens. Uses saved default when omitted.
            max_output_tokens (int, optional): Maximum number of output tokens for reasoning models.
            temperature (float, optional): Sampling temperature. Uses saved default when omitted.
            top_p (float, optional): Nucleus sampling value. Uses saved default when omitted.
            safety_settings (optional): Safety settings for supported providers.
            response_mime_type (optional): MIME type hint when supported.
            system_instruction (optional): Default system instruction payload.
            response_schema (optional): Structured response schema when supported.
            seed (int, optional): Deterministic seed when supported by the provider.
            response_logprobs (bool, optional): Request log probabilities when available.
            frequency_penalty (float, optional): Frequency penalty. Uses saved default when omitted.
            presence_penalty (float, optional): Presence penalty. Uses saved default when omitted.
            stream (bool, optional): Whether to stream the response. Uses saved default when omitted.
            current_persona (optional): The current persona.
            functions (optional): Functions to use.
            reasoning_effort (str, optional): Effort level for reasoning models.
            llm_call_type (str, optional): The type of LLM call.
            user (str, optional): Identifier for the active user.
            conversation_id (str, optional): Identifier for the active conversation.
            conversation_manager (optional): Conversation manager used for logging results.

        Returns:
            Union[str, AsyncIterator[str]]: The generated response or a stream of tokens.
        """
        # Determine provider and model
        requested_provider = provider if provider else self.current_llm_provider
        if not requested_provider:
            requested_provider = self.config_manager.get_default_provider()

        desired_provider = requested_provider
        if desired_provider != self.current_llm_provider:
            await self.switch_llm_provider(desired_provider)

        requested_provider = self.current_llm_provider or desired_provider

        defaults: Dict[str, Any] = {}
        if requested_provider == "OpenAI":
            defaults = self.get_openai_llm_settings()
        elif requested_provider == "Google":
            defaults = self.get_google_llm_settings()

        current_active_model = self.get_current_model()
        resolved_model = model or defaults.get("model") or current_active_model
        if not resolved_model:
            fallback_model = self.get_default_model_for_provider(requested_provider)
            resolved_model = fallback_model

        default_max_tokens = defaults.get("max_tokens")
        default_max_output_tokens = defaults.get("max_output_tokens")
        resolved_max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else default_max_output_tokens
        )
        if requested_provider == "Google":
            if max_tokens is not None:
                resolved_max_tokens = max_tokens
            elif resolved_max_output_tokens is not None:
                resolved_max_tokens = resolved_max_output_tokens
            else:
                resolved_max_tokens = default_max_tokens
        else:
            fallback_max_tokens = default_max_tokens
            if fallback_max_tokens is None and requested_provider == "OpenAI":
                fallback_max_tokens = 4000
            resolved_max_tokens = (
                max_tokens if max_tokens is not None else fallback_max_tokens
            )

        resolved_temperature = (
            temperature if temperature is not None else defaults.get("temperature", 0.0)
        )
        resolved_top_p = top_p if top_p is not None else defaults.get("top_p", 1.0)
        resolved_top_k = top_k if top_k is not None else defaults.get("top_k")
        resolved_candidate_count = (
            candidate_count if candidate_count is not None else defaults.get("candidate_count")
        )
        resolved_stop_sequences = (
            stop_sequences if stop_sequences is not None else defaults.get("stop_sequences")
        )
        resolved_safety_settings = (
            safety_settings if safety_settings is not None else defaults.get("safety_settings")
        )
        resolved_response_mime_type = (
            response_mime_type
            if response_mime_type is not None
            else defaults.get("response_mime_type")
        )
        resolved_system_instruction = (
            system_instruction
            if system_instruction is not None
            else defaults.get("system_instruction")
        )
        resolved_response_schema = (
            response_schema
            if response_schema is not None
            else defaults.get("response_schema")
        )
        if isinstance(resolved_response_schema, str) and not resolved_response_schema.strip():
            resolved_response_schema = None
        if isinstance(resolved_response_schema, dict) and not resolved_response_schema:
            resolved_response_schema = None
        resolved_seed = seed if seed is not None else defaults.get("seed")
        if isinstance(resolved_seed, str) and not str(resolved_seed).strip():
            resolved_seed = None
        resolved_response_logprobs = (
            response_logprobs
            if response_logprobs is not None
            else defaults.get("response_logprobs")
        )
        resolved_frequency_penalty = (
            frequency_penalty
            if frequency_penalty is not None
            else defaults.get("frequency_penalty", 0.0)
        )
        resolved_presence_penalty = (
            presence_penalty
            if presence_penalty is not None
            else defaults.get("presence_penalty", 0.0)
        )
        resolved_stream = stream if stream is not None else defaults.get("stream", True)
        resolved_function_calling = defaults.get("function_calling", True)
        if resolved_function_calling is None:
            resolved_function_calling = True
        resolved_parallel_tool_calls = defaults.get("parallel_tool_calls", True)
        resolved_tool_choice = defaults.get("tool_choice")
        if resolved_function_calling is False:
            resolved_tool_choice = "none"
        resolved_reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else defaults.get("reasoning_effort")
        )

        resolved_audio_enabled = defaults.get("audio_enabled", False)
        resolved_audio_voice = defaults.get("audio_voice")
        resolved_audio_format = defaults.get("audio_format")

        # Log the incoming parameters
        self.logger.info(
            "Generating response with Provider: %s, Model: %s, Persona: %s",
            requested_provider,
            resolved_model,
            current_persona,
        )

        # Switch model if different
        if resolved_model and resolved_model != current_active_model:
            await self.set_model(resolved_model)

        # Use current functions if not provided
        if functions is None:
            functions = self.current_functions

        if resolved_function_calling is False:
            functions = None

        start_time = time.time()
        self.logger.info(
            "Starting API call to %s with model %s for %s",
            requested_provider,
            resolved_model,
            llm_call_type,
        )

        try:
            if conversation_id:
                self.set_current_conversation_id(conversation_id)

            active_conversation_id = conversation_id or getattr(self, "current_conversation_id", None)
            active_conversation_manager = (
                conversation_manager
                if conversation_manager is not None
                else getattr(self, "conversation_manager", None)
            )

            if not self.generate_response_func:
                self.logger.error("No response generation function is set for the current provider.")
                raise ValueError("generate_response_func is None. Ensure the provider is properly initialized.")

            call_kwargs = {
                "messages": messages,
                "model": resolved_model,
                "max_tokens": resolved_max_tokens,
                "temperature": resolved_temperature,
                "stream": resolved_stream,
                "current_persona": current_persona,
                "functions": functions,
                "user": user,
                "conversation_id": active_conversation_id,
                "conversation_manager": active_conversation_manager,
            }

            if requested_provider == "OpenAI":
                call_kwargs.update(
                    top_p=resolved_top_p,
                    frequency_penalty=resolved_frequency_penalty,
                    presence_penalty=resolved_presence_penalty,
                    function_calling=resolved_function_calling,
                    parallel_tool_calls=resolved_parallel_tool_calls,
                    tool_choice=resolved_tool_choice,
                    max_output_tokens=resolved_max_output_tokens,
                    reasoning_effort=resolved_reasoning_effort,
                    audio_enabled=resolved_audio_enabled,
                    audio_voice=resolved_audio_voice,
                    audio_format=resolved_audio_format,
                )
            elif requested_provider == "Anthropic":
                call_kwargs.pop("max_tokens", None)
                call_kwargs.update(
                    top_p=resolved_top_p,
                    max_output_tokens=resolved_max_output_tokens,
                )
            elif requested_provider == "Google":
                call_kwargs.update(
                    top_p=resolved_top_p,
                    top_k=resolved_top_k,
                    candidate_count=resolved_candidate_count,
                    stop_sequences=resolved_stop_sequences,
                    safety_settings=resolved_safety_settings,
                    response_mime_type=resolved_response_mime_type,
                    system_instruction=resolved_system_instruction,
                    response_schema=resolved_response_schema,
                    enable_functions=bool(resolved_function_calling),
                )
                if resolved_seed is not None:
                    call_kwargs["seed"] = resolved_seed
                if resolved_response_logprobs is not None:
                    call_kwargs["response_logprobs"] = bool(resolved_response_logprobs)
                if resolved_max_output_tokens is not None and max_tokens is None:
                    call_kwargs["max_tokens"] = resolved_max_output_tokens

            response = await self._invoke_provider_callable(
                requested_provider,
                self.generate_response_func,
                dict(call_kwargs),
            )

            self.logger.info(f"API call completed in {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            self.logger.error(f"Error during API call to {requested_provider}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

            fallback_kwargs = dict(call_kwargs)
            fallback_kwargs.pop("messages", None)

            try:
                fallback_response = await self._use_fallback(
                    messages,
                    llm_call_type or "default",
                    **fallback_kwargs,
                )
            except Exception as fallback_exc:
                self.logger.error(
                    "Fallback provider invocation failed: %s", fallback_exc, exc_info=True
                )
                raise

            return fallback_response

    async def _use_fallback(self, messages: List[Dict[str, str]], llm_call_type: str, **kwargs) -> str:
        """
        Use a fallback provider to generate a response.
        This method handles cases where the primary provider fails.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            llm_call_type (str): The type of LLM call.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated response from the fallback provider.
        """
        fallback_getter = getattr(self.config_manager, "get_llm_fallback_config", None)
        if not callable(fallback_getter):
            raise ValueError("Fallback configuration is not available.")

        fallback_config = fallback_getter()
        if not isinstance(fallback_config, Mapping):
            raise ValueError("Fallback provider configuration is missing or invalid.")

        fallback_config = dict(fallback_config)
        fallback_provider = fallback_config.get('provider')

        if not fallback_provider:
            raise ValueError("Fallback provider is not defined in the configuration.")

        fallback_function = await self._ensure_provider_callable(fallback_provider)

        self.logger.info(f"Using fallback provider {fallback_provider} for {llm_call_type}")

        current_conversation_id = kwargs.get("conversation_id") or getattr(
            self, "current_conversation_id", None
        )
        conversation_manager = (
            kwargs.get("conversation_manager")
            if kwargs.get("conversation_manager") is not None
            else getattr(self, "conversation_manager", None)
        )

        fallback_kwargs = dict(kwargs)
        fallback_kwargs.update(
            {
                "messages": messages,
                "model": fallback_config.get('model', fallback_kwargs.get('model')),
                "max_tokens": fallback_config.get(
                    'max_tokens', fallback_kwargs.get('max_tokens', 4000)
                ),
                "temperature": fallback_config.get(
                    'temperature', fallback_kwargs.get('temperature', 0.0)
                ),
                "stream": fallback_config.get('stream', fallback_kwargs.get('stream', True)),
                "current_persona": fallback_config.get(
                    'current_persona', fallback_kwargs.get('current_persona')
                ),
                "functions": fallback_config.get('functions', fallback_kwargs.get('functions')),
                "response_schema": fallback_config.get(
                    'response_schema', fallback_kwargs.get('response_schema')
                ),
                "llm_call_type": llm_call_type,
                "user": kwargs.get("user"),
                "conversation_id": current_conversation_id,
                "conversation_manager": conversation_manager,
            }
        )

        if fallback_provider == "OpenAI":
            fallback_kwargs.update(
                top_p=fallback_config.get('top_p', fallback_kwargs.get('top_p', 1.0)),
                frequency_penalty=fallback_config.get(
                    'frequency_penalty', fallback_kwargs.get('frequency_penalty', 0.0)
                ),
                presence_penalty=fallback_config.get(
                    'presence_penalty', fallback_kwargs.get('presence_penalty', 0.0)
                ),
            )
        elif fallback_provider == "Anthropic":
            fallback_kwargs.pop("max_tokens", None)
            fallback_kwargs.update(
                top_p=fallback_config.get('top_p', fallback_kwargs.get('top_p', 1.0)),
                max_output_tokens=fallback_config.get(
                    'max_output_tokens', fallback_kwargs.get('max_output_tokens')
                ),
            )

        result = await self._invoke_provider_callable(
            fallback_provider,
            fallback_function,
            dict(fallback_kwargs),
        )

        self.logger.info(
            "Fallback provider %s succeeded for %s", fallback_provider, llm_call_type
        )
        return result

    async def process_streaming_response(self, response: AsyncIterator[Dict]) -> str:
        """
        Process a streaming response from the LLM provider.

        Args:
            response (AsyncIterator[Dict]): The streaming response.

        Returns:
            str: The full response.
        """
        try:
            if self.current_llm_provider == "HuggingFace":
                return await self.huggingface_generator.process_streaming_response(response)
            elif self.current_llm_provider == "Grok":
                return await self.grok_generator.process_streaming_response(response)
            elif self.process_streaming_response_func:
                return await self.process_streaming_response_func(response)
            else:
                raise ValueError(f"Streaming response processing not implemented for {self.current_llm_provider}")
        except Exception as e:
            self.logger.error(f"Error processing streaming response: {str(e)}")
            raise

    async def set_model(self, model: str):
        """
        Set the current model and load it if necessary.

        Args:
            model (str): The model to set.
        """
        if self.current_llm_provider == "HuggingFace":
            load_result = await self.load_hf_model(model)
            if not load_result.get("success"):
                raise ValueError(load_result.get("error", f"Failed to load HuggingFace model {model}."))
        else:
            self.current_model = model
            self.model_manager.set_model(model, self.current_llm_provider)
            self._provider_model_ready[self.current_llm_provider] = True
            self._pending_models.pop(self.current_llm_provider, None)
        self.logger.info(f"Model set to {model}")

    def switch_background_provider(self, background_provider: str):
        """
        Switch the background provider.

        Args:
            background_provider (str): The name of the background provider.
        """
        if background_provider not in self.AVAILABLE_PROVIDERS:
            self.logger.warning(f"Background provider {background_provider} is not available.")
            return

        self.current_background_provider = background_provider
        self.logger.info(f"Switched background provider to: {self.current_background_provider}")

    def get_current_provider(self) -> str:
        """
        Get the current LLM provider.

        Returns:
            str: The name of the current LLM provider.
        """
        return self.current_llm_provider

    def get_current_background_provider(self) -> str:
        """
        Get the current background provider.

        Returns:
            str: The name of the current background provider.
        """
        return self.current_background_provider

    def is_model_loaded(self, provider: Optional[str] = None) -> bool:
        """Return True if a model is marked as loaded for the given provider."""

        target = provider or self.current_llm_provider
        return bool(self._provider_model_ready.get(target))

    def get_pending_model_for_provider(self, provider: str) -> Optional[str]:
        """Return the recorded model awaiting load for the given provider, if any."""

        return self._pending_models.get(provider)

    def get_default_model_for_provider(self, provider: str) -> str:
        """
        Get the default model for a specific provider.

        Args:
            provider (str): The name of the provider.

        Returns:
            str: The default model name, or None if not available.
        """
        models = self.model_manager.get_available_models(provider).get(provider, [])
        if models:
            return models[0]

        self.logger.warning(
            "No cached models found for provider %s. Falling back to configured defaults.",
            provider,
        )

        def _safe_extract_model(getter_name: str) -> Optional[str]:
            getter = getattr(self, getter_name, None)
            if callable(getter):
                try:
                    settings = getter() or {}
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning(
                        "Failed to load %s for provider %s: %s",
                        getter_name,
                        provider,
                        exc,
                    )
                else:
                    model = settings.get("model")
                    if isinstance(model, str) and model.strip():
                        return model.strip()
            return None

        def _safe_extract_model_from_config(getter_name: str) -> Optional[str]:
            getter = getattr(self.config_manager, getter_name, None)
            if callable(getter):
                try:
                    settings = getter() or {}
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning(
                        "Failed to load %s for provider %s: %s",
                        getter_name,
                        provider,
                        exc,
                    )
                else:
                    model = settings.get("model")
                    if isinstance(model, str) and model.strip():
                        return model.strip()
            return None

        provider_specific_getters = {
            "OpenAI": "get_openai_llm_settings",
            "Google": "get_google_llm_settings",
        }
        config_specific_getters = {
            "Mistral": "get_mistral_llm_settings",
            "Anthropic": "get_anthropic_settings",
        }

        fallback_model = None

        getter_name = provider_specific_getters.get(provider)
        if getter_name:
            fallback_model = _safe_extract_model(getter_name)

        if not fallback_model:
            getter_name = config_specific_getters.get(provider)
            if getter_name:
                fallback_model = _safe_extract_model_from_config(getter_name)

        if not fallback_model:
            default_model_getter = getattr(self.config_manager, "get_default_model", None)
            if callable(default_model_getter):
                try:
                    candidate = default_model_getter()
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning(
                        "Failed to load global default model from configuration: %s",
                        exc,
                    )
                else:
                    if isinstance(candidate, str) and candidate.strip():
                        fallback_model = candidate.strip()

        if fallback_model:
            self.logger.info(
                "Using configured default model '%s' for provider %s.",
                fallback_model,
                provider,
            )
        else:
            self.logger.warning(
                "No configured default model available for provider %s.",
                provider,
            )

        return fallback_model

    async def get_available_models(self) -> List[str]:
        """
        Get available models for the current provider.

        Returns:
            List[str]: A list of available model names.
        """
        return self.model_manager.get_available_models(self.current_llm_provider).get(self.current_llm_provider, [])

    def set_conversation_manager(self, conversation_manager):
        """
        Set the conversation manager.

        Args:
            conversation_manager: The conversation manager instance.
        """
        self.conversation_manager = conversation_manager
        conversation_id_getter = None
        if conversation_manager is not None:
            conversation_id_getter = getattr(conversation_manager, "conversation_id", None)
            if callable(conversation_id_getter):
                try:
                    current_id = conversation_id_getter()
                except TypeError:
                    current_id = conversation_id_getter
                else:
                    self.current_conversation_id = current_id
                    return
            if conversation_id_getter is None:
                getter_method = getattr(conversation_manager, "get_conversation_id", None)
                if callable(getter_method):
                    try:
                        current_id = getter_method()
                    except Exception:  # pragma: no cover - defensive fallback
                        current_id = None
                    else:
                        if current_id is not None:
                            self.current_conversation_id = current_id
                            return
        if conversation_id_getter is not None and not callable(conversation_id_getter):
            self.current_conversation_id = conversation_id_getter

    def set_current_conversation_id(self, conversation_id: str):
        """
        Set the current conversation ID.

        Args:
            conversation_id (str): The conversation ID.
        """
        self.current_conversation_id = conversation_id

    async def close(self):
        """
        Perform cleanup operations, such as unloading models.
        """
        await self._cleanup_provider_generator("HuggingFace")
        await self._cleanup_provider_generator("OpenAI")
        await self._cleanup_provider_generator("Mistral")
        await self._cleanup_provider_generator("Anthropic")
        await self._cleanup_provider_generator("Grok")

        self.generate_response_func = None
        self.process_streaming_response_func = None
        self.current_model = None
        self.logger.info("ProviderManager closed and models unloaded.")
