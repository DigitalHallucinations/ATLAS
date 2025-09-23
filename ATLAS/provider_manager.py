# ATLAS/provider_manager.py

from typing import Any, Dict, List, Union, AsyncIterator, Optional
import asyncio
import json
import time
import traceback
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
from modules.Providers.OpenAI.OA_gen_response import generate_response as openai_generate_response
from modules.Providers.Mistral.Mistral_gen_response import generate_response as mistral_generate_response
from modules.Providers.Google.GG_gen_response import generate_response as google_generate_response
from modules.Providers.Anthropic.Anthropic_gen_response import generate_response as anthropic_generate_response


class ProviderManager:
    """
    Manages interactions with different LLM providers, ensuring only one instance exists.
    """

    AVAILABLE_PROVIDERS = ["OpenAI", "Mistral", "Google", "HuggingFace", "Anthropic", "Grok"]

    _instance = None  # Class variable to hold the singleton instance
    _lock = asyncio.Lock()  # Lock to ensure thread-safe instantiation

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
        self.current_functions = None
        self.providers = {}
        self.chat_session = None  

    @classmethod
    async def create(cls, config_manager: ConfigManager):
        """
        Asynchronous factory method to create or retrieve the singleton instance.
        
        Args:
            config_manager (ConfigManager): An instance of ConfigManager.
        
        Returns:
            ProviderManager: The singleton instance of ProviderManager.
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config_manager)
                await cls._instance.initialize_all_providers()
            return cls._instance

    async def initialize_all_providers(self):
        """
        Initializes the provider managers and sets the default provider.
        """
        await self.switch_llm_provider(self.current_llm_provider)

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
        stream: Optional[bool] = None,
        function_calling: Optional[bool] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
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
                stream=stream,
                function_calling=function_calling,
                base_url=base_url,
                organization=organization,
            )
        except ValueError as exc:
            self.logger.warning("Rejected OpenAI settings update: %s", exc)
            return self._build_result(False, error=str(exc))
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
            unique_models = prioritized

        return {
            "models": unique_models,
            "error": None,
            "base_url": effective_base_url,
            "organization": effective_org,
        }

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
        else:
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

        try:
            if llm_provider == "OpenAI":
                self.generate_response_func = openai_generate_response
                self.process_streaming_response_func = None
                self.grok_generator = None
                self.huggingface_generator = None

                default_model = self.get_default_model_for_provider("OpenAI")
                if default_model:
                    await self.set_model(default_model)
                else:
                    self.logger.error("No default model found for OpenAI. Ensure models are configured correctly.")
                    raise ValueError("No default model available for OpenAI provider.")

            elif llm_provider == "Mistral":
                self.generate_response_func = mistral_generate_response
                self.process_streaming_response_func = None
                self.grok_generator = None
                self.huggingface_generator = None
                default_model = self.get_default_model_for_provider("Mistral")
                if default_model:
                    await self.set_model(default_model)
                else:
                    self.logger.error("No default model found for Mistral. Ensure models are configured correctly.")
                    raise ValueError("No default model available for Mistral provider.")

            elif llm_provider == "Google":
                self.generate_response_func = google_generate_response
                self.process_streaming_response_func = None
                self.grok_generator = None
                self.huggingface_generator = None
                default_model = self.get_default_model_for_provider("Google")
                if default_model:
                    await self.set_model(default_model)
                else:
                    self.logger.error("No default model found for Google. Ensure models are configured correctly.")
                    raise ValueError("No default model available for Google provider.")

            elif llm_provider == "HuggingFace":
                self.grok_generator = None
                # Reset any previously selected model when switching to HuggingFace.
                self.current_model = None
                ensure_result = self.ensure_huggingface_ready()
                if not ensure_result.get("success"):
                    raise ValueError(ensure_result.get("error", "Failed to initialize HuggingFace generator."))

                self.generate_response_func = self.huggingface_generator.generate_response
                self.process_streaming_response_func = self.huggingface_generator.process_streaming_response
                default_model = self.get_default_model_for_provider("HuggingFace")
                if default_model:
                    load_result = await self.load_hf_model(default_model)
                    if not load_result.get("success"):
                        raise ValueError(load_result.get("error", "Failed to load default HuggingFace model."))
                else:
                    self.logger.warning(
                        "No default model found for HuggingFace. The provider is active without a loaded model."
                    )
                    # Ensure downstream logic knows no model is loaded yet.
                    if hasattr(self.model_manager, "current_model"):
                        self.model_manager.current_model = None
                    if hasattr(self.model_manager, "current_provider"):
                        self.model_manager.current_provider = "HuggingFace"
            elif llm_provider == "Anthropic":
                self.generate_response_func = anthropic_generate_response
                self.process_streaming_response_func = None
                self.grok_generator = None
                self.huggingface_generator = None
                default_model = self.get_default_model_for_provider("Anthropic")
                if default_model:
                    await self.set_model(default_model)
                else:
                    self.logger.error("No default model found for Anthropic. Ensure models are configured correctly.")
                    raise ValueError("No default model available for Anthropic provider.")

            elif llm_provider == "Grok":
                self.huggingface_generator = None
                self.grok_generator = GrokGenerator(self.config_manager)
                self.generate_response_func = self.grok_generator.generate_response
                self.process_streaming_response_func = self.grok_generator.process_streaming_response
                default_model = self.get_default_model_for_provider("Grok")
                if default_model:
                    self.current_model = default_model
                else:
                    self.logger.error("No default model found for Grok. Ensure models are configured correctly.")
                    raise ValueError("No default model available for Grok provider.")
                # Initialize Grok-specific settings if necessary

            else:
                self.logger.warning(f"Provider {llm_provider} is not recognized. Reverting to OpenAI.")
                self.generate_response_func = openai_generate_response
                self.process_streaming_response_func = None
                self.grok_generator = None
                self.huggingface_generator = None
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
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: Optional[bool] = None,
        current_persona=None,
        functions=None,
        llm_call_type: str = None
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate a response using the specified provider and model, or the current ones if not specified.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            model (str, optional): The model to use. If None, uses the current model.
            provider (str, optional): The provider to use. If None, uses the current provider.
            max_tokens (int, optional): Maximum number of tokens. Uses saved default when omitted.
            temperature (float, optional): Sampling temperature. Uses saved default when omitted.
            top_p (float, optional): Nucleus sampling value. Uses saved default when omitted.
            frequency_penalty (float, optional): Frequency penalty. Uses saved default when omitted.
            presence_penalty (float, optional): Presence penalty. Uses saved default when omitted.
            stream (bool, optional): Whether to stream the response. Uses saved default when omitted.
            current_persona (optional): The current persona.
            functions (optional): Functions to use.
            llm_call_type (str, optional): The type of LLM call.

        Returns:
            Union[str, AsyncIterator[str]]: The generated response or a stream of tokens.
        """
        # Determine provider and model
        requested_provider = provider if provider else self.current_llm_provider
        if not requested_provider:
            requested_provider = self.config_manager.get_default_provider()

        defaults: Dict[str, Any] = {}
        if requested_provider == "OpenAI":
            defaults = self.get_openai_llm_settings()

        resolved_model = model or defaults.get("model") or self.get_current_model()
        if not resolved_model:
            fallback_model = self.get_default_model_for_provider(requested_provider)
            resolved_model = fallback_model

        resolved_max_tokens = max_tokens if max_tokens is not None else defaults.get("max_tokens", 4000)
        resolved_temperature = (
            temperature if temperature is not None else defaults.get("temperature", 0.0)
        )
        resolved_top_p = top_p if top_p is not None else defaults.get("top_p", 1.0)
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

        # Log the incoming parameters
        self.logger.info(
            "Generating response with Provider: %s, Model: %s, Persona: %s",
            requested_provider,
            resolved_model,
            current_persona,
        )

        # Switch provider if different
        if requested_provider != self.current_llm_provider:
            await self.switch_llm_provider(requested_provider)

        # Switch model if different
        if resolved_model and resolved_model != self.current_model:
            await self.set_model(resolved_model)

        # Use current functions if not provided
        if functions is None:
            functions = self.current_functions

        start_time = time.time()
        self.logger.info(
            "Starting API call to %s with model %s for %s",
            requested_provider,
            resolved_model,
            llm_call_type,
        )

        try:
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
            }

            if requested_provider == "OpenAI":
                call_kwargs.update(
                    top_p=resolved_top_p,
                    frequency_penalty=resolved_frequency_penalty,
                    presence_penalty=resolved_presence_penalty,
                    function_calling=resolved_function_calling,
                )

            response = await self.generate_response_func(
                self.config_manager,
                **call_kwargs,
            )

            self.logger.info(f"API call completed in {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            self.logger.error(f"Error during API call to {requested_provider}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

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
        fallback_config = self.config_manager.get_llm_config('default')
        fallback_provider = fallback_config.get('provider')

        if not fallback_provider or fallback_provider not in self.providers:
            self.logger.error(f"Fallback provider {fallback_provider} is not configured or not available.")
            raise ValueError("Fallback provider configuration is missing or invalid.")

        fallback_function = self.providers.get(fallback_provider)
        if not fallback_function:
            self.logger.error(f"Fallback provider {fallback_provider} function not found.")
            raise ValueError("Fallback provider function is missing.")

        self.logger.info(f"Using fallback provider {fallback_provider} for {llm_call_type}")

        call_kwargs = {
            "messages": messages,
            "model": fallback_config.get('model'),
            "max_tokens": fallback_config.get('max_tokens', 4000),
            "temperature": fallback_config.get('temperature', 0.0),
            "stream": fallback_config.get('stream', True),
            "current_persona": fallback_config.get('current_persona'),
            "functions": fallback_config.get('functions'),
            "llm_call_type": llm_call_type,
            **kwargs,
        }

        if fallback_provider == "OpenAI":
            call_kwargs.update(
                top_p=fallback_config.get('top_p', 1.0),
                frequency_penalty=fallback_config.get('frequency_penalty', 0.0),
                presence_penalty=fallback_config.get('presence_penalty', 0.0),
            )

        return await fallback_function(
            self.config_manager,
            **call_kwargs,
        )

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

    def get_default_model_for_provider(self, provider: str) -> str:
        """
        Get the default model for a specific provider.

        Args:
            provider (str): The name of the provider.

        Returns:
            str: The default model name, or None if not available.
        """
        models = self.model_manager.get_available_models(provider).get(provider, [])
        if not models:
            self.logger.error(f"No models found for provider {provider}. Ensure the provider's models are loaded.")
            return None
        return models[0]

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
        if self.huggingface_generator:
            await self.unload_hf_model()
        if self.grok_generator:
            await self.grok_generator.unload_model()
        # Add any additional cleanup here
        self.logger.info("ProviderManager closed and models unloaded.")
