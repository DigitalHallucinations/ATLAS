# ATLAS/ATLAS.py

from concurrent.futures import Future
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from ATLAS.provider_manager import ProviderManager
from ATLAS.persona_manager import PersonaManager
from modules.Chat.chat_session import ChatSession
from modules.Speech_Services.speech_manager import SpeechManager
from modules.background_tasks import run_async_in_thread

class ATLAS:
    """
    The main ATLAS application class that manages configurations, providers, personas, and speech services.
    """

    def __init__(self):
        """
        Initialize the ATLAS instance with synchronous initialization.
        """
        self.config_manager = ConfigManager()
        self.logger = setup_logger(__name__)
        self.persona_path = self.config_manager.get_app_root()
        self.current_persona = None
        self.user = "Bib"  # Placeholder; replace with system user retrieval
        self.provider_manager = None
        self.persona_manager = None
        self.chat_session = None
        self.speech_manager = SpeechManager(self.config_manager)  # Instantiate SpeechManager with ConfigManager
        self._initialized = False
        self._provider_change_listeners: List[Callable[[Dict[str, str]], None]] = []
        self._message_dispatchers: List[Callable[[str, str], None]] = []
        self.message_dispatcher: Optional[Callable[[str, str], None]] = None
        self._default_status_tooltip = "Active LLM provider/model and TTS status"

    def _require_provider_manager(self) -> ProviderManager:
        """Return the initialized provider manager or raise an error if unavailable."""

        if self.provider_manager is None:
            raise RuntimeError("Provider manager is not initialized.")

        return self.provider_manager

    async def initialize(self):
        """
        Asynchronously initialize the ATLAS instance.
        """
        self.provider_manager = await ProviderManager.create(self.config_manager)
        self.persona_manager = PersonaManager(master=self, user=self.user)
        self.chat_session = ChatSession(self)
        
        default_provider = self.config_manager.get_default_provider()
        await self.provider_manager.set_current_provider(default_provider)
        
        self.logger.info(f"Default provider set to: {self.provider_manager.get_current_provider()}")
        self.logger.info(f"Default model set to: {self.provider_manager.get_current_model()}")
        self.logger.info("ATLAS initialized successfully.")
        
        # Initialize SpeechManager
        await self.speech_manager.initialize()  # Ensure SpeechManager is initialized
        self.logger.info("SpeechManager initialized successfully.")
        
        # Load TTS setting from configuration
        tts_enabled = self.config_manager.get_tts_enabled()
        self.speech_manager.set_tts_status(tts_enabled)
        self.logger.info(f"TTS enabled: {tts_enabled}")
        
        # Optionally, set default TTS provider if specified in config.yaml
        default_tts_provider = self.config_manager.get_config('DEFAULT_TTS_PROVIDER')
        if default_tts_provider:
            self.speech_manager.set_default_tts_provider(default_tts_provider)
            self.logger.info(f"Default TTS provider set to: {default_tts_provider}")
        
        self._initialized = True

    def is_initialized(self) -> bool:
        """
        Check if ATLAS is fully initialized.

        Returns:
            bool: True if ATLAS is initialized, False otherwise.
        """
        return self._initialized

    def _require_persona_manager(self) -> PersonaManager:
        """Return the initialized persona manager or raise an error if unavailable."""

        if self.persona_manager is None:
            raise RuntimeError("Persona manager is not initialized.")

        return self.persona_manager

    def get_persona_names(self) -> List[str]:
        """
        Retrieve persona names from the PersonaManager.

        Returns:
            List[str]: A list of persona names.
        """
        return self._require_persona_manager().persona_names

    def load_persona(self, persona: str):
        """
        Delegate loading a persona to the PersonaManager.

        Args:
            persona (str): The name of the persona to load.
        """
        self.logger.info(f"Loading persona: {persona}")
        manager = self._require_persona_manager()

        manager.updater(persona)
        self.current_persona = manager.current_persona  # Update the current_persona in ATLAS
        self.logger.info(f"Current persona set to: {self.current_persona}")

    def get_active_persona_name(self) -> str:
        """Return the human-friendly name of the active persona."""

        manager = self._require_persona_manager()
        persona = getattr(manager, "current_persona", None) or {}

        if isinstance(persona, dict):
            name = persona.get("name")
            if name:
                return str(name)

        default_name = getattr(manager, "default_persona_name", None)
        if default_name:
            return str(default_name)

        return "Assistant"

    def get_current_persona_prompt(self) -> Optional[str]:
        """Return the system prompt for the active persona when available."""

        manager = self._require_persona_manager()
        try:
            return manager.get_current_persona_prompt()
        except AttributeError:
            persona = getattr(manager, "current_persona", None)
            if isinstance(persona, dict):
                return persona.get("system_prompt")
            return None

    def get_persona_editor_state(self, persona_name: str) -> Optional[Dict[str, Any]]:
        """Fetch the structured editor state for the requested persona."""

        return self._require_persona_manager().get_editor_state(persona_name)

    def compute_persona_locked_content(
        self,
        persona_name: Optional[str] = None,
        *,
        general: Optional[Dict[str, Any]] = None,
        flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Compute start/end locked content previews for persona editors."""

        return self._require_persona_manager().compute_locked_content(
            persona_name,
            general=general,
            flags=flags,
        )

    def set_persona_flag(
        self,
        persona_name: str,
        flag: str,
        enabled: Any,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Toggle persona flags using the persona manager facade."""

        return self._require_persona_manager().set_flag(
            persona_name,
            flag,
            enabled,
            extras,
        )

    def update_persona_from_editor(
        self,
        persona_name: str,
        general: Optional[Dict[str, Any]] = None,
        persona_type: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        speech: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist persona updates captured from the editor views."""

        return self._require_persona_manager().update_persona_from_form(
            persona_name,
            general,
            persona_type,
            provider,
            speech,
        )

    def show_persona_message(self, role: str, message: str) -> None:
        """Proxy persona messages to the configured dispatcher."""

        self._require_persona_manager().show_message(role, message)

    def send_chat_message_async(
        self,
        message: str,
        *,
        on_success: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        thread_name: Optional[str] = None,
    ) -> Future[str]:
        """Dispatch ``message`` via the chat session on a background worker.

        The provided callbacks are invoked with the active persona's name along
        with the generated response (for successes) or the raised exception (for
        failures).  This keeps persona lookups consolidated within the backend
        layer so UI components only handle widget updates.
        """

        session = getattr(self, "chat_session", None)
        if session is None:
            raise RuntimeError("Chat session is not initialized.")

        def _invoke_success(response: str) -> None:
            if on_success is None:
                return
            persona_name = self.get_active_persona_name()
            try:
                on_success(persona_name, response)
            except Exception as callback_exc:  # noqa: BLE001 - log unexpected UI failures.
                self.logger.error(
                    "Chat success callback failed: %s", callback_exc, exc_info=True
                )

        def _invoke_error(exc: Exception) -> None:
            persona_name = self.get_active_persona_name()
            if on_error is None:
                self.logger.error("Chat background task failed: %s", exc, exc_info=True)
                return
            try:
                on_error(persona_name, exc)
            except Exception as callback_exc:  # noqa: BLE001 - log unexpected UI failures.
                self.logger.error(
                    "Chat error callback failed: %s", callback_exc, exc_info=True
                )

        return session.run_in_background(
            lambda: session.send_message(message),
            on_success=_invoke_success if on_success is not None else None,
            on_error=_invoke_error,
            thread_name=thread_name or "ChatResponseWorker",
        )

    def get_available_providers(self) -> List[str]:
        """
        Retrieve all available providers from the ProviderManager.

        Returns:
            List[str]: A list of provider names.
        """
        return self.provider_manager.get_available_providers()

    async def test_huggingface_token(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Validate a HuggingFace API token via the provider manager."""

        return await self.provider_manager.test_huggingface_token(token)

    def list_hf_models(self) -> Dict[str, Any]:
        """List installed HuggingFace models via the provider manager."""

        return self._require_provider_manager().list_hf_models()

    async def load_hf_model(self, model_name: str, force_download: bool = False) -> Dict[str, Any]:
        """Load a HuggingFace model using the provider manager helper."""

        return await self._require_provider_manager().load_hf_model(
            model_name, force_download=force_download
        )

    async def unload_hf_model(self) -> Dict[str, Any]:
        """Unload the active HuggingFace model via the provider manager."""

        return await self._require_provider_manager().unload_hf_model()

    async def remove_hf_model(self, model_name: str) -> Dict[str, Any]:
        """Remove a cached HuggingFace model through the provider manager."""

        return await self._require_provider_manager().remove_hf_model(model_name)

    async def download_hf_model(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        """Download a HuggingFace model through the provider manager."""

        return await self._require_provider_manager().download_huggingface_model(model_id, force=force)

    async def search_hf_models(
        self,
        search_query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search HuggingFace models using the provider manager helper."""

        return await self._require_provider_manager().search_huggingface_models(
            search_query,
            filters,
            limit=limit,
        )

    def update_hf_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Persist HuggingFace settings via the provider manager."""

        return self._require_provider_manager().update_huggingface_settings(settings)

    def clear_hf_cache(self) -> Dict[str, Any]:
        """Clear cached HuggingFace artefacts through the provider manager."""

        return self._require_provider_manager().clear_huggingface_cache()

    def save_hf_token(self, token: Optional[str]) -> Dict[str, Any]:
        """Save a Hugging Face token via the provider manager."""

        return self._require_provider_manager().save_huggingface_token(token)

    def get_provider_api_key_status(self, provider_name: str) -> Dict[str, Any]:
        """Fetch credential metadata for a provider using the provider manager."""

        return self._require_provider_manager().get_provider_api_key_status(provider_name)

    async def update_provider_api_key(
        self, provider_name: str, new_api_key: Optional[str]
    ) -> Dict[str, Any]:
        """Persist a provider API key through the provider manager facade."""

        return await self._require_provider_manager().update_provider_api_key(
            provider_name, new_api_key
        )

    def ensure_huggingface_ready(self) -> Dict[str, Any]:
        """Ensure the HuggingFace helper is initialized via the provider manager."""

        return self._require_provider_manager().ensure_huggingface_ready()

    def run_in_background(
        self,
        coroutine_factory: Callable[[], Awaitable[Any]],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        thread_name: Optional[str] = None,
    ) -> Future:
        """Execute an awaitable in a background thread using the shared task helper."""

        return run_async_in_thread(
            coroutine_factory,
            on_success=on_success,
            on_error=on_error,
            logger=self.logger,
            thread_name=thread_name,
        )

    def set_current_provider_in_background(
        self,
        provider: str,
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Future:
        """Schedule a provider switch without blocking the caller."""

        thread_name = f"set-provider-{provider}" if provider else None
        return self.run_in_background(
            lambda: self.set_current_provider(provider),
            on_success=on_success,
            on_error=on_error,
            thread_name=thread_name,
        )

    def update_provider_api_key_in_background(
        self,
        provider_name: str,
        new_api_key: Optional[str],
        *,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Future:
        """Persist provider credentials using a background worker thread."""

        thread_name = f"update-api-key-{provider_name}" if provider_name else None
        return self.run_in_background(
            lambda: self.update_provider_api_key(provider_name, new_api_key),
            on_success=on_success,
            on_error=on_error,
            thread_name=thread_name,
        )

    async def refresh_current_provider(
        self, provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reload the active provider configuration when the names align."""

        manager = self._require_provider_manager()
        active_provider = manager.get_current_provider()
        target_provider = provider_name or active_provider

        if not target_provider:
            return {
                "success": False,
                "error": "No active provider is configured.",
            }

        if target_provider != active_provider:
            return {
                "success": False,
                "error": f"Provider '{target_provider}' is not the active provider.",
                "active_provider": active_provider,
            }

        await manager.set_current_provider(active_provider)
        return {
            "success": True,
            "message": f"Provider {active_provider} refreshed.",
            "provider": active_provider,
        }

    async def set_current_provider(self, provider: str):
        """
        Asynchronously set the current provider in the ProviderManager.
        """
        try:
            await self.provider_manager.set_current_provider(provider)
        except Exception as exc:
            self.logger.error("Failed to set provider %s: %s", provider, exc, exc_info=True)
            raise

        self.chat_session.set_provider(provider)
        current_model = self.provider_manager.get_current_model()
        self.chat_session.set_model(current_model)

        # Log the updates
        self.logger.info(f"Current provider set to {provider} with model {current_model}")
        # Notify any observers (e.g., UI components) about the change
        self._notify_provider_change_listeners()

    def add_provider_change_listener(self, listener: Callable[[Dict[str, str]], None]) -> None:
        """Register a callback to be notified when the provider or model changes."""

        if not callable(listener):
            raise TypeError("listener must be callable")

        if listener in self._provider_change_listeners:
            return

        self._provider_change_listeners.append(listener)

    def remove_provider_change_listener(self, listener: Callable[[Dict[str, str]], None]) -> None:
        """Remove a previously registered provider change callback if present."""

        if listener in self._provider_change_listeners:
            self._provider_change_listeners.remove(listener)

    def _notify_provider_change_listeners(self) -> None:
        """Invoke all registered provider change callbacks."""

        summary = self.get_chat_status_summary()
        for listener in list(self._provider_change_listeners):
            try:
                listener(summary)
            except Exception as exc:
                self.logger.error(
                    "Provider change listener %s failed: %s", listener, exc, exc_info=True
                )

    def register_message_dispatcher(self, dispatcher: Callable[[str, str], None]) -> None:
        """Register a callback that handles persona-related messages from the backend."""
        if not callable(dispatcher):
            raise TypeError("dispatcher must be callable")

        if dispatcher in self._message_dispatchers:
            return

        self._message_dispatchers.append(dispatcher)
        self._refresh_message_dispatcher()

    def unregister_message_dispatcher(self, dispatcher: Callable[[str, str], None]) -> None:
        """Remove a previously registered persona message callback."""
        if dispatcher in self._message_dispatchers:
            self._message_dispatchers.remove(dispatcher)
            self._refresh_message_dispatcher()

    def _refresh_message_dispatcher(self) -> None:
        """Update the aggregated dispatcher exposed to backend components."""
        if not self._message_dispatchers:
            self.message_dispatcher = None
            return

        def aggregated(role: str, message: str) -> None:
            for callback in list(self._message_dispatchers):
                try:
                    callback(role, message)
                except Exception as exc:
                    self.logger.error(
                        "Message dispatcher %s failed: %s", callback, exc, exc_info=True
                    )

        self.message_dispatcher = aggregated

    def log_history(self):
        """
        Handle history-related functionality.
        """
        self.logger.info("History button clicked")
        print("History button clicked")

    def show_settings(self):
        """
        Handle settings-related functionality.
        """
        self.logger.info("Settings page clicked")
        print("Settings page clicked")

    def get_default_provider(self) -> str:
        """
        Get the default provider from the ProviderManager.

        Returns:
            str: The name of the default provider.
        """
        return self.provider_manager.get_current_provider()

    def get_default_model(self) -> str:
        """
        Get the default model from the ProviderManager.

        Returns:
            str: The name of the default model.
        """
        return self.provider_manager.get_current_model()

    def get_openai_llm_settings(self) -> Dict[str, Any]:
        """Expose the persisted OpenAI LLM defaults via the provider manager."""

        settings = self._require_provider_manager().get_openai_llm_settings()
        return dict(settings)

    async def list_openai_models(
        self,
        *,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch available OpenAI models through the provider manager facade."""

        return await self._require_provider_manager().list_openai_models(
            base_url=base_url,
            organization=organization,
        )

    def set_openai_llm_settings(
        self,
        *,
        model: str,
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
        """Persist OpenAI defaults through the provider manager facade."""

        manager = self._require_provider_manager()
        setter = getattr(manager, "set_openai_llm_settings", None)
        if not callable(setter):
            raise AttributeError("Provider manager does not support OpenAI settings updates.")

        return setter(
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

    def get_chat_status_summary(self) -> Dict[str, str]:
        """Return a consolidated snapshot of chat-related status information."""

        summary: Dict[str, str] = {
            "llm_provider": "Unknown",
            "llm_model": "No model selected",
            "tts_provider": "None",
            "tts_voice": "Not Set",
        }

        provider_manager = getattr(self, "provider_manager", None)
        if provider_manager is not None:
            try:
                provider_name = provider_manager.get_current_provider()
                if provider_name:
                    summary["llm_provider"] = provider_name
            except Exception as exc:
                self.logger.error("Failed to read current LLM provider: %s", exc, exc_info=True)

            try:
                model_name = provider_manager.get_current_model()
                if model_name:
                    summary["llm_model"] = model_name
            except Exception as exc:
                self.logger.error("Failed to read current LLM model: %s", exc, exc_info=True)

        speech_manager = getattr(self, "speech_manager", None)
        if speech_manager is not None:
            try:
                tts_provider, tts_voice = speech_manager.get_active_tts_summary()
            except Exception as exc:
                self.logger.error("Failed to read active TTS configuration: %s", exc, exc_info=True)
            else:
                summary["tts_provider"] = tts_provider or summary["tts_provider"]
                summary["tts_voice"] = tts_voice or summary["tts_voice"]

        return summary

    def format_chat_status(self, status_summary: Optional[Dict[str, str]] = None) -> str:
        """Generate the human-readable chat status message for display."""

        summary: Dict[str, str]
        if status_summary is None:
            try:
                summary = self.get_chat_status_summary()
            except Exception as exc:
                self.logger.error("Failed to obtain chat status summary: %s", exc, exc_info=True)
                summary = {}
        else:
            summary = status_summary

        llm_provider = summary.get("llm_provider") or "Unknown"
        llm_model = summary.get("llm_model") or "No model selected"
        tts_provider = summary.get("tts_provider") or "None"
        tts_voice = summary.get("tts_voice") or "Not Set"

        return (
            f"LLM: {llm_provider} • Model: {llm_model} • "
            f"TTS: {tts_provider} (Voice: {tts_voice})"
        )

    def get_speech_defaults(self) -> Dict[str, Any]:
        """Expose global speech defaults for UI consumers."""

        return self.speech_manager.describe_general_settings()

    def get_speech_provider_status(self, provider_name: str) -> Dict[str, Any]:
        """Return credential metadata for a speech provider."""

        return self.speech_manager.get_provider_credential_status(provider_name)

    def get_speech_voice_options(
        self, provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return the available voice options for a speech provider."""

        return self.speech_manager.list_tts_voice_options(provider)

    def get_active_speech_voice(self) -> Dict[str, Optional[str]]:
        """Return the active speech provider and voice name."""

        provider, voice = self.speech_manager.get_active_tts_summary()
        return {"provider": provider, "name": voice}

    def update_speech_defaults(
        self,
        *,
        tts_enabled: bool,
        tts_provider: Optional[str],
        stt_enabled: bool,
        stt_provider: Optional[str],
    ) -> Dict[str, Any]:
        """Persist global speech defaults via the speech manager."""

        self.speech_manager.configure_defaults(
            tts_enabled=bool(tts_enabled),
            tts_provider=tts_provider,
            stt_enabled=bool(stt_enabled),
            stt_provider=stt_provider,
        )

        return self.get_speech_defaults()

    def update_elevenlabs_settings(
        self,
        *,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update ElevenLabs credentials or active voice selection."""

        manager = self.speech_manager
        result = {
            "updated_api_key": False,
            "updated_voice": False,
        }

        if api_key:
            manager.set_elevenlabs_api_key(api_key)
            result["updated_api_key"] = True

        provider_key = manager.resolve_tts_provider(manager.get_default_tts_provider())

        if voice_id and provider_key:
            voices = manager.get_tts_voices(provider_key) or []
            selected_voice: Optional[Dict[str, Any]] = None
            for voice in voices:
                if not isinstance(voice, dict):
                    continue
                if voice.get("voice_id") == voice_id or voice.get("name") == voice_id:
                    selected_voice = voice
                    break

            if selected_voice is not None:
                manager.set_tts_voice(selected_voice, provider_key)
                result["updated_voice"] = True

        result["provider"] = provider_key
        return result

    def update_google_speech_settings(self, credentials_path: str) -> None:
        """Persist Google speech credentials via the speech manager."""

        self.speech_manager.set_google_credentials(credentials_path)

    def get_openai_speech_options(self) -> Dict[str, List[Tuple[str, Optional[str]]]]:
        """Return the OpenAI speech option sets for UI rendering."""

        return self.speech_manager.get_openai_option_sets()

    def get_openai_speech_configuration(self) -> Dict[str, Optional[str]]:
        """Return persisted OpenAI speech configuration values."""

        return self.speech_manager.get_openai_display_config()

    def update_openai_speech_settings(
        self, display_payload: Dict[str, Any]
    ) -> Dict[str, Optional[str]]:
        """Validate and persist OpenAI speech settings supplied by the UI."""

        prepared = self.speech_manager.normalize_openai_display_settings(display_payload)

        self.speech_manager.set_openai_speech_config(
            api_key=prepared.get("api_key"),
            stt_provider=prepared.get("stt_provider"),
            language=prepared.get("language"),
            task=prepared.get("task"),
            initial_prompt=prepared.get("initial_prompt"),
            tts_provider=prepared.get("tts_provider"),
        )

        return prepared

    def get_transcription_history(
        self, *, formatted: bool = False
    ) -> List[Dict[str, Any]]:
        """Return transcription history records from the speech manager."""

        return self.speech_manager.get_transcription_history(formatted=formatted)

    async def close(self):
        """
        Perform cleanup operations.
        """
        await self.provider_manager.close()
        await self.speech_manager.close()
        self.logger.info("ATLAS closed and all providers unloaded.")

    async def maybe_text_to_speech(self, response_text: str) -> None:
        """Run text-to-speech for the provided response when enabled.

        Args:
            response_text (str): The response to vocalize.

        Raises:
            RuntimeError: If text-to-speech is enabled but synthesis fails.
        """
        if not response_text:
            return

        if not self.speech_manager.get_tts_status():
            return

        self.logger.debug("TTS enabled; synthesizing response text.")

        try:
            await self.speech_manager.text_to_speech(response_text)
        except Exception as exc:
            self.logger.error("Text-to-speech failed: %s", exc, exc_info=True)
            raise RuntimeError("Text-to-speech failed") from exc

    def start_stt_listening(self) -> Dict[str, Any]:
        """Begin speech-to-text recording via the active provider.

        Returns:
            Dict[str, Any]: Structured payload describing the resulting state.
        """

        manager: Optional[SpeechManager] = getattr(self, "speech_manager", None)
        if manager is None:
            message = "Speech services unavailable."
            self.logger.error(message)
            return {
                "ok": False,
                "status_text": message,
                "provider": None,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self.get_chat_status_summary(),
            }

        provider_key = manager.get_active_stt_provider()
        if not provider_key:
            message = "No STT service configured."
            self.logger.error(message)
            return {
                "ok": False,
                "status_text": message,
                "provider": None,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self.get_chat_status_summary(),
            }

        try:
            started = manager.listen(provider_key)
        except Exception as exc:  # Defensive: listen() already handles errors.
            self.logger.error(
                "Failed to start STT provider %s: %s", provider_key, exc, exc_info=True
            )
            started = False

        if not started:
            message = "Failed to start listening."
            return {
                "ok": False,
                "status_text": message,
                "provider": provider_key,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self.get_chat_status_summary(),
            }

        self.logger.info("Listening started using provider '%s'.", provider_key)
        return {
            "ok": True,
            "status_text": "Listening…",
            "provider": provider_key,
            "listening": True,
            "spinner": False,
            "error": None,
            "status_tooltip": f"Listening via {provider_key}",
            "status_summary": self.get_chat_status_summary(),
        }

    def stop_stt_and_transcribe(self) -> Dict[str, Any]:
        """Stop recording (if active) and transcribe in the background.

        Returns:
            Dict[str, Any]: Structured payload with an attached transcription future.
        """

        manager: Optional[SpeechManager] = getattr(self, "speech_manager", None)
        if manager is None:
            message = "Speech services unavailable."
            self.logger.error(message)
            return {
                "ok": False,
                "status_text": message,
                "provider": None,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self.get_chat_status_summary(),
                "transcription_future": None,
            }

        provider_key = manager.get_active_stt_provider()
        if not provider_key:
            message = "No STT service configured."
            self.logger.error(message)
            return {
                "ok": False,
                "status_text": message,
                "provider": None,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self.get_chat_status_summary(),
                "transcription_future": None,
            }

        result_future: Future[Dict[str, Any]] = Future()

        def _finalize_payload(
            *, transcript: Optional[str] = None, error: Optional[Exception] = None
        ) -> Dict[str, Any]:
            summary = self.get_chat_status_summary()
            normalized_transcript = (transcript or "").strip()
            if error is not None:
                error_message = f"Transcription failed: {error}"
                payload = {
                    "ok": False,
                    "status_text": error_message,
                    "provider": provider_key,
                    "listening": False,
                    "spinner": False,
                    "transcript": "",
                    "error": error_message,
                    "status_tooltip": self._default_status_tooltip,
                    "status_summary": summary,
                }
            elif normalized_transcript:
                payload = {
                    "ok": True,
                    "status_text": "Transcription complete.",
                    "provider": provider_key,
                    "listening": False,
                    "spinner": False,
                    "transcript": normalized_transcript,
                    "error": None,
                    "status_tooltip": self._default_status_tooltip,
                    "status_summary": summary,
                }
            else:
                payload = {
                    "ok": True,
                    "status_text": "No transcription available.",
                    "provider": provider_key,
                    "listening": False,
                    "spinner": False,
                    "transcript": "",
                    "error": None,
                    "status_tooltip": self._default_status_tooltip,
                    "status_summary": summary,
                }

            return payload

        def _on_success(transcript: str) -> None:
            payload = _finalize_payload(transcript=transcript)
            if not result_future.done():
                result_future.set_result(payload)

        def _on_error(exc: Exception) -> None:
            self.logger.error("Unexpected transcription error: %s", exc, exc_info=True)
            payload = _finalize_payload(error=exc)
            if not result_future.done():
                result_future.set_result(payload)

        try:
            manager.stop_and_transcribe_in_background(
                provider_key,
                on_success=_on_success,
                on_error=_on_error,
                thread_name="SpeechTranscriptionWorker",
            )
        except Exception as exc:
            self.logger.error(
                "Failed to schedule transcription with provider %s: %s",
                provider_key,
                exc,
                exc_info=True,
            )
            payload = _finalize_payload(error=exc)
            if not result_future.done():
                result_future.set_result(payload)
            return {
                "ok": False,
                "status_text": payload["status_text"],
                "provider": provider_key,
                "listening": False,
                "spinner": False,
                "error": payload["error"],
                "status_tooltip": payload["status_tooltip"],
                "status_summary": payload["status_summary"],
                "transcription_future": result_future,
            }

        return {
            "ok": True,
            "status_text": "Transcribing…",
            "provider": provider_key,
            "listening": False,
            "spinner": True,
            "error": None,
            "status_tooltip": f"Transcribing via {provider_key}",
            "status_summary": self.get_chat_status_summary(),
            "transcription_future": result_future,
        }

    async def generate_response(self, messages: List[Dict[str, str]]) -> Union[str, AsyncIterator[str]]:
        """
        Generate a response using the current provider and model.
        Additionally, perform TTS generation if enabled.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.

        Returns:
            Union[str, AsyncIterator[str]]: The generated response or a stream of tokens.
        """
        if not self.current_persona:
            self.logger.error("No persona is currently loaded. Cannot generate response.")
            return "Error: No persona is currently loaded. Please select a persona."

        try:
            response = await self.provider_manager.generate_response(
                messages=messages,
                current_persona=self.current_persona,
                user=self.user,
                conversation_id=self.chat_session.conversation_id
            )

            # Perform TTS if enabled
            await self.maybe_text_to_speech(response)

            return response
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}", exc_info=True)
            return "Error: Failed to generate response. Please try again later."
