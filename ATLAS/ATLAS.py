# ATLAS/ATLAS.py

from concurrent.futures import Future
from typing import List, Dict, Union, AsyncIterator, Optional, Any, Callable
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from ATLAS.provider_manager import ProviderManager
from ATLAS.persona_manager import PersonaManager
from modules.Chat.chat_session import ChatSession
from modules.Speech_Services.speech_manager import SpeechManager  

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

    def get_persona_names(self) -> List[str]:
        """
        Retrieve persona names from the PersonaManager.

        Returns:
            List[str]: A list of persona names.
        """
        return self.persona_manager.persona_names

    def load_persona(self, persona: str):
        """
        Delegate loading a persona to the PersonaManager.

        Args:
            persona (str): The name of the persona to load.
        """
        self.logger.info(f"Loading persona: {persona}")
        self.persona_manager.updater(persona)
        self.current_persona = self.persona_manager.current_persona  # Update the current_persona in ATLAS
        self.logger.info(f"Current persona set to: {self.current_persona}")

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
