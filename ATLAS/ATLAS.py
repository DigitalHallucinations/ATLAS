# ATLAS/ATLAS.py

from typing import List, Dict, Union, AsyncIterator
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
    
    async def set_current_provider(self, provider: str):
        """
        Asynchronously set the current provider in the ProviderManager.
        """
        await self.provider_manager.set_current_provider(provider)
        self.chat_session.set_provider(provider)
        current_model = self.provider_manager.get_current_model()
        self.chat_session.set_model(current_model)
        
        # Log the updates
        self.logger.info(f"Current provider set to {provider} with model {current_model}")
        # Notify any observers (e.g., UI components) about the change
        self.notify_provider_changed(provider, current_model)

    def notify_provider_changed(self, provider: str, model: str):
        """
        Notify observers that the provider and model have changed.
        This method should be overridden by UI components that need to react to provider changes.

        Args:
            provider (str): The new provider.
            model (str): The new model.
        """
        # This method can be overridden or connected via signals in UI code
        pass

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
