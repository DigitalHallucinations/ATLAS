# modules/speech_services/speech_manager.py

"""
Speech Manager Module
-----------------------
This module provides a unified interface for managing Text-to-Speech (TTS)
and Speech-to-Text (STT) services. It supports multiple providers (e.g.,
Eleven Labs, Google, and Whisper) and allows dynamic selection, addition, and
removal of providers. The manager also handles asynchronous initialization and
clean-up of services.

Author:Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""

from typing import Dict, Any
from .elevenlabs_tts import ElevenLabsTTS
from .Google_tts import GoogleTTS
from .Google_stt import GoogleSTT
from modules.logging.logger import setup_logger
from ATLAS.config import ConfigManager

logger = setup_logger('speech_manager.py')

class SpeechManager:
    def __init__(self, config_manager: ConfigManager):
        """
        Initializes the SpeechManager.

        Args:
            config_manager (ConfigManager): Configuration manager for retrieving global settings.
        """
        self.config_manager = config_manager
        self.tts_services: Dict[str, Any] = {}
        self.stt_services: Dict[str, Any] = {}
        self.active_tts = None
        self.active_stt = None
        self._tts_enabled = False 

        self.initialize_services()

    async def initialize(self):
        """
        Perform any asynchronous initialization if required.
        Currently a placeholder for future asynchronous initialization.
        """
        pass

    def initialize_services(self):
        """
        Initializes available TTS and STT services.
        This method registers all providers and sets default active services.
        """
        logger.info("Initializing Speech Services...")

        # Initialize TTS services
        try:
            eleven_tts = ElevenLabsTTS()
            self.tts_services['eleven_labs'] = eleven_tts
            logger.info("Eleven Labs TTS initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Eleven Labs TTS: {e}")

        try:
            google_tts = GoogleTTS()
            self.tts_services['google'] = google_tts
            logger.info("Google TTS initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS: {e}")

        # Initialize STT services
        try:
            google_stt = GoogleSTT()
            self.stt_services['google'] = google_stt
            logger.info("Google STT initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Google STT: {e}")

        # Initialize Whisper STT services (local and online)
        try:
            from modules.Speech_Services.whisper_stt import WhisperSTT
            whisper_local = WhisperSTT(mode="local")
            self.stt_services['whisper_local'] = whisper_local
            logger.info("Whisper local STT initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper local STT: {e}")

        try:
            from modules.Speech_Services.whisper_stt import WhisperSTT
            whisper_online = WhisperSTT(mode="online")
            self.stt_services['whisper_online'] = whisper_online
            logger.info("Whisper online STT initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper online STT: {e}")

        # Set default active services
        if 'eleven_labs' in self.tts_services:
            self.active_tts = self.tts_services['eleven_labs']
            logger.info("Default TTS set to Eleven Labs.")
        elif 'google' in self.tts_services:
            self.active_tts = self.tts_services['google']
            logger.info("Default TTS set to Google.")

        # Default STT service is set to Google if available.
        if 'google' in self.stt_services:
            self.active_stt = self.stt_services['google']
            logger.info("Default STT set to Google.")

    # ----------------------- TTS Methods -----------------------

    async def text_to_speech(self, text: str, provider: str = None):
        """
        Converts text to speech using the specified provider.

        Args:
            text (str): The text to convert.
            provider (str, optional): The TTS provider key. If None, the default provider is used.
        """
        if not self.config_manager.get_tts_enabled():
            logger.info("TTS is disabled. Skipping TTS generation.")
            return

        provider = provider or self.get_default_tts_provider()
        tts = self.tts_services.get(provider)
        if not tts:
            logger.error(f"TTS provider '{provider}' not found.")
            return

        await tts.text_to_speech(text)

    def set_tts_voice(self, voice: dict, provider: str = None):
        """
        Sets the active voice for a given TTS provider.

        Args:
            voice (dict): Dictionary containing voice details.
            provider (str, optional): TTS provider key.
        """
        provider = provider or self.get_default_tts_provider()
        tts = self.tts_services.get(provider)
        if not tts:
            logger.error(f"TTS provider '{provider}' not found.")
            return
        tts.set_voice(voice)

    def get_tts_voices(self, provider: str = None) -> list:
        """
        Retrieves available voices from the specified TTS provider.

        Args:
            provider (str, optional): TTS provider key.

        Returns:
            list: List of available voices.
        """
        provider = provider or self.get_default_tts_provider()
        tts = self.tts_services.get(provider)
        if not tts:
            logger.error(f"TTS provider '{provider}' not found.")
            return []
        return tts.get_voices()

    def set_tts_status(self, value: bool, provider: str = None):
        """
        Enables or disables TTS for a specific provider or globally.

        Args:
            value (bool): True to enable TTS, False to disable.
            provider (str, optional): Specific TTS provider key. If None, applies globally.
        """
        if provider:
            tts = self.tts_services.get(provider)
            if not tts:
                logger.error(f"TTS provider '{provider}' not found.")
                return
            tts.set_tts(value)
            logger.info(f"TTS for provider '{provider}' set to {value}.")
        else:
            self._tts_enabled = value
            logger.info(f"Global TTS enabled set to {self._tts_enabled}.")

    def get_tts_status(self, provider: str = None) -> bool:
        """
        Gets the TTS enabled status for a specific provider or globally.

        Args:
            provider (str, optional): TTS provider key.

        Returns:
            bool: TTS enabled status.
        """
        if provider:
            tts = self.tts_services.get(provider)
            if not tts:
                logger.error(f"TTS provider '{provider}' not found.")
                return False
            return tts.get_tts()
        else:
            logger.info(f"Global TTS status: {self.config_manager.get_tts_enabled()}")
            return self.config_manager.get_tts_enabled()

    def add_tts_provider(self, name: str, tts_instance: Any):
        """
        Adds a new TTS provider.

        Args:
            name (str): Unique key for the provider.
            tts_instance (Any): Instance of the TTS provider.
        """
        if name in self.tts_services:
            logger.warning(f"TTS provider '{name}' already exists. Overwriting.")
        self.tts_services[name] = tts_instance
        logger.info(f"TTS provider '{name}' added.")

    def remove_tts_provider(self, name: str):
        """
        Removes a TTS provider.

        Args:
            name (str): Unique key for the provider.
        """
        if name in self.tts_services:
            del self.tts_services[name]
            logger.info(f"TTS provider '{name}' removed.")
        else:
            logger.warning(f"TTS provider '{name}' does not exist.")

    def get_default_tts_provider(self) -> str:
        """
        Retrieves the key for the default TTS provider.

        Returns:
            str: Default TTS provider key or None if not set.
        """
        if self.active_tts:
            for name, service in self.tts_services.items():
                if service == self.active_tts:
                    return name
        return None

    def set_default_tts_provider(self, provider: str):
        """
        Sets the default TTS provider.

        Args:
            provider (str): Unique key for the TTS provider.
        """
        tts = self.tts_services.get(provider)
        if not tts:
            logger.error(f"TTS provider '{provider}' not found.")
            return
        self.active_tts = tts
        logger.info(f"Default TTS provider set to '{provider}'.")

    # ----------------------- STT Methods -----------------------

    def listen(self, provider: str = None):
        """
        Starts speech recognition by invoking the active STT provider's listen method.

        Args:
            provider (str, optional): STT provider key.
        """
        provider = provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return
        stt.listen()

    def stop_listening(self, provider: str = None):
        """
        Stops speech recognition by invoking the active STT provider's stop_listening method.

        Args:
            provider (str, optional): STT provider key.
        """
        provider = provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return
        stt.stop_listening()

    def transcribe(self, audio_file: str, provider: str = None) -> str:
        """
        Transcribes the provided audio file using the active STT provider.

        Args:
            audio_file (str): The name or path of the audio file to transcribe.
            provider (str, optional): STT provider key.

        Returns:
            str: Transcribed text.
        """
        provider = provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return ""
        return stt.transcribe(audio_file)

    def add_stt_provider(self, name: str, stt_instance: Any):
        """
        Adds a new STT provider.

        Args:
            name (str): Unique key for the provider.
            stt_instance (Any): Instance of the STT provider.
        """
        if name in self.stt_services:
            logger.warning(f"STT provider '{name}' already exists. Overwriting.")
        self.stt_services[name] = stt_instance
        logger.info(f"STT provider '{name}' added.")

    def remove_stt_provider(self, name: str):
        """
        Removes an STT provider.

        Args:
            name (str): Unique key for the provider.
        """
        if name in self.stt_services:
            del self.stt_services[name]
            logger.info(f"STT provider '{name}' removed.")
        else:
            logger.warning(f"STT provider '{name}' does not exist.")

    def get_default_stt_provider(self) -> str:
        """
        Retrieves the key for the default STT provider.

        Returns:
            str: Default STT provider key or None if not set.
        """
        if self.active_stt:
            for name, service in self.stt_services.items():
                if service == self.active_stt:
                    return name
        return None

    def set_default_stt_provider(self, provider: str):
        """
        Sets the default STT provider.

        Args:
            provider (str): Unique key for the STT provider.
        """
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return
        self.active_stt = stt
        logger.info(f"Default STT provider set to '{provider}'.")

    async def close(self):
        """
        Performs cleanup operations for all services that implement a close method.
        """
        # Close TTS services
        for name, tts in self.tts_services.items():
            if hasattr(tts, 'close') and callable(tts.close):
                await tts.close()
                logger.info(f"Closed TTS provider '{name}'.")

        # Close STT services
        for name, stt in self.stt_services.items():
            if hasattr(stt, 'close') and callable(stt.close):
                await stt.close()
                logger.info(f"Closed STT provider '{name}'.")