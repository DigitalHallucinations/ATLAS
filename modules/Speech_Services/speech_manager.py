# modules/speech_services/speech_manager.py

from typing import Dict, Any
from .elevenlabs_tts import ElevenLabsTTS
from .Google_tts import GoogleTTS
from .Google_stt import GoogleSTT
from modules.logging.logger import setup_logger
from ATLAS.config import ConfigManager

logger = setup_logger('speech_manager.py')

class SpeechManager:
    def __init__(self, config_manager: ConfigManager):
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
        """
        # If any of the services require asynchronous initialization,
        # implement it here. Otherwise, pass.
        pass

    def initialize_services(self):
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

        # Set default active services
        if 'eleven_labs' in self.tts_services:
            self.active_tts = self.tts_services['eleven_labs']
            logger.info("Default TTS set to Eleven Labs.")
        elif 'google' in self.tts_services:
            self.active_tts = self.tts_services['google']
            logger.info("Default TTS set to Google.")

        if 'google' in self.stt_services:
            self.active_stt = self.stt_services['google']
            logger.info("Default STT set to Google.")

    # TTS Methods
    async def text_to_speech(self, text: str, provider: str = None):
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
        provider = provider or self.get_default_tts_provider()
        tts = self.tts_services.get(provider)
        if not tts:
            logger.error(f"TTS provider '{provider}' not found.")
            return
        tts.set_voice(voice)

    def get_tts_voices(self, provider: str = None) -> list:
        provider = provider or self.get_default_tts_provider()
        tts = self.tts_services.get(provider)
        if not tts:
            logger.error(f"TTS provider '{provider}' not found.")
            return []
        return tts.get_voices()

    def set_tts_status(self, value: bool, provider: str = None):
        """
        Enable or disable TTS for a specific provider or globally.

        Args:
            value (bool): True to enable TTS, False to disable.
            provider (str, optional): Specific TTS provider to set. If None, applies globally.
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
        Get the TTS status for a specific provider or globally.

        Args:
            provider (str, optional): Specific TTS provider to query. If None, returns global status.

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
        if name in self.tts_services:
            logger.warning(f"TTS provider '{name}' already exists. Overwriting.")
        self.tts_services[name] = tts_instance
        logger.info(f"TTS provider '{name}' added.")

    def remove_tts_provider(self, name: str):
        if name in self.tts_services:
            del self.tts_services[name]
            logger.info(f"TTS provider '{name}' removed.")
        else:
            logger.warning(f"TTS provider '{name}' does not exist.")

    def get_default_tts_provider(self) -> str:
        if self.active_tts:
            for name, service in self.tts_services.items():
                if service == self.active_tts:
                    return name
        return None

    def set_default_tts_provider(self, provider: str):
        tts = self.tts_services.get(provider)
        if not tts:
            logger.error(f"TTS provider '{provider}' not found.")
            return
        self.active_tts = tts
        logger.info(f"Default TTS provider set to '{provider}'.")

    # STT Methods
    def listen(self, provider: str = None):
        provider = provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return
        stt.listen()

    def stop_listening(self, provider: str = None):
        provider = provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return
        stt.stop_listening()

    def transcribe(self, audio_file: str, provider: str = None) -> str:
        provider = provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return ""
        return stt.transcribe(audio_file)

    def add_stt_provider(self, name: str, stt_instance: Any):
        if name in self.stt_services:
            logger.warning(f"STT provider '{name}' already exists. Overwriting.")
        self.stt_services[name] = stt_instance
        logger.info(f"STT provider '{name}' added.")

    def remove_stt_provider(self, name: str):
        if name in self.stt_services:
            del self.stt_services[name]
            logger.info(f"STT provider '{name}' removed.")
        else:
            logger.warning(f"STT provider '{name}' does not exist.")

    def get_default_stt_provider(self) -> str:
        if self.active_stt:
            for name, service in self.stt_services.items():
                if service == self.active_stt:
                    return name
        return None

    def set_default_stt_provider(self, provider: str):
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return
        self.active_stt = stt
        logger.info(f"Default STT provider set to '{provider}'.")

    async def close(self):
        """
        Perform cleanup operations for all services.
        """
        # Close all TTS services if they have a close method
        for name, tts in self.tts_services.items():
            if hasattr(tts, 'close') and callable(tts.close):
                await tts.close()
                logger.info(f"Closed TTS provider '{name}'.")

        # Close all STT services if they have a close method
        for name, stt in self.stt_services.items():
            if hasattr(stt, 'close') and callable(stt.close):
                await stt.close()
                logger.info(f"Closed STT provider '{name}'.")
