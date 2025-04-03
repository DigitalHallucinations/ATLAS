# modules/speech_services/speech_manager.py

"""
Speech Manager Module
-----------------------
This module provides a unified interface for managing Text-to-Speech (TTS)
and Speech-to-Text (STT) services. It supports multiple providers (e.g.,
Eleven Labs, Google, Whisper, and OpenAI's GPT-4o variants) and allows dynamic
selection, addition, and removal of providers. The manager also handles
asynchronous initialization, batch transcription, detailed transcription history
logging, and clean-up of services.

Author: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""

from typing import Dict, Any, List
import os
import asyncio
import time
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
        self.transcription_history: List[dict] = []  # Logs for transcription events

        self.initialize_services()

    async def initialize(self):
        """
        Perform any asynchronous initialization if required.
        Currently a placeholder for future asynchronous tasks.
        """
        pass

    def initialize_services(self):
        """
        Initializes available TTS and STT services.
        This method registers all providers and sets default active services.
        """
        logger.info("Initializing Speech Services...")

        # Initialize TTS providers
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

        # Initialize new GPT-4o TTS provider
        try:
            from modules.Speech_Services.gpt4o_tts import GPT4oTTS
            gpt4o_tts = GPT4oTTS(voice="default")
            self.tts_services['gpt4o_tts'] = gpt4o_tts
            logger.info("GPT-4o TTS initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4o TTS: {e}")

        # Initialize STT providers
        try:
            google_stt = GoogleSTT()
            self.stt_services['google'] = google_stt
            logger.info("Google STT initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Google STT: {e}")

        # Initialize Whisper STT providers (local and online)
        try:
            from modules.Speech_Services.whisper_stt import WhisperSTT
            whisper_local = WhisperSTT(
                mode="local",
                model_name=self.config_manager.config.get("WHISPER_MODEL", "base"),
                fs=int(self.config_manager.config.get("WHISPER_FS", 16000)),
                device=self.config_manager.config.get("WHISPER_DEVICE", None),
                noise_reduction=self.config_manager.config.get("WHISPER_NOISE_REDUCTION", False),
                fallback_online=self.config_manager.config.get("WHISPER_FALLBACK", False)
            )
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

        # Initialize new GPT-4o STT provider
        try:
            from modules.Speech_Services.gpt4o_stt import GPT4oSTT
            # You can choose between "gpt-4o" and "gpt-4o-mini". Here we use the full version.
            gpt4o_stt = GPT4oSTT(variant="gpt-4o")
            self.stt_services['gpt4o_stt'] = gpt4o_stt
            logger.info("GPT-4o STT initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4o STT: {e}")

        # Set default active providers
        if 'eleven_labs' in self.tts_services:
            self.active_tts = self.tts_services['eleven_labs']
            logger.info("Default TTS set to Eleven Labs.")
        elif 'google' in self.tts_services:
            self.active_tts = self.tts_services['google']
            logger.info("Default TTS set to Google.")

        if 'google' in self.stt_services:
            self.active_stt = self.stt_services['google']
            logger.info("Default STT set to Google.")

    # ----------------------- TTS Methods -----------------------

    async def text_to_speech(self, text: str, provider: str = None):
        """
        Converts text to speech using the specified provider.

        Args:
            text (str): The text to convert.
            provider (str, optional): The TTS provider key. If None, the default is used.
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
            value (bool): True to enable, False to disable.
            provider (str, optional): Specific provider key; if None, applies globally.
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
        Retrieves TTS enabled status for a provider or globally.

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
        Retrieves the default TTS provider key.

        Returns:
            str: Default TTS provider key or None if not set.
        """
        if self.active_tts:
            for key, service in self.tts_services.items():
                if service == self.active_tts:
                    return key
        return None

    def set_default_tts_provider(self, provider: str):
        """
        Sets the default TTS provider.

        Args:
            provider (str): Unique key for the provider.
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
        Stops speech recognition via the active STT provider.

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
        Transcribes an audio file using the active STT provider.

        Args:
            audio_file (str): Path to the audio file.
            provider (str, optional): STT provider key.

        Returns:
            str: Transcribed text.
        """
        provider = provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return ""
        # Synchronous transcription call wrapped for simplicity.
        transcript = stt.transcribe(audio_file)
        # Log transcription history
        self.transcription_history.append({
            "audio_file": audio_file,
            "transcript": transcript,
            "timestamp": time.time()
        })
        return transcript

    async def batch_transcribe(self, audio_files: List[str], provider: str = None, **kwargs) -> List[str]:
        """
        Asynchronously transcribes a batch of audio files.

        Args:
            audio_files (List[str]): List of audio file paths.
            provider (str, optional): STT provider key.
            kwargs: Additional parameters for transcription.

        Returns:
            List[str]: List of transcribed texts.
        """
        provider = provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return []
        tasks = [asyncio.to_thread(stt.transcribe, af, **kwargs) for af in audio_files]
        transcripts = await asyncio.gather(*tasks)
        for af, transcript in zip(audio_files, transcripts):
            self.transcription_history.append({
                "audio_file": af,
                "transcript": transcript,
                "timestamp": time.time()
            })
        return transcripts

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
        Retrieves the default STT provider key.

        Returns:
            str: Default STT provider key or None if not set.
        """
        if self.active_stt:
            for key, service in self.stt_services.items():
                if service == self.active_stt:
                    return key
        return None

    def set_default_stt_provider(self, provider: str):
        """
        Sets the default STT provider.

        Args:
            provider (str): Unique key for the provider.
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
