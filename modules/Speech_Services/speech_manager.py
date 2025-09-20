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

from typing import Dict, Any, List, Optional, Tuple, Callable
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
        self._active_stt_instance = None
        self._active_stt_key: Optional[str] = None
        self._recording_provider: Optional[str] = None
        self._last_audio_provider: Optional[str] = None
        self._stt_recording: bool = False
        self._last_audio_path: Optional[str] = None
        self._tts_enabled = False
        self.transcription_history: List[dict] = []  # Logs for transcription events

        self._google_tts_factory: Callable[[], Any] = lambda: GoogleTTS()
        self._google_stt_factory: Callable[[], Any] = lambda: GoogleSTT()
        self._openai_stt_factories: Dict[str, Callable[[], Any]] = {
            "Whisper Online": lambda: self._create_whisper_stt(),
            "GPT-4o STT": lambda: self._create_gpt4o_stt("gpt-4o"),
            "GPT-4o Mini STT": lambda: self._create_gpt4o_stt("gpt-4o-mini"),
        }
        self._openai_tts_factories: Dict[str, Callable[[], Any]] = {
            "GPT-4o Mini TTS": lambda: self._create_gpt4o_tts(),
        }

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
            google_tts = self._google_tts_factory()
            self.tts_services['google'] = google_tts
            logger.info("Google TTS initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS: {e}")

        # Initialize new GPT-4o TTS provider
        try:
            gpt4o_tts = self._create_gpt4o_tts()
            self.tts_services['gpt4o_tts'] = gpt4o_tts
            logger.info("GPT-4o TTS initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4o TTS: {e}")

        # Initialize STT providers
        try:
            google_stt = self._google_stt_factory()
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
            gpt4o_stt = self._create_gpt4o_stt("gpt-4o")
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

    @staticmethod
    def _restore_env_var(env_key: str, previous_value: Optional[str]):
        if previous_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous_value

    def _create_whisper_stt(self):
        from modules.Speech_Services.whisper_stt import WhisperSTT

        return WhisperSTT(mode="online")

    def _create_gpt4o_stt(self, variant: str):
        from modules.Speech_Services.gpt4o_stt import GPT4oSTT

        return GPT4oSTT(variant=variant)

    def _create_gpt4o_tts(self):
        from modules.Speech_Services.gpt4o_tts import GPT4oTTS

        return GPT4oTTS(voice="default")

    def set_google_credentials(self, credentials_path: str, persist: bool = True):
        """Update Google credentials and refresh provider instances."""

        if not credentials_path:
            raise ValueError("Google credentials path cannot be empty.")

        previous_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        was_default_tts = self.get_default_tts_provider() == 'google'
        was_default_stt = self.get_default_stt_provider() == 'google'

        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            new_tts = self._google_tts_factory()
            new_stt = self._google_stt_factory()
        except Exception as exc:
            self._restore_env_var("GOOGLE_APPLICATION_CREDENTIALS", previous_env)
            logger.error(f"Failed to initialize Google providers with new credentials: {exc}")
            raise

        if persist:
            try:
                self.config_manager.set_google_credentials(credentials_path)
            except Exception as exc:
                self._restore_env_var("GOOGLE_APPLICATION_CREDENTIALS", previous_env)
                logger.error(f"Failed to persist Google credentials: {exc}")
                raise
        else:
            self.config_manager.config['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        self.tts_services['google'] = new_tts
        self.stt_services['google'] = new_stt

        if was_default_tts:
            self.set_default_tts_provider('google')
        if was_default_stt:
            self.set_default_stt_provider('google')

    def set_elevenlabs_api_key(self, api_key: str, *, persist: bool = True):
        """Persist the ElevenLabs API key and refresh the provider instance."""

        if not api_key:
            raise ValueError("ElevenLabs API key cannot be empty.")

        previous_env = os.environ.get("XI_API_KEY")
        previous_service = self.tts_services.get('eleven_labs')
        was_default = self.get_default_tts_provider() == 'eleven_labs'

        previous_voice = None
        if previous_service:
            try:
                voices = previous_service.get_voices()
            except Exception as exc:
                logger.debug(f"Unable to capture existing ElevenLabs voice before refresh: {exc}")
                voices = []
            if voices:
                first_voice = voices[0]
                previous_voice = dict(first_voice) if isinstance(first_voice, dict) else first_voice

        try:
            os.environ["XI_API_KEY"] = api_key
            new_service = ElevenLabsTTS()
        except Exception as exc:
            self._restore_env_var("XI_API_KEY", previous_env)
            logger.error(f"Failed to initialize ElevenLabs TTS with new API key: {exc}")
            raise

        if persist:
            try:
                self.config_manager.set_elevenlabs_api_key(api_key)
            except Exception as exc:
                self._restore_env_var("XI_API_KEY", previous_env)
                logger.error(f"Failed to persist ElevenLabs API key: {exc}")
                raise
        else:
            if hasattr(self.config_manager, 'config'):
                self.config_manager.config['XI_API_KEY'] = api_key
            if hasattr(self.config_manager, 'env_config'):
                self.config_manager.env_config['XI_API_KEY'] = api_key

        if previous_voice:
            try:
                available_voices = new_service.get_voices()
            except Exception as exc:
                logger.debug(f"Unable to load ElevenLabs voices during refresh: {exc}")
                available_voices = []

            match = None
            if isinstance(previous_voice, dict):
                prev_id = previous_voice.get('voice_id')
                prev_name = previous_voice.get('name')
                for voice in available_voices:
                    if not isinstance(voice, dict):
                        continue
                    if prev_id and voice.get('voice_id') == prev_id:
                        match = voice
                        break
                    if prev_name and voice.get('name') == prev_name:
                        match = voice
                        break
            if match:
                try:
                    new_service.set_voice(match)
                except Exception as exc:
                    logger.debug(f"Failed to restore ElevenLabs voice selection: {exc}")

        self.tts_services['eleven_labs'] = new_service
        if was_default or self.active_tts is previous_service:
            self.active_tts = new_service

        logger.info("ElevenLabs API key updated and provider refreshed.")

    def set_openai_speech_config(
        self,
        *,
        api_key: Optional[str],
        stt_provider: Optional[str],
        language: Optional[str],
        task: Optional[str],
        initial_prompt: Optional[str],
        tts_provider: Optional[str],
        persist: bool = True,
    ):
        """Configure OpenAI-based providers and update persistence as needed."""

        previous_env = os.environ.get('OPENAI_API_KEY')
        stored_key = self.config_manager.get_config('OPENAI_API_KEY')
        target_api_key = api_key if api_key is not None else stored_key or previous_env

        if (stt_provider or tts_provider) and not target_api_key:
            raise ValueError("An OpenAI API key is required to configure speech services.")

        if target_api_key:
            os.environ['OPENAI_API_KEY'] = target_api_key

        new_stt = None
        new_tts = None
        new_stt_key: Optional[str] = None
        new_tts_key: Optional[str] = None

        if stt_provider:
            factory = self._openai_stt_factories.get(stt_provider)
            if not factory:
                self._restore_env_var('OPENAI_API_KEY', previous_env)
                raise ValueError(f"Unknown OpenAI STT provider '{stt_provider}'.")
            try:
                new_stt = factory()
                new_stt_key = 'openai_stt'
            except Exception as exc:
                self._restore_env_var('OPENAI_API_KEY', previous_env)
                logger.error(f"Failed to initialize OpenAI STT provider '{stt_provider}': {exc}")
                raise

        if tts_provider:
            factory = self._openai_tts_factories.get(tts_provider)
            if not factory:
                self._restore_env_var('OPENAI_API_KEY', previous_env)
                raise ValueError(f"Unknown OpenAI TTS provider '{tts_provider}'.")
            try:
                new_tts = factory()
                new_tts_key = 'openai_tts'
            except Exception as exc:
                self._restore_env_var('OPENAI_API_KEY', previous_env)
                logger.error(f"Failed to initialize OpenAI TTS provider '{tts_provider}': {exc}")
                raise

        if persist:
            try:
                self.config_manager.set_openai_speech_config(
                    api_key=api_key,
                    stt_provider=stt_provider,
                    tts_provider=tts_provider,
                    language=language,
                    task=task,
                    initial_prompt=initial_prompt,
                )
            except Exception as exc:
                self._restore_env_var('OPENAI_API_KEY', previous_env)
                logger.error(f"Failed to persist OpenAI speech configuration: {exc}")
                raise
        else:
            if api_key is not None:
                self.config_manager.config['OPENAI_API_KEY'] = api_key
            if stt_provider is not None:
                self.config_manager.config['OPENAI_STT_PROVIDER'] = stt_provider
            if tts_provider is not None:
                self.config_manager.config['OPENAI_TTS_PROVIDER'] = tts_provider
            if language is not None:
                self.config_manager.config['OPENAI_LANGUAGE'] = language
            if task is not None:
                self.config_manager.config['OPENAI_TASK'] = task
            if initial_prompt is not None:
                self.config_manager.config['OPENAI_INITIAL_PROMPT'] = initial_prompt

        if new_stt_key:
            self.stt_services[new_stt_key] = new_stt
        if new_tts_key:
            self.tts_services[new_tts_key] = new_tts

        self.set_default_speech_providers(tts_provider=new_tts_key, stt_provider=new_stt_key)

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
            if hasattr(self.config_manager, 'set_tts_enabled'):
                try:
                    self.config_manager.set_tts_enabled(value)
                except Exception as exc:
                    logger.error(f"Failed to persist TTS status: {exc}")
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

    def get_tts_provider_names(self) -> Tuple[str, ...]:
        """Return the registered TTS provider keys in insertion order."""

        return tuple(self.tts_services.keys())

    def resolve_tts_provider(self, preferred: Optional[str]) -> Optional[str]:
        """Resolve a TTS provider from a preferred key with sensible fallbacks."""

        providers = self.get_tts_provider_names()
        if preferred and preferred in providers:
            return preferred

        for fallback in ("eleven_labs",):
            if fallback in providers:
                return fallback

        return providers[0] if providers else None

    def get_default_tts_provider_index(self) -> Optional[int]:
        """Return the index of the default TTS provider within the provider list."""

        provider = self.get_default_tts_provider()
        if not provider:
            return None
        try:
            return self.get_tts_provider_names().index(provider)
        except ValueError:
            return None

    def set_default_speech_providers(self, tts_provider: Optional[str] = None, stt_provider: Optional[str] = None):
        """Update the default TTS and/or STT providers in a single call."""

        if tts_provider:
            if tts_provider in self.tts_services:
                self.set_default_tts_provider(tts_provider)
                if hasattr(self.config_manager, 'config'):
                    self.config_manager.config['DEFAULT_TTS_PROVIDER'] = tts_provider
                logger.info(f"Persisted default TTS provider '{tts_provider}'.")
            else:
                logger.error(f"Cannot set unknown TTS provider '{tts_provider}'.")

        if stt_provider:
            if stt_provider in self.stt_services:
                self.set_default_stt_provider(stt_provider)
                if hasattr(self.config_manager, 'config'):
                    self.config_manager.config['DEFAULT_STT_PROVIDER'] = stt_provider
                logger.info(f"Persisted default STT provider '{stt_provider}'.")
            else:
                logger.error(f"Cannot set unknown STT provider '{stt_provider}'.")

    def disable_stt(self):
        """Disable speech-to-text by clearing the active provider and state."""

        if self._stt_recording and self._recording_provider:
            try:
                self.stop_listening(self._recording_provider)
            except Exception as exc:
                logger.debug(f"Error stopping provider '{self._recording_provider}' during disable: {exc}")

        self._stt_recording = False
        self._recording_provider = None
        self._last_audio_provider = None
        self._last_audio_path = None
        self.active_stt = None
        if hasattr(self.config_manager, 'config'):
            self.config_manager.config['DEFAULT_STT_PROVIDER'] = None
        logger.info("STT disabled and default provider cleared.")

    def configure_openai_speech(
        self,
        api_key: Optional[str],
        stt_provider: str,
        language: Optional[str],
        task: Optional[str],
        initial_prompt: Optional[str],
        tts_provider: Optional[str],
    ):
        """Backward-compatible wrapper for legacy callers."""

        self.set_openai_speech_config(
            api_key=api_key,
            stt_provider=stt_provider,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            tts_provider=tts_provider,
        )

    def get_active_tts_summary(self) -> Tuple[str, str]:
        """Return a tuple describing the active TTS provider and voice label."""
        provider_label = "None"
        voice_label = "Not Set"

        try:
            provider_key = self.get_default_tts_provider()
        except Exception as exc:
            logger.error(f"Failed to determine default TTS provider: {exc}")
            provider_key = None

        active_service = None
        if provider_key:
            provider_label = provider_key
            active_service = self.tts_services.get(provider_key)
            if active_service is None:
                logger.error(f"Active TTS provider '{provider_key}' is not registered.")

        if active_service is None:
            fallback_service = getattr(self, "active_tts", None)
            if fallback_service:
                active_service = fallback_service
                if provider_label == "None":
                    for key, service in self.tts_services.items():
                        if service == active_service:
                            provider_label = key
                            break

        if active_service is None:
            return provider_label, voice_label

        try:
            voice_label = self._extract_voice_label(active_service, voice_label)
        except Exception as exc:
            logger.error(f"Failed to resolve active TTS voice: {exc}")

        return provider_label, voice_label

    def _extract_voice_label(self, tts_service: Any, default: str = "Not Set") -> str:
        """Derive a human readable voice label for the provided TTS service."""

        voice_label = default

        getter = getattr(tts_service, "get_current_voice", None)
        if callable(getter):
            try:
                voice_value = getter()
            except Exception as exc:
                logger.error(f"Error retrieving current voice from provider: {exc}")
            else:
                normalized = self._normalize_voice_value(voice_value, default)
                if normalized != default:
                    return normalized
                voice_label = normalized

        voice_ids = getattr(tts_service, "voice_ids", None)
        if isinstance(voice_ids, list) and voice_ids:
            normalized = self._normalize_voice_value(voice_ids[0], default)
            if normalized != default:
                return normalized
            voice_label = normalized

        for attr_name in ("voice", "current_voice", "selected_voice"):
            voice_value = getattr(tts_service, attr_name, None)
            if voice_value is not None:
                normalized = self._normalize_voice_value(voice_value, default)
                if normalized != default:
                    return normalized
                voice_label = normalized

        return voice_label

    def _normalize_voice_value(self, value: Any, default: str = "Not Set") -> str:
        """Convert various voice metadata representations into display text."""
        if value is None:
            return default

        if isinstance(value, str):
            text = value.strip()
            return text if text else default

        if isinstance(value, dict):
            for key in ("name", "label", "voice_name", "voice_id", "id"):
                entry = value.get(key)
                if entry:
                    return str(entry)
            return str(value)

        for attr in ("name", "label", "voice_name"):
            attr_value = getattr(value, attr, None)
            if attr_value:
                return str(attr_value)

        return str(value)

    # ----------------------- STT Methods -----------------------

    @property
    def active_stt(self):
        return self._active_stt_instance

    @active_stt.setter
    def active_stt(self, value):
        self._active_stt_instance = value
        if value is None:
            self._active_stt_key = None
        else:
            if hasattr(self, 'stt_services'):
                for key, service in self.stt_services.items():
                    if service == value:
                        self._active_stt_key = key
                        break

    def is_listening(self) -> bool:
        """Return True if an STT provider is actively recording."""
        return self._stt_recording

    def get_active_stt_provider(self) -> Optional[str]:
        """Return the provider currently used for recording or the default provider."""
        return self._recording_provider or self._active_stt_key

    def get_last_audio_path(self) -> Optional[str]:
        """Return the most recent audio file reference produced by stop_listening."""
        return self._last_audio_path

    def listen(self, provider: str = None) -> bool:
        """
        Starts speech recognition by invoking the active STT provider's listen method.

        Args:
            provider (str, optional): STT provider key.
        """
        if self._stt_recording:
            logger.info("STT recording is already active.")
            return True

        provider = provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return False

        try:
            stt.listen()
        except Exception as exc:
            logger.error(f"Failed to start STT provider '{provider}': {exc}")
            return False

        self._stt_recording = True
        self._recording_provider = provider
        self._last_audio_path = None
        self._last_audio_provider = None
        logger.info(f"Listening started using provider '{provider}'.")
        return True

    def stop_listening(self, provider: str = None) -> Optional[str]:
        """
        Stops speech recognition via the active STT provider.

        Args:
            provider (str, optional): STT provider key.
        """
        provider = provider or self._recording_provider or self.get_default_stt_provider()
        stt = self.stt_services.get(provider)
        if not stt:
            logger.error(f"STT provider '{provider}' not found.")
            return None

        try:
            stt.stop_listening()
        except Exception as exc:
            logger.error(f"Failed to stop STT provider '{provider}': {exc}")

        audio_reference = self._resolve_audio_reference(provider, stt)
        self._last_audio_path = audio_reference
        self._last_audio_provider = provider
        self._stt_recording = False
        self._recording_provider = None
        logger.info(f"Listening stopped for provider '{provider}'.")
        return audio_reference

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
        if self._active_stt_key:
            return self._active_stt_key
        if self.active_stt:
            for key, service in self.stt_services.items():
                if service == self.active_stt:
                    self._active_stt_key = key
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
        self._active_stt_key = provider
        self.active_stt = stt
        logger.info(f"Default STT provider set to '{provider}'.")

    def get_stt_provider_names(self) -> Tuple[str, ...]:
        """Return the registered STT provider keys in insertion order."""

        return tuple(self.stt_services.keys())

    def get_default_stt_provider_index(self) -> Optional[int]:
        """Return the index of the default STT provider within the provider list."""

        provider = self.get_default_stt_provider()
        if not provider:
            return None
        try:
            return self.get_stt_provider_names().index(provider)
        except ValueError:
            return None

    def _resolve_audio_reference(self, provider: Optional[str], stt_instance: Any) -> Optional[str]:
        """Infer the audio reference produced by an STT provider after recording."""
        if not stt_instance:
            return None

        audio_file = getattr(stt_instance, 'audio_file', None)
        if isinstance(audio_file, str) and audio_file:
            return audio_file

        if hasattr(stt_instance, 'get_audio_file') and callable(stt_instance.get_audio_file):
            try:
                audio_file = stt_instance.get_audio_file()
                if isinstance(audio_file, str) and audio_file:
                    return audio_file
            except Exception as exc:
                logger.debug(f"Error retrieving audio file from provider '{provider}': {exc}")

        if provider == 'google':
            return 'output.wav'

        return None

    async def stop_and_transcribe(self, provider: str | None = None) -> str:
        """Stop recording (if active) and return the resulting transcript string."""
        provider_key = provider or self._recording_provider or self.get_default_stt_provider()
        if not provider_key:
            logger.error("No STT provider configured for transcription.")
            return ""

        stt = self.stt_services.get(provider_key)
        if not stt:
            logger.error(f"STT provider '{provider_key}' not found for transcription.")
            return ""

        audio_reference: Optional[str] = None
        if self._stt_recording:
            audio_reference = await asyncio.to_thread(self.stop_listening, provider_key)
        else:
            audio_reference = self._last_audio_path or self._resolve_audio_reference(provider_key, stt)

        if not audio_reference:
            logger.warning("No audio available to transcribe.")
            return ""

        try:
            transcript = await asyncio.to_thread(self.transcribe, audio_reference, provider_key)
            return transcript or ""
        except Exception as exc:
            logger.error(f"Error transcribing audio with provider '{provider_key}': {exc}")
            return ""

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
