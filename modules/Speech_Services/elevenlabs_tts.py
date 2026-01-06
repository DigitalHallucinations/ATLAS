# modules/Speech_Services/Eleven_Labs/elevenlabs_tts.py

"""
Module: elevenlabs_tts.py
Description:
    Enterprise production–ready implementation of Eleven Labs Text-to-Speech integration.
    This provider now checks for an API key configuration. If the API key is not provided,
    the service logs a warning and disables TTS functionality without breaking the application.
    
Usage:
    The API key can be set via the environment (or via the ConfigManager in the Speech Settings UI).
    If not provided, the provider remains disabled.
    
Author: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""

import asyncio
import os
import threading
from typing import Optional

import requests
from datetime import datetime
from dotenv import load_dotenv

from core.config import ConfigManager
from modules.audio import AudioEngine
from modules.logging.logger import setup_logger

from .base import BaseTTS

logger = setup_logger('eleven_labs_tts.py')

CHUNK_SIZE = 1024

try:
    RequestsTimeout = requests.exceptions.Timeout
except AttributeError:  # pragma: no cover - fallback for minimal request stubs
    base_exception = getattr(requests.exceptions, "RequestException", Exception)

    class RequestsTimeout(base_exception):
        """Fallback timeout when requests library lacks a dedicated exception."""

    setattr(requests.exceptions, "Timeout", RequestsTimeout)

class ElevenLabsTTS(BaseTTS):
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        *,
        audio_engine: AudioEngine | None = None,
    ):
        self._use_tts = True
        self.voice_ids = []
        self.configured = False  # Flag indicating whether API key is configured.
        self.audio_engine = audio_engine

        # Bootstrap state
        self._initialization_error: Optional[Exception] = None
        self._initialization_lock: Optional[asyncio.Lock] = None
        self._lock_guard = threading.Lock()
        self._config_manager = config_manager or ConfigManager()

    def _load_voices_sync(self) -> None:
        """Load the available voices from Eleven Labs synchronously."""

        load_dotenv()
        xi_api_key = os.getenv("XI_API_KEY")
        if not xi_api_key:
            self.configured = False
            self.voice_ids = []
            logger.error("XI_API_KEY not found. Eleven Labs TTS is disabled. Please configure the API key.")
            raise RuntimeError("XI_API_KEY is not configured for Eleven Labs TTS.")

        url = "https://api.elevenlabs.io/v1/voices"
        headers = {
            "Accept": "application/json",
            "xi-api-key": xi_api_key,
            "Content-Type": "application/json",
        }

        logger.debug("Fetching voices from Eleven Labs API...")
        try:
            response = requests.get(url, headers=headers, timeout=30)
        except Exception as exc:  # noqa: BLE001 - propagate precise error upstream
            logger.error("Failed to fetch voices from Eleven Labs: %s", exc)
            self.voice_ids = []
            self.configured = False
            raise RuntimeError("Unable to contact Eleven Labs voice API.") from exc

        if not response.ok:
            logger.error("Failed to fetch voices: %s", response.text)
            self.voice_ids = []
            self.configured = False
            raise RuntimeError(f"Eleven Labs voice request failed with status {response.status_code}.")

        data = response.json()
        voices = [
            {"voice_id": voice["voice_id"], "name": voice["name"]}
            for voice in data.get("voices", [])
        ]

        if not voices:
            logger.error("No voices found in Eleven Labs response.")
            self.voice_ids = []
            self.configured = False
            raise RuntimeError("Eleven Labs returned no available voices.")

        self.voice_ids = voices
        self.configured = True
        logger.debug("Loaded %s voices from Eleven Labs.", len(self.voice_ids))

    async def ensure_ready(self) -> None:
        """Ensure the provider has loaded voices before use."""

        if self.configured and self.voice_ids:
            return
        if self._initialization_error is not None:
            raise self._initialization_error

        # Lazily create the asyncio lock when a loop is available.
        if self._initialization_lock is None:
            with self._lock_guard:
                if self._initialization_lock is None:
                    self._initialization_lock = asyncio.Lock()

        assert self._initialization_lock is not None  # For type checkers

        async with self._initialization_lock:
            if self.configured and self.voice_ids:
                return
            if self._initialization_error is not None:
                raise self._initialization_error

            try:
                await asyncio.to_thread(self._load_voices_sync)
            except Exception as exc:  # noqa: BLE001 - propagate to caller
                self._initialization_error = exc
                raise
            else:
                self._initialization_error = None

    def contains_code(self, text: str) -> bool:
        """
        Checks if the provided text contains any code blocks.
        
        Args:
            text (str): Text to check.
        
        Returns:
            bool: True if code is detected, False otherwise.
        """
        logger.debug(f"Checking if text contains code: {text}")
        return text != self.strip_code_blocks(text)

    async def text_to_speech(self, text: str):
        """
        Converts text to speech by sending a request to the Eleven Labs API.
        If TTS is disabled or the service is not configured, the method logs the state and returns.

        Args:
            text (str): Text to be converted to speech.
        """
        try:
            await self.ensure_ready()
        except Exception:
            logger.exception("Eleven Labs TTS failed to initialize; skipping speech synthesis.")
            return

        if not self._use_tts:
            logger.debug("TTS is turned off.")
            return

        if not self.configured:
            logger.error("Eleven Labs TTS is not configured (missing API key).")
            return

        stripped_text = self.strip_code_blocks(text)
        if stripped_text != text:
            logger.debug("Skipping code snippets in Eleven Labs TTS request payload.")
            text = stripped_text

        if not text:
            logger.debug("No text remaining after removing code; skipping Eleven Labs TTS synthesis.")
            return

        if not self.voice_ids:
            logger.error("No voice IDs available for Eleven Labs TTS.")
            return

        voice_id = self.voice_ids[0]['voice_id']
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
       
        headers = {
            "Accept": "application/json",
            "xi-api-key": os.getenv("XI_API_KEY")
        }

        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }

        def download_and_save_audio():
            logger.debug("Sending TTS request to Eleven Labs with text: %s", text)
            connect_timeout = 30
            read_timeout = 30

            try:
                response = requests.post(
                    tts_url,
                    headers=headers,
                    json=data,
                    stream=True,
                    timeout=(connect_timeout, read_timeout),
                )
            except RequestsTimeout as exc:
                logger.warning(
                    "Eleven Labs TTS request timed out after %s/%s seconds (connect/read): %s",
                    connect_timeout,
                    read_timeout,
                    exc,
                )
                raise
            except Exception as exc:  # noqa: BLE001 - propagate precise error upstream
                logger.error("Failed to send Eleven Labs TTS request: %s", exc)
                raise

            logger.debug("Eleven Labs API response status: %s", response.status_code)
            if not response.ok:
                logger.error(f"Eleven Labs TTS Error: {response.text}")
                return None

            logger.debug("Received TTS response successfully.")
            output_dir = self._resolve_output_dir()
            try:
                os.makedirs(output_dir, exist_ok=True)
            except PermissionError as exc:
                logger.error(
                    "Permission denied when creating Eleven Labs speech cache directory '%s': %s",
                    output_dir,
                    exc,
                )
                return None
            except OSError as exc:
                logger.error(
                    "Failed to create Eleven Labs speech cache directory '%s': %s",
                    output_dir,
                    exc,
                )
                return None
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(output_dir, f"output_{timestamp}.mp3")

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            logger.debug("Audio saved to %s", output_path)
            return output_path

        try:
            output_path = await asyncio.to_thread(download_and_save_audio)
        except RequestsTimeout:
            logger.warning("Eleven Labs TTS request timed out; skipping speech synthesis.")
            return
        except Exception:
            logger.exception("Unexpected error while generating audio with Eleven Labs TTS.")
            return

        if not output_path:
            return

        await asyncio.to_thread(self.play_audio, output_path)

    def play_audio(self, filename):
        """Retained for backwards compatibility with tests and callers."""

        logger.debug("Playing audio file: %s", filename)
        self.play_audio_file(
            filename,
            logger=logger,
            audio_engine=self.audio_engine,
        )

    def _resolve_output_dir(self) -> str:
        """Determine where Eleven Labs speech files should be written."""

        config_dir = None

        if self._config_manager is not None:
            try:
                config_dir = self._config_manager.get_config(
                    'ELEVENLABS_SPEECH_CACHE_DIR',
                    None,
                )
            except Exception as exc:  # noqa: BLE001 - log and continue with fallback
                logger.warning(
                    "Error accessing ELEVENLABS_SPEECH_CACHE_DIR configuration: %s",
                    exc,
                )
                config_dir = None

            if not config_dir:
                getter = getattr(self._config_manager, 'get_speech_cache_dir', None)
                if callable(getter):
                    try:
                        config_dir = getter()
                    except Exception as exc:  # noqa: BLE001 - log and continue
                        logger.warning(
                            "Error resolving speech cache directory from configuration: %s",
                            exc,
                        )
                        config_dir = None

        if not config_dir:
            app_root = None
            if self._config_manager is not None:
                try:
                    app_root = self._config_manager.get_app_root()
                except Exception as exc:  # noqa: BLE001 - log and fall back to cwd
                    logger.warning(
                        "Error retrieving APP_ROOT from configuration: %s",
                        exc,
                    )
            if not app_root:
                app_root = os.getcwd()
            config_dir = os.path.join(app_root, 'data', 'audio', 'tts_output')

        return config_dir

    def set_voice(self, voice: dict):
        """
        Sets the active voice to the one matching the provided voice dictionary.
        
        Args:
            voice (dict): Dictionary containing voice details.
        """
        for i, v in enumerate(self.voice_ids):
            if v['name'] == voice['name'] and v['voice_id'] == voice['voice_id']:
                self.voice_ids.pop(i)
                self.voice_ids.insert(0, voice)
                logger.debug(
                    "Voice set to: %s with ID: %s",
                    voice['name'],
                    voice['voice_id'],
                )
                return
        logger.error(f"Voice {voice['name']} not found in Eleven Labs voices.")

    def get_voices(self) -> list:
        """
        Returns a list of available voices.
        
        Returns:
            list: Available voices.
        """
        return self.voice_ids

    def set_tts(self, value: bool):
        """
        Enables or disables TTS functionality.
        
        Args:
            value (bool): True to enable, False to disable.
        """
        self._use_tts = value
        logger.info(f"Eleven Labs TTS set to: {self._use_tts}")

    def get_tts(self) -> bool:
        """
        Gets the current TTS enabled status.

        Returns:
            bool: True if enabled, False otherwise.
        """
        return self._use_tts
