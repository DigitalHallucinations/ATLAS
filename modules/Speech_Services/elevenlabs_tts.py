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
import re
import pygame
import requests
from datetime import datetime
from dotenv import load_dotenv
from modules.logging.logger import setup_logger
from .base import BaseTTS

logger = setup_logger('eleven_labs_tts.py')

CHUNK_SIZE = 1024

class ElevenLabsTTS(BaseTTS):
    def __init__(self):
        self._use_tts = True
        self.voice_ids = []
        self.configured = False  # Flag indicating whether API key is configured.
        self._mixer_failed = False
        self._mixer_failure_logged = False
        self.load_voices()

    def _ensure_mixer_ready(self) -> bool:
        """Ensures the pygame mixer is initialized before playback."""
        if self._mixer_failed:
            return False

        if pygame.mixer.get_init():
            return True

        try:
            pygame.mixer.init()
        except Exception as exc:
            if not self._mixer_failure_logged:
                logger.error("Failed to initialize pygame mixer: %s", exc)
                self._mixer_failure_logged = True
            self._mixer_failed = True
            return False

        return True

    def load_voices(self):
        """
        Loads available voices from the Eleven Labs API.
        Reads the XI_API_KEY from the environment.
        If the API key is not provided, logs a warning and disables further functionality.
        """
        load_dotenv()
        XI_API_KEY = os.getenv("XI_API_KEY")
        if not XI_API_KEY:
            logger.error("XI_API_KEY not found. Eleven Labs TTS is disabled. Please configure the API key.")
            self.configured = False
            self.voice_ids = []
            return  # Do not raise an exception; allow the app to continue.
        else:
            self.configured = True

        url = "https://api.elevenlabs.io/v1/voices"
        headers = {
            "Accept": "application/json",
            "xi-api-key": XI_API_KEY,
            "Content-Type": "application/json"
        }

        logger.info("Fetching voices from Eleven Labs API...")
        response = requests.get(url, headers=headers)
        if response.ok:
            data = response.json()
            self.voice_ids = [{'voice_id': voice['voice_id'], 'name': voice['name']} for voice in data.get('voices', [])]
            if self.voice_ids:
                logger.info(f"Loaded {len(self.voice_ids)} voices from Eleven Labs.")
            else:
                logger.error("No voices found in Eleven Labs.")
        else:
            logger.error(f"Failed to fetch voices: {response.text}")
            self.voice_ids = []

    def play_audio(self, filename):
        """
        Plays the specified audio file.
        """
        logger.info(f"Playing audio file: {filename}")
        if not self._ensure_mixer_ready():
            logger.warning("Skipping audio playback because the mixer could not be initialized.")
            return

        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            logger.info("Audio playback finished.")
        except Exception:
            logger.exception("Error occurred during audio playback.")
        finally:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()

    def contains_code(self, text: str) -> bool:
        """
        Checks if the provided text contains any code blocks.
        
        Args:
            text (str): Text to check.
        
        Returns:
            bool: True if code is detected, False otherwise.
        """
        logger.debug(f"Checking if text contains code: {text}")
        return "<code>" in text

    async def text_to_speech(self, text: str):
        """
        Converts text to speech by sending a request to the Eleven Labs API.
        If TTS is disabled or the service is not configured, the method logs the state and returns.
        
        Args:
            text (str): Text to be converted to speech.
        """
        if not self._use_tts:
            logger.info("TTS is turned off.")
            return

        if not self.configured:
            logger.error("Eleven Labs TTS is not configured (missing API key).")
            return

        if self.contains_code(text):
            logger.info("Skipping TTS as the text contains code.")
            text = re.sub(r"`[^`]*`", "", text)

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
            logger.info(f"Sending TTS request to Eleven Labs with text: {text}")
            response = requests.post(tts_url, headers=headers, json=data, stream=True)

            logger.info(f"Eleven Labs API response status: {response.status_code}")
            if not response.ok:
                logger.error(f"Eleven Labs TTS Error: {response.text}")
                return None

            logger.info("Received TTS response successfully.")
            output_dir = "assets/SCOUT/tts_mp3/"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(output_dir, f"output_{timestamp}.mp3")

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Audio saved to {output_path}")
            return output_path

        try:
            output_path = await asyncio.to_thread(download_and_save_audio)
        except Exception:
            logger.exception("Unexpected error while generating audio with Eleven Labs TTS.")
            return

        if not output_path:
            return

        await asyncio.to_thread(self.play_audio, output_path)

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
                logger.info(f"Voice set to: {voice['name']} with ID: {voice['voice_id']}")
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