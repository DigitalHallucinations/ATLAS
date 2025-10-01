# modules/Speech_Services/Google/tts.py
# Description: Text to speech module using Google Cloud Text-to-Speech

import os
import re
import pygame
import threading
from typing import Any, Optional, Tuple
from google.cloud import texttospeech
from modules.logging.logger import setup_logger
from .base import BaseTTS

logger = setup_logger('google_tts.py')

CHUNK_SIZE = 1024
OUTPUT_PATH = "assets/SCOUT/tts_mp3/output.mp3"

class GoogleTTS(BaseTTS):
    def __init__(self):
        self._use_tts = False
        self._voice_config = {
            "language_code": "en-US",
            "name": "en-US-Wavenet-A",
        }
        self.voice = texttospeech.VoiceSelectionParams(**self._voice_config)
        self.client = texttospeech.TextToSpeechClient()

    def play_audio(self, filename):
        logger.info(f"Playing audio file: {filename}")
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        logger.info("Audio playback finished.")

    def contains_code(self, text: str) -> bool:
        logger.debug(f"Checking if text contains code: {text}")
        return "<code>" in text

    async def text_to_speech(self, text: str):
        if not self._use_tts:
            logger.info("TTS is turned off.")
            return

        if self.contains_code(text):
            logger.info("Skipping TTS as the text contains code.")
            text = re.sub(r"`[^`]*`", "", text)

        synthesis_input = texttospeech.SynthesisInput(text=text)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        try:
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=audio_config
            )
            logger.info("Google TTS response received successfully.")
        except Exception as e:
            logger.error(f"Google TTS synthesis error: {e}")
            return

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "wb") as out:
            out.write(response.audio_content)
            logger.info(f'Audio content written to "{OUTPUT_PATH}"')

        threading.Thread(target=self.play_audio, args=(OUTPUT_PATH,)).start()

    def _normalize_voice_payload(self, voice: Any) -> Tuple[Optional[str], Optional[str]]:
        if isinstance(voice, dict):
            name = voice.get("name") or voice.get("voice_id") or voice.get("id")
            language_code = (
                voice.get("language")
                or voice.get("language_code")
                or (
                    voice.get("language_codes")[0]
                    if isinstance(voice.get("language_codes"), (list, tuple)) and voice.get("language_codes")
                    else None
                )
            )
            if isinstance(language_code, str):
                return name, language_code
            return name, None

        if isinstance(voice, str):
            parts = voice.split("-")
            language_code = "-".join(parts[:2]) if len(parts) >= 2 else None
            return voice, language_code

        return None, None

    def _update_voice(self, *, name: Optional[str], language_code: Optional[str]):
        config = self._voice_config.copy()
        if name:
            config["name"] = name
        if language_code:
            config["language_code"] = language_code
        self._voice_config = config
        self.voice = texttospeech.VoiceSelectionParams(**config)

    def set_voice(self, voice: Any):
        name, language_code = self._normalize_voice_payload(voice)
        if not name and not language_code:
            logger.warning("Invalid voice payload supplied to Google TTS; keeping current voice.")
            return
        self._update_voice(name=name, language_code=language_code)
        logger.info(
            "Google TTS voice set to: %s (language: %s)",
            self._voice_config.get("name"),
            self._voice_config.get("language_code"),
        )

    def get_voices(self) -> list:
        voices = []
        try:
            response = self.client.list_voices()
            for voice in response.voices:
                voices.append({
                    'name': voice.name,
                    'language_codes': list(voice.language_codes),
                    'ssml_gender': texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                    'natural_sample_rate_hertz': voice.natural_sample_rate_hertz,
                })
            logger.info(f"Google TTS: Found {len(voices)} voices.")
        except Exception as e:
            logger.error(f"Error fetching Google TTS voices: {e}")
        return voices

    def set_tts(self, value: bool):
        self._use_tts = value
        logger.info(f"Google TTS set to: {self._use_tts}")

    def get_tts(self) -> bool:
        return self._use_tts
