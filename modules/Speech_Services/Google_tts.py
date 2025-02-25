# modules/Speech_Services/Google/tts.py
# Description: Text to speech module using Google Cloud Text-to-Speech

import os
import re
import pygame
import threading
from google.cloud import speech_v1p1beta1 as speech
from modules.logging.logger import setup_logger
from .base import BaseTTS

logger = setup_logger('google_tts.py')

CHUNK_SIZE = 1024
OUTPUT_PATH = "assets/SCOUT/tts_mp3/output.mp3"

class GoogleTTS(BaseTTS):
    def __init__(self):
        self._use_tts = False
        self.voice = speech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-A"
        )
        self.client = speech.SpeechClient()

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

        synthesis_input = speech.SynthesisInput(text=text)
        audio_config = speech.AudioConfig(
            audio_encoding=speech.AudioEncoding.MP3
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

    def set_voice(self, voice_name: str):
        self.voice = speech.VoiceSelectionParams(
            language_code=voice_name.split('-')[0] + '-' + voice_name.split('-')[1],
            name=voice_name,
        )
        logger.info(f"Google TTS voice set to: {voice_name}")

    def get_voices(self) -> list:
        voices = []
        try:
            response = self.client.list_voices()
            for voice in response.voices:
                for language_code in voice.language_codes:
                    voices.append({
                        'name': voice.name,
                        'language': language_code,
                        'ssml_gender': speech.SsmlVoiceGender(voice.ssml_gender).name,
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
