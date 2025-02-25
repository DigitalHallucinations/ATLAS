# modules/Speech_Services/Eleven_Labs/elevenlabs_tts.py

import os
import re
import pygame
import threading
import requests
from datetime import datetime
from dotenv import load_dotenv
from modules.logging.logger import setup_logger
from .base import BaseTTS

logger = setup_logger('eleven_labs_tts.py')

CHUNK_SIZE = 1024 

class ElevenLabsTTS(BaseTTS):
    def __init__(self):
        self._use_tts = False
        self.voice_ids = []
        self.load_voices()

    def load_voices(self):
        load_dotenv()
        XI_API_KEY = os.getenv("XI_API_KEY")
        if XI_API_KEY is None:
            logger.error("API key not found. Please set the XI_API_KEY environment variable.")
            raise ValueError("API key not found. Please set the XI_API_KEY environment variable.")

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

        logger.info(f"Sending TTS request to Eleven Labs with text: {text}")
        response = requests.post(tts_url, headers=headers, json=data, stream=True)

        logger.info(f"Eleven Labs API response status: {response.status_code}")
        if response.ok:
            logger.info("Received TTS response successfully.")
            output_dir = "assets/SCOUT/tts_mp3/"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = os.path.join(output_dir, f"output_{timestamp}.mp3")
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
            logger.info(f"Audio saved to {output_path}")
            threading.Thread(target=self.play_audio, args=(output_path,)).start()
        else:
            logger.error(f"Eleven Labs TTS Error: {response.text}")

    def set_voice(self, voice: dict):
        for i, v in enumerate(self.voice_ids):
            if v['name'] == voice['name'] and v['voice_id'] == voice['voice_id']:
                self.voice_ids.pop(i)
                self.voice_ids.insert(0, voice)
                logger.info(f"Voice set to: {voice['name']} with ID: {voice['voice_id']}")
                return
        logger.error(f"Voice {voice['name']} not found in Eleven Labs voices.")

    def get_voices(self) -> list:
        return self.voice_ids

    def set_tts(self, value: bool):
        self._use_tts = value
        logger.info(f"Eleven Labs TTS set to: {self._use_tts}")

    def get_tts(self) -> bool:
        return self._use_tts
