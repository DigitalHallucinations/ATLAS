# modules/Speech_Services/gpt4o_stt.py

"""
Module: gpt4o_stt.py
Description:
    Implements a Speech-to-Text provider using OpenAI's GPT-4o (or GPT-4o Mini) models.
    This provider calls the new audio transcription API using either the full or mini variant.
    
Author: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""
import os
import logging

logger = logging.getLogger(__name__)

class GPT4oSTT:
    """
    Speech-to-Text provider using OpenAI's GPT-4o or GPT-4o Mini models.
    """
    def __init__(self, variant="gpt-4o"):
        """
        Initializes the GPT-4o STT provider.
        
        Args:
            variant (str): Either "gpt-4o" or "gpt-4o-mini". Defaults to "gpt-4o".
        """
        self.variant = variant
        import openai
        self.openai = openai
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY is not set for GPT4o STT.")
            raise Exception("OPENAI_API_KEY is required for GPT4o STT")
        self.openai.api_key = self.api_key
        logger.debug("Initialized GPT4o STT provider (variant: %s).", self.variant)
        
    def transcribe(self, audio_file: str, language=None) -> str:
        """
        Transcribes an audio file using GPT-4o (or GPT-4o Mini) transcription.
        
        Args:
            audio_file (str): Path to the audio file.
            language (str): Optional ISO language code.
        
        Returns:
            str: The transcribed text.
        """
        try:
            with open(audio_file, "rb") as f:
                params = {
                    "model": "gpt-4o-transcribe" if self.variant == "gpt-4o" else "gpt-4o-mini-transcribe",
                    "file": f,
                }
                if language:
                    params["language"] = language
                result = self.openai.Audio.transcribe(**params)
            logger.debug("GPT4o STT transcription complete.")
            return result.get("text", "")
        except Exception as e:
            logger.error(f"GPT4o STT transcription error: {e}")
            return ""
