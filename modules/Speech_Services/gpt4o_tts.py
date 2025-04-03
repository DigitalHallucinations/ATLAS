# modules/Speech_Services/gpt4o_tts.py

"""
Module: gpt4o_tts.py
Description:
    Implements a Text-to-Speech provider using OpenAI's GPT-4o Mini TTS model.
    This provider converts text to speech via the new GPT-4o Mini TTS API.
    
Author: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""
import os
import logging

logger = logging.getLogger(__name__)

class GPT4oTTS:
    """
    Text-to-Speech provider using OpenAI's GPT-4o Mini TTS.
    """
    def __init__(self, voice="default"):
        """
        Initializes the GPT-4o TTS provider.
        
        Args:
            voice (str): The voice to use for synthesis.
        """
        self.voice = voice
        import openai
        self.openai = openai
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY is not set for GPT4o TTS.")
            raise Exception("OPENAI_API_KEY is required for GPT4o TTS")
        self.openai.api_key = self.api_key
        logger.info("Initialized GPT4o TTS provider.")
        
    async def text_to_speech(self, text: str):
        """
        Converts text to speech using the GPT-4o Mini TTS API.
        Saves the resulting audio to a file.
        
        Args:
            text (str): The text to synthesize.
        """
        try:
            # Assuming the new API uses openai.Audio.create for TTS with model "gpt-4o-mini-tts"
            response = self.openai.Audio.create(
                model="gpt-4o-mini-tts",
                input=text,
                voice=self.voice
            )
            audio_data = response.get("data")
            output_file = "gpt4o_tts_output.mp3"
            with open(output_file, "wb") as f:
                f.write(audio_data)
            logger.info("GPT4o TTS synthesis complete. Audio saved to %s", output_file)
            # Optionally, you can return the output file path.
            return output_file
        except Exception as e:
            logger.error(f"GPT4o TTS synthesis error: {e}")
