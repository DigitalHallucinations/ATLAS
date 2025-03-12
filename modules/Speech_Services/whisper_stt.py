# modules/Speech_Services/whisper_stt.py

"""
Module: whisper_stt.py
Description:
    Implements a speech-to-text provider using OpenAI Whisper.
    Supports both local transcription (via the whisper Python package)
    and online transcription (via OpenAI's API). If the API key for online mode
    is not configured, online transcription is disabled.
    
Author: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""

import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

class WhisperSTT:
    """
    Speech-to-Text provider using OpenAI Whisper.
    Supports both local and online modes.
    
    Args:
        mode (str): Either "local" or "online". If "online", an API key must be set.
        model_name (str): The Whisper model to load in local mode.
        fs (int): The sampling frequency.
    """
    def __init__(self, mode="local", model_name="base", fs=16000):
        self.mode = mode.lower()  # "local" or "online"
        self.fs = fs
        self.frames = []
        self.recording = False
        self.audio_file = None

        # Initialize microphone stream.
        self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=self.fs)

        if self.mode == "local":
            try:
                import whisper
                self.model = whisper.load_model(model_name)
                logger.info(f"Loaded local Whisper model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load local Whisper model: {e}")
                raise e
        else:
            try:
                import openai
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    logger.error("OPENAI_API_KEY not set. Whisper online mode is disabled.")
                    self.online_configured = False
                    # Online mode disabled; you can decide to either fall back or raise an exception.
                else:
                    openai.api_key = self.api_key
                    self.online_configured = True
                    logger.info("Configured OpenAI API for Whisper online mode.")
            except Exception as e:
                logger.error(f"Failed to initialize Whisper online mode: {e}")
                raise e

    def callback(self, indata, frames, time, status):
        """
        Callback for the audio input stream to store audio frames.
        """
        if self.recording:
            self.frames.append(indata.copy())

    def listen(self):
        """
        Starts recording audio from the microphone.
        """
        self.frames = []
        self.recording = True
        self.stream.start()
        logger.info("Whisper STT listening...")

    def stop_listening(self):
        """
        Stops recording and saves the audio file.
        """
        if self.recording:
            self.stream.stop()
            self.recording = False
            self.save_recording()
            logger.info("Whisper STT stopped listening.")
        self.stream.close()
        self.frames = []

    def save_recording(self, filename="whisper_output.wav"):
        """
        Saves the recorded audio to a file.
        
        Args:
            filename (str): The name of the file to save.
        """
        output_dir = os.path.join("assets", "user", "sst_whisper")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)
        if self.frames:
            data = np.concatenate(self.frames)
            sf.write(output_file, data, self.fs)
            logger.info(f"Saved Whisper audio to {output_file}")
            self.audio_file = output_file
        else:
            logger.warning("No audio frames to save.")

    def transcribe(self, audio_file=None) -> str:
        """
        Transcribes the audio file using the selected Whisper mode.
        In online mode, if the API key is not configured, returns an error message.
        
        Args:
            audio_file (str): Path to the audio file; if not provided, uses the last recorded file.
            
        Returns:
            str: The transcribed text.
        """
        if not audio_file:
            if not self.audio_file:
                logger.error("No audio file available for transcription.")
                return ""
            audio_file = self.audio_file

        if self.mode == "local":
            try:
                result = self.model.transcribe(audio_file)
                logger.info("Local Whisper transcription complete.")
                return result.get("text", "")
            except Exception as e:
                logger.error(f"Error during local transcription: {e}")
                return ""
        else:
            if not getattr(self, "online_configured", False):
                logger.error("Online mode is not configured due to missing API key.")
                return "Online transcription unavailable (API key missing)"
            try:
                import openai
                with open(audio_file, "rb") as f:
                    result = openai.Audio.transcribe("whisper-1", f)
                logger.info("Online Whisper transcription complete.")
                return result.get("text", "")
            except Exception as e:
                logger.error(f"Error during online transcription: {e}")
                return ""