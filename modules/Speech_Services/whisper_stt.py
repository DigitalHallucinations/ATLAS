# modules/Speech_Services/whisper_stt.py

"""
Module: whisper_stt.py
Description:
    Implements an enterprise-grade speech-to-text provider using OpenAI Whisper.
    Supports both local and online transcription with advanced error handling,
    asynchronous and batch operations, noise reduction, audio format conversion,
    and enhanced logging/telemetry.
    
Author: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import logging
import tempfile
import time
import asyncio

# Optional noise reduction support
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

# Optional audio conversion support via pydub
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)

class WhisperSTT:
    """
    Speech-to-Text provider using OpenAI Whisper.
    Supports local and online modes with extended configuration.
    """
    def __init__(self, mode="local", model_name="base", fs=16000, device=None, noise_reduction=False, fallback_online=False):
        self.mode = mode.lower()  # "local" or "online"
        self.fs = fs
        self.frames = []
        self.recording = False
        self.audio_file = None
        self.noise_reduction = noise_reduction
        self.fallback_online = fallback_online

        # Device selection: auto-detect if not provided.
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Initialize the microphone stream.
        try:
            self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=self.fs)
        except Exception as e:
            logger.error(f"Error initializing microphone stream: {e}")
            raise e

        if self.mode == "local":
            try:
                import whisper
                self.model = whisper.load_model(model_name, device=self.device)
                logger.info(f"Loaded local Whisper model: {model_name} on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load local Whisper model: {e}")
                if self.fallback_online:
                    logger.info("Falling back to online mode due to local model load failure.")
                    self.mode = "online"
                    self._init_online()
                else:
                    raise e
        else:
            self._init_online()

    def _init_online(self):
        try:
            import openai
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.error("OPENAI_API_KEY not set. Online mode is disabled.")
                self.online_configured = False
            else:
                openai.api_key = self.api_key
                self.online_configured = True
                logger.info("Configured OpenAI API for Whisper online mode.")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper online mode: {e}")
            self.online_configured = False
            if not self.fallback_online:
                raise e

    def callback(self, indata, frames, time_info, status):
        """
        Callback for storing audio frames.
        """
        if self.recording:
            self.frames.append(indata.copy())

    def listen(self):
        """
        Starts recording audio.
        """
        self.frames = []
        self.recording = True
        try:
            self.stream.start()
            logger.info("Whisper STT listening...")
        except Exception as e:
            logger.error(f"Error starting stream: {e}")

    def stop_listening(self):
        """
        Stops recording and saves the audio file.
        """
        if self.recording:
            try:
                self.stream.stop()
                self.recording = False
                self.save_recording()
                logger.info("Whisper STT stopped listening.")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
        try:
            self.stream.close()
        except Exception as e:
            logger.error(f"Error closing stream: {e}")
        self.frames = []

    def save_recording(self, filename="whisper_output.wav"):
        """
        Saves recorded audio to a file.
        """
        output_dir = os.path.join("assets", "user", "sst_whisper")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)
        if self.frames:
            try:
                data = np.concatenate(self.frames)
                sf.write(output_file, data, self.fs)
                logger.info(f"Saved Whisper audio to {output_file}")
                self.audio_file = output_file
            except Exception as e:
                logger.error(f"Error saving recording: {e}")
        else:
            logger.warning("No audio frames to save.")

    def _apply_noise_reduction(self, audio_data):
        """
        Applies noise reduction to the audio data if enabled.
        """
        if self.noise_reduction and NOISEREDUCE_AVAILABLE:
            try:
                reduced_noise = nr.reduce_noise(y=audio_data, sr=self.fs)
                logger.info("Noise reduction applied.")
                return reduced_noise
            except Exception as e:
                logger.error(f"Noise reduction failed: {e}")
                return audio_data
        return audio_data

    def _ensure_wav_format(self, audio_file):
        """
        Converts audio to WAV format if needed using pydub.
        """
        if not audio_file.lower().endswith(".wav"):
            if PYDUB_AVAILABLE:
                try:
                    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sound = AudioSegment.from_file(audio_file)
                    sound.export(temp_wav.name, format="wav")
                    logger.info(f"Converted {audio_file} to WAV format.")
                    return temp_wav.name
                except Exception as e:
                    logger.error(f"Audio conversion failed: {e}")
                    return audio_file
            else:
                logger.warning("pydub not available. Cannot convert audio to WAV.")
        return audio_file

    def transcribe(self, audio_file=None, language=None, task="transcribe", initial_prompt=None, return_segments=False) -> str:
        """
        Transcribes an audio file with optional noise reduction and format conversion.
        Also logs the transcription duration.
        """
        if not audio_file:
            if not self.audio_file:
                logger.error("No audio file available for transcription.")
                return ""
            audio_file = self.audio_file

        # Ensure audio is in WAV format.
        audio_file = self._ensure_wav_format(audio_file)

        # If noise reduction is enabled, process audio and use a temporary file.
        temp_file = None
        if self.noise_reduction:
            try:
                data, samplerate = sf.read(audio_file)
                processed_data = self._apply_noise_reduction(data)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                    sf.write(tf.name, processed_data, samplerate)
                    temp_file = tf.name
                audio_file = temp_file
            except Exception as e:
                logger.error(f"Error during noise reduction: {e}")

        start_time = time.time()
        transcript = ""
        if self.mode == "local":
            try:
                result = self.model.transcribe(
                    audio_file,
                    language=language,
                    task=task,
                    initial_prompt=initial_prompt
                )
                transcript = result['segments'] if return_segments else result.get("text", "")
                logger.info("Local Whisper transcription complete.")
            except Exception as e:
                logger.error(f"Error during local transcription: {e}")
                if self.fallback_online and self.online_configured:
                    logger.info("Falling back to online transcription.")
                    transcript = self._transcribe_online(audio_file, language, task, initial_prompt)
        else:
            if not getattr(self, "online_configured", False):
                logger.error("Online mode is not configured (missing API key).")
                transcript = "Online transcription unavailable (API key missing)"
            else:
                transcript = self._transcribe_online(audio_file, language, task, initial_prompt)
        duration = time.time() - start_time
        logger.info(f"Transcription duration: {duration:.2f} seconds.")
        # Clean up temporary file if created.
        if temp_file:
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {e}")
        return transcript

    def _transcribe_online(self, audio_file, language, task, initial_prompt):
        try:
            import openai
            with open(audio_file, "rb") as f:
                params = {
                    "model": "whisper-1",
                    "file": f,
                    "task": task,
                }
                if language:
                    params["language"] = language
                if initial_prompt:
                    params["initial_prompt"] = initial_prompt
                result = openai.Audio.transcribe(**params)
            logger.info("Online Whisper transcription complete.")
            return result.get("text", "")
        except Exception as e:
            logger.error(f"Error during online transcription: {e}")
            return ""

    async def async_transcribe(self, *args, **kwargs):
        """
        Asynchronous wrapper for transcribe.
        """
        return await asyncio.to_thread(self.transcribe, *args, **kwargs)

    async def batch_transcribe(self, audio_files, language=None, task="transcribe", initial_prompt=None, return_segments=False):
        """
        Asynchronously transcribes a batch of audio files.
        Returns a list of transcripts.
        """
        tasks = [
            self.async_transcribe(
                audio_file=af,
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                return_segments=return_segments
            ) for af in audio_files
        ]
        return await asyncio.gather(*tasks)

    def stream_transcribe(self, chunk_duration=5, **kwargs):
        """
        Stub for real-time streaming transcription.
        Future implementation: Process audio in overlapping chunks.
        """
        logger.info("Streaming transcription not yet implemented.")
        return ""
