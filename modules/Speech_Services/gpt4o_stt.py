# modules/Speech_Services/gpt4o_stt.py

"""OpenAI GPT-4o speech-to-text provider with live microphone capture."""

from __future__ import annotations

import os
import time
import logging
from typing import Optional

try:  # Optional microphone dependencies
    import sounddevice as sd
except ImportError:  # pragma: no cover - exercised in runtime environments
    sd = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None  # type: ignore

logger = logging.getLogger(__name__)


class GPT4oSTT:
    """
    Speech-to-Text provider using OpenAI's GPT-4o or GPT-4o Mini models.
    """
    def __init__(self, variant="gpt-4o", sample_rate: int = 16000):
        """
        Initializes the GPT-4o STT provider.

        Args:
            variant (str): Either "gpt-4o" or "gpt-4o-mini". Defaults to "gpt-4o".
        """
        if sd is None or np is None or sf is None:
            raise RuntimeError(
                "sounddevice, numpy, and soundfile must be installed to capture audio with GPT-4o STT."
            )

        self.variant = variant
        self.sample_rate = sample_rate
        self.frames = []
        self.recording = False
        self.stream = None
        self.audio_file: Optional[str] = None
        self.last_audio_path: Optional[str] = None
        self.output_dir = os.path.join("data", "audio", "stt_output")

        import openai
        self.openai = openai
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY is not set for GPT4o STT.")
            raise Exception("OPENAI_API_KEY is required for GPT4o STT")
        self.openai.api_key = self.api_key
        logger.debug("Initialized GPT4o STT provider (variant: %s).", self.variant)

        try:
            self.stream = sd.InputStream(
                callback=self._callback,
                channels=1,
                samplerate=self.sample_rate,
            )
        except Exception as exc:
            logger.error("Failed to initialize microphone stream: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Microphone capture helpers
    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def listen(self):
        """Begin capturing microphone audio for GPT-4o transcription."""
        if self.stream is None:
            self.stream = sd.InputStream(
                callback=self._callback,
                channels=1,
                samplerate=self.sample_rate,
            )

        self.frames = []
        self.recording = True
        self.audio_file = None
        self.last_audio_path = None

        try:
            self.stream.start()
            logger.debug("GPT4o STT listening (sample rate: %s Hz).", self.sample_rate)
        except Exception as exc:
            self.recording = False
            logger.error("Failed to start GPT4o STT recording: %s", exc)
            raise

    def stop_listening(self) -> Optional[str]:
        """Stop capturing audio and persist the latest recording to disk."""
        if not self.stream:
            logger.warning("GPT4o STT stop requested without an active stream.")
            self.recording = False
            return None

        audio_reference: Optional[str] = None

        if self.recording:
            try:
                self.stream.stop()
            except Exception as exc:
                logger.error("Failed to stop GPT4o STT stream: %s", exc)
            finally:
                self.recording = False
                audio_reference = self._save_recording()
        else:
            audio_reference = self._save_recording()

        try:
            self.stream.close()
        except Exception as exc:
            logger.debug("Error closing GPT4o STT stream: %s", exc)
        finally:
            self.stream = None
            self.frames = []

        if audio_reference:
            self.audio_file = audio_reference
            self.last_audio_path = audio_reference
            logger.debug("GPT4o STT saved recording to %s", audio_reference)
        else:
            logger.debug("GPT4o STT produced no audio recording.")

        return audio_reference

    def _save_recording(self) -> Optional[str]:
        """Persist buffered frames to a timestamped WAV file."""
        if not self.frames:
            return None

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"gpt4o_capture_{timestamp}.wav"
        output_path = os.path.join(self.output_dir, filename)

        try:
            data = np.concatenate(self.frames)
            sf.write(output_path, data, self.sample_rate)
            return output_path
        except Exception as exc:
            logger.error("Failed to save GPT4o STT recording: %s", exc)
            return None

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

    def get_audio_file(self) -> Optional[str]:
        """Return the most recent audio file path saved by the provider."""

        return self.last_audio_path
