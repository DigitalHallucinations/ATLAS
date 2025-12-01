"""Base classes and shared helpers for speech services."""

from abc import ABC, abstractmethod
import os
import re
from typing import Callable, Optional

from modules.audio import AudioEngine, PlaybackRequest, SoundDeviceEngine

_INLINE_CODE_PATTERN = re.compile(r"`[^`]*`")
_CODE_BLOCK_PATTERN = re.compile(r"<code>.*?</code>", re.IGNORECASE | re.DOTALL)

_DEFAULT_AUDIO_ENGINE: AudioEngine | None = None

class BaseTTS(ABC):
    @abstractmethod
    async def text_to_speech(self, text: str):
        """Convert ``text`` into audio output."""

    def play_audio_file(
        self,
        filename: str,
        *,
        logger,
        audio_engine: Optional[AudioEngine] = None,
        cleanup: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Play ``filename`` with the shared audio engine while handling cleanup."""

        def _run_cleanup() -> None:
            if cleanup is None:
                return
            try:
                cleanup(filename)
            except FileNotFoundError:
                logger.debug("Temporary audio file already removed: %s", filename)
            except OSError as exc:
                logger.warning("Failed to remove audio file %s: %s", filename, exc)

        try:
            engine = _resolve_audio_engine(audio_engine)
        except Exception as exc:  # noqa: BLE001 - propagate precise failure details
            logger.error("Audio engine unavailable; skipping playback: %s", exc)
            _run_cleanup()
            return

        try:
            handle = engine.play(PlaybackRequest(path=filename))
            logger.debug("Audio playback started for: %s", filename)
            handle.wait()
            logger.debug("Audio playback finished for: %s", filename)
        except Exception as exc:  # noqa: BLE001 - propagate playback failure details
            logger.error("Error during audio playback: %s", exc)
        finally:
            _run_cleanup()

    @abstractmethod
    def set_voice(self, voice: dict):
        """Set the active voice used for synthesis."""

    @abstractmethod
    def get_voices(self) -> list:
        """Return available voices for the provider."""

    @abstractmethod
    def set_tts(self, value: bool):
        """Enable or disable TTS playback."""

    @abstractmethod
    def get_tts(self) -> bool:
        """Return whether TTS playback is enabled."""

    @staticmethod
    def strip_code_blocks(text: str) -> str:
        """Remove inline and fenced code segments from ``text``.

        This helper ensures text presented to a TTS backend does not include
        code snippets that would otherwise produce undesirable speech output.
        """

        without_inline = _INLINE_CODE_PATTERN.sub("", text)
        without_blocks = _CODE_BLOCK_PATTERN.sub("", without_inline)
        return without_blocks.strip()

    @staticmethod
    def safe_remove_file(path: str, *, logger) -> None:
        """Best-effort removal of ``path`` that logs any failures."""

        try:
            os.remove(path)
            logger.debug("Removed temporary audio file: %s", path)
        except FileNotFoundError:
            logger.debug("Temporary audio file already removed: %s", path)
        except OSError as exc:
            logger.warning("Failed to remove temporary audio file %s: %s", path, exc)


def _resolve_audio_engine(audio_engine: AudioEngine | None = None) -> AudioEngine:
    """Return the shared audio engine or the supplied override."""

    global _DEFAULT_AUDIO_ENGINE

    if audio_engine is not None:
        return audio_engine

    if _DEFAULT_AUDIO_ENGINE is None:
        _DEFAULT_AUDIO_ENGINE = SoundDeviceEngine()

    return _DEFAULT_AUDIO_ENGINE

class BaseSTT(ABC):
    @abstractmethod
    def listen(self):
        pass

    @abstractmethod
    def stop_listening(self):
        pass

    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        pass
