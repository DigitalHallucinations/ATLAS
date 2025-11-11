"""Base classes and shared helpers for speech services."""

from abc import ABC, abstractmethod
import os
import re
import time
from typing import Callable, Optional

import pygame

_INLINE_CODE_PATTERN = re.compile(r"`[^`]*`")
_CODE_BLOCK_PATTERN = re.compile(r"<code>.*?</code>", re.IGNORECASE | re.DOTALL)

class BaseTTS(ABC):
    @abstractmethod
    async def text_to_speech(self, text: str):
        """Convert ``text`` into audio output."""

    def play_audio_file(
        self,
        filename: str,
        *,
        logger,
        ensure_mixer_ready: Optional[Callable[[], bool]] = None,
        cleanup: Optional[Callable[[str], None]] = None,
        clock_hz: int = 10,
    ) -> None:
        """Play ``filename`` with pygame while handling mixer failures.

        Parameters
        ----------
        filename:
            Path to the audio file that should be played.
        logger:
            Logger used for diagnostic output.
        ensure_mixer_ready:
            Optional callable that prepares the pygame mixer. If supplied and it
            returns ``False``, playback is skipped. When omitted the helper will
            initialise the mixer on-demand.
        cleanup:
            Optional callable invoked with ``filename`` during cleanup. Useful
            for removing temporary audio files.
        clock_hz:
            Frequency (in Hz) used when polling ``pygame.mixer.music.get_busy``.
        """

        def _run_cleanup() -> None:
            if cleanup is None:
                return
            try:
                cleanup(filename)
            except FileNotFoundError:
                logger.debug("Temporary audio file already removed: %s", filename)
            except OSError as exc:
                logger.warning("Failed to remove audio file %s: %s", filename, exc)

        mixer = getattr(pygame, "mixer", None)
        if mixer is None:
            logger.error("pygame mixer module unavailable; skipping audio playback.")
            _run_cleanup()
            return

        music = getattr(mixer, "music", None)
        if music is None:
            logger.error("pygame mixer music controls unavailable; skipping audio playback.")
            _run_cleanup()
            return

        try:
            if ensure_mixer_ready is not None:
                try:
                    if not ensure_mixer_ready():
                        logger.warning(
                            "Skipping audio playback because the mixer could not be initialized."
                        )
                        _run_cleanup()
                        return
                except Exception as exc:  # noqa: BLE001 - propagate precise error to logs
                    logger.error("Mixer preparation failed: %s", exc)
                    _run_cleanup()
                    return
            else:
                get_init = getattr(mixer, "get_init", None)
                needs_init = True
                if callable(get_init):
                    needs_init = not get_init()
                init = getattr(mixer, "init", None)
                if needs_init:
                    if not callable(init):
                        raise AttributeError("pygame.mixer.init is unavailable")
                    init()
        except Exception as exc:  # noqa: BLE001 - log detailed mixer failure
            logger.error("Failed to initialize pygame mixer: %s", exc)
            _run_cleanup()
            return

        try:
            load_fn = getattr(music, "load", None)
            play_fn = getattr(music, "play", None)
            get_busy = getattr(music, "get_busy", None)
            if not callable(load_fn) or not callable(play_fn):
                raise AttributeError("pygame.mixer.music lacks load/play functions")

            load_fn(filename)
            play_fn()
            logger.debug("Audio playback started for: %s", filename)
            clock_module = getattr(pygame, "time", None)
            clock_factory = getattr(clock_module, "Clock", None) if clock_module else None
            clock = clock_factory() if callable(clock_factory) else None
            sleep_interval = 1.0 / clock_hz if clock_hz > 0 else 0.1

            while callable(get_busy) and get_busy():
                if clock is not None and hasattr(clock, "tick"):
                    clock.tick(clock_hz)
                else:
                    time.sleep(sleep_interval)
            logger.debug("Audio playback finished for: %s", filename)
        except Exception as exc:  # noqa: BLE001 - propagate playback failure details
            logger.error("Error during audio playback: %s", exc)
        finally:
            get_init = getattr(mixer, "get_init", None)
            mixer_ready = bool(get_init()) if callable(get_init) else True
            stop_fn = getattr(music, "stop", None)
            if mixer_ready and callable(stop_fn):
                stop_fn()
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
