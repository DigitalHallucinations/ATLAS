"""Interfaces for unified audio playback, routing, and mixing backends.

This module defines contracts for a sounddevice/soundfile-oriented backend that
can replace the current ``pygame`` helpers while supporting more advanced use
cases such as multi-channel routing, concurrent stems, and device selection.
The interfaces focus on three layers:

1. **Load** (decode or accept PCM buffers)
2. **Mix** (combine multiple sources into buses with volume/pan controls)
3. **Output** (push frames to the selected hardware device)

Implementations can wrap ``sounddevice`` streams or other engines while keeping
existing TTS providers isolated from backend-specific details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

# Type alias for interleaved PCM32F frames shaped (frames, channels).
AudioBuffer = np.ndarray


class PlaybackEvent(Enum):
    """High-level lifecycle notifications emitted by playback handles."""

    STARTED = auto()
    FINISHED = auto()
    STOPPED = auto()
    FAILED = auto()


@dataclass(slots=True)
class StreamConfig:
    """Desired output format for a playback or mix bus."""

    sample_rate: int = 44100
    channels: int = 2
    blocksize: int = 0
    dtype: str = "float32"


@dataclass(slots=True)
class AudioDevice:
    """Describes a concrete output device candidates can choose from."""

    id: int
    name: str
    max_output_channels: int
    default_samplerate: float
    hostapi: str | None = None
    supports_mmap: bool = False


@dataclass(slots=True)
class PlaybackRequest:
    """Inputs for starting playback or scheduling a stream on a mix bus."""

    buffer: AudioBuffer | None = None
    path: str | None = None
    config: StreamConfig = field(default_factory=StreamConfig)
    device: int | AudioDevice | None = None
    gain: float = 1.0
    loop: bool = False
    bus: str = "main"

    def __post_init__(self) -> None:
        if self.buffer is None and self.path is None:
            msg = "Either 'buffer' or 'path' must be provided"
            raise ValueError(msg)


class PlaybackHandle(Protocol):
    """Controller returned by playback operations."""

    def wait(self, timeout: float | None = None) -> None:
        """Block until playback finishes or timeout expires."""

    def stop(self) -> None:
        """Stop playback and release any underlying stream resources."""

    def is_active(self) -> bool:
        """Return True while frames are still being sent to the device."""

    def on(self, event: PlaybackEvent, callback: Callable[[PlaybackRequest], None]) -> None:
        """Register callbacks for lifecycle events (optional to implement)."""


class AudioMixer(Protocol):
    """Mix engine capable of handling multiple concurrent sources."""

    def register_bus(self, name: str, config: StreamConfig | None = None) -> None:
        """Create or update a logical bus (e.g., main, music, sfx, ui)."""

    def attach(self, request: PlaybackRequest) -> PlaybackHandle:
        """Attach a request to the mixer; returns a handle for control."""

    def set_gain(self, bus: str, gain: float) -> None:
        """Adjust gain for an entire bus without altering individual sources."""

    def set_pan(self, bus: str, pan: float) -> None:
        """Adjust pan for stereo buses; -1 is left, 0 is center, +1 is right."""

    def render(self, frames: int) -> AudioBuffer:
        """Return mixed frames for a pull-based output stream."""


class AudioEngine(Protocol):
    """Facade combining decode, mix, and device selection responsibilities."""

    def list_output_devices(self) -> Sequence[AudioDevice]:
        """Enumerate available output devices."""

    def default_output_device(self) -> AudioDevice | None:
        """Return the default output device if the backend exposes one."""

    def play(self, request: PlaybackRequest) -> PlaybackHandle:
        """Play a file or buffer using the implicit default mix bus."""

    def play_inline(self, buffer: AudioBuffer, *, config: StreamConfig | None = None) -> PlaybackHandle:
        """Start playback for in-memory PCM data without touching disk."""

    def mixer(self) -> AudioMixer:
        """Access the mixer to attach stems, register buses, or stream pull-mode audio."""

    def close(self) -> None:
        """Shut down devices and stop all active playback."""


class MixBus:
    """Default mix bus behavior usable by both push and pull architectures."""

    def __init__(self, name: str, config: StreamConfig):
        self.name = name
        self.config = config
        self._sources: list[PlaybackRequest] = []
        self.gain = 1.0
        self.pan = 0.0

    def attach(self, request: PlaybackRequest) -> None:
        """Register a playback request with the bus."""

        self._sources.append(request)

    def iter_sources(self) -> Iterable[PlaybackRequest]:
        """Iterate over queued sources for inspection or rendering."""

        return tuple(self._sources)

    def clear(self) -> None:
        """Remove all queued sources from the bus."""

        self._sources.clear()
