"""Sounddevice-backed audio engine for playback and mixing.

This module provides a concrete implementation of the audio interfaces defined
in :mod:`modules.audio.interface` using ``sounddevice`` for output and
``soundfile`` for decoding. It supports file-backed playback, inline PCM, bus
routing with gain/pan controls, and basic looped stems.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from .interface import (
    AudioBuffer,
    AudioDevice,
    AudioEngine,
    AudioMixer,
    MixBus,
    PlaybackEvent,
    PlaybackHandle,
    PlaybackRequest,
    StreamConfig,
)


@dataclass(slots=True)
class _SourceState:
    request: PlaybackRequest
    buffer: AudioBuffer
    position: int = 0
    finished: bool = False


class _PlaybackHandle(PlaybackHandle):
    def __init__(self, mixer: "_SimpleAudioMixer", source_id: str, request: PlaybackRequest):
        self._mixer = mixer
        self._source_id = source_id
        self._request = request
        self._callbacks: dict[PlaybackEvent, list[Callable[[PlaybackRequest], None]]] = {}
        self._done = threading.Event()

    def wait(self, timeout: float | None = None) -> None:
        self._done.wait(timeout=timeout)

    def stop(self) -> None:
        self._mixer.remove(self._source_id)
        self._done.set()
        self._emit(PlaybackEvent.STOPPED)

    def is_active(self) -> bool:
        return not self._done.is_set()

    def on(self, event: PlaybackEvent, callback: Callable[[PlaybackRequest], None]) -> None:
        self._callbacks.setdefault(event, []).append(callback)

    def _emit(self, event: PlaybackEvent) -> None:
        for callback in self._callbacks.get(event, []):
            callback(self._request)

    def mark_finished(self) -> None:
        if not self._done.is_set():
            self._done.set()
            self._emit(PlaybackEvent.FINISHED)

    def mark_started(self) -> None:
        self._emit(PlaybackEvent.STARTED)

    def mark_failed(self) -> None:
        if not self._done.is_set():
            self._done.set()
        self._emit(PlaybackEvent.FAILED)


class _SimpleAudioMixer(AudioMixer):
    def __init__(self, default_config: StreamConfig):
        self._default_config = default_config
        self._buses: dict[str, MixBus] = {"main": MixBus("main", default_config)}
        self._bus_sources: Dict[str, Dict[str, _SourceState]] = {"main": {}}
        self._handles: Dict[str, _PlaybackHandle] = {}
        self._lock = threading.RLock()
        self._counter = 0

    def register_bus(self, name: str, config: StreamConfig | None = None) -> None:
        with self._lock:
            bus_config = config or self._default_config
            if bus_config.channels != self._default_config.channels:
                msg = "Bus channel count must match output channel count"
                raise ValueError(msg)
            if bus_config.sample_rate != self._default_config.sample_rate:
                msg = "Bus sample rate must match output sample rate"
                raise ValueError(msg)
            self._buses[name] = MixBus(name, bus_config)
            self._bus_sources.setdefault(name, {})

    def attach(self, request: PlaybackRequest) -> PlaybackHandle:
        with self._lock:
            if request.bus not in self._buses:
                self.register_bus(request.bus)
            bus_config = self._buses[request.bus].config
        buffer, _ = self._load_buffer(request, target_config=bus_config)
        with self._lock:
            source_id = f"src-{self._counter}"
            self._counter += 1
            state = _SourceState(request=request, buffer=buffer)
            self._bus_sources[request.bus][source_id] = state
            handle = _PlaybackHandle(self, source_id, request)
            self._handles[source_id] = handle
        handle.mark_started()
        return handle

    def _load_buffer(self, request: PlaybackRequest, *, target_config: StreamConfig) -> tuple[AudioBuffer, StreamConfig]:
        if request.buffer is not None:
            buffer = np.array(request.buffer, copy=True)
            config = request.config
        elif request.path is not None:
            data, sample_rate = sf.read(request.path, dtype=request.config.dtype)
            buffer = np.asarray(data, dtype=request.config.dtype)
            config = StreamConfig(sample_rate=int(sample_rate), channels=buffer.shape[1] if buffer.ndim > 1 else 1)
        else:
            msg = "PlaybackRequest requires buffer or path"
            raise ValueError(msg)
        buffer = self._ensure_2d(buffer)
        buffer = self._ensure_channels(buffer, target_config.channels)
        if config.sample_rate != target_config.sample_rate:
            buffer = self._resample(buffer, config.sample_rate, target_config.sample_rate)
        return buffer.astype(target_config.dtype), target_config

    def _ensure_2d(self, buffer: AudioBuffer) -> AudioBuffer:
        if buffer.ndim == 1:
            return buffer.reshape(-1, 1)
        return buffer

    def _ensure_channels(self, buffer: AudioBuffer, channels: int) -> AudioBuffer:
        if buffer.shape[1] == channels:
            return buffer
        if buffer.shape[1] == 1 and channels == 2:
            return np.repeat(buffer, 2, axis=1)
        if buffer.shape[1] > channels:
            return buffer[:, :channels]
        repeats = channels // buffer.shape[1]
        return np.repeat(buffer, repeats, axis=1)

    def _resample(self, buffer: AudioBuffer, source_rate: int, target_rate: int) -> AudioBuffer:
        if source_rate == target_rate:
            return buffer
        ratio = target_rate / source_rate
        new_length = int(buffer.shape[0] * ratio)
        indices = np.linspace(0, buffer.shape[0] - 1, new_length)
        resampled = np.vstack([np.interp(indices, np.arange(buffer.shape[0]), buffer[:, ch]) for ch in range(buffer.shape[1])])
        return resampled.T

    def remove(self, source_id: str) -> None:
        with self._lock:
            for bus_name, bus_sources in self._bus_sources.items():
                if source_id in bus_sources:
                    del bus_sources[source_id]
                    break
            handle = self._handles.pop(source_id, None)
        if handle:
            handle.mark_finished()

    def set_gain(self, bus: str, gain: float) -> None:
        with self._lock:
            if bus not in self._buses:
                self.register_bus(bus)
            self._buses[bus].gain = gain

    def set_pan(self, bus: str, pan: float) -> None:
        with self._lock:
            if bus not in self._buses:
                self.register_bus(bus)
            self._buses[bus].pan = float(np.clip(pan, -1.0, 1.0))

    def render(self, frames: int) -> AudioBuffer:
        with self._lock:
            output = np.zeros((frames, self._default_config.channels), dtype=self._default_config.dtype)
            completed: list[str] = []
            for bus_name, bus in self._buses.items():
                bus_sources = self._bus_sources.get(bus_name, {})
                if not bus_sources:
                    continue
                bus_mix = np.zeros_like(output)
                for source_id, state in bus_sources.items():
                    bus_mix, done = self._mix_source(bus_mix, state, frames)
                    if done:
                        completed.append(source_id)
                if bus.config.channels == 2:
                    bus_mix = self._apply_pan(bus_mix, bus.pan)
                output += bus_mix * bus.gain
            for source_id in completed:
                self.remove(source_id)
            return output

    def _mix_source(self, bus_mix: AudioBuffer, state: _SourceState, frames: int) -> tuple[AudioBuffer, bool]:
        buffer = state.buffer
        start = state.position
        end = min(start + frames, buffer.shape[0])
        slice_ = buffer[start:end]
        slice_frames = slice_.shape[0]
        bus_mix[:slice_frames] += slice_ * state.request.gain
        state.position += slice_frames
        finished = state.position >= buffer.shape[0]
        if finished and state.request.loop:
            remaining = frames - slice_frames
            if remaining > 0:
                loop_frames = min(remaining, buffer.shape[0])
                loop_slice = buffer[:loop_frames]
                bus_mix[slice_frames : slice_frames + loop_frames] += loop_slice * state.request.gain
                state.position = loop_frames
                finished = False
            else:
                state.position = 0
                finished = False
        return bus_mix, finished

    def _apply_pan(self, buffer: AudioBuffer, pan: float) -> AudioBuffer:
        if buffer.shape[1] < 2:
            return buffer
        left = buffer[:, 0] * (1 - pan) * 0.5
        right = buffer[:, 1] * (1 + pan) * 0.5
        buffer[:, 0] = left
        buffer[:, 1] = right
        return buffer

    def iter_sources(self, bus: str) -> Iterable[PlaybackRequest]:
        with self._lock:
            return tuple(state.request for state in self._bus_sources.get(bus, {}).values())

    def iter_buses(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._buses.keys())


class SoundDeviceEngine(AudioEngine):
    def __init__(self, default_config: StreamConfig | None = None):
        self._default_config = default_config or StreamConfig()
        self._mixer = _SimpleAudioMixer(self._default_config)
        self._stream: Optional[sd.OutputStream] = None
        self._stream_lock = threading.RLock()

    def list_output_devices(self) -> list[AudioDevice]:
        devices = []
        for idx, device in enumerate(sd.query_devices()):
            if device["max_output_channels"] <= 0:
                continue
            devices.append(
                AudioDevice(
                    id=idx,
                    name=device["name"],
                    max_output_channels=device["max_output_channels"],
                    default_samplerate=device.get("default_samplerate", self._default_config.sample_rate),
                    hostapi=str(device.get("hostapi")) if "hostapi" in device else None,
                    supports_mmap=bool(device.get("hostapi") == "ALSA" and device.get("hostapi")),
                )
            )
        return devices

    def default_output_device(self) -> AudioDevice | None:
        default_id = sd.default.device[1] if sd.default.device else None
        if default_id is None:
            return None
        devices = sd.query_devices()
        device = devices[default_id]
        return AudioDevice(
            id=default_id,
            name=device["name"],
            max_output_channels=device["max_output_channels"],
            default_samplerate=device.get("default_samplerate", self._default_config.sample_rate),
            hostapi=str(device.get("hostapi")) if "hostapi" in device else None,
            supports_mmap=bool(device.get("hostapi") == "ALSA" and device.get("hostapi")),
        )

    def play(self, request: PlaybackRequest) -> PlaybackHandle:
        self._ensure_stream(device=request.device, config=request.config)
        handle = self._mixer.attach(request)
        return handle

    def play_inline(self, buffer: AudioBuffer, *, config: StreamConfig | None = None) -> PlaybackHandle:
        request = PlaybackRequest(buffer=buffer, config=config or self._default_config)
        return self.play(request)

    def mixer(self) -> AudioMixer:
        return self._mixer

    def close(self) -> None:
        with self._stream_lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def _ensure_stream(self, device: int | AudioDevice | None, config: StreamConfig) -> None:
        with self._stream_lock:
            if self._stream is not None and self._stream.active:
                return
            device_id = None
            if isinstance(device, AudioDevice):
                device_id = device.id
            elif isinstance(device, int):
                device_id = device
            self._stream = sd.OutputStream(
                samplerate=self._default_config.sample_rate,
                channels=self._default_config.channels,
                blocksize=self._default_config.blocksize,
                dtype=self._default_config.dtype,
                device=device_id,
                callback=self._audio_callback,
            )
            self._stream.start()

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:  # type: ignore[override]
        del time_info  # unused
        if status:
            # Dropouts or warnings are exposed via status; mix anyway.
            pass
        mix = self._mixer.render(frames)
        if mix.shape[0] < frames:
            padded = np.zeros((frames, mix.shape[1]), dtype=mix.dtype)
            padded[: mix.shape[0]] = mix
            mix = padded
        outdata[:] = mix


__all__ = ["SoundDeviceEngine"]
