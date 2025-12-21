---
audience: Backend/audio developers
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/audio/interface.py; modules/audio/engine.py
---

# Audio engine for playback and mixing

This document describes the backend-neutral audio surface now implemented in
`modules.audio`. It replaces the `pygame.mixer` helper with a
`sounddevice`-powered engine that supports concurrent stems, multichannel
output, and device selection. Providers (e.g., ElevenLabs TTS) can depend on
the narrow facade without caring whether the runtime uses the default
`SoundDeviceEngine` or a test double.

## Core types
- **`StreamConfig`** — captures sample rate, channel count, and dtype defaults to
  keep format negotiation explicit.
- **`AudioDevice`** — describes discoverable output devices; supports preselecting
  default devices or enumerating host APIs.
- **`PlaybackRequest`** — single payload that supports file-based and in-memory
  PCM, optional looping, bus selection, and per-request gain.
- **`PlaybackHandle`** — returned controller for stopping, polling, and waiting
  on playback completion. Optional event hooks allow logging or cleanup.
- **`AudioMixer`** — handles bus creation, per-bus gain/pan, and attaches
  `PlaybackRequest` instances. Exposes `render(frames)` for pull-based backends.
- **`AudioEngine`** — high-level facade that exposes `play`/`play_inline`, device
  enumeration, and access to the shared mixer.
- **`MixBus`** — simple default bus implementation usable for queue inspection or
  trivial mixes.

See `modules/audio/interface.py` for the contracts and `modules/audio/engine.py`
for the `SoundDeviceEngine` implementation.

## Minimal usage examples

### Simple playback (file-backed)
```python
from modules.audio import PlaybackRequest, SoundDeviceEngine

engine = SoundDeviceEngine()
request = PlaybackRequest(path="/tmp/tts.mp3")
handle = engine.play(request)
handle.wait()  # blocks until finished
```

### Inline PCM playback (e.g., streaming TTS)
```python
pcm = numpy.zeros((1024, 2), dtype="float32")  # placeholder frames
handle = engine.play_inline(pcm)
```

### Bus-aware mixing
```python
mixer = engine.mixer()
mixer.register_bus("music", StreamConfig(sample_rate=48000, channels=2))
mixer.set_gain("music", 0.5)

mixer.attach(
    PlaybackRequest(path="/tmp/music.ogg", bus="music", loop=True)
)
mixer.attach(
    PlaybackRequest(path="/tmp/tts.mp3", bus="main", gain=0.8)
)
```

### Looping stems and per-request gain
```python
looped = PlaybackRequest(path="/tmp/ambience.wav", loop=True, gain=0.35, bus="ambience")
mixer.register_bus("ambience", StreamConfig(sample_rate=48000, channels=2))
mixer.attach(looped)
```

### Device selection and validation
```python
for device in engine.list_output_devices():
    print(device.name, device.max_output_channels)

surround_device = next(
    d for d in engine.list_output_devices() if d.max_output_channels >= 6
)
request = PlaybackRequest(path="/tmp/ambisonic.wav", device=surround_device)
engine.play(request)
```

### Adapter for existing TTS providers
`BaseTTS.play_audio_file` can be replaced by a thin adapter that builds a
`PlaybackRequest` and delegates to `AudioEngine.play`. Tests can stub
`PlaybackHandle` interfaces without relying on `pygame` globals, or inject a
fake engine that satisfies `AudioEngine`.

## Implementation notes
- **Backend selection**: The default `SoundDeviceEngine` wraps a
  `sounddevice.OutputStream`, while alternate engines (e.g., mock or offline
  renderers) can implement the same interfaces for testing and CI.
- **Resource lifecycle**: `AudioEngine.close()` flushes active streams and
  detaches from devices. Handles emit `PlaybackEvent` signals for lifecycle
  hooks (e.g., cleaning up temporary files or informing UI overlays).
- **Threading model**: The mixer renders within the sounddevice callback,
  enabling low-latency mixing for multiple stems. It supports looped playback
  and per-bus gain/pan for stereo buses.
- **Channel layouts**: `StreamConfig.channels` covers both stereo and surround
  scenarios; bus registration allows per-output shaping without burdening call
  sites.
- **Bus compatibility**: The engine enforces that bus sample rate and channel
  counts align with the active output stream to avoid surprises at mix time.
