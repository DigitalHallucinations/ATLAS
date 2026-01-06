---
audience: Developers maintaining legacy speech stack
status: legacy
last_verified: 2025-12-21
last_updated_hint: Flagged the page as legacy context and pointed readers to the current audio engine docs.
source_of_truth: modules/Speech_Services/base.py; modules/Speech_Services/elevenlabs_tts.py; modules/Speech_Services/Google_tts.py; modules/Speech_Services/Google_stt.py; modules/Speech_Services/gpt4o_tts.py; modules/Speech_Services/gpt4o_stt.py; modules/Speech_Services/whisper_stt.py; modules/Speech_Services/speech_manager.py
---

# Speech Services legacy snapshot

The former `modules/Speech_Services/complete speech services` file stitched together
examples of the speech interfaces and provider implementations that now live in
individual modules. That aggregate file no longer carries runnable code; the
production-ready providers continue to be maintained under
`modules/Speech_Services/`. This page keeps the useful historical context
without duplicating code. Refer to [audio/interface.md](../audio/interface.md)
for the current audio engine, playback, and mixer guidance used in production.

## Shared abstractions

* `BaseTTS` defines the asynchronous playback contract that all text-to-speech
  providers implement, including helpers for cleaning inline code from prompts
  and safely managing temporary audio files before delegating to the shared
  `SoundDeviceEngine` for playback.【F:modules/Speech_Services/base.py†L1-L87】【F:modules/Speech_Services/base.py†L89-L139】
* `BaseSTT` captures the minimal interface for speech-to-text providers so the
  speech manager can work with any backend that supports recording, stopping,
  and transcription hooks.【F:modules/Speech_Services/base.py†L110-L120】

## Provider highlights

* **ElevenLabs TTS** lazily loads the available voices, verifies that the
  `XI_API_KEY` environment variable is configured, uses the shared audio engine
  for playback, and strips code blocks before sending text to the API.【F:modules/Speech_Services/elevenlabs_tts.py†L1-L111】【F:modules/Speech_Services/elevenlabs_tts.py†L142-L208】
* **Google TTS** relies on the Cloud Text-to-Speech client, sanitises prompts,
  writes the audio response to a temporary MP3, and reuses the shared playback
  helper to stream the file.【F:modules/Speech_Services/Google_tts.py†L1-L64】【F:modules/Speech_Services/Google_tts.py†L69-L108】
* **Google STT** records audio from `sounddevice`, stores samples under
  `data/audio/stt_output`, and submits them to Cloud Speech-to-Text before
  cleaning up the captured file.【F:modules/Speech_Services/Google_stt.py†L1-L77】【F:modules/Speech_Services/Google_stt.py†L80-L123】
* **OpenAI GPT-4o providers** cover both synthesis and transcription: the TTS
  wrapper streams audio from the async OpenAI SDK while the STT helper manages
  microphone capture, timestamped WAV storage, and API interactions that require
  an `OPENAI_API_KEY`.【F:modules/Speech_Services/gpt4o_tts.py†L1-L87】【F:modules/Speech_Services/gpt4o_stt.py†L1-L92】
* **Whisper STT** supports local and online inference with optional noise
  reduction and multiple export formats, recording microphone input before
  running the selected pipeline.【F:modules/Speech_Services/whisper_stt.py†L1-L88】【F:modules/Speech_Services/whisper_stt.py†L91-L152】

## Orchestration

The `speech_manager` module coordinates these providers. It exposes option
normalisation utilities for the UI, wires the configured providers into the
runtime, and ensures asynchronous initialization does not block the main
application thread.【F:modules/Speech_Services/speech_manager.py†L1-L107】

Use the individual modules above when making functional changes; this document
is only a lightweight reference to explain how the removed aggregate file was
originally organised.
