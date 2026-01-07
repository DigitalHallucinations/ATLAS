# Speech Service

> **Status**: 📋 Planning  
> **Priority**: Low  
> **Complexity**: Low  
> **Effort**: 2-3 days  
> **Created**: 2026-01-07

---

## Overview

Consolidate `modules/Speech_Services/` into `core/services/speech/`:

- Text-to-Speech (TTS)
- Speech-to-Text (STT)
- Voice and language management

---

## Phases

### Phase 1: Service Creation

- [ ] **1.1** Create `core/services/speech/` package
- [ ] **1.2** Implement SpeechService:
  - `text_to_speech(text, voice, options)` - TTS
  - `speech_to_text(audio, options)` - STT
  - `list_voices()` - Available TTS voices
  - `list_languages()` - Supported languages
  - `get_default_voice()` - Current default
  - `set_default_voice(actor, voice_id)` - Change default
  - `get_provider_status()` - Speech provider health
- [ ] **1.3** Add MessageBus events:
  - `speech.tts_started`
  - `speech.tts_completed`
  - `speech.stt_started`
  - `speech.stt_completed`
- [ ] **1.4** Update audio subsystem to use service
- [ ] **1.5** Write unit tests

---

## Service Methods

```python
class SpeechService:
    # TTS
    async def text_to_speech(
        self,
        actor: Actor,
        text: str,
        voice: str | None = None,
        options: TTSOptions | None = None
    ) -> OperationResult[AudioFile]: ...
    
    # STT
    async def speech_to_text(
        self,
        actor: Actor,
        audio: bytes | Path,
        options: STTOptions | None = None
    ) -> OperationResult[Transcript]: ...
    
    # Voice management
    def list_voices(self) -> OperationResult[list[Voice]]: ...
    def get_default_voice(self) -> OperationResult[Voice]: ...
    def set_default_voice(self, actor: Actor, voice_id: str) -> OperationResult[Voice]: ...
    
    # Language support
    def list_languages(self) -> OperationResult[list[Language]]: ...
    
    # Status
    def get_provider_status(self) -> OperationResult[ProviderStatus]: ...
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `speech.tts_started` | `TTSEvent` | SpeechService |
| `speech.tts_completed` | `TTSResultEvent` | SpeechService |
| `speech.stt_started` | `STTEvent` | SpeechService |
| `speech.stt_completed` | `STTResultEvent` | SpeechService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/speech/__init__.py` | Package exports |
| `core/services/speech/types.py` | Dataclasses, events |
| `core/services/speech/service.py` | SpeechService |
| `tests/services/speech/` | Service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/Speech_Services/` - Speech providers
- `core/messaging/` - MessageBus for events

---

## Success Criteria

1. Speech operations centralized
2. Provider abstraction working
3. Voice/language management functional
4. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| TTS audio format? | WAV / MP3 / Configurable | TBD |
| STT streaming support? | Yes / No | TBD |
| Offline TTS fallback? | Yes (piper) / No | TBD |
| Audio caching strategy? | None / LRU / Configurable | TBD |
