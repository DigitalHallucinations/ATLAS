"""Speech service facade used by :class:`ATLAS.ATLAS`."""

from __future__ import annotations

from collections.abc import AsyncIterator as AbcAsyncIterator
from concurrent.futures import Future
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from modules.Speech_Services.speech_manager import SpeechManager


class SpeechService:
    """Facade responsible for orchestrating speech related interactions."""

    def __init__(
        self,
        *,
        speech_manager: SpeechManager | None,
        logger,
        status_summary_getter: Callable[[], Dict[str, Any]] | None,
        default_status_tooltip: str,
    ) -> None:
        self._speech_manager = speech_manager
        self._logger = logger
        self._status_summary_getter = status_summary_getter
        self._default_status_tooltip = default_status_tooltip
        self._transcript_export_preferences: Dict[str, Any] = {
            "formats": [],
            "directory": None,
        }

    # ------------------------------------------------------------------
    # helpers
    def _log_debug(self, message: str, *args: Any) -> None:
        if self._logger is not None:
            self._logger.debug(message, *args)

    def _log_error(self, message: str, *args: Any, exc_info: bool | None = None) -> None:
        if self._logger is not None:
            self._logger.error(message, *args, exc_info=exc_info)

    def _resolve_manager(self) -> SpeechManager:
        if self._speech_manager is None:
            raise RuntimeError("Speech services unavailable.")
        return self._speech_manager

    def _status_summary(self) -> Dict[str, Any]:
        getter = self._status_summary_getter
        if getter is None:
            return {}
        try:
            summary = getter()
        except Exception as exc:  # pragma: no cover - defensive guard
            self._log_error("Status summary callback failed: %s", exc, exc_info=True)
            return {}
        return summary or {}

    # ------------------------------------------------------------------
    # default + configuration queries
    def get_speech_defaults(self) -> Dict[str, Any]:
        return self._resolve_manager().describe_general_settings()

    def get_speech_provider_status(self, provider_name: str) -> Dict[str, Any]:
        return self._resolve_manager().get_provider_credential_status(provider_name)

    def get_speech_voice_options(
        self, provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self._resolve_manager().list_tts_voice_options(provider)

    def get_active_speech_voice(self) -> Dict[str, Optional[str]]:
        provider, voice = self._resolve_manager().get_active_tts_summary()
        return {"provider": provider, "voice": voice}

    def update_speech_defaults(
        self,
        *,
        tts_enabled: bool,
        tts_provider: Optional[str],
        stt_enabled: bool,
        stt_provider: Optional[str],
    ) -> Dict[str, Any]:
        manager = self._resolve_manager()
        manager.configure_defaults(
            tts_enabled=bool(tts_enabled),
            tts_provider=tts_provider,
            stt_enabled=bool(stt_enabled),
            stt_provider=stt_provider,
        )
        return manager.describe_general_settings()

    # ------------------------------------------------------------------
    # provider specific configuration
    def update_elevenlabs_settings(
        self,
        *,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        manager = self._resolve_manager()
        result = {"updated_api_key": False, "updated_voice": False}

        if api_key:
            manager.set_elevenlabs_api_key(api_key)
            result["updated_api_key"] = True

        provider_key = manager.resolve_tts_provider(manager.get_default_tts_provider())

        if voice_id and provider_key:
            voices: Iterable[Dict[str, Any]] = manager.get_tts_voices(provider_key) or []
            selected_voice: Optional[Dict[str, Any]] = None
            for voice in voices:
                if not isinstance(voice, dict):
                    continue
                if voice.get("voice_id") == voice_id or voice.get("name") == voice_id:
                    selected_voice = voice
                    break

            if selected_voice is not None:
                manager.set_tts_voice(selected_voice, provider_key)
                result["updated_voice"] = True

        result["provider"] = provider_key
        return result

    def update_google_speech_settings(
        self,
        credentials_path: str,
        *,
        tts_voice: Optional[str] = None,
        stt_language: Optional[str] = None,
        auto_punctuation: Optional[bool] = None,
    ) -> None:
        manager = self._resolve_manager()
        manager.set_google_credentials(
            credentials_path,
            voice_name=tts_voice,
            stt_language=stt_language,
            auto_punctuation=auto_punctuation,
        )

    def get_google_speech_credentials_path(self) -> Optional[str]:
        return self._resolve_manager().get_google_credentials_path()

    def get_google_speech_settings(self) -> Dict[str, Any]:
        return self._resolve_manager().get_google_speech_settings()

    def get_openai_speech_options(self) -> Dict[str, List[Tuple[str, Optional[str]]]]:
        return self._resolve_manager().get_openai_option_sets()

    def get_openai_speech_configuration(self) -> Dict[str, Optional[str]]:
        return self._resolve_manager().get_openai_display_config()

    def update_openai_speech_settings(
        self, display_payload: Dict[str, Any]
    ) -> Dict[str, Optional[str]]:
        manager = self._resolve_manager()
        prepared = manager.normalize_openai_display_settings(display_payload)
        manager.set_openai_speech_config(
            api_key=prepared.get("api_key"),
            stt_provider=prepared.get("stt_provider"),
            language=prepared.get("language"),
            task=prepared.get("task"),
            initial_prompt=prepared.get("initial_prompt"),
            tts_provider=prepared.get("tts_provider"),
        )
        return prepared

    def get_transcription_history(self, *, formatted: bool = False) -> List[Dict[str, Any]]:
        return self._resolve_manager().get_transcription_history(formatted=formatted)

    # ------------------------------------------------------------------
    # transcript export preferences
    def get_transcript_export_preferences(self) -> Dict[str, Any]:
        preferences = self._transcript_export_preferences
        return {
            "formats": list(preferences.get("formats") or []),
            "directory": preferences.get("directory"),
        }

    def update_transcript_export_preferences(
        self,
        *,
        formats: Iterable[str] | None = None,
        directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_formats: List[str] = []
        if formats is not None:
            seen = set()
            for fmt in formats:
                if not isinstance(fmt, str):
                    continue
                normalized = fmt.strip().lower()
                if not normalized or normalized in seen:
                    continue
                if normalized not in {"txt", "json", "srt"}:
                    raise ValueError(f"Unsupported transcript export format: {fmt}")
                seen.add(normalized)
                normalized_formats.append(normalized)
        else:
            normalized_formats = list(self._transcript_export_preferences.get("formats") or [])

        normalized_directory = None
        if directory is None:
            normalized_directory = self._transcript_export_preferences.get("directory")
        elif isinstance(directory, str):
            stripped = directory.strip()
            normalized_directory = stripped or None
        else:
            raise ValueError("directory must be a string path or None")

        self._transcript_export_preferences = {
            "formats": normalized_formats,
            "directory": normalized_directory,
        }

        return self.get_transcript_export_preferences()

    def _build_export_payload(
        self,
        *,
        export_formats: Optional[Iterable[str]],
        export_directory: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        formats = None
        if export_formats is not None:
            seen = set()
            normalized: List[str] = []
            for fmt in export_formats:
                if not isinstance(fmt, str):
                    continue
                stripped = fmt.strip().lower()
                if not stripped or stripped in seen:
                    continue
                seen.add(stripped)
                normalized.append(stripped)
            formats = normalized
        else:
            formats = list(self._transcript_export_preferences.get("formats") or [])

        if not formats:
            return None

        directory = export_directory
        if directory is None:
            directory = self._transcript_export_preferences.get("directory")

        payload: Dict[str, Any] = {"formats": formats}
        if directory:
            payload["destination"] = directory
        return payload

    # ------------------------------------------------------------------
    # stt orchestration
    def start_stt_listening(self) -> Dict[str, Any]:
        try:
            manager = self._resolve_manager()
        except RuntimeError as exc:
            message = str(exc)
            return {
                "ok": False,
                "status_text": message,
                "provider": None,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self._status_summary(),
            }

        provider_key = manager.get_active_stt_provider()
        if not provider_key:
            message = "No STT service configured."
            self._log_error(message)
            return {
                "ok": False,
                "status_text": message,
                "provider": None,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self._status_summary(),
            }

        try:
            started = manager.listen(provider_key)
        except Exception as exc:  # pragma: no cover - manager already logs
            self._log_error(
                "Failed to start STT provider %s: %s", provider_key, exc, exc_info=True
            )
            started = False

        if not started:
            message = "Failed to start listening."
            return {
                "ok": False,
                "status_text": message,
                "provider": provider_key,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self._status_summary(),
            }

        self._log_debug("Listening started using provider '%s'.", provider_key)
        return {
            "ok": True,
            "status_text": "Listening…",
            "provider": provider_key,
            "listening": True,
            "spinner": False,
            "error": None,
            "status_tooltip": f"Listening via {provider_key}",
            "status_summary": self._status_summary(),
        }

    def stop_stt_and_transcribe(
        self,
        *,
        export_formats: Optional[Iterable[str]] = None,
        export_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            manager = self._resolve_manager()
        except RuntimeError as exc:
            message = str(exc)
            return {
                "ok": False,
                "status_text": message,
                "provider": None,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self._status_summary(),
                "transcription_future": None,
            }

        provider_key = manager.get_active_stt_provider()
        if not provider_key:
            message = "No STT service configured."
            self._log_error(message)
            return {
                "ok": False,
                "status_text": message,
                "provider": None,
                "listening": False,
                "spinner": False,
                "error": message,
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self._status_summary(),
                "transcription_future": None,
            }

        result_future: Future[Dict[str, Any]] = Future()

        def _finalize_payload(
            *, transcript: Optional[str] = None, error: Optional[Exception] = None
        ) -> Dict[str, Any]:
            summary = self._status_summary()
            normalized_transcript = (transcript or "").strip()
            if error is not None:
                error_message = f"Transcription failed: {error}"
                payload = {
                    "ok": False,
                    "status_text": error_message,
                    "provider": provider_key,
                    "listening": False,
                    "spinner": False,
                    "transcript": "",
                    "error": error_message,
                    "status_tooltip": self._default_status_tooltip,
                    "status_summary": summary,
                }
            elif normalized_transcript:
                payload = {
                    "ok": True,
                    "status_text": "Transcription complete.",
                    "provider": provider_key,
                    "listening": False,
                    "spinner": False,
                    "transcript": normalized_transcript,
                    "error": None,
                    "status_tooltip": self._default_status_tooltip,
                    "status_summary": summary,
                }
            else:
                payload = {
                    "ok": True,
                    "status_text": "No transcription available.",
                    "provider": provider_key,
                    "listening": False,
                    "spinner": False,
                    "transcript": "",
                    "error": None,
                    "status_tooltip": self._default_status_tooltip,
                    "status_summary": summary,
                }
            return payload

        export_payload = self._build_export_payload(
            export_formats=export_formats,
            export_directory=export_directory,
        )

        def _on_success(transcript: str) -> None:
            payload = _finalize_payload(transcript=transcript)
            if not result_future.done():
                result_future.set_result(payload)

        def _on_error(exc: Exception) -> None:
            self._log_error("Unexpected transcription error: %s", exc, exc_info=True)
            payload = _finalize_payload(error=exc)
            if not result_future.done():
                result_future.set_result(payload)

        try:
            manager.stop_and_transcribe_in_background(
                provider_key,
                export_options=export_payload,
                on_success=_on_success,
                on_error=_on_error,
                thread_name="SpeechTranscriptionWorker",
            )
        except Exception as exc:
            self._log_error(
                "Failed to schedule transcription with provider %s: %s",
                provider_key,
                exc,
                exc_info=True,
            )
            payload = _finalize_payload(error=exc)
            if not result_future.done():
                result_future.set_result(payload)
            return {
                "ok": False,
                "status_text": payload["status_text"],
                "provider": provider_key,
                "listening": False,
                "spinner": False,
                "error": payload["error"],
                "status_tooltip": payload["status_tooltip"],
                "status_summary": payload["status_summary"],
                "transcription_future": result_future,
            }

        return {
            "ok": True,
            "status_text": "Transcribing…",
            "provider": provider_key,
            "listening": False,
            "spinner": True,
            "error": None,
            "status_tooltip": f"Transcribing via {provider_key}",
            "status_summary": self._status_summary(),
            "transcription_future": result_future,
        }

    # ------------------------------------------------------------------
    # text to speech orchestration
    async def maybe_text_to_speech(self, response_text: Any) -> None:
        manager = self._resolve_manager()

        if isinstance(response_text, AbcAsyncIterator):
            self._log_debug(
                "Skipping text-to-speech for streaming async iterator response."
            )
            return

        payload_text = ""
        audio_available = False

        if isinstance(response_text, dict):
            payload_text = str(response_text.get("text") or "")
            audio_available = bool(response_text.get("audio"))
        else:
            payload_text = str(response_text or "")

        if not payload_text:
            return

        if audio_available:
            return

        if not manager.get_tts_status():
            return

        self._log_debug("TTS enabled; synthesizing response text.")

        try:
            await manager.text_to_speech(payload_text)
        except Exception as exc:
            self._log_error("Text-to-speech failed: %s", exc, exc_info=True)


__all__ = ["SpeechService"]
