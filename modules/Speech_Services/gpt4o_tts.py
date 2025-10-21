# modules/Speech_Services/gpt4o_tts.py

"""Async wrapper for GPT-4o Mini text-to-speech synthesis."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional

try:  # pragma: no cover - import guard for environments without the client
    from openai import AsyncOpenAI
except Exception:  # noqa: BLE001 - broad import guard to keep module importable in tests
    AsyncOpenAI = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class GPT4oTTS:
    """Text-to-Speech provider using OpenAI's GPT-4o Mini TTS."""

    def __init__(self, voice: str = "default", *, audio_format: str = "mp3"):
        """Initialize the GPT-4o TTS provider with an async OpenAI client."""

        self.voice = voice
        self.audio_format = audio_format
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY is not set for GPT4o TTS.")
            raise RuntimeError("OPENAI_API_KEY is required for GPT4o TTS")

        if AsyncOpenAI is None:
            raise RuntimeError("AsyncOpenAI client is unavailable in the installed OpenAI SDK")

        self._client = AsyncOpenAI(api_key=self.api_key)
        logger.debug("Initialized GPT4o TTS provider.")

    async def text_to_speech(
        self,
        text: str,
        *,
        output_path: Optional[os.PathLike[str]] = None,
        model: str = "gpt-4o-mini-tts",
    ) -> str:
        """Convert ``text`` to speech asynchronously and persist the audio to disk."""

        if not text:
            raise ValueError("Text to synthesize must be provided for GPT4o TTS")

        destination = Path(output_path) if output_path is not None else Path("gpt4o_tts_output.mp3")

        try:
            response = await self._client.audio.speech.create(
                model=model,
                voice=self.voice,
                input=text,
                format=self.audio_format,
            )

            audio_bytes = await self._gather_audio_bytes(response)
        except Exception as exc:  # noqa: BLE001 - propagate meaningful errors to caller
            logger.error("GPT4o TTS synthesis error: %s", exc, exc_info=True)
            raise

        if not audio_bytes:
            message = "GPT4o TTS response did not contain audio data."
            logger.error(message)
            raise RuntimeError(message)

        await asyncio.to_thread(destination.write_bytes, audio_bytes)
        logger.debug("GPT4o TTS synthesis complete. Audio saved to %s", destination)
        return str(destination)

    async def _gather_audio_bytes(self, response: Any) -> bytes:
        """Extract raw audio bytes from a variety of GPT-4o TTS response shapes."""

        if hasattr(response, "iter_bytes") and callable(response.iter_bytes):
            iterator = response.iter_bytes()
            if hasattr(iterator, "__aiter__"):
                chunks = [bytes(chunk) async for chunk in iterator if chunk]
            else:
                chunks = [bytes(chunk) for chunk in iterator if chunk]
            return b"".join(chunks)

        if hasattr(response, "model_dump"):
            try:
                return self._collect_audio_chunks(response.model_dump())
            except Exception:  # noqa: BLE001 - fallback to other strategies
                logger.debug("Failed to parse model_dump response for GPT4o TTS", exc_info=True)

        if hasattr(response, "to_dict"):
            try:
                return self._collect_audio_chunks(response.to_dict())
            except Exception:  # noqa: BLE001
                logger.debug("Failed to parse dict response for GPT4o TTS", exc_info=True)

        return self._collect_audio_chunks(response)

    def _collect_audio_chunks(self, payload: Any) -> bytes:
        chunks = list(self._iterate_audio_payloads(payload))
        return b"".join(chunks)

    def _iterate_audio_payloads(self, payload: Any, *, _visited: Optional[set[int]] = None) -> Iterable[bytes]:
        if payload is None:
            return

        if isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
            if data:
                yield data
            return

        if isinstance(payload, str):
            decoded = self._decode_base64(payload)
            if decoded:
                yield decoded
            return

        if isinstance(payload, (list, tuple, set)):
            for item in payload:
                yield from self._iterate_audio_payloads(item, _visited=_visited)
            return

        if _visited is None:
            _visited = set()

        obj_id = id(payload)
        if obj_id in _visited:
            return
        _visited.add(obj_id)

        if hasattr(payload, "model_dump"):
            try:
                yield from self._iterate_audio_payloads(payload.model_dump(), _visited=_visited)
                return
            except Exception:  # noqa: BLE001
                logger.debug("Failed to walk model_dump payload for GPT4o TTS", exc_info=True)

        if hasattr(payload, "to_dict"):
            try:
                yield from self._iterate_audio_payloads(payload.to_dict(), _visited=_visited)
                return
            except Exception:  # noqa: BLE001
                logger.debug("Failed to walk dict payload for GPT4o TTS", exc_info=True)

        if isinstance(payload, dict):
            for key in ("b64_json", "base64", "content", "data", "audio", "value", "body"):
                if key in payload:
                    yield from self._iterate_audio_payloads(payload[key], _visited=_visited)
            return

        for attr in ("b64_json", "base64", "content", "data", "audio", "value", "body"):
            if hasattr(payload, attr):
                yield from self._iterate_audio_payloads(getattr(payload, attr), _visited=_visited)

    @staticmethod
    def _decode_base64(value: Any) -> bytes:
        if isinstance(value, (bytes, bytearray)):
            candidate = bytes(value)
        elif isinstance(value, str):
            candidate = value.encode("utf-8")
        else:
            return b""

        candidate = candidate.strip()
        if not candidate:
            return b""

        try:
            return base64.b64decode(candidate, validate=True)
        except Exception:  # noqa: BLE001 - fall back to permissive decode
            try:
                return base64.b64decode(candidate)
            except Exception:  # noqa: BLE001 - non-base64 content
                return b""
