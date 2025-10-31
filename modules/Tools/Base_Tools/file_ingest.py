"""Filesystem ingestion helper that captures lightweight metadata previews."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from modules.logging.logger import setup_logger

__all__ = ["FileIngestor", "FileIngestError"]


logger = setup_logger(__name__)

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_SAMPLE_BYTES = 2048


class FileIngestError(RuntimeError):
    """Raised when a file cannot be ingested safely."""


@dataclass(frozen=True)
class FilePreview:
    encoding: str
    content: str
    size: int


class FileIngestor:
    """Read files from disk with guardrails around size and encoding."""

    def __init__(
        self,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        sample_bytes: int = _DEFAULT_SAMPLE_BYTES,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if sample_bytes <= 0:
            raise ValueError("sample_bytes must be positive")
        self._max_bytes = max_bytes
        self._sample_bytes = sample_bytes

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser().resolve()
        if not candidate.exists():
            raise FileIngestError(f"File '{path}' does not exist")
        if not candidate.is_file():
            raise FileIngestError(f"Path '{path}' is not a file")
        size = candidate.stat().st_size
        if size > self._max_bytes:
            raise FileIngestError(
                f"File '{candidate}' exceeds max allowed size of {self._max_bytes} bytes"
            )
        return candidate

    async def run(self, *, path: str, as_text: bool | None = None) -> Mapping[str, object]:
        """Return a structured preview of the requested file."""

        target = self._resolve(path)
        mime_type, _ = mimetypes.guess_type(target.name)
        file_bytes = await asyncio.to_thread(target.read_bytes)
        digest = hashlib.sha256(file_bytes).hexdigest()
        sample = file_bytes[: self._sample_bytes]

        preview = self._build_preview(sample, as_text=as_text, mime_type=mime_type)

        payload = {
            "path": str(target),
            "size_bytes": len(file_bytes),
            "sha256": digest,
            "mime_type": mime_type,
            "preview": preview.__dict__,
        }
        logger.info("Ingested %s (%d bytes)", target, payload["size_bytes"])
        return payload

    def _build_preview(
        self,
        sample: bytes,
        *,
        as_text: bool | None,
        mime_type: str | None,
    ) -> FilePreview:
        if not sample:
            return FilePreview(encoding="text", content="", size=0)

        if as_text is True or (as_text is None and mime_type and mime_type.startswith("text/")):
            content = sample.decode("utf-8", errors="replace")
            encoding = "text"
        else:
            content = base64.b64encode(sample).decode("ascii")
            encoding = "base64"
        return FilePreview(encoding=encoding, content=content, size=len(sample))
