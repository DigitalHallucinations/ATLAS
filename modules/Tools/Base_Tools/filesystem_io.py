"""Filesystem sandbox utilities for tool usage.

This module exposes asynchronous helpers for reading, writing, and listing
files within a sandboxed directory. The sandbox root is configurable via
the ``ATLAS_FILESYSTEM_SANDBOX`` environment variable and enforces several
safety constraints:

* Path traversal attempts are rejected.
* File size and directory listing quotas are respected.
* Operations are executed within a configurable timeout budget.
* MIME detection is performed using :mod:`mimetypes`.

The public functions return JSON-serializable dictionaries to simplify tool
integration.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

__all__ = [
    "FilesystemError",
    "SandboxViolationError",
    "QuotaExceededError",
    "FilesystemTimeoutError",
    "read_file",
    "write_file",
    "list_dir",
]


_DEFAULT_SANDBOX_ROOT = Path(__file__).resolve().parents[3]
_ENV_ROOT = "ATLAS_FILESYSTEM_SANDBOX"
_ENV_MAX_READ = "ATLAS_FILESYSTEM_MAX_READ_BYTES"
_ENV_MAX_WRITE = "ATLAS_FILESYSTEM_MAX_WRITE_BYTES"
_ENV_MAX_LIST = "ATLAS_FILESYSTEM_MAX_LIST_ENTRIES"
_ENV_TIMEOUT = "ATLAS_FILESYSTEM_OPERATION_TIMEOUT"
_ENV_MAX_TOTAL = "ATLAS_FILESYSTEM_MAX_TOTAL_BYTES"

_DEFAULT_MAX_READ_BYTES = 256 * 1024
_DEFAULT_MAX_WRITE_BYTES = 256 * 1024
_DEFAULT_MAX_LIST_ENTRIES = 512
_DEFAULT_OPERATION_TIMEOUT = 2.0
_DEFAULT_MAX_TOTAL_BYTES = 10 * 1024 * 1024


class FilesystemError(RuntimeError):
    """Base class for filesystem sandbox failures."""


class SandboxViolationError(FilesystemError):
    """Raised when a path escapes or invalidates the sandbox."""


class QuotaExceededError(FilesystemError):
    """Raised when a configured quota is violated."""


class FilesystemTimeoutError(FilesystemError):
    """Raised when an operation exceeds the allotted timeout."""


@dataclass(frozen=True)
class _SandboxConfig:
    root: Path
    max_read_bytes: int
    max_write_bytes: int
    max_list_entries: int
    operation_timeout: float
    max_total_bytes: int


def _read_int_env(name: str, default: int, minimum: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise QuotaExceededError(f"Environment variable {name} must be an integer.") from exc
    if parsed < minimum:
        raise QuotaExceededError(
            f"Environment variable {name} must be at least {minimum}, got {parsed}."
        )
    return parsed


def _read_float_env(name: str, default: float, minimum: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise QuotaExceededError(f"Environment variable {name} must be a number.") from exc
    if parsed < minimum:
        raise QuotaExceededError(
            f"Environment variable {name} must be at least {minimum}, got {parsed}."
        )
    return parsed


def _current_config() -> _SandboxConfig:
    root_candidate = os.environ.get(_ENV_ROOT)
    if root_candidate:
        try:
            root = Path(root_candidate).expanduser().resolve()
        except OSError as exc:
            raise SandboxViolationError(
                f"Unable to resolve sandbox root '{root_candidate}'."
            ) from exc
    else:
        root = _DEFAULT_SANDBOX_ROOT

    if not root.exists() or not root.is_dir():
        raise SandboxViolationError(
            f"Sandbox root '{root}' does not exist or is not a directory."
        )

    max_read_bytes = _read_int_env(_ENV_MAX_READ, _DEFAULT_MAX_READ_BYTES, 1)
    max_write_bytes = _read_int_env(_ENV_MAX_WRITE, _DEFAULT_MAX_WRITE_BYTES, 1)
    max_list_entries = _read_int_env(_ENV_MAX_LIST, _DEFAULT_MAX_LIST_ENTRIES, 1)
    operation_timeout = _read_float_env(_ENV_TIMEOUT, _DEFAULT_OPERATION_TIMEOUT, 0.1)
    max_total_bytes = _read_int_env(_ENV_MAX_TOTAL, _DEFAULT_MAX_TOTAL_BYTES, 1)

    return _SandboxConfig(
        root=root,
        max_read_bytes=max_read_bytes,
        max_write_bytes=max_write_bytes,
        max_list_entries=max_list_entries,
        operation_timeout=operation_timeout,
        max_total_bytes=max_total_bytes,
    )


def _resolve_path(path: str, *, allow_missing: bool = False) -> Path:
    config = _current_config()
    candidate = Path(path)
    target = candidate if candidate.is_absolute() else config.root.joinpath(candidate)
    try:
        resolved = target.resolve(strict=not allow_missing)
    except FileNotFoundError:
        raise SandboxViolationError(f"Path '{path}' does not exist within sandbox.")
    except RuntimeError as exc:  # pragma: no cover - defensive guard for symlink loops
        raise SandboxViolationError("Unable to resolve requested path.") from exc

    try:
        resolved.relative_to(config.root)
    except ValueError as exc:
        raise SandboxViolationError(
            f"Path '{path}' escapes sandbox root '{config.root}'."
        ) from exc
    return resolved


def _enforce_total_quota(
    config: _SandboxConfig, additional_bytes: int, *, previous_size: int = 0
) -> None:
    total = 0
    for entry in config.root.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:  # pragma: no cover - skip unreadable files
                continue
        if total > config.max_total_bytes:
            break
    adjusted_total = max(0, total - previous_size)
    if adjusted_total + additional_bytes > config.max_total_bytes:
        raise QuotaExceededError(
            "Writing the requested content would exceed the sandbox total size quota."
        )


async def _run_with_timeout(callable_obj, config: _SandboxConfig):
    try:
        return await asyncio.wait_for(asyncio.to_thread(callable_obj), timeout=config.operation_timeout)
    except asyncio.TimeoutError as exc:
        raise FilesystemTimeoutError("Filesystem operation exceeded timeout quota.") from exc


def _detect_mime(path: Path) -> tuple[str | None, str | None]:
    mimetype, encoding = mimetypes.guess_type(path.name)
    return mimetype, encoding


def _encode_payload(data: bytes, mimetype: str | None) -> tuple[str, str]:
    if mimetype and mimetype.startswith("text/"):
        try:
            return data.decode("utf-8"), "text"
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace"), "text"
    return base64.b64encode(data).decode("ascii"), "base64"


async def read_file(path: str, *, max_bytes: int | None = None) -> dict[str, object]:
    """Read a file from the sandbox respecting size quotas."""

    config = _current_config()
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise SandboxViolationError(f"Path '{path}' is not a file.")

    size = resolved.stat().st_size
    limit = config.max_read_bytes if max_bytes is None else min(config.max_read_bytes, max(1, max_bytes))

    def _reader() -> tuple[bytes, bool]:
        with resolved.open("rb") as handle:
            data = handle.read(limit)
        truncated = size > len(data)
        return data, truncated

    data, truncated = await _run_with_timeout(_reader, config)
    mimetype, encoding = _detect_mime(resolved)
    payload, content_encoding = _encode_payload(data, mimetype)

    return {
        "path": str(resolved.relative_to(config.root)),
        "size": size,
        "content": payload,
        "content_encoding": content_encoding,
        "mime_type": mimetype,
        "mime_encoding": encoding,
        "truncated": truncated,
    }


async def write_file(
    path: str,
    content: str,
    *,
    mode: Literal["text", "base64"] = "text",
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> dict[str, object]:
    """Write content to a file within the sandbox respecting quotas."""

    config = _current_config()
    resolved = _resolve_path(path, allow_missing=True)
    parent = resolved.parent
    if not parent.exists():
        raise SandboxViolationError(
            f"Parent directory '{parent.relative_to(config.root)}' does not exist."
        )
    if not parent.is_dir():
        raise SandboxViolationError("Parent path is not a directory.")

    if resolved.exists() and not overwrite:
        raise SandboxViolationError("Refusing to overwrite existing file without overwrite=True.")

    existing_size = resolved.stat().st_size if resolved.exists() else 0

    if mode == "base64":
        try:
            payload = base64.b64decode(content, validate=True)
        except (ValueError, binascii.Error) as exc:  # pragma: no cover - defensive guard
            raise FilesystemError("Provided base64 payload is invalid.") from exc
    elif mode == "text":
        payload = content.encode(encoding)
    else:  # pragma: no cover - defensive guard
        raise FilesystemError(f"Unsupported mode '{mode}'.")

    if len(payload) > config.max_write_bytes:
        raise QuotaExceededError("Content exceeds per-operation write quota.")

    _enforce_total_quota(config, len(payload), previous_size=existing_size)

    def _writer() -> dict[str, object]:
        with resolved.open("wb") as handle:
            handle.write(payload)
        stat = resolved.stat()
        mimetype, mime_encoding = _detect_mime(resolved)
        return {
            "path": str(resolved.relative_to(config.root)),
            "size": stat.st_size,
            "mime_type": mimetype,
            "mime_encoding": mime_encoding,
        }

    return await _run_with_timeout(_writer, config)


async def list_dir(
    path: str | None = None,
    *,
    include_hidden: bool = False,
    max_entries: int | None = None,
) -> dict[str, Sequence[dict[str, object]]]:
    """List directory contents within the sandbox."""

    config = _current_config()
    resolved = _resolve_path(path or ".")
    if not resolved.is_dir():
        raise SandboxViolationError(f"Path '{path or '.'}' is not a directory.")

    limit = config.max_list_entries
    if max_entries is not None:
        limit = min(limit, max(1, max_entries))

    def _collector() -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for entry in sorted(resolved.iterdir(), key=lambda p: p.name.lower()):
            if not include_hidden and entry.name.startswith("."):
                continue
            mimetype, mime_encoding = _detect_mime(entry)
            entry_info = {
                "name": entry.name,
                "path": str(entry.relative_to(config.root)),
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else None,
                "mime_type": mimetype if entry.is_file() else None,
                "mime_encoding": mime_encoding if entry.is_file() else None,
            }
            entries.append(entry_info)
            if len(entries) > limit:
                raise QuotaExceededError("Directory listing exceeds configured quota.")
        return entries

    entries = await _run_with_timeout(_collector, config)
    return {"path": str(resolved.relative_to(config.root)), "entries": entries}
