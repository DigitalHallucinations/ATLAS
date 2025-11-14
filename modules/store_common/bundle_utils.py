"""Shared helpers for signed archive bundles used across asset types."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Tuple, Type


BUNDLE_ALGORITHM = "HS256"


def utcnow_isoformat() -> str:
    """Return the current UTC timestamp formatted for bundle metadata."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return canonical JSON bytes for ``payload`` suitable for signing."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def ensure_signing_key(signing_key: str, *, error_cls: Type[Exception]) -> bytes:
    """Validate and normalize the provided ``signing_key`` for bundle usage."""

    key = (signing_key or "").encode("utf-8")
    if not key:
        raise error_cls("Signing key is required for bundle operations.")
    return key


def sign_payload(
    payload: Mapping[str, Any], *, signing_key: str, error_cls: Type[Exception]
) -> str:
    """Return an HMAC signature for ``payload`` using ``signing_key``."""

    key = ensure_signing_key(signing_key, error_cls=error_cls)
    digest = hmac.new(key, canonical_json_bytes(payload), hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def verify_signature(
    payload: Mapping[str, Any], *, signature: str, signing_key: str, error_cls: Type[Exception]
) -> None:
    """Verify that ``signature`` matches ``payload`` for ``signing_key``."""

    expected = sign_payload(payload, signing_key=signing_key, error_cls=error_cls)
    if not hmac.compare_digest(expected, signature):
        raise error_cls("Bundle signature verification failed.")


def build_archive_payload(
    directory: Path,
    *,
    error_cls: Type[Exception],
    arcname: Optional[str] = None,
) -> Mapping[str, Any]:
    """Encode ``directory`` as a base64 ``tar.gz`` archive payload."""

    if not directory.exists() or not directory.is_dir():
        raise error_cls("Bundle source directory is missing for export.")

    buffer = io.BytesIO()
    try:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(directory, arcname=arcname or directory.name)
    except (OSError, tarfile.TarError) as exc:
        raise error_cls("Failed to build bundle archive for export.") from exc

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "format": "tar.gz",
        "encoding": "base64",
        "data": encoded,
    }


def decode_archive_payload(
    archive_payload: Mapping[str, Any], *, error_cls: Type[Exception]
) -> Tuple[bytes, str]:
    """Decode ``archive_payload`` returning raw archive bytes and format."""

    format_name = str(archive_payload.get("format") or "tar.gz").strip().lower()
    encoding = str(archive_payload.get("encoding") or "base64").strip().lower()
    data = archive_payload.get("data")

    if encoding not in {"base64", "base64url"}:
        raise error_cls("Unsupported bundle archive encoding.")
    if format_name not in {"tar", "tar.gz", "tgz"}:
        raise error_cls("Unsupported bundle archive format.")
    if not isinstance(data, str) or not data.strip():
        raise error_cls("Bundle archive data is missing.")

    try:
        if encoding == "base64url":
            archive_bytes = base64.urlsafe_b64decode(data.encode("ascii"))
        else:
            archive_bytes = base64.b64decode(data.encode("ascii"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise error_cls("Bundle archive data is not valid base64.") from exc

    return archive_bytes, format_name


def safe_extract_tar_archive(
    tar: tarfile.TarFile, destination: Path, *, error_cls: Type[Exception]
) -> None:
    """Extract ``tar`` into ``destination`` with basic security checks."""

    destination = destination.resolve()
    for member in tar.getmembers():
        name = member.name or ""
        member_path = Path(name)
        if member_path.is_absolute() or any(part == ".." for part in member_path.parts):
            raise error_cls("Bundle archive contains unsafe paths.")
        if member.islnk() or member.issym():
            raise error_cls("Bundle archive contains unsupported links.")
        resolved_member = (destination / member_path).resolve()
        if not str(resolved_member).startswith(str(destination)):
            raise error_cls("Bundle archive resolves outside the extraction directory.")

    tar.extractall(path=destination)


def extract_archive_to_tempdir(
    archive_payload: Mapping[str, Any], *, error_cls: Type[Exception]
) -> Tuple[tempfile.TemporaryDirectory, Path]:
    """Decode and extract ``archive_payload`` into a temporary directory."""

    archive_bytes, format_name = decode_archive_payload(archive_payload, error_cls=error_cls)
    mode = "r:gz" if format_name in {"tar.gz", "tgz"} else "r:"

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    buffer = io.BytesIO(archive_bytes)
    try:
        with tarfile.open(fileobj=buffer, mode=mode) as archive:
            safe_extract_tar_archive(archive, temp_path, error_cls=error_cls)
    except (OSError, tarfile.TarError) as exc:
        temp_dir.cleanup()
        raise error_cls("Bundle archive could not be unpacked.") from exc

    return temp_dir, temp_path


def normalize_archive_payload(payload: MutableMapping[str, Any]) -> None:
    """Ensure ``payload`` only contains supported archive keys."""

    allowed_keys = {"format", "encoding", "data"}
    for key in list(payload.keys()):
        if key not in allowed_keys:
            payload.pop(key, None)

