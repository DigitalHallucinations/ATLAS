"""Utilities for tracking completion of the ATLAS setup process."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MARKER_FILENAME = "setup_complete.json"


def _default_marker_path() -> Path:
    return Path(__file__).resolve().parent / "config" / DEFAULT_MARKER_FILENAME


def get_setup_marker_path() -> Path:
    """Return the configured path for the setup completion marker."""

    override = os.environ.get("ATLAS_SETUP_MARKER")
    if override:
        return Path(override).expanduser()
    return _default_marker_path()


def is_setup_complete(marker_path: Path | None = None) -> bool:
    """Return ``True`` if the setup completion marker exists."""

    path = marker_path or get_setup_marker_path()
    return path.is_file()


def write_setup_marker(
    data: Mapping[str, Any] | None = None,
    *,
    marker_path: Path | None = None,
) -> Path:
    """Persist the setup completion marker.

    Args:
        data: Optional JSON-serializable payload to store in the marker file.
        marker_path: Optional override for the marker path.

    Returns:
        Path: The path of the written marker file.
    """

    path = marker_path or get_setup_marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data if data is not None else {"setup_complete": True}
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


__all__ = [
    "DEFAULT_MARKER_FILENAME",
    "get_setup_marker_path",
    "is_setup_complete",
    "write_setup_marker",
]
