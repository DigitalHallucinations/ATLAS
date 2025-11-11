"""Shared helpers for manifest loader modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


__all__ = ["ensure_yaml_stub", "get_manifest_logger", "resolve_app_root"]


def ensure_yaml_stub() -> None:
    """Provide a minimal stub for :mod:`yaml` when the dependency is absent."""

    if "yaml" not in sys.modules:
        sys.modules["yaml"] = SimpleNamespace(
            safe_load=lambda *_args, **_kwargs: {},
            dump=lambda *_args, **_kwargs: None,
        )


ensure_yaml_stub()


from modules.logging.logger import setup_logger

try:  # ConfigManager may not be available in certain test scenarios
    from ATLAS.config import ConfigManager as _ConfigManager
except Exception:  # pragma: no cover - defensive import guard
    _ConfigManager = None  # type: ignore


def get_manifest_logger(name: str) -> logging.Logger:
    """Return a logger configured for manifest loader modules."""

    ensure_yaml_stub()
    return setup_logger(name)


def resolve_app_root(config_manager=None, *, logger: Optional[logging.Logger] = None) -> Path:
    """Resolve the application root using :class:`ConfigManager` fallbacks."""

    ensure_yaml_stub()
    active_logger = logger or get_manifest_logger(__name__)

    if config_manager is not None:
        getter = getattr(config_manager, "get_app_root", None)
        if callable(getter):
            try:
                root = getter()
                candidate = _validate_app_root(root)
                if candidate is not None:
                    return candidate
            except Exception:  # pragma: no cover - defensive guard
                active_logger.warning(
                    "Failed to resolve app root from supplied config manager", exc_info=True
                )

    if _ConfigManager is not None:
        try:
            manager = config_manager or _ConfigManager()
            getter = getattr(manager, "get_app_root", None)
            if callable(getter):
                candidate = _validate_app_root(getter())
                if candidate is not None:
                    return candidate
        except Exception:  # pragma: no cover - defensive guard
            active_logger.warning("Unable to resolve app root via ConfigManager", exc_info=True)

    fallback = Path(__file__).resolve().parents[2]
    active_logger.debug("Falling back to computed app root at %s", fallback)
    return fallback


def _validate_app_root(root: Optional[str]) -> Optional[Path]:
    if not root:
        return None

    candidate = Path(root).expanduser().resolve()
    modules_dir = candidate / "modules"
    if modules_dir.exists():
        return candidate
    return None
